"""
result_viewer.py - "Camera | Depth Map" rebuilt from the Nexys4 DDR FPGA's result packets.

All processing (U-Net, objects, ToF distance per object) happens on the FPGA; this script only
draws what arrives over the USB-UART (2 Mbaud), plus the same temporal smoothing / hold as
the original PC pipeline (that part stays on the PC).

    live (board connected, the Nexys4 DDR USB serial port, e.g. COM3):
        python sw\\result_viewer.py --port COM5
    replay a stream captured by the simulation (no board needed):
        python sw\\result_viewer.py --replay golden_model\\data\\rtl_simulation\\uart_capture.bin

Keys: l labels   z ToF zones   s snapshot   q quit
SW0 on the board = include the camera image in each packet (about 8 fps); SW0 off = results only
(faster), the left side then shows the object map in grey.
"""
import argparse, os, sys, threading, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                                                    # sw/ (outputs: logs, captures, snapshots)
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "hw", "tools"))
import fpga_model as fm

PW, PH = 640, 480
WIN = "FPGA: Camera | Depth Map"
DMIN, DMAX = 200.0, 3000.0
HOLD_S = 3.0


def depth_color(d):
    v = np.clip((d - DMIN) / (DMAX - DMIN), 0, 1)
    return tuple(int(c) for c in cv2.applyColorMap(np.array([[255 - v * 255]], np.uint8), cv2.COLORMAP_JET)[0, 0])


class PacketReader(threading.Thread):
    """Finds A5 5A 46 50, parses packets with fpga_model.parse_uart_packet; keeps the newest."""
    def __init__(self, src, replay_fps=0.0):
        super().__init__(daemon=True)
        self.src, self.replay_fps = src, replay_fps
        self.lock = threading.Lock()
        self.latest = None; self.seq = 0
        self.n_ok = self.n_bad = 0; self.running = True; self.alive = True; self.eof = False

    def run(self):
        buf = bytearray()
        try:
            while self.running:
                chunk = self.src.read(65536 if self.replay_fps else 4096)
                if not chunk:
                    if self.replay_fps:
                        self.eof = True
                        break
                    continue
                buf += chunk
                while True:
                    i = buf.find(fm.UART_SYNC)
                    if i < 0:
                        del buf[:-3]
                        break
                    if i:
                        del buf[:i]
                    try:
                        pkt, n = fm.parse_uart_packet(bytes(buf))
                    except ValueError:
                        self.n_bad += 1
                        del buf[:4]
                        continue
                    if pkt is None:
                        break
                    del buf[:n]
                    self.n_ok += 1
                    with self.lock:
                        self.latest = pkt; self.seq += 1
                    if self.replay_fps:
                        time.sleep(1.0 / self.replay_fps)
        except Exception as e:
            print("[reader]", e)
        self.alive = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="Nexys4 DDR USB serial port, e.g. COM3")
    ap.add_argument("--baud", type=int, default=2_000_000)
    ap.add_argument("--replay", default=None, help="binary file of captured UART bytes")
    ap.add_argument("--replay-fps", type=float, default=2.0)
    ap.add_argument("--zones", default=os.path.join(REPO, "hw", "mem", "zone_tables.npz"))
    a = ap.parse_args()

    if a.replay:
        src = open(a.replay, "rb"); fps_replay = a.replay_fps
    else:
        if not a.port:
            sys.exit("give --port COMx (the Nexys4 DDR USB serial port) or --replay file")
        import serial
        src = serial.Serial(a.port, a.baud, timeout=0.05)
        src.reset_input_buffer(); fps_replay = 0.0
    rd = PacketReader(src, fps_replay); rd.start()
    zt = fm.load_zone_tables(a.zones) if os.path.exists(a.zones) else None

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    print("keys: l labels | z ToF zones | s snapshot | q quit")
    labels, show_zones = True, False
    last_img = None; seq = -1; pkt = None
    tracks = []
    n_frames = 0; fps = 0.0; t0 = time.perf_counter(); results = []
    try:
        while True:
            with rd.lock:
                p, s = rd.latest, rd.seq
            if p is not None and s != seq:
                seq, pkt = s, p; n_frames += 1
                if pkt["image"] is not None:
                    last_img = pkt["image"]
                now = time.perf_counter()
                # ---- temporal smoothing / hold, exactly as the original PC pipeline ----
                results = []
                km = pkt["kmap"]
                for o in pkt["objs"]:
                    mask = km == o.k
                    d = float(o.dist) if o.dist else None
                    src_ = "tof" if d is not None else ""
                    names = [fm.CLASSES[o.cls]]; t_meas = now if d is not None else None
                    for (pm, pd, pnames, pt) in tracks:
                        inter = np.logical_and(pm, mask).sum(); uni = np.logical_or(pm, mask).sum()
                        if uni and inter / uni > 0.4:
                            if pd is not None and d is not None and abs(d - pd) < 0.3 * pd:
                                d = 0.6 * pd + 0.4 * d
                            elif d is None and pd is not None and pt is not None and now - pt < HOLD_S:
                                d, src_, t_meas = pd, "hold", pt
                            names = (pnames + names)[-10:]
                            break
                    results.append((o, mask, d, src_, names, t_meas))
                tracks = [(m, d, nm, tm) for o, m, d, s_, nm, tm in results]
            now = time.perf_counter()
            if now - t0 >= 1:
                fps = n_frames / (now - t0); n_frames = 0; t0 = now

            # ---- draw ----
            if pkt is None:
                left = np.full((PH, PW, 3), 30, np.uint8); right = np.zeros((PH, PW, 3), np.uint8)
                msg = f"waiting for FPGA packets ({a.port or a.replay})"
                cv2.putText(left, msg, (20, PH // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                if last_img is not None:
                    left = cv2.resize(cv2.cvtColor(last_img, cv2.COLOR_GRAY2BGR), (PW, PH), interpolation=cv2.INTER_LINEAR)
                else:
                    g = (pkt["kmap"] > 0).astype(np.uint8) * 120 + 30
                    left = cv2.resize(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), (PW, PH), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(left, "SW0 off: no camera image", (16, PH - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
                right = np.zeros((PH, PW, 3), np.uint8)
                lab = []
                for o, mask, d, src_, names in [(r[0], r[1], r[2], r[3], r[4]) for r in
                                                sorted(results, key=lambda t: -t[0].area)]:
                    col = depth_color(d) if d is not None else (110, 110, 110)
                    m = cv2.resize(mask.astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST).astype(bool)
                    right[m] = col
                    ys, xs = np.nonzero(m)
                    if xs.size:
                        name = max(set(names), key=names.count)
                        txt = name + (" ?" if d is None else f" {'~' if src_ == 'hold' else ''}{d / 1000:.2f} m")
                        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        lab.append((txt, (int(xs.mean()) - tw // 2, max(th + 2, int(ys.min()) - 6))))
                if labels:
                    for txt, (x, y) in lab:
                        x = min(max(0, x), PW - 10)
                        cv2.putText(right, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                        cv2.putText(right, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                if show_zones and zt is not None:
                    ok = fm.tof_valid(pkt["tof_dist"], pkt["tof_status"])
                    sx, sy = PW / fm.LAB_W, PH / fm.LAB_H
                    for i in range(64):
                        if not ok[i]:
                            continue
                        d = int(pkt["tof_dist"][i])
                        x0, y0, x1, y1, v = zt.rects[i, fm.depth_bin(d, zt)]
                        if v:
                            cv2.rectangle(left, (int(x0 * sx), int(y0 * sy)), (int((x1 + 1) * sx) - 1, int((y1 + 1) * sy) - 1),
                                          depth_color(d), 1)
                            cv2.putText(left, f"{d / 1000:.1f}", (int(x0 * sx) + 2, int(y0 * sy) + 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            for img, t in ((left, "Camera"), (right, "Depth Map")):
                cv2.putText(img, t, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
                cv2.putText(img, t, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            body = np.hstack([left, np.zeros((PH, 2, 3), np.uint8), right])
            if pkt is not None:
                f = pkt["flags"]
                status = (f"FPGA {fps:.1f} fps | camera #{pkt['cam_no']} | ToF #{pkt['tof_no']}"
                          f"{'' if f & 2 else ' (no ToF yet)'} | U-Net {pkt['infer_cycles'] / 1e5:.1f} ms | "
                          f"objects {pkt['post_cycles'] / 1e5:.1f} ms | {len(pkt['objs'])} objects"
                          f"{' | >63 objects!' if f & 4 else ''}{' | CC not converged' if f & 8 else ''}"
                          f" | packets {rd.n_ok} ok, {rd.n_bad} bad")
            else:
                status = "starting"
            bar = np.zeros((24, body.shape[1], 3), np.uint8)
            cv2.putText(bar, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 0), 1)
            frame = np.vstack([bar, body])
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("l"):
                labels = not labels
            elif key == ord("z"):
                show_zones = not show_zones
            elif key == ord("s"):
                os.makedirs(os.path.join(ROOT, "snapshots"), exist_ok=True)
                fn = os.path.join(ROOT, "snapshots", time.strftime("fpga_view_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(fn, frame); print("saved", fn)
            if not rd.alive and not a.replay:
                print("[error] serial reader stopped - is the port right and free?"); break
            if a.replay and rd.eof and rd.latest is not None and cv2.getWindowProperty(WIN, 0) < 0:
                break
    finally:
        rd.running = False
        cv2.destroyAllWindows()
        try:
            src.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
