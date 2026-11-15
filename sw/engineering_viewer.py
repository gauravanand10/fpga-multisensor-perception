"""
engineering_viewer.py - "Camera | Depth Map" + IMU / servo panel, from the Nexys4 DDR FPGA's UART stream.

Extends sw\\result_viewer.py (same depth-map drawing, same smoothing / hold) plus:
  * IMU packets (A5 5A 49 4D) between the result packets: 2 IMUs x (accel + gyro) = 12 values,
    shown in g and deg/s, ok flags, WHO_AM_I, small scrolling plots
  * servo mode and pulse width / angle (target and now)
  * CSV logging of every IMU sample (key c)

    live (board connected, the Nexys4 DDR USB serial port, e.g. COM3):
        python sw\\engineering_viewer.py --port COM3
    replay a stream captured by the simulation (no board needed):
        python sw\\engineering_viewer.py --replay golden_model\\data\\rtl_simulation\\uart_capture.bin
    replay without a window, save the last view as a PNG (used for the checks):
        ... --replay golden_model\\data\\rtl_simulation\\uart_capture.bin --headless-out build\\viewer_replay.png --no-image

Keys: l labels   z ToF zones   c CSV log on/off   s snapshot   q quit
SW0 on the board = include the camera image in each packet; SW1 = servo sweep; SW[15:8] = servo angle.
--no-image never draws the camera picture (object map in grey instead), e.g. for figures.
"""
import argparse, collections, csv, os, sys, threading, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                                                    # sw/ (outputs: logs, captures, snapshots)
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "hw", "tools"))
import fpga_model as fm

PW, PH = 640, 480
IMU_H = 236
WIN = "FPGA: Camera | Depth Map | IMU"
DMIN, DMAX = 200.0, 3000.0
HOLD_S = 3.0
PLOT_N = 500                      # samples in the scrolling plots (5 s at 100 Hz)
WHO_NAMES = {0x70: "MPU-6500", 0x71: "MPU-9250", 0x73: "MPU-9255", 0x68: "MPU-6000", 0xFF: "no answer", 0x00: "-"}


def depth_color(d):
    v = np.clip((d - DMIN) / (DMAX - DMIN), 0, 1)
    return tuple(int(c) for c in cv2.applyColorMap(np.array([[255 - v * 255]], np.uint8), cv2.COLORMAP_JET)[0, 0])


class StreamReader(threading.Thread):
    """Finds result packets (A5 5A 46 50) and IMU packets (A5 5A 49 4D); keeps the newest result,
    the newest IMU status and the last PLOT_N IMU samples."""
    def __init__(self, src, replay_fps=0.0):
        super().__init__(daemon=True)
        self.src, self.replay_fps = src, replay_fps
        self.lock = threading.Lock()
        self.latest = None; self.seq = 0
        self.n_ok = self.n_bad = 0
        self.imu = None; self.imu_seq = 0
        self.n_imu_ok = self.n_imu_bad = 0
        self.n_smp = 0; self.n_lost = 0; self.n_ovf = 0; self.last_sno = None
        self.samples = collections.deque(maxlen=PLOT_N)
        self.rate = 0.0; self._rt0 = time.perf_counter(); self._rn = 0
        self.csv = None
        self.running = True; self.alive = True; self.eof = False

    def _imu(self, p):
        now = time.perf_counter()
        with self.lock:
            self.imu = p; self.imu_seq += 1; self.n_imu_ok += 1
            if p["overflow"]:
                self.n_ovf += 1
            for sno, v in p["samples"]:
                if self.last_sno is not None:
                    self.n_lost += ((sno - self.last_sno) & 0xFFFF) - 1
                self.last_sno = sno
                self.samples.append((sno, v, p["imu_ok"]))
                self.n_smp += 1; self._rn += 1
                if self.csv is not None:
                    acc = [x / fm.ACC_LSB_PER_G for x in v]
                    row = [f"{now:.4f}", sno, p["flags"], p["who1"], p["who2"], p["servo_mode"],
                           p["servo_target_us"], p["servo_now_us"]] + list(v)
                    row += [f"{v[i] / fm.ACC_LSB_PER_G:.5f}" if (i % 6) < 3 else f"{v[i] / fm.GYRO_LSB_PER_DPS:.3f}"
                            for i in range(12)]
                    self.csv[1].writerow(row)
            if now - self._rt0 >= 1.0:
                self.rate = self._rn / (now - self._rt0); self._rn = 0; self._rt0 = now

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
                    i1, i2 = buf.find(fm.UART_SYNC), buf.find(fm.IMU_SYNC)
                    cand = [i for i in (i1, i2) if i >= 0]
                    if not cand:
                        del buf[:-3]
                        break
                    i = min(cand)
                    if i:
                        del buf[:i]
                    is_res = (i == i1)
                    try:
                        pkt, n = (fm.parse_uart_packet if is_res else fm.parse_imu_packet)(bytes(buf))
                    except ValueError:
                        if is_res:
                            self.n_bad += 1
                        else:
                            self.n_imu_bad += 1
                        del buf[:4]
                        continue
                    if pkt is None:
                        break
                    del buf[:n]
                    if is_res:
                        self.n_ok += 1
                        with self.lock:
                            self.latest = pkt; self.seq += 1
                        if self.replay_fps:
                            time.sleep(1.0 / self.replay_fps)
                    else:
                        self._imu(pkt)
        except Exception as e:
            print("[reader]", e)
        self.alive = False


# ---------------------------------------------------------------- IMU panel
COLS = [(90, 90, 255), (90, 220, 90), (255, 160, 60)]          # x red, y green, z blue (BGR)


def put(img, txt, x, y, col=(230, 230, 230), s=0.45, th=1):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def plot(img, x0, y0, w, h, series, title, unit, fixed=None, ok=True):
    cv2.rectangle(img, (x0, y0), (x0 + w - 1, y0 + h - 1), (70, 70, 70), 1)
    cv2.line(img, (x0, y0 + h // 2), (x0 + w - 1, y0 + h // 2), (55, 55, 55), 1)
    if series is not None and len(series):
        a = np.asarray(series, np.float64)
        lim = fixed if fixed else max(10.0, float(np.abs(a).max()) * 1.2)
        n = a.shape[0]
        xs = x0 + 1 + (np.arange(n) * (w - 3) / max(PLOT_N - 1, 1)).astype(np.int32) + (w - 3) * (PLOT_N - n) // PLOT_N
        for k in range(3):
            ys = (y0 + h / 2 - a[:, k] / lim * (h / 2 - 3)).astype(np.int32)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, COLS[k], 1, cv2.LINE_AA)
        put(img, f"+-{lim:g} {unit}", x0 + w - 92, y0 + h - 6, (150, 150, 150), 0.38)
    put(img, title, x0 + 4, y0 + 14, (220, 220, 220) if ok else (90, 90, 200), 0.42)


def imu_panel(width, rd):
    img = np.full((IMU_H, width, 3), 18, np.uint8)
    with rd.lock:
        p = rd.imu
        smp = list(rd.samples)
        stats = (rd.n_imu_ok, rd.n_imu_bad, rd.n_smp, rd.n_lost, rd.n_ovf, rd.rate, rd.csv)
    n_ok, n_bad, n_smp, n_lost, n_ovf, rate, log = stats
    if p is None:
        put(img, "waiting for IMU packets (A5 5A 49 4D) ...", 12, 30, (0, 200, 255), 0.6)
        return img
    last = smp[-1][1] if smp else [0] * 12
    y = 22
    for i in range(2):
        okb = bool(p["imu_ok"] >> i & 1)
        who = p["who1"] if i == 0 else p["who2"]
        name = WHO_NAMES.get(who, "unknown")
        put(img, f"IMU{i + 1}", 12, y, (255, 255, 255), 0.55, 2)
        put(img, ("OK" if okb else "MISSING") + f"   WHO_AM_I 0x{who:02X} ({name})", 70, y,
            (90, 230, 90) if okb else (80, 80, 255), 0.5, 1)
        v = last[6 * i:6 * i + 6]
        ax, ay, az = (q / fm.ACC_LSB_PER_G for q in v[:3])
        gx, gy, gz = (q / fm.GYRO_LSB_PER_DPS for q in v[3:])
        put(img, f"accel g    x {ax:+6.3f}  y {ay:+6.3f}  z {az:+6.3f}   |a| {np.sqrt(ax*ax + ay*ay + az*az):5.3f}", 24, y + 22)
        put(img, f"gyro deg/s x {gx:+7.2f} y {gy:+7.2f} z {gz:+7.2f}", 24, y + 42)
        y += 70
    tgt, now = p["servo_target_us"], p["servo_now_us"]
    put(img, f"servo {fm.SERVO_MODES.get(p['servo_mode'], p['servo_mode'])}: target {tgt} us "
             f"({fm.servo_us_to_deg(tgt):.1f} deg)  now {now} us ({fm.servo_us_to_deg(now):.1f} deg)", 12, y + 4,
        (0, 220, 255), 0.47)
    bx, bw = 12, 400
    cv2.rectangle(img, (bx, y + 14), (bx + bw, y + 24), (70, 70, 70), 1)
    fx = bx + int(np.clip((now - 1000) / 1000.0, 0, 1) * bw)
    tx = bx + int(np.clip((tgt - 1000) / 1000.0, 0, 1) * bw)
    cv2.rectangle(img, (bx + 1, y + 15), (fx, y + 23), (0, 160, 220), -1)
    cv2.line(img, (tx, y + 11), (tx, y + 27), (255, 255, 255), 1)
    put(img, f"IMU packets {n_ok} ok, {n_bad} bad | samples {n_smp} ({rate:.0f}/s), lost {n_lost}, "
             f"overflow {n_ovf} | CSV {'ON ' + os.path.basename(log[2]) if log else 'off (c)'}", 12, IMU_H - 10,
        (0, 255, 0), 0.42)
    # plots
    px = 440
    pw = (width - px - 8) // 4 - 6
    ph = IMU_H - 16
    arr = np.asarray([s[1] for s in smp], np.float64) if smp else None
    for j, (title, sl, scale, unit, fixed) in enumerate([
            ("IMU1 accel", slice(0, 3), fm.ACC_LSB_PER_G, "g", 2.0),
            ("IMU1 gyro", slice(3, 6), fm.GYRO_LSB_PER_DPS, "dps", None),
            ("IMU2 accel", slice(6, 9), fm.ACC_LSB_PER_G, "g", 2.0),
            ("IMU2 gyro", slice(9, 12), fm.GYRO_LSB_PER_DPS, "dps", None)]):
        okb = bool(p["imu_ok"] >> (j // 2) & 1)
        plot(img, px + j * (pw + 6), 8, pw, ph, None if arr is None else arr[:, sl] / scale,
             title + ("  (x y z)" if okb else "  missing"), unit, fixed, okb)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="Nexys4 DDR USB serial port, e.g. COM3")
    ap.add_argument("--baud", type=int, default=2_000_000)
    ap.add_argument("--replay", default=None, help="binary file of captured UART bytes")
    ap.add_argument("--replay-fps", type=float, default=2.0)
    ap.add_argument("--zones", default=os.path.join(REPO, "hw", "mem", "zone_tables.npz"))
    ap.add_argument("--no-image", action="store_true", help="never draw the camera picture")
    ap.add_argument("--csv", action="store_true", help="start CSV logging at once")
    ap.add_argument("--headless-out", default=None, help="replay only: no window, save the final view to this PNG")
    a = ap.parse_args()

    if a.replay:
        src = open(a.replay, "rb"); fps_replay = a.replay_fps if not a.headless_out else 1e9
    else:
        if not a.port:
            sys.exit("give --port COMx (the Nexys4 DDR USB serial port) or --replay file")
        import serial
        src = serial.Serial(a.port, a.baud, timeout=0.05)
        src.reset_input_buffer(); fps_replay = 0.0
    rd = StreamReader(src, fps_replay)

    def csv_on():
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        fn = os.path.join(ROOT, "logs", time.strftime("imu_%Y%m%d_%H%M%S.csv"))
        fh = open(fn, "w", newline="")
        w = csv.writer(fh)
        w.writerow(["t_pc_s", "sample_no", "flags", "who1", "who2", "servo_mode", "servo_target_us", "servo_now_us"]
                   + [f + "_raw" for f in fm.IMU_FIELDS]
                   + [f + ("_g" if (i % 6) < 3 else "_dps") for i, f in enumerate(fm.IMU_FIELDS)])
        with rd.lock:
            rd.csv = (fh, w, fn)
        print("CSV log ->", fn)

    def csv_off():
        with rd.lock:
            c, rd.csv = rd.csv, None
        if c:
            c[0].close(); print("CSV log closed:", c[2])

    if a.csv:
        csv_on()
    if a.headless_out:
        rd.run()                                         # synchronous: whole file
    else:
        rd.start()
    zt = fm.load_zone_tables(a.zones) if os.path.exists(a.zones) else None

    if not a.headless_out:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        print("keys: l labels | z ToF zones | c CSV log | s snapshot | q quit")
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
                if pkt["image"] is not None and not a.no_image:
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
                msg = f"waiting for FPGA result packets ({a.port or a.replay})"
                cv2.putText(left, msg, (20, PH // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            else:
                if last_img is not None:
                    left = cv2.resize(cv2.cvtColor(last_img, cv2.COLOR_GRAY2BGR), (PW, PH), interpolation=cv2.INTER_LINEAR)
                else:
                    g = (pkt["kmap"] > 0).astype(np.uint8) * 120 + 30
                    left = cv2.resize(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), (PW, PH), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(left, "--no-image" if a.no_image else "SW0 off: no camera image", (16, PH - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
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
                status = f"no result packets yet | IMU packets {rd.n_imu_ok} ok, {rd.n_imu_bad} bad"
            bar = np.zeros((24, body.shape[1], 3), np.uint8)
            cv2.putText(bar, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 0), 1)
            frame = np.vstack([bar, body, np.zeros((2, body.shape[1], 3), np.uint8), imu_panel(body.shape[1], rd)])

            if a.headless_out:
                os.makedirs(os.path.dirname(os.path.abspath(a.headless_out)), exist_ok=True)
                cv2.imwrite(a.headless_out, frame)
                print(f"result packets {rd.n_ok} ok, {rd.n_bad} bad | IMU packets {rd.n_imu_ok} ok, {rd.n_imu_bad} bad | "
                      f"samples {rd.n_smp}, lost {rd.n_lost}, overflow {rd.n_ovf}")
                if rd.imu:
                    q = rd.imu
                    print(f"last IMU packet: ok {q['imu_ok']:02b}, WHO_AM_I {q['who1']:#04x}/{q['who2']:#04x}, servo "
                          f"{fm.SERVO_MODES.get(q['servo_mode'])} {q['servo_target_us']}/{q['servo_now_us']} us, "
                          f"last sample {rd.samples[-1][1] if rd.samples else None}")
                print("saved", a.headless_out)
                break
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("l"):
                labels = not labels
            elif key == ord("z"):
                show_zones = not show_zones
            elif key == ord("c"):
                csv_off() if rd.csv else csv_on()
            elif key == ord("s"):
                os.makedirs(os.path.join(ROOT, "snapshots"), exist_ok=True)
                fn = os.path.join(ROOT, "snapshots", time.strftime("fpga_view_imu_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(fn, frame); print("saved", fn)
            if not rd.alive and not a.replay:
                print("[error] serial reader stopped - is the port right and free?"); break
            if a.replay and rd.eof and rd.latest is not None and cv2.getWindowProperty(WIN, 0) < 0:
                break
    finally:
        rd.running = False
        csv_off()
        if not a.headless_out:
            cv2.destroyAllWindows()
        try:
            src.close()
        except Exception:
            pass
    ok = rd.n_bad == 0 and rd.n_imu_bad == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
