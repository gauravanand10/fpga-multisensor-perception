"""
laptop_host.py - stage 7: the laptop side (sw/result_viewer.py)

Reads the UART stream, finds the sync word, checks the checksum and draws the
"Camera | Depth Map" view: every object filled with a colour for its distance (red = near,
blue = far, grey = no ToF distance) and labelled "class  x.xx m". The live viewer also smooths the
distance over time and holds it for 3 s; this single-frame model draws the raw per-frame result.
"""
import cv2
import numpy as np

from . import fpga_model as fm

PW, PH = 640, 480
DMIN, DMAX = 200.0, 3000.0


def depth_color(d):
    v = np.clip((d - DMIN) / (DMAX - DMIN), 0, 1)
    return tuple(int(c) for c in cv2.applyColorMap(np.array([[255 - v * 255]], np.uint8), cv2.COLORMAP_JET)[0, 0])


def find_packets(stream: bytes):
    """All complete, checksum-valid packets in a byte stream (like the viewer's reader thread)."""
    out, i = [], 0
    while True:
        j = stream.find(fm.UART_SYNC, i)
        if j < 0:
            break
        try:
            pkt, n = fm.parse_uart_packet(stream[j:])
        except ValueError:
            i = j + 1; continue
        if pkt is None:
            break
        out.append((j, pkt, n)); i = j + n
    return out


def render_view(pkt, zt=None, show_zones=False, labels=True, title=True):
    """Packet dict -> BGR image (PH+24, 2*PW+2): status bar, Camera | Depth Map."""
    if pkt["image"] is not None:
        left = cv2.resize(cv2.cvtColor(pkt["image"], cv2.COLOR_GRAY2BGR), (PW, PH), interpolation=cv2.INTER_LINEAR)
    else:
        g = (pkt["kmap"] > 0).astype(np.uint8) * 120 + 30
        left = cv2.resize(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), (PW, PH), interpolation=cv2.INTER_NEAREST)
    right = np.zeros((PH, PW, 3), np.uint8)
    texts = []
    for o in sorted(pkt["objs"], key=lambda o: -o.area):
        d = float(o.dist) if o.dist else None
        col = depth_color(d) if d is not None else (110, 110, 110)
        m = cv2.resize((pkt["kmap"] == o.k).astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST).astype(bool)
        right[m] = col
        ys, xs = np.nonzero(m)
        if xs.size:
            t = fm.CLASSES[o.cls] + (" ?" if d is None else f" {d / 1000:.2f} m")
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            texts.append((t, (int(xs.mean()) - tw // 2, max(th + 2, int(ys.min()) - 6))))
    if labels:
        for t, (x, y) in texts:
            x = min(max(0, x), PW - 10)
            cv2.putText(right, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(right, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
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
    if title:
        for img, t in ((left, "Camera"), (right, "Depth Map")):
            cv2.putText(img, t, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
            cv2.putText(img, t, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    body = np.hstack([left, np.zeros((PH, 2, 3), np.uint8), right])
    bar = np.zeros((24, body.shape[1], 3), np.uint8)
    s = (f"camera #{pkt['cam_no']} | ToF #{pkt['tof_no']} | U-Net {pkt['infer_cycles'] / 1e5:.1f} ms | "
         f"{len(pkt['objs'])} objects")
    cv2.putText(bar, s, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 0), 1)
    return np.vstack([bar, body])
