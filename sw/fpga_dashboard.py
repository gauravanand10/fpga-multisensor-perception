"""
fpga_dashboard.py - operator / presentation console for the Nexys 4 DDR multi-sensor system:
camera + U-Net objects + ToF distance (computed on the FPGA), two MPU-6500 IMUs, servo actuator.

Same packet reader as engineering_viewer.py (result packets A5 5A 46 50, IMU packets A5 5A 49 4D); only the display differs.
The frame is drawn at the screen's true resolution (DPI-aware, 1:1 pixels), so text stays sharp on 1080p .. 4K.

    live:      python sw\\fpga_dashboard.py --port COM3
    projector: ... --port COM3 --theme light
    replay:    ... --replay captures\\file.bin          PNG: ... --replay file.bin --headless-out view.png

Keys: F full screen  L labels  Z ToF zones on camera  O boxes
      B zero gyroscopes (IMUs at rest)  R raw capture (captures\\*.bin + .tim)  C IMU CSV  S snapshot  Q / Esc quit
Board: SW0 camera image in packets, SW1 servo sweep, SW15..SW8 servo angle (manual).
Roll / pitch are estimated on the PC with a complementary filter (98 % gyroscope, 2 % accelerometer).
"""
import argparse, csv, math, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                                                    # sw/ (outputs: logs, captures, snapshots)
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "hw", "tools"))
sys.path.insert(0, HERE)
import fpga_model as fm
from engineering_viewer import StreamReader as _BaseReader

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

WIN = "Multi-Sensor FPGA Perception System"
W, H = 1600, 900                 # logical layout units; drawn at W*SC x H*SC real pixels
DMIN, DMAX = 200.0, 3000.0
HOLD_S = 3.0
TWIN = 5.0
AA = cv2.LINE_AA
SC = 1.0


def P(v):
    return int(round(v * SC))


def hexc(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


THEMES = {
    "dark": dict(bg="#111316", panel="#1a1d22", head="#1f2329", border="#2d333b", text="#e6e8eb", muted="#98a2ad",
                 dim="#5c6670", grid="#262b32", grid2="#353c45", plot="#16191d", accent="#4c9be8", ok="#3fb950",
                 warn="#d29922", err="#f85149", imgbg="#0d0f12", band="#22364d", edge="#ffffff"),
    "light": dict(bg="#eef0f3", panel="#ffffff", head="#f6f7f9", border="#d0d5dc", text="#1f2328", muted="#57606a",
                  dim="#8c959f", grid="#eceff2", grid2="#d8dde3", plot="#fbfcfd", accent="#0969da", ok="#1a7f37",
                  warn="#9a6700", err="#cf222e", imgbg="#f2f4f7", band="#dbe9f8", edge="#1f2328"),
}
C = {}
AXIS = [hexc("#e15759"), hexc("#59a14f"), hexc("#4e79a7")]
CLS_COL = [None, hexc("#f28e2b"), hexc("#4e79a7"), hexc("#59a14f"), hexc("#b07aa1"), hexc("#e15759")]
VIRIDIS = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_VIRIDIS)[:, 0, :]

# ---------------------------------------------------------------- layout (logical units)
M, G = 14, 12
HDR = 44
IMG_W, IMG_H = 528, 396
PW = IMG_W + 24
X1, X2 = M, M + PW + G
XR = X2 + PW + G
RW = W - M - XR
Y1 = HDR + 10
PH1 = 466
Y2 = Y1 + PH1 + G
PH2 = H - 10 - Y2
PIPE_H = 214
SRV_Y, SRV_H = Y1 + PIPE_H + G, PH1 - PIPE_H - G


def screen_size():
    """True screen pixels (process made DPI-aware so Windows does not bitmap-stretch the window)."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return None


# ---------------------------------------------------------------- text (PIL, one pass per frame, real pixels)
class Text:
    FONTS = {
        "reg": ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/lato/Lato-Regular.ttf", "DejaVuSans.ttf"],
        "semi": ["C:/Windows/Fonts/seguisb.ttf", "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf",
                 "DejaVuSans-Bold.ttf"],
        "mono": ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "DejaVuSansMono.ttf"],
        "disp": ["C:/Windows/Fonts/segoeuil.ttf", "C:/Windows/Fonts/segoeui.ttf",
                 "/usr/share/fonts/truetype/lato/Lato-Light.ttf", "DejaVuSans.ttf"],
    }

    def __init__(self):
        self.q, self.cache, self.mcache = [], {}, {}

    def font(self, kind, px):
        key = (kind, px)
        if key not in self.cache:
            f = None
            if Image is not None:
                for p in self.FONTS[kind]:
                    try:
                        f = ImageFont.truetype(p, px); break
                    except OSError:
                        continue
                if f is None:
                    f = ImageFont.load_default()
            self.cache[key] = f
        return self.cache[key]

    def width_px(self, s, kind, px):
        f = self.font(kind, px)
        if f is None:
            return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, px / 30.0, 1)[0][0]
        return int(math.ceil(f.getlength(s)))

    def width(self, s, kind, size):                      # logical units
        return self.width_px(s, kind, max(6, int(round(size * SC)))) / SC

    def __call__(self, x, y, s, size=13, col=(230, 230, 230), kind="reg", anchor="la"):
        self.q.append((P(x), P(y), s, max(6, int(round(size * SC))), col, kind, anchor))

    def px(self, x, y, s, px, col, kind="reg", anchor="la"):
        self.q.append((int(x), int(y), s, int(px), col, kind, anchor))

    def _mask(self, s, px, kind, anchor):
        """anti-aliased coverage mask of a string (cached), plus its offset from the anchor point"""
        key = (s, px, kind, anchor)
        m = self.mcache.get(key)
        if m is None:
            if not hasattr(self, "_meas"):
                self._meas = ImageDraw.Draw(Image.new("L", (1, 1)))
            f = self.font(kind, px)
            l, t, r, b = self._meas.textbbox((0, 0), s, font=f, anchor=anchor)
            l, t = int(math.floor(l)) - 1, int(math.floor(t)) - 1
            w, h = int(math.ceil(r)) - l + 2, int(math.ceil(b)) - t + 2
            im = Image.new("L", (max(1, w), max(1, h)), 0)
            ImageDraw.Draw(im).text((-l, -t), s, font=f, fill=255, anchor=anchor)
            m = ((np.asarray(im, np.float32) / 255.0)[..., None], l, t)
            if len(self.mcache) > 6000:
                self.mcache.clear()
            self.mcache[key] = m
        return m

    def render(self, img):
        if not self.q:
            return
        if Image is None:
            for x, y, s, px, col, kind, anchor in self.q:
                cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, px / 30.0, col, 1, AA)
        else:
            Hh, Ww = img.shape[:2]
            for x, y, s, px, col, kind, anchor in self.q:
                m, l, t = self._mask(s, px, kind, anchor)
                x0, y0 = x + l, y + t
                mx0, my0 = max(0, -x0), max(0, -y0)
                x1, y1 = min(Ww, x0 + m.shape[1]), min(Hh, y0 + m.shape[0])
                if x1 <= x0 + mx0 or y1 <= y0 + my0:
                    continue
                mm = m[my0:my0 + (y1 - y0 - my0), mx0:mx0 + (x1 - x0 - mx0)]
                roi = img[y0 + my0:y1, x0 + mx0:x1]
                roi[:] = (roi * (1.0 - mm) + np.array(col, np.float32) * mm).astype(np.uint8)
        self.q.clear()


# ---------------------------------------------------------------- drawing helpers (logical coordinates)
def ln(img, x0, y0, x1, y1, col, t=1):
    cv2.line(img, (P(x0), P(y0)), (P(x1), P(y1)), col, max(1, int(round(t * SC))), AA)


def rc(img, x0, y0, x1, y1, col, t=1):
    cv2.rectangle(img, (P(x0), P(y0)), (P(x1), P(y1)), col, -1 if t < 0 else max(1, int(round(t * SC))))


def dot(img, x, y, r, col):
    cv2.circle(img, (P(x), P(y)), max(2, P(r)), col, -1, AA)


def blend_px(img, x0, y0, x1, y1, col, a):
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - a) + np.array(col, np.float32) * a).astype(np.uint8)


def panel(img, T, x, y, w, h, title, right=None, right_col=None):
    rc(img, x, y, x + w - 1, y + h - 1, C["panel"], -1)
    rc(img, x, y, x + w - 1, y + 29, C["head"], -1)
    rc(img, x, y, x + w - 1, y + h - 1, C["border"], 1)
    ln(img, x, y + 29, x + w - 1, y + 29, C["border"])
    T(x + 12, y + 15, title, 13, C["text"], "semi", "lm")
    if right:
        T(x + w - 12, y + 15, right, 12, right_col or C["muted"], "reg", "rm")


def kv(T, x, y, w, k, v, vcol=None):
    T(x, y, k, 12, C["muted"], "reg", "lm")
    T(x + w, y, v, 13, vcol or C["text"], "mono", "rm")


def depth_idx(d):
    return int(np.clip(255 * (1 - (d - DMIN) / (DMAX - DMIN)), 0, 255))


def depth_color(d):
    return tuple(int(c) for c in VIRIDIS[depth_idx(d)])


def nice_ticks(lo, hi, n=4):
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    t0 = math.ceil(lo / step - 1e-9) * step
    k = int(math.floor((hi - t0) / step + 1e-9))
    s = f"{step:g}"
    dec = len(s.split(".")[1]) if "." in s and "e" not in s else 0
    return [t0 + i * step for i in range(k + 1)], dec


def plot_ts(img, T, x, y, w, h, t, arr, title, min_span, xlabels=False, legend=False, msg=None):
    rc(img, x, y, x + w, y + h, C["plot"], -1)
    for s_ in range(-int(TWIN), 1):
        xx = x + (s_ + TWIN) / TWIN * w
        ln(img, xx, y, xx, y + h, C["grid"])
        if xlabels:
            T(xx, y + h + 10, f"{s_} s", 10, C["muted"], "reg", "mm")
    if arr is not None and len(arr) > 1:
        lo, hi = float(arr.min()), float(arr.max())
        mid, span = (lo + hi) / 2, max(hi - lo, min_span) * 1.2
        lo, hi = mid - span / 2, mid + span / 2
        ticks, dec = nice_ticks(lo, hi, 3)
        for tv in ticks:
            yy = y + h - (tv - lo) / (hi - lo) * h
            ln(img, x, yy, x + w, yy, C["grid2"] if abs(tv) < 1e-9 else C["grid"])
            T(x - 6, yy, f"{tv:.{dec}f}", 10, C["muted"], "mono", "rm")
        keep = t >= -TWIN
        xs = ((x + (t[keep] + TWIN) / TWIN * w) * SC).astype(np.int32)
        for k in range(arr.shape[1]):
            ys = ((y + h - (arr[keep, k] - lo) / (hi - lo) * h) * SC).astype(np.int32)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, AXIS[k], max(1, int(round(1.2 * SC))), AA)
    else:
        T(x + w / 2, y + h / 2, msg or "no data", 12, C["dim"], "reg", "mm")
    rc(img, x, y, x + w, y + h, C["grid2"], 1)
    tw = T.width(title, "semi", 11)
    rc(img, x + 1, y + 1, x + tw + 12, y + 17, C["plot"], -1)
    T(x + 6, y + 9, title, 11, C["muted"], "semi", "lm")
    if legend:
        for k, nm in enumerate("xyz"):
            lx = x + w - 92 + k * 30
            rc(img, lx - 2, y + 2, lx + 26, y + 16, C["plot"], -1)
            ln(img, lx, y + 9, lx + 12, y + 9, AXIS[k], 2)
            T(lx + 16, y + 9, nm, 11, C["muted"], "reg", "lm")


def tof_field(pkt, zt):
    """ToF zones projected into the 80x60 label grid exactly like the FPGA (nearest zone wins),
    camera-axis distance in mm, inf where no zone covers the cell."""
    dist, status = pkt["tof_dist"], pkt["tof_status"]
    ok = fm.tof_valid(dist, status)
    zb = np.full((fm.LAB_H, fm.LAB_W), np.inf)
    for i in np.nonzero(ok)[0]:
        d = int(dist[i])
        dc = (d * int(zt.rz_q15[i]) + (1 << 14)) >> 15
        x0, y0, x1, y1, v = (int(q) for q in zt.rects[i, fm.depth_bin(d, zt)])
        if v:
            sl = zb[y0:y1 + 1, x0:x1 + 1]
            np.minimum(sl, dc, out=sl)
    return zb


# ---------------------------------------------------------------- raw capture (key R) + reader
class TeeSource:
    """Wraps the serial port / replay file. While recording, every byte read is appended to
    captures/<name>.bin (replayable with --replay) and '<pc time s> <byte offset>' lines go to <name>.tim,
    so pc/make_figures.py can rebuild figures and timing statistics from exactly what the board sent."""
    def __init__(self, src):
        self.src, self.fh, self.tim, self.off, self.path, self.t0 = src, None, None, 0, None, 0.0

    def read(self, n):
        b = self.src.read(n)
        fh, tim = self.fh, self.tim
        if b and fh is not None:
            fh.write(b)
            tim.write(f"{time.perf_counter():.6f} {self.off}\n")
            self.off += len(b)
        return b

    def start(self, path):
        self.off, self.path, self.t0 = 0, path, time.perf_counter()
        self.tim = open(path[:-4] + ".tim", "w")
        self.fh = open(path, "wb")

    def stop(self):
        fh, tim, self.fh, self.tim = self.fh, self.tim, None, None
        if fh:
            fh.close(); tim.close()
        return self.path, self.off

    def close(self):
        self.stop()
        self.src.close()



class Reader(_BaseReader):
    def __init__(self, src, replay_fps=0.0):
        super().__init__(src, replay_fps)
        self.att = [None, None]                    # (roll, pitch) deg per IMU
        self.att_sno = None
        self.bias = np.zeros(12)                   # raw counts, gyro fields only
        self.resyncs = 0
        self.t_imu = 0.0
        self.spark = [np.zeros((0, 3)), np.zeros((0, 3))]

    def _imu(self, p):
        with self.lock:
            prev, lost0 = self.last_sno, self.n_lost
        super()._imu(p)
        with self.lock:
            self.n_lost = lost0                    # recount: ignore backward jumps / resets
            last = prev
            for sno, v in p["samples"]:
                if last is not None:
                    gap = ((sno - last) & 0xFFFF) - 1
                    if 0 <= gap < 1000:
                        self.n_lost += gap
                    else:
                        self.resyncs += 1
                last = sno
                self._attitude(sno, np.asarray(v, np.float64), p["imu_ok"])
            self.t_imu = time.perf_counter()

    def _attitude(self, sno, v, ok):
        dt = 0.01
        if self.att_sno is not None:
            dt = min(0.1, max(0.001, (((sno - self.att_sno) & 0xFFFF) or 1) * 0.01))
        self.att_sno = sno
        for i in range(2):
            if not (ok >> i & 1):
                self.att[i] = None
                continue
            ax, ay, az = v[6 * i:6 * i + 3] / fm.ACC_LSB_PER_G
            gx, gy = (v[6 * i + 3:6 * i + 5] - self.bias[6 * i + 3:6 * i + 5]) / fm.GYRO_LSB_PER_DPS
            r_acc = math.degrees(math.atan2(ay, az))
            p_acc = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
            if self.att[i] is None:
                self.att[i] = (r_acc, p_acc)
            else:
                r, p = self.att[i]
                r = r + gx * dt
                if r - r_acc > 180: r -= 360
                if r - r_acc < -180: r += 360
                self.att[i] = (0.98 * r + 0.02 * r_acc, 0.98 * (p + gy * dt) + 0.02 * p_acc)

    def zero_gyros(self):
        with self.lock:
            if len(self.samples) < 20:
                return False
            a = np.asarray([s[1] for s in list(self.samples)[-200:]], np.float64)
            self.bias = np.zeros(12)
            for i in range(2):
                self.bias[6 * i + 3:6 * i + 6] = a[:, 6 * i + 3:6 * i + 6].mean(0)
            self.att = [None, None]
        return True



# ---------------------------------------------------------------- panels drawn in real pixels
def draw_camera(img, T, pkt, last_img, results, zt, opt):
    ix, iy = X1 + 12, Y1 + 40
    vx, vy, vw, vh = P(ix), P(iy), P(IMG_W), P(IMG_H)
    view = img[vy:vy + vh, vx:vx + vw]
    view[:] = C["imgbg"]
    cell = vw / fm.LAB_W
    if pkt is None:
        T(ix + IMG_W / 2, iy + IMG_H / 2, "No video signal", 16, C["muted"], "reg", "mm")
        return
    if last_img is not None:
        view[:] = cv2.cvtColor(cv2.resize(last_img, (vw, vh), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    else:
        m = cv2.resize((pkt["kmap"] > 0).astype(np.uint8), (vw, vh), interpolation=cv2.INTER_NEAREST)
        view[m > 0] = C["grid2"]
        T(ix + IMG_W / 2, iy + IMG_H - 18, "Camera image off (SW0)", 12, C["muted"], "reg", "mm")
    if opt["zones"] and zt is not None:
        ok = fm.tof_valid(pkt["tof_dist"], pkt["tof_status"])
        for i in range(64):
            if not ok[i]:
                continue
            dmm = int(pkt["tof_dist"][i])
            x0, y0, x1, y1, v = zt.rects[i, fm.depth_bin(dmm, zt)]
            if v:
                cv2.rectangle(view, (int(x0 * cell), int(y0 * cell)), (int((x1 + 1) * cell) - 1, int((y1 + 1) * cell) - 1),
                              depth_color(dmm), max(1, int(SC)))
                T.px(vx + x0 * cell + P(3), vy + y0 * cell + P(3), f"{dmm / 1000:.2f}", P(10), (245, 245, 245), "mono")
    if opt["boxes"]:
        fpx = P(11)
        for o, mask, d, src_, names, _ in results:
            col = CLS_COL[o.cls] or C["muted"]
            b0, b1 = (int(o.x0 * cell), int(o.y0 * cell)), (int((o.x1 + 1) * cell) - 1, int((o.y1 + 1) * cell) - 1)
            cv2.rectangle(view, b0, b1, col, max(2, int(round(1.5 * SC))))
            if opt["labels"]:
                nm = max(set(names), key=names.count)
                s_ = nm + ("" if d is None else f"  {d / 1000:.2f} m")
                tw, th = T.width_px(s_, "semi", fpx) + P(10), P(17)
                lx = min(b0[0], vw - tw)
                ly = b0[1] - th if b0[1] >= th else b0[1]
                cv2.rectangle(view, (lx, ly), (lx + tw, ly + th), col, -1)
                T.px(vx + lx + P(5), vy + ly + th // 2, s_, fpx, (255, 255, 255), "semi", "lm")


def draw_depth(img, T, pkt, last_img, results, zt, opt):
    """same look as the previous version: flat object colours on a grid, labels above the objects"""
    ix, iy = X2 + 12, Y1 + 40
    vx, vy, vw, vh = P(ix), P(iy), P(IMG_W), P(IMG_H)
    view = img[vy:vy + vh, vx:vx + vw]
    view[:] = C["imgbg"]
    lw = max(1, int(round(SC)))
    for k in range(9):
        xx = int(k * vw / 8)
        cv2.line(view, (xx, 0), (xx, vh), C["grid"], lw)
    for k in range(7):
        yy = int(k * vh / 6)
        cv2.line(view, (0, yy), (vw, yy), C["grid"], lw)
    if pkt is None:
        T(ix + IMG_W / 2, iy + IMG_H / 2, "No data", 16, C["muted"], "reg", "mm")
        return
    tags = []
    for o, mask, d, src_, names, _ in sorted(results, key=lambda r: -r[0].area):
        col = depth_color(d) if d is not None else C["dim"]
        m = cv2.resize(mask.astype(np.uint8), (vw, vh), interpolation=cv2.INTER_NEAREST)
        view[m > 0] = col
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, cs, -1, C["panel"], lw)
        ys_, xs_ = np.nonzero(m)
        if xs_.size:
            nm = max(set(names), key=names.count)
            dt_ = "" if d is None else (("held " if src_ == "hold" else "") + f"{d / 1000:.2f} m")
            tags.append((nm, dt_, int(xs_.mean()), int(ys_.min())))
    if opt["labels"]:
        fpx = P(12)
        for nm, dt_, cx_, top in tags:
            w1 = T.width_px(nm, "semi", fpx)
            w2 = T.width_px(dt_, "mono", fpx) if dt_ else 0
            tw = w1 + (w2 + P(6) if dt_ else 0) + P(16)
            th = P(21)
            lx = min(max(P(2), cx_ - tw // 2), vw - tw - P(2))
            ly = max(P(2), top - P(26))
            blend_px(view, lx, ly, lx + tw, ly + th, C["panel"], 0.9)
            cv2.rectangle(view, (lx, ly), (lx + tw, ly + th), C["border"], max(1, int(SC)))
            T.px(vx + lx + P(8), vy + ly + th // 2, nm, fpx, C["text"], "semi", "lm")
            if dt_:
                T.px(vx + lx + P(14) + w1, vy + ly + th // 2, dt_, fpx, C["text"], "mono", "lm")


# ---------------------------------------------------------------- main
def main():
    global SC
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="Nexys 4 DDR USB serial port, e.g. COM3")
    ap.add_argument("--baud", type=int, default=2_000_000)
    ap.add_argument("--replay", default=None, help="binary file of captured UART bytes")
    ap.add_argument("--replay-fps", type=float, default=8.0)
    ap.add_argument("--zones", default=os.path.join(REPO, "hw", "mem", "zone_tables.npz"))
    ap.add_argument("--theme", choices=sorted(THEMES), default="dark")
    ap.add_argument("--fullscreen", action="store_true", help="start in full screen")
    ap.add_argument("--no-image", action="store_true", help="never draw the camera picture")
    ap.add_argument("--csv", action="store_true", help="start IMU CSV logging at once")
    ap.add_argument("--scale", type=float, default=None,
                    help="render scale (1.0 = 1600x900, 2.4 = 3840x2160); default: fit the screen")
    ap.add_argument("--headless-out", default=None, help="replay only: no window, save the final view to this PNG")
    a = ap.parse_args()
    C.update({k: hexc(v) for k, v in THEMES[a.theme].items()})
    scr = None if a.headless_out else screen_size()

    def sc_for(full):
        if a.scale:
            return a.scale
        if scr is None:
            return 1.0
        sw, sh = scr
        return min(sw / W, sh / H) if full else min((sw - 24) / W, (sh - 110) / H)

    if a.replay:
        src = open(a.replay, "rb"); rfps = a.replay_fps if not a.headless_out else 1e9
    else:
        if not a.port:
            sys.exit("give --port COMx (the Nexys 4 DDR USB serial port) or --replay file")
        import serial
        src = serial.Serial(a.port, a.baud, timeout=0.05)
        try:
            src.set_buffer_size(rx_size=4 * 1024 * 1024)
        except Exception:
            pass
        src.reset_input_buffer(); rfps = 0.0
    src = TeeSource(src)
    rd = Reader(src, rfps)
    msg = ["", 0.0]

    def note(s):
        msg[:] = [s, time.perf_counter()]
        print(s)

    print("keys: F full screen  L labels  Z ToF zones  O boxes  B zero gyros  "
          "R raw capture  C IMU CSV  S snapshot  Q quit")

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
        note("IMU CSV logging to " + os.path.relpath(fn, ROOT))

    def csv_off():
        with rd.lock:
            c, rd.csv = rd.csv, None
        if c:
            c[0].close(); note("IMU CSV saved: " + os.path.relpath(c[2], ROOT))

    if a.csv:
        csv_on()
    if a.headless_out:
        rd.run()
    else:
        rd.start()
    zt = fm.load_zone_tables(a.zones) if os.path.exists(a.zones) else None

    full = a.fullscreen
    bg_cache = {}

    def open_window(fs):
        global SC
        try:
            cv2.destroyWindow(WIN)
        except cv2.error:
            pass
        SC = sc_for(fs)
        if fs:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)            # 1:1 pixels, no resampling

    if a.headless_out:
        SC = a.scale or 1.0
    else:
        open_window(full)

    def background():
        if SC not in bg_cache:
            bg = np.empty((P(H), P(W), 3), np.uint8); bg[:] = C["bg"]
            rc(bg, 0, 0, W, HDR, C["panel"], -1)
            ln(bg, 0, HDR, W, HDR, C["border"])
            bg_cache.clear(); bg_cache[SC] = bg
        return bg_cache[SC]

    T = Text()
    dcache = {}
    opt = dict(labels=True, zones=False, boxes=True)
    last_img, seq, pkt, tracks, results = None, -1, None, [], []
    n_fr, fps, t0, t_res = 0, 0.0, time.perf_counter(), 0.0
    tof_last = [None, time.perf_counter()]

    try:
        while True:
            tnow = time.perf_counter()
            with rd.lock:
                p, s = rd.latest, rd.seq
                ip = rd.imu
                smp = list(rd.samples)
                att = list(rd.att)
                bias = rd.bias.copy()
                cnt = dict(ok=rd.n_ok, bad=rd.n_bad, iok=rd.n_imu_ok, ibad=rd.n_imu_bad, rate=rd.rate,
                           lost=rd.n_lost, ovf=rd.n_ovf, t_imu=rd.t_imu)
            if p is not None and s != seq:
                seq, pkt = s, p; n_fr += 1; t_res = tnow
                if pkt["image"] is not None and not a.no_image:
                    last_img = pkt["image"]
                results = []
                km = pkt["kmap"]
                for o in pkt["objs"]:
                    mask = km == o.k
                    d = float(o.dist) if o.dist else None
                    src_ = "tof" if d is not None else ""
                    names = [fm.CLASSES[o.cls]]; t_meas = tnow if d is not None else None
                    for (pm, pd, pnames, pt) in tracks:
                        inter = np.logical_and(pm, mask).sum(); uni = np.logical_or(pm, mask).sum()
                        if uni and inter / uni > 0.4:
                            if pd is not None and d is not None and abs(d - pd) < 0.3 * pd:
                                d = 0.6 * pd + 0.4 * d
                            elif d is None and pd is not None and pt is not None and tnow - pt < HOLD_S:
                                d, src_, t_meas = pd, "hold", pt
                            names = (pnames + names)[-10:]
                            break
                    results.append((o, mask, d, src_, names, t_meas))
                tracks = [(m, d, nm, tm) for o, m, d, s_, nm, tm in results]
                if pkt["tof_no"] != tof_last[0]:
                    tof_last[:] = [pkt["tof_no"], tnow]
            if tnow - t0 >= 1:
                fps = n_fr / (tnow - t0); n_fr = 0; t0 = tnow
            con = (pkt is not None and (bool(a.replay) or tnow - t_res < 1.5)) or \
                  (ip is not None and (bool(a.replay) or tnow - cnt["t_imu"] < 1.0))
            tof_stale = pkt is not None and not a.replay and tnow - tof_last[1] > 2.0

            img = background().copy()

            # ---------------- header
            T(M, HDR / 2, "Multi-Sensor FPGA Perception System", 18, C["text"], "semi", "lm")
            if src.fh is not None:
                st, sc_ = f"Recording  {tnow - src.t0:.0f} s", C["err"]
            elif msg[1] and tnow - msg[1] < 4:
                st, sc_ = msg[0], C["muted"]
            else:
                st, sc_ = ("Connected" if con else "No data"), (C["ok"] if con else C["err"])
            xr = W - M
            T(xr, HDR / 2, time.strftime("%H:%M:%S"), 14, C["text"], "mono", "rm")
            xr -= T.width("00:00:00", "mono", 14) + 24
            T(xr, HDR / 2, st, 13, sc_, "reg", "rm")
            dot(img, xr - T.width(st, "reg", 13) - 10, HDR / 2, 4, sc_)

            # ---------------- camera + depth
            panel(img, T, X1, Y1, PW, PH1, "Camera")
            draw_camera(img, T, pkt, last_img, results, zt, opt)
            rc(img, X1 + 11, Y1 + 39, X1 + 12 + IMG_W, Y1 + 40 + IMG_H, C["border"], 1)

            x, y = X2, Y1
            panel(img, T, x, y, PW, PH1, "Depth map",
                  f"{len(results)} object{'s' if len(results) != 1 else ''}" if pkt is not None else None)
            dkey = (seq, SC, opt["labels"])
            dv = (slice(P(Y1 + 40), P(Y1 + 40) + P(IMG_H)), slice(P(X2 + 12), P(X2 + 12) + P(IMG_W)))
            if dcache.get("key") != dkey:
                nq = len(T.q)
                draw_depth(img, T, pkt, last_img, results, zt, opt)
                dcache.update(key=dkey, pix=img[dv].copy(), txt=T.q[nq:])
            else:
                img[dv] = dcache["pix"]; T.q.extend(dcache["txt"])
            ix, iy = x + 12, y + 40
            rc(img, ix - 1, iy - 1, ix + IMG_W, iy + IMG_H, C["border"], 1)
            cy0 = y + PH1 - 27
            T(ix, cy0 + 4, "Distance (m)", 11, C["muted"], "reg", "lm")
            bx0, bw = ix + 86, 300
            gw = P(bx0 + bw) - P(bx0)
            grad = VIRIDIS[np.linspace(255, 0, gw).astype(np.uint8)][None, :, :]
            img[P(cy0):P(cy0 + 8), P(bx0):P(bx0) + gw] = np.repeat(grad, P(cy0 + 8) - P(cy0), 0)
            rc(img, bx0, cy0, bx0 + bw, cy0 + 8, C["border"], 1)
            for dm in (0.2, 1.0, 2.0, 3.0):
                xx = bx0 + (dm * 1000 - DMIN) / (DMAX - DMIN) * bw
                ln(img, xx, cy0 + 8, xx, cy0 + 11, C["muted"])
                T(xx, cy0 + 18, f"{dm:g}", 10, C["muted"], "reg", "mm")
            rc(img, bx0 + bw + 26, cy0, bx0 + bw + 36, cy0 + 8, C["dim"], -1)
            T(bx0 + bw + 42, cy0 + 4, "no distance", 11, C["muted"], "reg", "lm")

            # ---------------- FPGA processing
            x, y, w = XR, Y1, RW
            panel(img, T, x, y, w, PIPE_H, "FPGA processing")
            fv = f"{fps:.1f}" if pkt is not None else "–"
            T(x + 12, y + 70, fv, 34, C["text"], "disp", "ls")
            T(x + 18 + T.width(fv, "disp", 34), y + 70, "frames / s", 12, C["muted"], "reg", "ls")
            rows = [("U-Net inference", f"{pkt['infer_cycles'] / 1e5:.1f} ms" if pkt else "–", None),
                    ("Post-processing", f"{pkt['post_cycles'] / 1e5:.1f} ms" if pkt else "–", None),
                    ("Objects detected", f"{len(pkt['objs'])}" if pkt else "–", None)]
            if tof_stale or (pkt is not None and not pkt["flags"] & 2):
                rows.append(("ToF sensor", "no new data", C["warn"]))
            for k, (l_, v_, c_) in enumerate(rows):
                yy = y + 104 + k * 28
                if k:
                    ln(img, x + 12, yy - 14, x + w - 12, yy - 14, C["grid"])
                kv(T, x + 12, yy, w - 24, l_, v_, c_)

            # ---------------- servo
            x, y, w, h = XR, SRV_Y, RW, SRV_H
            if ip is not None:
                tgt, now = ip["servo_target_us"], ip["servo_now_us"]
                a_now, a_tgt = fm.servo_us_to_deg(now), fm.servo_us_to_deg(tgt)
                mode = {0: "Manual", 1: "Sweep", 2: "External"}.get(ip["servo_mode"], "?")
                moving = abs(tgt - now) > 2
                panel(img, T, x, y, w, h, "Servo", "moving" if moving else "holding", C["warn"] if moving else C["ok"])
                T(x + 12, y + 70, f"{a_now:.1f}°", 34, C["text"], "disp", "ls")
                for k, (l_, v_) in enumerate([("Mode", mode), ("Target", f"{a_tgt:.0f}°"), ("Pulse width", f"{now} µs")]):
                    yy = y + 98 + k * 26
                    if k:
                        ln(img, x + 12, yy - 13, x + w - 12, yy - 13, C["grid"])
                    kv(T, x + 12, yy, w - 24, l_, v_)
                sx0, sx1, sy_ = x + 24, x + w - 24, y + h - 46
                d2x = lambda dg: sx0 + dg / 180.0 * (sx1 - sx0)
                rc(img, d2x(45), sy_ - 7, d2x(135), sy_ + 7, C["band"], -1)
                ln(img, sx0, sy_, sx1, sy_, C["grid2"], 2)
                for dg in range(0, 181, 15):
                    major = dg % 45 == 0
                    ln(img, d2x(dg), sy_ + 2, d2x(dg), sy_ + (10 if major else 6), C["muted"])
                    if major:
                        T(d2x(dg), sy_ + 21, f"{dg}°", 10, C["muted"], "reg", "mm")
                nx_ = d2x(max(0, min(180, a_now)))
                ln(img, nx_, sy_ - 9, nx_, sy_ + 9, C["accent"], 3)
            else:
                panel(img, T, x, y, w, h, "Servo")
                T(x + w / 2, y + h / 2, "No data", 14, C["muted"], "reg", "mm")

            # ---------------- IMUs
            t_rel = sarr = None
            if smp:
                sno = np.array([q[0] for q in smp], np.int64)
                t_rel = (((sno - sno[-1] + 32768) & 0xFFFF) - 32768) * 0.01
                sarr = np.asarray([q[1] for q in smp], np.float64) - bias
            for i, x in enumerate((X1, X2)):
                y = Y2
                okb = bool(ip["imu_ok"] >> i & 1) if ip else False
                panel(img, T, x, y, PW, PH2, f"IMU {i + 1}", "online" if okb else "offline",
                      C["ok"] if okb else C["err"])
                cols = [x + 214, x + 296, x + 378, x + 470]
                for c_, hd in zip(cols, ("X", "Y", "Z", "Norm")):
                    T(c_, y + 44, hd, 11, C["muted"], "semi", "rm")
                ln(img, x + 12, y + 54, x + PW - 12, y + 54, C["grid2"])
                v = sarr[-1] if (sarr is not None and okb) else None
                acc = v[6 * i:6 * i + 3] / fm.ACC_LSB_PER_G if v is not None else None
                gyr = v[6 * i + 3:6 * i + 6] / fm.GYRO_LSB_PER_DPS if v is not None else None
                rr = att[i]
                rows = [("Acceleration (g)", [f"{q:+.3f}" for q in acc] + [f"{np.linalg.norm(acc):.3f}"] if acc is not None else None),
                        ("Angular rate (°/s)", [f"{q:+.2f}" for q in gyr] + [f"{np.linalg.norm(gyr):.2f}"] if gyr is not None else None),
                        ("Roll / pitch (°)", [f"{rr[0]:+.1f}", f"{rr[1]:+.1f}", "", ""] if (rr is not None and okb) else None)]
                for k, (l_, vals) in enumerate(rows):
                    yy = y + 68 + k * 22
                    T(x + 12, yy, l_, 12, C["muted"], "reg", "lm")
                    for c_, vv in zip(cols, vals or ["–"] * 3 + [""]):
                        T(c_, yy, vv, 13, C["text"], "mono", "rm")
                px0, pw_ = x + 54, PW - 54 - 14
                g_ok = sarr is not None and okb
                plot_ts(img, T, px0, y + 128, pw_, 94, t_rel if g_ok else None,
                        (sarr[:, 6 * i:6 * i + 3] / fm.ACC_LSB_PER_G) if g_ok else None, "Acceleration (g)", 0.2,
                        legend=True, msg="offline")
                plot_ts(img, T, px0, y + 236, pw_, 94, t_rel if g_ok else None,
                        (sarr[:, 6 * i + 3:6 * i + 6] / fm.GYRO_LSB_PER_DPS) if g_ok else None, "Angular rate (°/s)",
                        4.0, xlabels=True, msg="offline")

            # ---------------- data link
            x, y, w, h = XR, Y2, RW, PH2
            panel(img, T, x, y, w, h, "Data link")
            rows = [("Result packets", f"{cnt['ok']}", None),
                    ("IMU packets", f"{cnt['iok']}", None),
                    ("Errors", f"{cnt['bad'] + cnt['ibad']}", C["err"] if cnt["bad"] + cnt["ibad"] else None),
                    ("IMU sample rate", f"{cnt['rate']:.0f} Hz" if not a.replay else "–", None),
                    ("Samples lost", f"{cnt['lost']}", C["warn"] if cnt["lost"] or cnt["ovf"] else None)]
            for k, (l_, v_, c_) in enumerate(rows):
                yy = y + 56 + k * 30
                if k:
                    ln(img, x + 12, yy - 15, x + w - 12, yy - 15, C["grid"])
                kv(T, x + 12, yy, w - 24, l_, v_, c_)

            T.render(img)

            if a.headless_out:
                os.makedirs(os.path.dirname(os.path.abspath(a.headless_out)), exist_ok=True)
                cv2.imwrite(a.headless_out, img)
                print(f"{img.shape[1]}x{img.shape[0]} | result packets {rd.n_ok} ok, {rd.n_bad} bad | IMU packets "
                      f"{rd.n_imu_ok} ok, {rd.n_imu_bad} bad | samples {rd.n_smp}, lost {rd.n_lost}")
                print("saved", a.headless_out)
                break
            cv2.imshow(WIN, img)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                full = not full; open_window(full)
                continue
            toggles = {ord("l"): ("labels", "Labels"), ord("z"): ("zones", "ToF zones on camera"),
                       ord("o"): ("boxes", "Object boxes")}
            if key in toggles:
                k_, nm_ = toggles[key]
                opt[k_] = not opt[k_]; note(f"{nm_} {'on' if opt[k_] else 'off'}")
            elif key == ord("b"):
                note("Gyroscope offsets removed" if rd.zero_gyros() else "Not enough IMU samples yet")
            elif key == ord("c"):
                csv_off() if rd.csv else csv_on()
            elif key == ord("r"):
                if src.fh is None:
                    os.makedirs(os.path.join(ROOT, "captures"), exist_ok=True)
                    fn = os.path.join(ROOT, "captures", time.strftime("demo_%Y%m%d_%H%M%S.bin"))
                    src.start(fn); note("Raw capture started: " + os.path.relpath(fn, ROOT))
                else:
                    fn, nb = src.stop(); note(f"Raw capture saved: {os.path.relpath(fn, ROOT)} ({nb / 1e6:.1f} MB)")
            elif key == ord("s"):
                os.makedirs(os.path.join(ROOT, "snapshots"), exist_ok=True)
                fn = os.path.join(ROOT, "snapshots", time.strftime("dashboard_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(fn, img); note("Snapshot saved: " + os.path.relpath(fn, ROOT))
            if not a.replay and not rd.alive:
                print("[error] serial reader stopped - is the COM port right and free?"); break
            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
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


if __name__ == "__main__":
    main()
