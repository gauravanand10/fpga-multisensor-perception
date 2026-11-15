"""
make_figures.py - architecture figures of the FPGA perception system in the style of IEEE accelerator papers:
gradient-filled blocks, 3-D registers, operator arrays, hollow block arrows, coloured nested regions.

    python docs/make_figures.py      -> docs/figures/*.png (300 dpi, 7.2 in wide)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Calibri", "Carlito", "Arial", "Liberation Sans", "DejaVu Sans"]})
S = 0.04                                   # inches per unit (180 units = 7.2 in)
PT = S * 72                                # points per unit
INK = "#1a1a1a"
FS = 8.5


def canvas(w, h):
    fig = plt.figure(figsize=(w * S, h * S), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    return fig, ax


def _grad(ax, patch, top, bot, z):
    t, b = np.array(mcolors.to_rgb(top)), np.array(mcolors.to_rgb(bot))
    g = np.linspace(0, 1, 128)[:, None, None]
    img = (t * (1 - g) + b * g) * np.ones((1, 2, 3))
    x0, y0, x1, y1 = patch.get_extents().bounds if False else (None,) * 4
    return img


def gbox(ax, x, y, w, h, text="", top="#ffffff", bot="#d6e0ef", edge="#3c3c3c", r=1.2, fs=FS, bold=False,
         tc=INK, z=2, lw=0.9, cut=False):
    if cut:                                  # buffer with a cut corner
        c = min(3, w * 0.25)
        patch = Polygon([(x, y), (x + w - c, y), (x + w, y + c), (x + w, y + h), (x, y + h)], closed=True,
                        fc="none", ec=edge, lw=lw, zorder=z + 0.2)
    else:
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec=edge, lw=lw,
                               zorder=z + 0.2)
    ax.add_patch(patch)
    t, b = np.array(mcolors.to_rgb(top)), np.array(mcolors.to_rgb(bot))
    g = np.linspace(0, 1, 128)[:, None, None]
    img = t * (1 - g) + b * g
    im = ax.imshow(np.repeat(img, 2, 1), extent=(x, x + w, y + h, y), aspect="auto", zorder=z, interpolation="bilinear")
    im.set_clip_path(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1,
                fontweight="bold" if bold else "normal", linespacing=1.05)


def sbox(ax, x, y, w, h, text="", fc="#f2f2f2", edge="#404040", fs=FS, tc=INK, bold=True, z=2, lw=0.9):
    """solid block (nested-region style)"""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=edge, lw=lw, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1,
                fontweight="bold" if bold else "normal", linespacing=1.05)


def region(ax, x, y, w, h, label, edge="#1f3864", lw=2.2, ls="-", lc=None, r=0.0, loc="right", fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec=edge, lw=lw,
                                ls=ls, zorder=1))
    if label:
        lx = x + w - 1.8 if loc == "right" else x + 1.8
        ax.text(lx, y + 3.2, label, ha=loc, va="center", fontsize=fs, fontweight="bold", style="italic",
                color=lc or edge, zorder=1.1)


def cuboid(ax, x, y, w, h, text="", d=1.6, fs=6.2, z=3):
    gbox(ax, x, y, w, h, top="#fbfbfb", bot="#c9c9c9", r=0.01, z=z, lw=0.6, edge="#5a5a5a")
    ax.add_patch(Polygon([(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)], closed=True, fc="#ececec", ec="#5a5a5a",
                         lw=0.6, zorder=z + 0.3))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d), (x + w, y + h)], closed=True, fc="#a9a9a9",
                         ec="#5a5a5a", lw=0.6, zorder=z + 0.3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=z + 1, linespacing=1.0)


def sq(ax, cx, cy, s, a=4.2, fs=8, z=4):
    ax.add_patch(Rectangle((cx - a / 2, cy - a / 2), a, a, fc="white", ec=INK, lw=0.8, zorder=z))
    ax.text(cx, cy + 0.1, s, ha="center", va="center", fontsize=fs, zorder=z + 1)


def dashed(ax, x, y, w, h, label=None, r=0.8, lw=0.9, lab_below=True, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec=INK, lw=lw,
                                ls=(0, (3, 2)), zorder=2.5))
    if label:
        ax.text(x + w / 2, y + h + 2.4 if lab_below else y - 2.2, label, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, pts, lw=1.6, c=INK, ms=7):
    """thin/thick black data arrow"""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=c, lw=lw, zorder=2.2, solid_capstyle="butt", solid_joinstyle="miter")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c, zorder=2.3,
                                 shrinkA=0, shrinkB=0))


def barrow(ax, pts, w=2.2, fc="#dae3f3", ec="#2f5597", both=False, z=1.8):
    """hollow block arrow along a polyline (axis-aligned segments), head(s) at the end(s)"""
    pts = [np.array(p, float) for p in pts]
    hl, hw = 1.5 * w, 2.1 * w

    def head(tip, prev):
        d = tip - prev; d = d / np.linalg.norm(d); n = np.array([-d[1], d[0]])
        base = tip - d * hl
        ax.add_patch(Polygon([tip, base + n * hw / 2, base - n * hw / 2], closed=True, fc=fc, ec=ec, lw=0.8,
                             zorder=z + 0.2, joinstyle="miter"))
        return base
    p = list(pts)
    p[-1] = head(pts[-1], pts[-2])
    if both:
        p[0] = head(pts[0], pts[1])
    xs, ys = [q[0] for q in p], [q[1] for q in p]
    ax.plot(xs, ys, color=ec, lw=w * PT + 1.6, zorder=z, solid_capstyle="butt", solid_joinstyle="miter")
    ax.plot(xs, ys, color=fc, lw=w * PT, zorder=z + 0.1, solid_capstyle="butt", solid_joinstyle="miter")


def lbl(ax, x, y, s, fs=7.5, ha="center", c=INK, rot=0, it=False):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=c, rotation=rot, zorder=6, style="italic" if it else "normal")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig); print("wrote docs/figures/" + name + ".png")


GOLD, GOLD_E = "#c9a227", "#7f6000"
BLK = "#1e1e1e"
BG, BG_E = "#adb9ca", "#44546a"
GRN, ORG, LGR = "#70ad47", "#ed7d31", "#e7e6e6"


# ------------------------------------------------------------------ Fig. 1 system architecture
def fig_system():
    fig, ax = canvas(182, 100)
    region(ax, 1, 4, 28, 94, "Sensors", edge="#1f3864")
    for t, y in [("ESP32-CAM", 12), ("ToF\nVL53L5CX", 48), ("IMU ×2\nMPU-6500", 66), ("Servo", 84)]:
        sbox(ax, 4, y, 22, 12, t, fc=LGR)
    region(ax, 34, 1, 120, 97, "FPGA (Artix-7)", edge="#2e75b6")
    for t, y in [("SPI RX", 12), ("SPI RX", 48), ("SPI\nmaster", 66), ("PWM", 84)]:
        sbox(ax, 38, y, 16, 12, t, fc=GOLD, edge=GOLD_E, fs=8)
    region(ax, 59, 5, 69, 39, "U-Net core", edge="#c00000", lw=1.8, ls=(0, (5, 3)))
    sbox(ax, 63, 12, 16, 12, "Frame\nbuffer", fc=BLK, tc="white", fs=7.8)
    sbox(ax, 84, 12, 21, 12, "Activations\n+ weights", fc=BLK, tc="white", fs=7.8)
    sbox(ax, 110, 12, 14, 12, "Re-\nquant", fc=ORG, edge="#843c0c", tc="white", fs=7.8)
    sbox(ax, 63, 29, 16, 11, "Layer\ncontrol", fc=BG, edge=BG_E, fs=7.8)
    sbox(ax, 84, 29, 21, 11, "PE array\n16 × 8", fc=LGR, fs=7.8)
    sbox(ax, 110, 29, 14, 11, "Adder\ntrees", fc=GRN, edge="#375623", tc="white", fs=7.8)
    sbox(ax, 59, 48, 34, 12, "Post-processing", fc=ORG, edge="#843c0c", tc="white", fs=8)
    sbox(ax, 59, 66, 34, 12, "IMU FIFO", fc=LGR, fs=8)
    sbox(ax, 100, 48, 28, 30, "Packet\narbiter\n+ UART", fc=GOLD, edge=GOLD_E, fs=8.5)
    sbox(ax, 160, 52, 21, 22, "Host PC", fc=BLK, tc="white", fs=9)
    w = 2.0
    barrow(ax, [(26, 18), (38, 18)], w); barrow(ax, [(26, 54), (38, 54)], w)
    barrow(ax, [(26, 72), (38, 72)], w, both=True); barrow(ax, [(38, 90), (26, 90)], w)
    barrow(ax, [(54, 18), (63, 18)], w); barrow(ax, [(79, 18), (84, 18)], w)
    barrow(ax, [(94.5, 24), (94.5, 29)], w); barrow(ax, [(79, 34.5), (84, 34.5)], w)
    barrow(ax, [(105, 34.5), (110, 34.5)], w); barrow(ax, [(117, 29), (117, 24)], w)
    barrow(ax, [(110, 18), (105, 18)], w)
    barrow(ax, [(76, 44), (76, 48)], w)
    barrow(ax, [(54, 54), (59, 54)], w); barrow(ax, [(54, 72), (59, 72)], w)
    barrow(ax, [(93, 54), (100, 54)], w); barrow(ax, [(93, 72), (100, 72)], w)
    barrow(ax, [(128, 63), (160, 63)], w)
    lbl(ax, 144, 59.5, "UART 2 Mbaud", fs=7, it=True)
    lbl(ax, 76, 46.0 - 0.2, "", fs=6)
    save(fig, "fig1_system_architecture")


# ------------------------------------------------------------------ Fig. 2 accelerator
def fig_accelerator():
    fig, ax = canvas(182, 104)
    gbox(ax, 2, 8, 22, 30, "Weight\nROM", cut=True, fs=9)
    gbox(ax, 2, 44, 22, 38, "Activation\nbanks", cut=True, fs=9)
    dashed(ax, 32, 2, 148, 84)
    ax.text(178, 5.2, "Accelerator", ha="right", va="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((36, 9), 118, 72, boxstyle="round,pad=0,rounding_size=3", fc="none", ec=INK, lw=1.0, zorder=2))
    ax.text(39.5, 12.8, "Convolution engine", ha="left", va="center", fontsize=9.5)
    cuboid(ax, 41, 22, 13, 8, "Weight\nregister")
    cuboid(ax, 41, 58, 13, 8, "Data\nregister")
    arrow(ax, [(24, 26), (41, 26)], lw=1.8); lbl(ax, 28, 23.3, "Weight", fs=6.5)
    arrow(ax, [(24, 62), (41, 62)], lw=1.8); lbl(ax, 28, 59.3, "Data", fs=6.5)
    # multipliers
    dashed(ax, 61, 18, 10, 56, "PE array", lab_below=False)
    for yy in (23, 31, 61, 69):
        sq(ax, 66, yy, "×")
    for yy in (40, 45, 50):
        ax.text(66, yy, "·", ha="center", va="center", fontsize=10)
    arrow(ax, [(55.6, 26), (61, 26)], lw=1.4); arrow(ax, [(55.6, 62), (61, 62)], lw=1.4)
    # adder tree
    dashed(ax, 78, 18, 30, 56, "Adder tree", lab_below=False)
    l1 = [23, 31, 61, 69]; l2 = [27, 65]
    for yy in l1:
        sq(ax, 83, yy, "+")
        arrow(ax, [(68.1, yy), (80.9, yy)], lw=0.9, ms=5)
    for yy, (a, b) in zip(l2, [(23, 31), (61, 69)]):
        sq(ax, 92, yy, "+")
        for s_ in (a, b):
            ax.plot([85.1, 89.9], [s_, yy], color=INK, lw=0.8, zorder=2.2)
    for yy in (40, 45, 50):
        ax.text(83, yy, "·", ha="center", va="center", fontsize=10); ax.text(92, yy, "·", ha="center", va="center", fontsize=10)
    sq(ax, 102, 46, "+")
    for yy in l2:
        ax.plot([94.1, 99.9], [yy, 46], color=INK, lw=0.8, zorder=2.2)
    # accumulator
    cuboid(ax, 113, 41, 11, 9, "ACC", fs=7)
    arrow(ax, [(104.1, 46), (113, 46)], lw=1.4)
    # epilogue
    dashed(ax, 131, 18, 18, 56, "Epilogue", lab_below=False)
    ops = ["×M", "+", ">>", "clip", "ReLU"]
    ys = [23, 34, 45, 56, 67]
    for s_, yy in zip(ops, ys):
        ax.add_patch(Rectangle((134, yy - 3), 12, 6, fc="white", ec=INK, lw=0.8, zorder=4))
        ax.text(140, yy, s_, ha="center", va="center", fontsize=7.5, zorder=5)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, [(140, a + 3), (140, b - 3)], lw=0.9, ms=5)
    arrow(ax, [(125.6, 46), (128, 46), (128, 23), (134, 23)], lw=1.4)
    gbox(ax, 160, 54, 19, 14, "Argmax", fs=8.5)
    arrow(ax, [(146, 67), (160, 61)], lw=1.4)
    lbl(ax, 169.5, 74, "label map", fs=7.5, it=True)
    # write-back
    arrow(ax, [(140, 70), (140, 78), (13, 78), (13, 82)], lw=1.8) if False else None
    ax.plot([152, 152, 13], [67, 88, 88], color=INK, lw=1.8, zorder=2.2)
    ax.plot([146, 152], [67, 67], color=INK, lw=1.8, zorder=2.2)
    arrow(ax, [(13, 88), (13, 82)], lw=1.8)
    lbl(ax, 90, 90.5, "write-back", fs=7.5, it=True)
    # controller
    gbox(ax, 32, 95, 148, 8, "Controller (layer sequencer)", fs=9, r=0.5)
    for xx in (66, 93, 140):
        barrow(ax, [(xx, 95), (xx, 74.5)], w=2.6, fc="#bfbfbf", ec="#7f7f7f")
    barrow(ax, [(9, 95), (9, 83)], w=2.4, fc="#bfbfbf", ec="#7f7f7f")
    barrow(ax, [(17, 83), (17, 95)], w=2.4, fc="#bfbfbf", ec="#7f7f7f")
    save(fig, "fig2_accelerator_microarchitecture")


# ------------------------------------------------------------------ Fig. 3 datapath
def fig_datapath():
    fig, ax = canvas(184, 46)
    groups = [("Address", [("+", 1), ("+", 1), ("+", 1)]), ("Memory", [("RAM", 1), ("", 1)]),
              ("Input requant", [("LUT", 1), ("×", 1), ("+", 1), (">>", 1), ("clip", 1)]),
              ("MAC", [("×", 1), ("Σ", 1), ("+", 1)]),
              ("Output requant", [("×", 1), ("+", 1), ("+", 1), (">>", 1), ("clip", 1)]),
              ("Argmax", [("max", 1), ("max", 1)])]
    step = 178 / 20
    x, si = 2, 0
    for gi, (name, items) in enumerate(groups):
        n = sum(k for _, k in items)
        dashed(ax, x + 0.4, 9, n * step - 0.8, 22, None)
        lbl(ax, x + n * step / 2, 35, name, fs=7.5)
        xx = x
        for s_, k in items:
            cx = xx + k * step / 2 - 1.3
            if not s_:
                pass
            elif len(s_) <= 2:
                sq(ax, cx, 20, s_, a=4.6, fs=8)
            else:
                ax.add_patch(Rectangle((cx - k * 2.6 - 0.6, 17), k * 5.2 + 1.2, 6, fc="white", ec=INK, lw=0.8, zorder=4))
                ax.text(cx, 20, s_, ha="center", va="center", fontsize=7, zorder=5)
            for j in range(k):
                si += 1
                rx = xx + (j + 1) * step - 1.3
                gbox(ax, rx - 0.7, 12, 1.4, 16, top="#d9d9d9", bot="#7f7f7f", r=0.01, lw=0.4, z=4)
                lbl(ax, rx, 6, str(si), fs=6.3, c="#404040")
            xx += k * step
        x += n * step
    ax.plot([3, 2 + 20 * step - 1.3], [20, 20], color=INK, lw=0.9, zorder=1.5)
    ax.add_patch(FancyArrowPatch((2 + 20 * step - 1.3, 20), (2 + 20 * step + 1.5, 20), arrowstyle="-|>",
                                 mutation_scale=5, lw=0.9, color=INK, zorder=1.6, shrinkA=0, shrinkB=0))
    lbl(ax, 1.2, 6, "stage", fs=6.3, ha="left", c="#404040")
    save(fig, "fig3_convolution_datapath")


# ------------------------------------------------------------------ Fig. 4 network
def fig_network():
    fig, ax = canvas(182, 84)
    lv = {120: 6, 60: 18, 30: 36, 15: 52}
    hh = {120: 40, 60: 30, 30: 18, 15: 11}
    wid = lambda c: 1.4 + c * 0.055
    BL, GN, GY = ("#ffffff", "#bdd7ee"), ("#ffffff", "#c5e0b4"), ("#ffffff", "#d9d9d9")

    def bar(x, res, ch, col=BL, lab=True, text=None):
        w = wid(ch)
        gbox(ax, x, lv[res], w, hh[res], top=col[0], bot=col[1], r=0.01, lw=0.7)
        if lab:
            lbl(ax, x + w / 2, lv[res] - 2, text or str(ch), fs=6.8)
        return x + w

    RED, GRN_, BLU, GRY, PUR = "#c00000", "#548235", "#2f5597", "#7f7f7f", "#7030a0"

    def a(p0, p1, c, ls="-"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, lw=1.1, color=c, zorder=3, shrinkA=0,
                                     shrinkB=0, linestyle=ls))
    e0 = bar(4, 120, 1, GY)
    lbl(ax, 2, lv[120] + hh[120] / 2, "120×160", fs=6.5, rot=90)
    a((e0 + 0.3, lv[120] + hh[120] - 3), (14, lv[60] + hh[60] / 2), RED)
    p = bar(14, 60, 16); a((p, lv[60] + 15), (p + 5, lv[60] + 15), BLU)
    b = bar(p + 5, 60, 16); a((b - 1, lv[60] + hh[60]), (b + 3, lv[30] + 9), RED)
    c_ = bar(b + 3, 30, 32); a((c_, lv[30] + 9), (c_ + 5, lv[30] + 9), BLU)
    d = bar(c_ + 5, 30, 32); a((d - 1, lv[30] + hh[30]), (d + 3, lv[15] + 5.5), RED)
    e_ = bar(d + 3, 15, 80); a((e_, lv[15] + 5.5), (e_ + 4, lv[15] + 5.5), BLU)
    f_ = bar(e_ + 4, 15, 80); a((f_, lv[15] + 5.5), (f_ + 4, lv[15] + 5.5), BLU)
    g = bar(f_ + 4, 15, 80)
    up1 = g + 6
    bar(up1, 30, 80, GN, lab=False); bar(up1 + wid(80), 30, 32, lab=False)
    lbl(ax, up1 + (wid(80) + wid(32)) / 2, lv[30] - 2, "80+32", fs=6.8)
    a((g - 1, lv[15]), (up1 + 2, lv[30] + hh[30]), GRN_)
    a((d - 2, lv[30] + 3), (up1 + wid(80) + 1, lv[30] + 3), GRY, (0, (3, 2)))
    h = up1 + wid(80) + wid(32); a((h, lv[30] + 9), (h + 5, lv[30] + 9), BLU)
    i_ = bar(h + 5, 30, 32); a((i_, lv[30] + 9), (i_ + 5, lv[30] + 9), BLU)
    j = bar(i_ + 5, 30, 32)
    up2 = j + 6
    bar(up2, 60, 32, GN, lab=False); bar(up2 + wid(32), 60, 16, lab=False)
    lbl(ax, up2 + (wid(32) + wid(16)) / 2, lv[60] - 2, "32+16", fs=6.8)
    a((j - 1, lv[30]), (up2 + 1, lv[60] + hh[60]), GRN_)
    a((b - 1, lv[60] + 3), (up2 + wid(32) + 1, lv[60] + 3), GRY, (0, (3, 2)))
    k = up2 + wid(32) + wid(16); a((k, lv[60] + 15), (k + 5, lv[60] + 15), BLU)
    l_ = bar(k + 5, 60, 16); a((l_, lv[60] + 15), (l_ + 5, lv[60] + 15), BLU)
    m = bar(l_ + 5, 60, 16); a((m, lv[60] + 15), (m + 5, lv[60] + 15), PUR)
    n = bar(m + 5, 60, 6, ("#ffffff", "#f8cbad"))
    a((n, lv[60] + 15), (n + 5, lv[60] + 15), INK)
    gbox(ax, n + 5, lv[60] + 9, 15, 12, "80×60\nlabels", top="#ffffff", bot="#c5e0b4", fs=7)
    for res, lab in [(60, "60×80"), (30, "30×40"), (15, "15×20")]:
        lbl(ax, 180, lv[res] + hh[res] / 2, lab, fs=6.5, ha="right", c="#595959")
    L = [(BLU, "-", "conv 3×3 + ReLU"), (RED, "-", "conv 3×3, stride 2"), (GRN_, "-", "upsample ×2"),
         (GRY, (0, (3, 2)), "skip (concat)"), (PUR, "-", "conv 1×1")]
    for idx, (cc, ls, t) in enumerate(L):
        lx = 22 + idx * 31
        a((lx, 78), (lx + 6, 78), cc, ls)
        lbl(ax, lx + 7.5, 78, t, fs=7, ha="left")
    save(fig, "fig4_tiny_unet")


# ------------------------------------------------------------------ Fig. 5 verification flow
def fig_verification():
    fig, ax = canvas(182, 40)
    steps = ["PyTorch\nmodel", "int8\nquantisation", "Golden\nmodel", "Test\nvectors", "RTL\nsimulation",
             "Implemen-\ntation", "FPGA\nboard"]
    cols = ["#f8cbad", "#f8cbad", "#ffe699", "#c5e0b4", "#bdd7ee", "#d9c3e9", "#d9d9d9"]
    w, gap = 20.5, 5.4
    for i, (t, c) in enumerate(zip(steps, cols)):
        x = 1 + i * (w + gap)
        gbox(ax, x, 4, w, 14, t, top="#ffffff", bot=c, fs=8)
        if i:
            barrow(ax, [(x - gap + 0.4, 11), (x - 0.3, 11)], w=1.8, fc="#bfbfbf", ec="#7f7f7f")
    xg, xr = 1 + 2 * (w + gap) + w / 2, 1 + 4 * (w + gap) + w / 2
    xm = (xg + xr) / 2
    ax.plot([xg, xg, xm - 3], [18, 30, 30], color="#2f5597", lw=1.3, zorder=2)
    ax.plot([xr, xr, xm + 3], [18, 30, 30], color="#2f5597", lw=1.3, zorder=2)
    sq(ax, xm, 30, "=", a=6, fs=10)
    lbl(ax, xm, 36.3, "bit-exact compare", fs=7.5, c="#2f5597", it=True)
    save(fig, "fig5_verification_flow")


if __name__ == "__main__":
    fig_system(); fig_accelerator(); fig_datapath(); fig_network(); fig_verification()
