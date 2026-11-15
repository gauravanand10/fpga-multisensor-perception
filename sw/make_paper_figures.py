"""
make_figures.py - publication-quality figures and a summary table from a raw capture of the FPGA stream.

Record a capture in the dashboard (key R, again R to stop) -> captures\\demo_YYYYmmdd_HHMMSS.bin + .tim, then:
    python sw\\make_paper_figures.py captures\\demo_YYYYmmdd_HHMMSS.bin

Output in figures\\<capture name>\\ (each figure as vector PDF + 600 dpi PNG, IEEE column widths):
    fig_scene      (a) camera + detections, (b) semantic objects, (c) per-object distance, (d) raw 8x8 ToF
    fig_imu        accelerometer, gyroscope and attitude of both IMUs over time
    fig_servo      servo command vs. actual angle (slew-limited PWM)
    fig_timing     per-frame time budget (FPGA compute vs. UART) and frame-interval distribution
    table_summary.tex / .md / .csv   key numbers (booktabs LaTeX table)

Privacy: frames in which an object of class 'person' was detected are never used for the camera panel
(unless --include-person-frames is given), and --no-camera drops the camera panel entirely.
"""
import argparse, csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                                                    # sw/ (outputs: logs, captures, snapshots)
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "hw", "tools"))
import fpga_model as fm

COL1, COL2 = 3.5, 7.16                       # IEEE single / double column width, inches
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          verm="#D55E00", purple="#CC79A7", grey="#7F7F7F")     # Okabe-Ito (colour-blind safe)
AXC = [OI["verm"], OI["green"], OI["blue"]]
CLS_COL = ["#F2F2F2", OI["orange"], OI["sky"], OI["green"], OI["purple"], OI["verm"]]
FRAME_BUDGET_MS = 1000.0 / 14.0              # camera rate


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 0.9, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "axes.grid": True, "grid.linewidth": 0.3, "grid.alpha": 0.45, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("  wrote", name + ".pdf / .png")


def sub(ax, s):
    ax.text(0.5, -0.06, s, transform=ax.transAxes, ha="center", va="top", fontsize=8)


# ---------------------------------------------------------------- capture loading
def load(path):
    buf = open(path, "rb").read()
    tp = path[:-4] + ".tim"
    if os.path.exists(tp):
        a = np.loadtxt(tp, ndmin=2)
        ct, co = a[:, 0] - a[0, 0], a[:, 1].astype(np.int64)
        cend = np.append(co[1:], len(buf))                  # end offset of each read chunk
        arrival = lambda p: float(ct[min(np.searchsorted(cend, p), len(ct) - 1)])
    else:
        arrival = None
    res, imu, bad_res, bad_imu, pos = [], [], 0, 0, 0
    while True:
        i = buf.find(b"\xA5\x5A", pos)
        if i < 0 or len(buf) - i < 4:
            break
        sync = buf[i:i + 4]
        if sync not in (fm.UART_SYNC, fm.IMU_SYNC):
            pos = i + 1
            continue
        try:
            pkt, n = (fm.parse_uart_packet if sync == fm.UART_SYNC else fm.parse_imu_packet)(buf[i:])
        except ValueError:
            if sync == fm.UART_SYNC: bad_res += 1
            else: bad_imu += 1
            pos = i + 4
            continue
        if pkt is None:
            break
        t = arrival(i + n) if arrival else None
        (res if sync == fm.UART_SYNC else imu).append((t, n, pkt))
        pos = i + n
    return dict(res=res, imu=imu, bad_res=bad_res, bad_imu=bad_imu, nbytes=len(buf), timed=arrival is not None,
                dur=(float(ct[-1]) if arrival else None))


def imu_series(imu):
    sno, val, ok, pk_i = [], [], [], []
    srv = []
    for k, (_, _, p) in enumerate(imu):
        for s, v in p["samples"]:
            sno.append(s); val.append(v); ok.append(p["imu_ok"]); pk_i.append(k)
        srv.append((len(sno) - 1, p["servo_target_us"], p["servo_now_us"], p["servo_mode"]))
    if not sno:
        return None
    sno = np.asarray(sno, np.int64)
    d = np.diff(sno) & 0xFFFF
    lost = int(d[(d >= 1) & (d < 1000)].sum() - ((d >= 1) & (d < 1000)).sum())
    resync = int(((d == 0) | (d >= 1000)).sum())
    d = np.where((d >= 1) & (d < 1000), d, 1)
    t = np.concatenate([[0], np.cumsum(d)]) * 0.01
    val = np.asarray(val, np.float64)
    acc = np.stack([val[:, 0:3], val[:, 6:9]]) / fm.ACC_LSB_PER_G          # (2, N, 3)
    gyr = np.stack([val[:, 3:6], val[:, 9:12]]) / fm.GYRO_LSB_PER_DPS
    ok = np.asarray(ok)
    att = np.full((2, len(t), 2), np.nan)
    for i in range(2):
        r = p_ = None
        for n in range(len(t)):
            if not (ok[n] >> i & 1):
                r = None; continue
            ax, ay, az = acc[i, n]
            ra, pa = math.degrees(math.atan2(ay, az)), math.degrees(math.atan2(-ax, math.hypot(ay, az)))
            if r is None:
                r, p_ = ra, pa
            else:
                dt = t[n] - t[n - 1]
                r = 0.98 * (r + gyr[i, n, 0] * dt) + 0.02 * ra
                p_ = 0.98 * (p_ + gyr[i, n, 1] * dt) + 0.02 * pa
            att[i, n] = (r, p_)
    srv = np.asarray(srv, np.float64)
    return dict(t=t, acc=acc, gyr=gyr, ok=ok, att=att, lost=lost, resync=resync,
                srv_t=t[srv[:, 0].astype(int)], srv_tgt=fm.servo_us_to_deg(srv[:, 1]),
                srv_now=fm.servo_us_to_deg(srv[:, 2]), srv_mode=srv[:, 3].astype(int))


# ---------------------------------------------------------------- figures
def fig_scene(cap, out, a):
    res = cap["res"]
    if not res:
        print("  no result packets: fig_scene skipped"); return None
    def has_person(p): return any(o.cls == 1 for o in p["objs"])
    def score(p): return sum(1 for o in p["objs"] if o.dist) * 10 + len(p["objs"])
    if a.frame is not None:
        idx = a.frame
    else:
        cand = [k for k, (_, _, p) in enumerate(res)
                if p["image"] is not None and (a.include_person_frames or not has_person(p))]
        if not cand:
            cand = list(range(len(res)))
        idx = max(cand, key=lambda k: (score(res[k][2]), k))
    p = res[idx][2]
    show_cam = (not a.no_camera and p["image"] is not None and (a.include_person_frames or not has_person(p)))
    if p["image"] is not None and not show_cam and not a.no_camera:
        print("  camera panel left out (person detected in the chosen frame or no image)")
    km = p["kmap"]
    cls = np.zeros_like(km)
    dep = np.full(km.shape, np.nan)
    for o in p["objs"]:
        cls[km == o.k] = o.cls
        if o.dist:
            dep[km == o.k] = o.dist / 1000.0
    tv = fm.tof_valid(p["tof_dist"], p["tof_status"])
    tof = np.where(tv, p["tof_dist"] / 1000.0, np.nan).reshape(8, 8)
    vmax = float(np.nanmax(np.concatenate([dep[np.isfinite(dep)], tof[np.isfinite(tof)], [1.0]])))
    vmin = 0.0
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#E6E6E6")

    n = 4 if show_cam else 3
    fig = plt.figure(figsize=(COL2, 1.95))
    widths = [1, 1, 1, 0.75] if show_cam else [1, 1, 0.75]
    gs = fig.add_gridspec(1, n + 1, width_ratios=widths + [0.05], wspace=0.12)
    axs = [fig.add_subplot(gs[0, k]) for k in range(n)]
    k = 0
    if show_cam:
        ax = axs[0]; k = 1
        ax.imshow(p["image"], cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        for o in p["objs"]:
            c = CLS_COL[o.cls]
            ax.add_patch(Rectangle((o.x0 * 2 - 0.5, o.y0 * 2 - 0.5), (o.x1 - o.x0 + 1) * 2, (o.y1 - o.y0 + 1) * 2,
                                   fill=False, ec=c, lw=1.0))
            lab = fm.CLASSES[o.cls] + (f" {o.dist / 1000:.2f} m" if o.dist else "")
            ax.text(o.x0 * 2, o.y0 * 2 - 1.5, lab, fontsize=5.5, color="white", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.15", fc=c, ec="none"))
        sub(ax, "(a) Camera input + detections")
    ax = axs[k]
    ax.imshow(cls, cmap=ListedColormap(CLS_COL), vmin=-0.5, vmax=5.5, interpolation="nearest")
    present = sorted({o.cls for o in p["objs"]})
    if present:
        ax.legend(handles=[Patch(fc=CLS_COL[c], label=fm.CLASSES[c]) for c in present], loc="lower center", ncol=3,
                  fontsize=5.5, handlelength=0.8, handleheight=0.8, borderaxespad=0.2, frameon=True,
                  framealpha=0.85, edgecolor="none")
    sub(ax, f"({'ab'[k]}) U-Net objects (80×60)")
    ax = axs[k + 1]
    im = ax.imshow(dep, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    sub(ax, f"({'bc'[k]}) Object distance")
    ax = axs[k + 2]
    ax.imshow(tof, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for yy in range(8):
        for xx in range(8):
            if np.isfinite(tof[yy, xx]):
                ax.text(xx, yy, f"{tof[yy, xx]:.1f}", ha="center", va="center", fontsize=4.2,
                        color="white" if tof[yy, xx] < 0.6 * vmax else "black")
    sub(ax, f"({'cd'[k]}) ToF 8×8 (m)")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, n]))
    cb.set_label("Distance (m)"); cb.outline.set_linewidth(0.5)
    save(fig, out, "fig_scene")
    return idx


def fig_imu(s, out, a):
    t = s["t"]
    fig, axs = plt.subplots(3, 2, figsize=(COL2, 4.2), sharex=True)
    for i in range(2):
        okm = (s["ok"] >> i & 1).astype(bool)
        for kx in range(3):
            axs[0, i].plot(t, np.where(okm, s["acc"][i, :, kx], np.nan), color=AXC[kx], label="xyz"[kx])
            axs[1, i].plot(t, np.where(okm, s["gyr"][i, :, kx], np.nan), color=AXC[kx], label="xyz"[kx])
        axs[2, i].plot(t, s["att"][i, :, 0], color=OI["purple"], label="roll")
        axs[2, i].plot(t, s["att"][i, :, 1], color=OI["orange"], label="pitch", ls="--")
        axs[0, i].set_title(f"IMU {i + 1}" + ("" if okm.any() else " (no data)"))
    axs[0, 0].set_ylabel("Acceleration (g)")
    axs[1, 0].set_ylabel("Angular rate (°/s)")
    axs[2, 0].set_ylabel("Attitude (°)")
    for i in range(2):
        axs[2, i].set_xlabel("Time (s)")
    for ax in axs.ravel():
        ax.set_xlim(t[0], t[-1])
    fig.align_ylabels(axs[:, 0])
    fig.tight_layout(h_pad=0.6, w_pad=1.0, rect=(0, 0, 1, 0.95))
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    h2, l2 = axs[2, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, [f"{q}-axis" for q in l1] + l2, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "fig_imu")


def fig_servo(s, out, a):
    fig, ax = plt.subplots(figsize=(COL1, 1.7))
    ax.step(s["srv_t"], s["srv_tgt"], where="post", color=OI["orange"], ls="--", label="command")
    ax.plot(s["srv_t"], s["srv_now"], color=OI["blue"], label="PWM output")
    ax.axhspan(45, 135, color=OI["sky"], alpha=0.12, lw=0, label="allowed range")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Servo angle (°)")
    ax.set_xlim(s["srv_t"][0], s["srv_t"][-1]); ax.set_ylim(30, 150)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    save(fig, out, "fig_servo")


def fig_timing(cap, out, a):
    res = cap["res"]
    if not res:
        return None
    unet = np.array([p["infer_cycles"] for _, _, p in res]) / 1e5
    post = np.array([p["post_cycles"] for _, _, p in res]) / 1e5
    nb = np.array([n for _, n, _ in res])
    img = np.array([p["image"] is not None for _, _, p in res])
    uart = nb * 10.0 / a.baud * 1e3
    rows = []
    for name, m in (("with image", img), ("results only", ~img)):
        if m.any():
            rows.append((name, unet[m].mean(), post[m].mean(), uart[m].mean()))
    timed = cap["timed"] and len(res) > 2
    fig, axs = plt.subplots(1, 2 if timed else 1, figsize=(COL2 if timed else COL1, 1.6),
                            gridspec_kw=dict(width_ratios=[1.3, 1]) if timed else None)
    ax = axs[0] if timed else axs
    y = np.arange(len(rows))
    l0 = np.array([r[1] for r in rows]); l1 = np.array([r[2] for r in rows]); l2 = np.array([r[3] for r in rows])
    ax.barh(y, l0, color=OI["blue"], height=0.5, label="U-Net (FPGA)")
    ax.barh(y, l1, left=l0, color=OI["sky"], height=0.5, label="post-processing (FPGA)")
    ax.barh(y, l2, left=l0 + l1, color=OI["orange"], height=0.5, label="UART transfer")
    ax.axvline(FRAME_BUDGET_MS, color="k", lw=0.8, ls=":")
    ax.text(FRAME_BUDGET_MS + 1.5, len(rows) - 0.42, "14 fps budget", fontsize=6, va="center", ha="left")
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlabel("Time per frame (ms)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.42), fontsize=6)
    ax.set_ylim(-0.6, len(rows) - 0.1); ax.grid(axis="y", visible=False)
    sub(ax, "")
    if timed:
        ta = np.array([t for t, _, _ in res])
        dt = np.diff(ta) * 1e3
        dt = dt[(dt > 0) & (dt < 2000)]
        ax = axs[1]
        ax.hist(dt, bins=40, color=OI["blue"], alpha=0.85)
        ax.axvline(np.median(dt), color=OI["verm"], lw=0.9, label=f"median {np.median(dt):.0f} ms "
                                                                     f"({1000 / np.median(dt):.1f} fps)")
        ax.set_xlabel("Interval between result packets (ms)"); ax.set_ylabel("Count")
        ax.legend(loc="upper right")
    fig.tight_layout(w_pad=2.0)
    save(fig, out, "fig_timing")


# ---------------------------------------------------------------- table
def table(cap, s, out, a, name):
    res = cap["res"]
    R = []
    def add(group, k, v): R.append((group, k, v))
    dur = cap["dur"] if cap["timed"] else (s["t"][-1] if s is not None else None)
    add("Capture", "Duration", f"{dur:.1f} s" if dur else "--")
    add("Capture", "UART bytes", f"{cap['nbytes'] / 1e6:.2f} MB at {a.baud / 1e6:g} Mbaud")
    if res:
        unet = np.array([p["infer_cycles"] for _, _, p in res]) / 1e5
        post = np.array([p["post_cycles"] for _, _, p in res]) / 1e5
        nobj = np.array([len(p["objs"]) for _, _, p in res])
        img = np.array([p["image"] is not None for _, _, p in res])
        add("Vision", "Result packets (bad)", f"{len(res)} ({cap['bad_res']})")
        add("Vision", "Packet error rate", f"{cap['bad_res'] / max(1, len(res) + cap['bad_res']) * 100:.2f} %")
        if cap["timed"] and len(res) > 2:
            ta = np.array([t for t, _, _ in res])
            add("Vision", "Result rate", f"{(len(res) - 1) / (ta[-1] - ta[0]):.1f} fps "
                                         f"({img.mean() * 100:.0f} % with camera image)")
        add("Vision", "U-Net latency (FPGA)", f"{unet.mean():.2f} ± {unet.std():.2f} ms")
        add("Vision", "Post-processing latency", f"{post.mean():.2f} ± {post.std():.2f} ms")
        add("Vision", "Objects per frame", f"{nobj.mean():.1f} (max {nobj.max()})")
        add("Vision", "ToF frames used", f"{len({p['tof_no'] for _, _, p in res})}")
    if s is not None:
        n = len(s["t"])
        add("IMU", "IMU packets (bad)", f"{len(cap['imu'])} ({cap['bad_imu']})")
        add("IMU", "Samples / rate", f"{n} / {n / max(1e-9, s['t'][-1] - s['t'][0]):.1f} Hz")
        add("IMU", "Samples lost (resyncs)", f"{s['lost']} ({s['resync']})")
        last = cap["imu"][-1][2]
        add("IMU", "WHO_AM_I", f"0x{last['who1']:02X} / 0x{last['who2']:02X}")
        for i in range(2):
            okm = (s["ok"] >> i & 1).astype(bool)
            if okm.sum() > 10:
                am = np.linalg.norm(s["acc"][i][okm], axis=1)
                gs_ = s["gyr"][i][okm].std(0)
                add("IMU", f"IMU {i + 1} |a|", f"{am.mean():.3f} ± {am.std():.3f} g")
                add("IMU", f"IMU {i + 1} gyro σ (x, y, z)", f"{gs_[0]:.2f}, {gs_[1]:.2f}, {gs_[2]:.2f} °/s")
        add("Servo", "Angle range (command)", f"{s['srv_tgt'].min():.0f}–{s['srv_tgt'].max():.0f}°")
        modes = sorted({fm.SERVO_MODES.get(m, str(m)) for m in s["srv_mode"]})
        add("Servo", "Modes", ", ".join(modes))
    with open(os.path.join(out, "table_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([("group", "quantity", "value")] + R)
    with open(os.path.join(out, "table_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(f"| Group | Quantity | Value |\n|---|---|---|\n")
        fh.writelines(f"| {g} | {k} | {v} |\n" for g, k, v in R)
    tex = lambda x: (x.replace("%", r"\%").replace("±", r"$\pm$").replace("°", r"$^\circ$")
                      .replace("σ", r"$\sigma$").replace("|a|", r"$|a|$").replace("–", "--").replace("_", r"\_"))
    with open(os.path.join(out, "table_summary.tex"), "w", encoding="utf-8") as fh:
        fh.write("% needs \\usepackage{booktabs}\n\\begin{table}[t]\n\\centering\n"
                 f"\\caption{{Measured performance of the FPGA sensing system ({tex(name)})}}\n"
                 "\\label{tab:fpga_summary}\n\\begin{tabular}{llr}\n\\toprule\n"
                 "Group & Quantity & Value \\\\\n\\midrule\n")
        prev = None
        for g, k, v in R:
            if prev is not None and g != prev:
                fh.write("\\midrule\n")
            fh.write(f"{tex(g) if g != prev else ''} & {tex(k)} & {tex(v)} \\\\\n")
            prev = g
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("  wrote table_summary.tex / .md / .csv")
    for g, k, v in R:
        print(f"    {g:7s} {k:28s} {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="captures\\demo_*.bin (from the dashboard, key R)")
    ap.add_argument("--out", default=None, help="output folder (default figures\\<capture name>)")
    ap.add_argument("--baud", type=float, default=2e6)
    ap.add_argument("--frame", type=int, default=None, help="result-packet index for fig_scene (default: best)")
    ap.add_argument("--no-camera", action="store_true", help="never put the camera image in a figure")
    ap.add_argument("--include-person-frames", action="store_true",
                    help="allow camera frames with a detected person (default: not used)")
    a = ap.parse_args()
    style()
    name = os.path.splitext(os.path.basename(a.capture))[0]
    out = a.out or os.path.join(ROOT, "figures", name)
    os.makedirs(out, exist_ok=True)
    print("reading", a.capture)
    cap = load(a.capture)
    print(f"  {len(cap['res'])} result packets ({cap['bad_res']} bad), {len(cap['imu'])} IMU packets "
          f"({cap['bad_imu']} bad), timing {'yes' if cap['timed'] else 'no (.tim missing)'}")
    s = imu_series(cap["imu"])
    idx = fig_scene(cap, out, a)
    if idx is not None:
        print(f"  scene = result packet #{idx} (camera frame {cap['res'][idx][2]['cam_no']})")
    if s is not None:
        fig_imu(s, out, a)
        fig_servo(s, out, a)
    fig_timing(cap, out, a)
    table(cap, s, out, a, name)
    print("figures in", out)


if __name__ == "__main__":
    main()
