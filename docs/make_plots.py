"""
make_plots.py - every result graph of the project, drawn from stored data in one journal style

    python docs/make_plots.py          (run from the repository root, about 10 s)

Reads only files already in the repository; nothing is measured or simulated here.

  model/results/           training_curve, class_iou_training, iou_per_class, pixel_agreement, confusion_matrix
  golden_model/results/end_to_end/   distance_error
  hw/results/              resource_utilization, resource_by_module, timing_slack, frame_latency

Inputs: model/results/training_log.csv, confusion_matrix.csv; golden_model/results/accuracy/
float_vs_int8_metrics.json, per_image_metrics.csv; golden_model/results/end_to_end/end_to_end_summary.json;
hw/vivado/reports/utilization.rpt, utilization_hierarchical.rpt, summary.txt.
"""
import csv, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)

INK, INK2, GRID, SHADE = "#1a1a1a", "#555555", "#e2e2e2", "#efefef"
CLASSES = ["background", "person", "chair", "tv", "laptop", "bottle"]
LABEL = {"background": "Background", "person": "Person", "chair": "Chair", "tv": "TV",
         "laptop": "Laptop", "bottle": "Bottle"}
CLASS_COL = {"background": "#505050", "person": "#2a78d6", "chair": "#eb6834", "tv": "#1baf7a",
             "laptop": "#eda100", "bottle": "#e87ba4"}
FLOAT_COL, INT8_COL = "#9bb8dc", "#2a5a9e"      # one hue, two steps: float32 light, int8 dark
BAR_COL = "#2a78d6"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def save(fig, *path):
    out = P(*path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("  wrote", os.path.relpath(out, ROOT).replace(os.sep, "/"))


# ------------------------------------------------------------------ model
def training():
    rows = list(csv.DictReader(open(P("model", "results", "training_log.csv"))))
    ep = np.arange(1, len(rows) + 1)          # float epochs, then QAT epochs (QAT restarts its own count)
    loss = np.array([float(r["loss"]) for r in rows])
    miou = np.array([float(r["coco_miou"]) for r in rows])
    qat0 = sum(r["phase"] == "float" for r in rows) + 0.5

    def shade(ax):
        ax.axvspan(qat0, ep.max() + 0.5, color=SHADE, lw=0, zorder=0)
        ax.set_xlim(0.5, ep.max() + 0.5)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True)
    a1.plot(ep, loss, color=BAR_COL, lw=1.4)
    a1.set_ylabel("Training loss")
    a2.plot(ep, miou, color=BAR_COL, lw=1.4)
    a2.set_ylabel("mIoU (COCO val)")
    a2.set_xlabel("Epoch")
    for a in (a1, a2):
        shade(a)
    a1.text(qat0 - 1, 0.95, "QAT \u2192", transform=a1.get_xaxis_transform(), ha="right", va="top", fontsize=7.5, color=INK2)
    fig.align_ylabels((a1, a2))
    fig.tight_layout(h_pad=0.6)
    save(fig, "model", "results", "training_curve.png")

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for c in CLASSES:
        ax.plot(ep, [float(r["coco_" + c]) for r in rows], color=CLASS_COL[c], lw=1.3,
                label=LABEL[c])
    shade(ax)
    ax.text(qat0 - 1, 0.97, "QAT \u2192", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=7.5, color=INK2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU (COCO val)"); ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.6)
    fig.tight_layout()
    save(fig, "model", "results", "class_iou_training.png")


def accuracy():
    m = json.load(open(P("golden_model", "results", "accuracy", "float_vs_int8_metrics.json")))
    cls = CLASSES[1:]
    names = [LABEL[c] for c in cls] + ["Mean"]
    f = [m["iou_float"][c] for c in cls] + [m["miou5_float"]]
    q = [m["iou_int8"][c] for c in cls] + [m["miou5_int8"]]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.bar(x - w / 2 - 0.01, f, w, color=FLOAT_COL, label="float32", zorder=3)
    ax.bar(x + w / 2 + 0.01, q, w, color=INT8_COL, label="int8 (FPGA)", zorder=3)
    for xi, v in zip(x + w / 2 + 0.01, q):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.axvline(len(cls) - 0.5, color=GRID, lw=0.8)
    ax.set_xticks(x, names); ax.grid(axis="x", visible=False)
    ax.set_ylabel("IoU (COCO val)"); ax.set_ylim(0, 0.6)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "model", "results", "iou_per_class.png")

    rows = list(csv.DictReader(open(P("golden_model", "results", "accuracy", "per_image_metrics.csv"))))
    ag = np.array([100 * float(r["agreement_float_int8"]) for r in rows])
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    lo = max(80, np.floor(ag.min()))
    ax.hist(np.clip(ag, lo, 100), bins=np.arange(lo, 100.01, 0.5), color=BAR_COL, edgecolor="white",
            linewidth=0.6, zorder=3)
    med = np.median(ag)
    ax.axvline(med, color=INK, lw=0.9, ls="--")
    ax.text(med - 0.3, ax.get_ylim()[1] * 0.93, f"median {med:.1f} %", ha="right", va="top", fontsize=7.5,
            color=INK2)
    ax.set_xlabel("Pixel agreement, float32 vs int8 (%)"); ax.set_ylabel("Images")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "model", "results", "pixel_agreement.png")


def confusion():
    rows = [r for r in csv.reader(open(P("model", "results", "confusion_matrix.csv"))) if r and not r[0].startswith("#")]
    cm = np.array(rows, float)
    pct = 100 * cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if pct[i, j] > 55 else INK)
    lab = [LABEL[c] for c in CLASSES]
    ax.set_xticks(range(6), lab, rotation=40, ha="right"); ax.set_yticks(range(6), lab)
    ax.set_xlabel("Predicted (int8)"); ax.set_ylabel("Ground truth")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Row %", fontsize=8); cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "model", "results", "confusion_matrix.png")


# ------------------------------------------------------------------ golden model
def distance():
    s = json.load(open(P("golden_model", "results", "end_to_end", "end_to_end_summary.json")))
    objs = [o for o in s["objects"] if o.get("error_mm") is not None]
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    lim = [0.5, 3.6]
    ax.fill_between(lim, [v * 0.95 for v in lim], [v * 1.05 for v in lim], color=SHADE, lw=0, zorder=1,
                    label="±5 %")
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--", zorder=2)
    for c in CLASSES[1:]:
        pts = [(o["true_depth_mm"] / 1000, o["fpga_distance_mm"] / 1000) for o in objs if o["cls"] == c]
        if pts:
            ax.scatter(*zip(*pts), s=30, color=CLASS_COL[c], edgecolor="white", linewidth=1.0, zorder=3,
                       label=LABEL[c])
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("True depth (m)"); ax.set_ylabel("FPGA distance (m)")
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, "golden_model", "results", "end_to_end", "distance_error.png")


# ------------------------------------------------------------------ hardware
def hier_table():
    t = {}
    for line in open(P("hw", "vivado", "reports", "utilization_hierarchical.rpt")):
        if not line.startswith("|") or "Instance" in line:
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 10:
            continue
        try:
            v = [float(x) for x in c[2:10]]
        except ValueError:
            continue
        t[c[0]] = dict(lut=v[0], ff=v[4], bram=v[5] + v[6] / 2, dsp=v[7])
    return t


def resources():
    used = {}
    for line in open(P("hw", "vivado", "reports", "utilization.rpt")):
        for key, name in (("Slice LUTs ", "LUT"), ("Slice Registers ", "FF"), ("Block RAM Tile", "BRAM"),
                          ("DSPs ", "DSP"), ("Bonded IOB", "I/O")):
            if line.startswith("| " + key) and name not in used:
                c = [x.strip() for x in line.strip().strip("|").split("|")]
                used[name] = (float(c[1]), float(c[-2]))
    order = ["LUT", "FF", "BRAM", "DSP", "I/O"]
    pct = [100 * used[k][0] / used[k][1] for k in order]
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    y = np.arange(len(order))[::-1]
    ax.barh(y, [100] * len(order), 0.56, color=SHADE, zorder=2)
    ax.barh(y, pct, 0.56, color=BAR_COL, zorder=3)
    for yi, k, p in zip(y, order, pct):
        u, a = used[k]
        ax.text(101.5, yi, f"{p:.0f} %", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, order); ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Utilization of xc7a100t (%)"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "hw", "results", "resource_utilization.png")

    t = hier_table()
    def add(*names):
        return {k: sum(t[n][k] for n in names) for k in ("lut", "ff", "bram", "dsp")}
    banks = [n for n in t if n.startswith("g_bank[")]
    groups = [("U-Net engine", add("u_eng")),
              ("Weight ROM", add("u_wrom")),
              ("Feature memories", add(*banks)),
              ("U-Net control", add("(u_core)", "u_prom")),
              ("Post-processing", add("u_pp")),
              ("Camera SPI rx", add("u_cam")),
              ("ToF SPI rx", add("u_tof")),
              ("UART", add("u_pk", "u_tx", "u_arb")),
              ("IMU", add("u_imu", "u_ififo", "u_ipk")),
              ("Servo", add("u_servo", "u_sctl")),
              ("Top level", add("(nexys4_top)", "(u_sys)"))]
    kinds = [("lut", "LUT"), ("ff", "FF"), ("bram", "BRAM tiles"), ("dsp", "DSP")]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True)
    y = np.arange(len(groups))[::-1]
    for ax, (k, title) in zip(axs, kinds):
        v = [g[1][k] for g in groups]
        ax.barh(y, v, 0.62, color=BAR_COL, zorder=3)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, max(v) * 1.08)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
        ax.locator_params(axis="x", nbins=3)
    axs[0].set_yticks(y, [g[0] for g in groups])
    fig.tight_layout(w_pad=0.8)
    save(fig, "hw", "results", "resource_by_module.png")


def timing():
    rows = []
    for line in open(P("hw", "vivado", "reports", "summary.txt")):
        mt = re.match(r"\s+(.+?)\s+worst slack\s+([-\d.]+) ns", line)
        if mt and mt.group(1) != "system control":
            rows.append((mt.group(1), float(mt.group(2))))
    rows.sort(key=lambda r: r[1])
    names = [r[0][0].upper() + r[0][1:] for r in rows]
    names = [n.replace("Spi", "SPI").replace("Imu", "IMU").replace("Uart", "UART").replace("Tof", "ToF") for n in names]
    v = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, v, 0.62, color=BAR_COL, zorder=3)
    for yi, s in zip(y, v):
        ax.text(s + 0.1, yi, f"{s:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, names); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(v) * 1.15)
    ax.set_xlabel("Worst setup slack at 100 MHz (ns)"); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "hw", "results", "timing_slack.png")


def latency():
    s = json.load(open(P("golden_model", "results", "end_to_end", "end_to_end_summary.json")))
    short = {"ESP32-CAM": "Camera SPI (8 MHz)", "ToF ESP32": "ToF SPI (4 MHz)", "U-Net": "U-Net accelerator",
             "Objects": "Objects + distance"}
    rows = []
    for t in s["timing_budget_ms"]:
        st = t["stage"]
        name = next((v for k, v in short.items() if st.startswith(k)), None)
        if name is None:
            name = "UART, with image" if "with image" in st else "UART, no image"
        rows.append((name, t["ms"], "sim" if "simulation" in t["source"] else "link"))
    fig, ax = plt.subplots(figsize=(3.8, 2.4))
    y = np.arange(len(rows))[::-1]
    col = {"link": "#9bb8dc", "sim": "#2a5a9e"}
    for yi, (n, ms, src) in zip(y, rows):
        ax.barh(yi, ms, 0.62, color=col[src], zorder=3)
        ax.text(ms + 1.5, yi, f"{ms:.1f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    ax.set_xlabel("Time per frame (ms)"); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col["sim"], label="RTL simulation"), Patch(color=col["link"], label="Link rate")],
              loc="upper right")
    fig.tight_layout()
    save(fig, "hw", "results", "frame_latency.png")


if __name__ == "__main__":
    print("make_plots.py: drawing result graphs from stored data")
    training(); accuracy(); confusion(); distance(); resources(); timing(); latency()
    print("done")
