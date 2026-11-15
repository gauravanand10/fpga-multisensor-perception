"""
run_ref.py - run / check the pure-numpy FPGA reference models and write golden test vectors.

    cd <repository root>\\model\\quantization
    python fpga_ref\\run_ref.py --check            numpy fp32 == ONNX Runtime ?  (should print ~1e-6)
    python fpga_ref\\run_ref.py                    int8 vs ONNX Runtime on one camera frame -> picture
    python fpga_ref\\run_ref.py --image x.png      same on an image file
    python fpga_ref\\run_ref.py --report           per-layer table (MACs, weights, activations, int8 error)
    python fpga_ref\\run_ref.py --dump             golden vectors: every int8 tensor of one frame (.npz)
    python fpga_ref\\run_ref.py --dump --mem       + $readmemh .mem files per layer for HDL testbenches
Options: --model yolo|midas|both (default both), --port COM7, --no-show

Outputs go to fpga_ref\\out\\.
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from engine import Net   # noqa: E402

FPGA = os.path.join(ROOT, "models", "fpga")
OUT = os.path.join(HERE, "out")
MODELS = {"yolo": ("yolov8n_seg", "yolov8n-seg.onnx"), "midas": ("model_small", "model-small.onnx")}


# ------------------------------------------------------------------ input frame
def get_frame(a):
    if a.image:
        img = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"cannot read {a.image}")
        return img
    if not a.no_camera:
        try:
            from calibrate import grab_camera
            return grab_camera(1, a.port, a.baud)[0]
        except SystemExit as e:
            print("[camera]", e, "- using a saved calibration frame instead")
        except Exception as e:
            print("[camera] not available:", e, "- using a saved calibration frame instead")
    fs = sorted(glob.glob(os.path.join(HERE, "calib_frames", "*.png")))
    if not fs:
        raise SystemExit("no frame: connect the camera, use --image, or run calibrate.py --camera 40 first")
    last = os.path.basename(fs[-1]).rsplit("_", 1)[0]               # frames of the latest calibration run
    fs = [f for f in fs if os.path.basename(f).startswith(last)]
    # the frame where the float model sees the most objects (a frame of the ceiling tests nothing)
    from seg_teacher import SegTeacher
    seg = SegTeacher()
    best, score = fs[len(fs) // 2], -1.0
    for f in fs:
        objs = seg(cv2.imread(f, cv2.IMREAD_UNCHANGED))
        sc = len(objs) + sum(o["score"] for o in objs) * 0.01
        if sc > score:
            best, score = f, sc
    print(f"[frame] {os.path.basename(best)}  ({int(score)} objects, best of {len(fs)} saved frames)")
    return cv2.imread(best, cv2.IMREAD_UNCHANGED)


def make(kind, backend):
    if kind == "yolo":
        from seg_teacher import SegTeacher
        return SegTeacher(backend=backend)
    from mono_depth import MonoDepth
    return MonoDepth("midas", backend=backend)


def need_int8(name):
    if not os.path.exists(os.path.join(FPGA, name + "_int8.npz")):
        raise SystemExit(f"{name}_int8.npz missing - run: python fpga_ref\\calibrate.py --camera 40")


# ------------------------------------------------------------------ --check
def check(kinds):
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    for k in kinds:
        name, onnx_f = MODELS[k]
        net = Net(os.path.join(FPGA, name), "fp32")
        x = rng.random([1] + net.input_shapes[0][1:], dtype=np.float32)
        t = time.perf_counter(); y = net.run(x); dt = time.perf_counter() - t
        s = ort.InferenceSession(os.path.join(ROOT, "models", onnx_f), providers=["CPUExecutionProvider"])
        r = s.run(None, {s.get_inputs()[0].name: x})
        for i, (p, q) in enumerate(zip(y, r)):
            rel = float(np.abs(p - q).max() / (np.abs(q).max() + 1e-12))
            print(f"{name} output {i} {tuple(p.shape)}: max relative difference numpy-fp32 vs ONNX Runtime = {rel:.2e}"
                  f"  {'OK' if rel < 1e-4 else 'MISMATCH'}   ({dt:.2f} s numpy)")


# ------------------------------------------------------------------ compare picture
def draw_objs(img, objs, title):
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    rng = np.random.default_rng(3)
    for o in objs:
        c = tuple(int(x) for x in rng.integers(60, 255, 3))
        cnts, _ = cv2.findContours(o["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cnts, -1, c, 2)
        x0, y0 = int(max(0, o["box"][0])), int(max(12, o["box"][1]))
        cv2.putText(v, f"{o['name']} {o['score']:.2f}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(v, f"{o['name']} {o['score']:.2f}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    cv2.putText(v, title, (6, v.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return v


def depth_vis(r, title):
    r = (r - r.min()) / (np.ptp(r) + 1e-9)
    v = cv2.applyColorMap((r * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    cv2.putText(v, title, (6, v.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return v


def compare(kinds, img, show):
    rows = []
    if "yolo" in kinds:
        need_int8("yolov8n_seg")
        ref, q = make("yolo", "ort"), make("yolo", "int8")
        a = ref(img); t = time.perf_counter(); b = q(img); dt = time.perf_counter() - t
        print(f"\n[YOLOv8n-seg]  ONNX Runtime float: {len(a)} objects | numpy int8: {len(b)} objects ({dt:.1f} s)")
        for o in a:
            best = max(b, key=lambda p: (p["mask"] & o["mask"]).sum() / max(1, (p["mask"] | o["mask"]).sum()), default=None)
            iou = (best["mask"] & o["mask"]).sum() / max(1, (best["mask"] | o["mask"]).sum()) if best else 0.0
            print(f"   {o['name']:13s} {o['score']:.2f}  ->  int8: "
                  + (f"{best['name']:13s} {best['score']:.2f}  mask IoU {iou:.2f}" if best and iou > 0.3 else "not found"))
        rows.append(np.hstack([draw_objs(img, a, "ONNX Runtime (float)"), draw_objs(img, b, "numpy int8 (FPGA maths)")]))
    if "midas" in kinds:
        need_int8("model_small")
        ref, q = make("midas", "ort"), make("midas", "int8")
        a = ref(img); t = time.perf_counter(); b = q(img); dt = time.perf_counter() - t
        c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        print(f"\n[MiDaS small]  correlation int8 vs float = {c:.4f}  ({dt:.1f} s)")
        rows.append(np.hstack([depth_vis(a, "ONNX Runtime (float)"), depth_vis(b, f"numpy int8  corr {c:.3f}")]))
    if rows:
        w = max(r.shape[1] for r in rows)
        pic = np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows])
        os.makedirs(OUT, exist_ok=True)
        fn = os.path.join(OUT, time.strftime("compare_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(fn, pic); print("\nsaved", fn)
        if show:
            cv2.imshow("float (left) vs int8 FPGA reference (right) - any key", pic); cv2.waitKey(0); cv2.destroyAllWindows()


# ------------------------------------------------------------------ --report / --dump
def run_both(kind, img):
    """float and int8 runs of the same pre-processed input; returns (int8 net, fp32 tensors, int8 tensors, input)."""
    name = MODELS[kind][0]
    need_int8(name)
    m = make(kind, "int8"); m.sess.keep = True
    m(img)
    x = m.sess.last_input
    f = Net(os.path.join(FPGA, name), "fp32")
    _, Tf = f.run(x, keep=True)
    return m.sess.net, Tf, m.sess.last, x


def report(kind, img):
    q, Tf, Tq, x = run_both(kind, img)
    name = MODELS[kind][0]
    rows, tot_mac, tot_w = [], 0, 0
    for i, L in enumerate(q.layers):
        o = L["outputs"][0]
        a, b = Tf[o], Tq[o]
        mac = 0; wbytes = 0
        if L["op"] in ("Conv", "ConvTranspose"):
            W = q.P[o + "__w"]
            if L["op"] == "Conv":           # every output value: Cin/groups * kh * kw MACs
                mac = int(np.prod(b.shape[1:])) * int(np.prod(W.shape[1:]))
            else:                           # transposed conv (kernel == stride): every input pixel * all weights
                mac = int(np.prod(Tf[L["inputs"][0]].shape[2:])) * int(W.size)
            wbytes = W.size + 4 * W.shape[0 if L["op"] == "Conv" else 1] * 3     # weights + bias, M, sh
        snr = ""
        if a.dtype.kind == "f" and a.size > 1:
            bq = q.deq(o, b).astype(np.float64)
            err = ((a - bq) ** 2).mean()
            snr = f"{10 * np.log10((a.astype(np.float64) ** 2).mean() / max(err, 1e-30)):.1f}"
        tot_mac += mac; tot_w += wbytes
        rows.append(dict(idx=i, op=L["op"], output=o, shape="x".join(map(str, b.shape)),
                         kernel="x".join(map(str, L["attrs"].get("kernel_shape", []))),
                         stride="x".join(map(str, L["attrs"].get("strides", []))), groups=L["attrs"].get("group", ""),
                         int8=q.kind.get(o) == "q", MACs=mac, weight_bytes=wbytes,
                         act_bytes=int(b.size * b.dtype.itemsize), snr_dB_vs_float=snr))
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"{name}_layers.csv")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    conv = [r for r in rows if r["MACs"]]
    big = max(rows, key=lambda r: r["act_bytes"])
    print(f"\n[{name}] {len(rows)} layers ({len(conv)} conv), {tot_mac/1e6:.0f} M MACs per frame, "
          f"{tot_w/1e6:.2f} MB weights+params, largest activation {big['act_bytes']/1024:.0f} KB ({big['output']})")
    print(f"   at 100 MHz with N parallel int8 MACs: {tot_mac/1e8:.1f}/N s per frame "
          f"(e.g. 256 MACs -> {tot_mac/1e8/256*1000:.0f} ms ideal)")
    print("   saved", fn)


def mem_write(fn, arr):
    """$readmemh: one two's-complement hex value per line, C order."""
    a = np.asarray(arr).ravel()
    bits = a.dtype.itemsize * 8
    digits = bits // 4
    with open(fn, "w") as f:
        f.write("\n".join(format(int(v) & ((1 << bits) - 1), f"0{digits}x") for v in a))
        f.write("\n")


def dump(kind, img, mem):
    q, Tf, Tq, x = run_both(kind, img)
    name = MODELS[kind][0]
    d = os.path.join(OUT, f"golden_{name}"); os.makedirs(d, exist_ok=True)
    first = next(L for L in q.layers if L["op"] == "Conv")
    from engine import quantize
    xq = quantize(Tq[first["inputs"][0]], q.scale(first["inputs"][0]), q.qmax)
    tensors = {"input_float": x.astype(np.float32), "input_q": xq}
    index = []
    for i, L in enumerate(q.layers):
        for o in L["outputs"]:
            t = Tq[o]
            tensors[o] = t
            index.append(dict(idx=i, op=L["op"], tensor=o, shape=list(t.shape), dtype=str(t.dtype),
                              scale=(q.scale(o) if q.kind.get(o) == "q" else None),
                              inputs=L["inputs"]))
    np.savez_compressed(os.path.join(d, "tensors.npz"), **tensors)
    with open(os.path.join(d, "index.json"), "w") as f:
        json.dump(dict(model=name, qmax=q.qmax, input_tensor=first["inputs"][0], layers=index), f, indent=1)
    cv2.imwrite(os.path.join(d, "frame.png"), img)
    print(f"[{name}] golden vectors: {len(tensors)} tensors -> {os.path.join(d, 'tensors.npz')} (+ index.json, frame.png)")
    if mem:
        md = os.path.join(d, "mem"); os.makedirs(md, exist_ok=True)
        mem_write(os.path.join(md, "input_q.mem"), xq)
        n = 0
        for i, L in enumerate(q.layers):
            o = L["outputs"][0]
            if q.kind.get(o) != "q":
                continue
            tag = f"L{i:03d}_{L['op']}"
            for suf in ("__w", "__b", "__M", "__sh", "__lut", "__M0", "__sh0", "__M1", "__sh1"):
                if o + suf in q.P:
                    mem_write(os.path.join(md, f"{tag}{suf.replace('__', '_')}.mem"), q.P[o + suf])
            for k in range(len(L["inputs"])):
                if f"{o}__M{k}" in q.P and L["op"] == "Concat":
                    mem_write(os.path.join(md, f"{tag}_M{k}.mem"), q.P[f"{o}__M{k}"])
                    mem_write(os.path.join(md, f"{tag}_sh{k}.mem"), q.P[f"{o}__sh{k}"])
            for j, oo in enumerate(L["outputs"]):
                mem_write(os.path.join(md, f"{tag}_out{j}.mem"), Tq[oo])
            n += 1
        with open(os.path.join(md, "README.txt"), "w") as f:
            f.write("One file per value array, $readmemh format, C order (N,C,H,W activations; O,I,kh,kw weights;\n"
                    "ConvTranspose weights I,O,kh,kw). Two's complement hex: int8 = 2 digits, int16 = 4, int32 = 8.\n"
                    "L<idx>_<op>: idx = layer index in index.json / <model>_graph.json.\n"
                    "_w weights, _b bias, _M/_sh requant multiplier/shift (per output channel), _lut SiLU table\n"
                    "(index = q + qmax + 1), _M0/_sh0/_M1/_sh1 Add rescale, _M<k>/_sh<k> Concat input k rescale,\n"
                    "_out<j> expected layer output. input_q.mem = quantised network input.\n")
        print(f"   .mem files for {n} integer layers -> {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["yolo", "midas", "both"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--image", default=None)
    ap.add_argument("--no-camera", action="store_true", help="use a saved calibration frame")
    ap.add_argument("--port", default="COM7"); ap.add_argument("--baud", type=int, default=921_600)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    kinds = ["yolo", "midas"] if a.model == "both" else [a.model]
    if a.check:
        check(kinds); return
    img = get_frame(a)
    if a.report or a.dump:
        for k in kinds:
            if a.report:
                report(k, img)
            if a.dump:
                dump(k, img, a.mem)
        return
    compare(kinds, img, not a.no_show)


if __name__ == "__main__":
    main()
