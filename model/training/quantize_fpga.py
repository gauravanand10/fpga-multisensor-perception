"""
quantize_fpga.py - the trained U-Net -> int8 FPGA package + golden test vectors + size report.

    python quantize_fpga.py

Uses the FPGA tools in ..\\model\\quantization\\fpga_ref (the same ones used for YOLO):
  1. export_graph : tiny_unet.onnx -> models\\fpga\\tiny_unet_graph.json + tiny_unet_fp32.npz
  2. calibration  : runs the float model on your own frames + COCO validation photos,
                    records the range of every tensor -> models\\fpga\\tiny_unet_int8.npz
                    (int8 weights per output channel, int32 bias, multipliers / shifts)
  3. check        : int8 result vs float result on the validation frames
  4. golden files : one of your frames through the int8 network, every layer saved as
                    $readmemh .mem files -> ..\\golden_model\\data\\golden_vectors\\
  5. verify_mem   : recomputes every integer layer from the .mem files (must be 0 mismatches)
  6. report       : parameters, MACs, BRAM needed, speed estimate on the Nexys4 DDR
"""
import argparse, json, os, subprocess, sys
import numpy as np
import cv2

from config import CLASSES, DATA, FUSED, FPGA_MODELS, IGNORE, INPUT_HW, OUTPUT_HW, RUNS, ensure_fused_on_path

NAME = "tiny_unet"


def frames(limit_coco=200):
    """calibration / test frames (uint8 320x240) and labels."""
    X, Y = [], []
    for name, lim in (("own", None), ("coco_val", limit_coco)):
        p = os.path.join(DATA, f"{name}_images.npy")
        if os.path.exists(p):
            n = int(json.load(open(os.path.join(DATA, f"{name}_info.json")))["n"])
            xi = np.load(p, mmap_mode="r"); yi = np.load(os.path.join(DATA, f"{name}_labels.npy"), mmap_mode="r")
            idx = np.arange(n) if lim is None or n <= lim else np.linspace(0, n - 1, lim).astype(int)
            X += [np.array(xi[i]) for i in idx]; Y += [np.array(yi[i]) for i in idx]
    return X, Y


def peak_memory(net, T):
    """bytes of int8 tensors alive at the same time (each layer's inputs + outputs + kept skips)."""
    size = {t: v.nbytes for t, v in T.items() if net.kind.get(t) == "q" and t not in net.src}
    last = {}
    for i, L in enumerate(net.layers):
        for t in L["inputs"]:
            t = net.src.get(t, t)
            if t in size:
                last[t] = i
    alive, peak = set(), 0
    for i, L in enumerate(net.layers):
        alive |= {o for o in L["outputs"] if o in size}
        peak = max(peak, sum(size[t] for t in alive))
        alive = {t for t in alive if last.get(t, -1) > i}
    return peak, max(size.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, default=99.99, help="activation range percentile (calibration)")
    ap.add_argument("--abits", type=int, default=8, choices=[8, 16])
    a = ap.parse_args()
    ensure_fused_on_path()
    import export_graph, calibrate, run_ref
    from engine import Net, quantize
    from seg_unet import preprocess

    onnx_path = os.path.join(FUSED, "models", "tiny_unet.onnx")
    if not os.path.exists(onnx_path):
        raise SystemExit("models\\tiny_unet.onnx missing - run export_onnx.py first")
    # 1. graph + float package
    export_graph.export(onnx_path)
    base = os.path.join(FPGA_MODELS, NAME)

    # 2. calibration
    X, Y = frames()
    if not X:
        raise SystemExit("no frames in data\\ - run prepare_coco.py / label_own_frames.py first")
    st = calibrate.Stats(a.pct)
    fnet = Net(base, "fp32")
    for i, img in enumerate(X):
        fnet.run(preprocess(img, INPUT_HW), record=st)
        print(f"\r[calib] float pass {i + 1}/{len(X)}", end="")
    print()
    pkg = calibrate.build(base, st.amax, a.abits)
    np.savez(base + "_int8.npz", **pkg)
    print(f"[calib] written {base}_int8.npz")

    # 3. int8 vs float
    qnet = Net(base, "int8")
    C = len(CLASSES); agree = []; iou = {k: [np.zeros(C), np.zeros(C)] for k in ("float", "int8")}
    for img, lab in zip(X, Y):
        x = preprocess(img, INPUT_HW)
        pf = fnet.run(x)[0][0].argmax(0); pq = qnet.run(x)[0][0].argmax(0)
        agree.append((pf == pq).mean())
        y = cv2.resize(lab, (OUTPUT_HW[1], OUTPUT_HW[0]), interpolation=cv2.INTER_NEAREST); v = y != IGNORE
        for k, p in (("float", pf), ("int8", pq)):
            for c in range(C):
                iou[k][0][c] += ((p == c) & (y == c) & v).sum(); iou[k][1][c] += (((p == c) | (y == c)) & v).sum()
    miou = {k: float((i / np.maximum(u, 1))[1:][u[1:] > 0].mean()) for k, (i, u) in iou.items()}
    print(f"[check] int8 and float give the same class on {np.mean(agree) * 100:.2f} % of pixels; "
          f"mean IoU float {miou['float']:.3f}, int8 {miou['int8']:.3f}  ({len(X)} frames)")

    # 4. golden vectors: the frame with the most object pixels (own frames first)
    k = int(np.argmax([((y > 0) & (y != IGNORE)).mean() for y in Y]))
    x = preprocess(X[k], INPUT_HW)
    _, T = qnet.run(x, keep=True)
    d = os.path.join(FUSED, "fpga_ref", "out", f"golden_{NAME}"); md = os.path.join(d, "mem")
    os.makedirs(md, exist_ok=True)
    first = next(L for L in qnet.layers if L["op"] == "Conv")
    xq = quantize(T[first["inputs"][0]], qnet.scale(first["inputs"][0]), qnet.qmax)
    tensors = {"input_float": x, "input_q": xq}
    index = []
    for i, L in enumerate(qnet.layers):
        for o in L["outputs"]:
            tensors[o] = T[o]
            index.append(dict(idx=i, op=L["op"], tensor=o, shape=list(T[o].shape), dtype=str(T[o].dtype),
                              scale=(qnet.scale(o) if qnet.kind.get(o) == "q" else None), inputs=L["inputs"]))
    np.savez_compressed(os.path.join(d, "tensors.npz"), **tensors)
    with open(os.path.join(d, "index.json"), "w") as f:
        json.dump(dict(model=NAME, qmax=qnet.qmax, input_tensor=first["inputs"][0], layers=index), f, indent=1)
    cv2.imwrite(os.path.join(d, "frame.png"), X[k])
    run_ref.mem_write(os.path.join(md, "input_q.mem"), xq)
    n_mem = 0
    for i, L in enumerate(qnet.layers):
        o = L["outputs"][0]
        if qnet.kind.get(o) != "q":
            continue
        tag = f"L{i:03d}_{L['op']}"
        for suf in ("__w", "__b", "__M", "__sh", "__lut", "__M0", "__sh0", "__M1", "__sh1"):
            if o + suf in qnet.P:
                run_ref.mem_write(os.path.join(md, f"{tag}{suf.replace('__', '_')}.mem"), qnet.P[o + suf])
        if L["op"] == "Concat":
            for j in range(len(L["inputs"])):
                run_ref.mem_write(os.path.join(md, f"{tag}_M{j}.mem"), qnet.P[f"{o}__M{j}"])
                run_ref.mem_write(os.path.join(md, f"{tag}_sh{j}.mem"), qnet.P[f"{o}__sh{j}"])
        for j, oo in enumerate(L["outputs"]):
            run_ref.mem_write(os.path.join(md, f"{tag}_out{j}.mem"), T[oo])
        n_mem += 1
    print(f"[golden] {n_mem} integer layers -> {md}")

    # 5. independent bit-exact check
    r = subprocess.run([sys.executable, os.path.join(FUSED, "fpga_ref", "verify_mem.py"), "--model", "unet"],
                       capture_output=True, text=True)
    print("[verify]", (r.stdout.strip().splitlines() or ["(no output)"])[-1], r.stderr.strip()[-400:])

    # 6. report
    macs = 0
    for L in qnet.layers:
        if L["op"] == "Conv":
            W = qnet.P[L["outputs"][0] + "__w"]
            macs += int(np.prod(T[L["outputs"][0]].shape[1:])) * int(np.prod(W.shape[1:]))
    wbytes = sum(v.nbytes for kk, v in qnet.P.items() if kk.endswith("__w"))
    pbytes = sum(v.nbytes for kk, v in qnet.P.items() if kk.endswith(("__b", "__M", "__sh")))
    peak, biggest = peak_memory(qnet, T)
    lines = [f"Tiny U-Net on the Nexys4 DDR (Artix-7 XC7A100T: 4,860 Kbit = 607 KB BRAM, 240 DSP)",
             f"  classes            {CLASSES}",
             f"  input / output     {INPUT_HW[1]}x{INPUT_HW[0]} grey -> {OUTPUT_HW[1]}x{OUTPUT_HW[0]} label map",
             f"  int8 weights       {wbytes / 1024:.0f} KB  (+ bias/multipliers/shifts {pbytes / 1024:.1f} KB)",
             f"  activations        peak {peak / 1024:.0f} KB alive at once, largest tensor {biggest / 1024:.0f} KB",
             f"  BRAM total         about {(wbytes + pbytes + peak) / 1024:.0f} KB of 607 KB "
             f"({(wbytes + pbytes + peak) / 607 / 1024 * 100:.0f} %) - everything on-chip, no DDR needed",
             f"  multiply-adds      {macs / 1e6:.0f} M per frame",
             f"  speed (100 MHz)    64 MACs/cycle -> {macs / 64 / 1e5:.1f} ms;  128 -> {macs / 128 / 1e5:.1f} ms;  "
             f"200 -> {macs / 200 / 1e5:.1f} ms per frame (ideal)",
             f"  int8 vs float      same class on {np.mean(agree) * 100:.2f} % of pixels; mean IoU float {miou['float']:.3f}, int8 {miou['int8']:.3f}"]
    print("\n" + "\n".join(lines))
    os.makedirs(os.path.join(RUNS, NAME), exist_ok=True)
    with open(os.path.join(RUNS, NAME, "fpga_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved {os.path.join(RUNS, NAME, 'fpga_report.txt')}")


if __name__ == "__main__":
    main()
