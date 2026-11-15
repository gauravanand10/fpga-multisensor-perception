"""
verify_mem.py - a Python 'testbench' that recomputes the integer layers ONLY from the
$readmemh .mem files written by  run_ref.py --dump --mem  and the rules in INT8_SPEC.md,
with its own simple loop code (independent of engine.py). Every layer must match exactly.
An HDL testbench does the same: load _w/_b/_M/_sh + the input .mem, run the hardware,
compare with _out0.mem.

    python fpga_ref\\verify_mem.py                  (YOLOv8n-seg)
    python fpga_ref\\verify_mem.py --model midas
    python fpga_ref\\verify_mem.py --model unet     (the small U-Net, after model\\training\\quantize_fpga.py)
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(path, bits):
    v = np.array([int(l, 16) for l in open(path).read().split()], np.int64)
    return np.where(v >= 1 << (bits - 1), v - (1 << bits), v)


def requant(acc, M, sh, qmax):
    """q = clamp((acc * M + 2^(sh-1)) >> sh, -qmax, qmax)   (arithmetic shift right)"""
    return np.clip((acc * M + (np.int64(1) << (sh - 1))) >> sh, -qmax, qmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo", choices=["yolo", "midas", "unet"])
    ap.add_argument("--layers", type=int, default=0, help="stop after this many checked layers (0 = all)")
    a = ap.parse_args()
    name = {"yolo": "yolov8n_seg", "midas": "model_small", "unet": "tiny_unet"}[a.model]
    d = os.path.join(HERE, "out", f"golden_{name}")
    md = os.path.join(d, "mem")
    if not os.path.isdir(md):
        raise SystemExit(f"{md} missing - run: python fpga_ref\\run_ref.py --dump --mem --model {a.model}")
    idx = json.load(open(os.path.join(d, "index.json")))
    graph = json.load(open(os.path.join(HERE, "..", "models", "fpga", f"{name}_graph.json")))
    qmax = idx["qmax"]; abits = 8 if qmax <= 127 else 16
    shape = {e["tensor"]: e["shape"] for e in idx["layers"]}
    shape.setdefault(idx["input_tensor"], graph["io"]["inputs"][0][1])   # quantised input of the first conv
    where = {idx["input_tensor"]: os.path.join(md, "input_q.mem")}      # tensor -> .mem file
    outs_of = {}
    for e in idx["layers"]:
        outs_of.setdefault(e["idx"], []).append(e["tensor"])
    for i, L in enumerate(graph["layers"]):
        for j, t in enumerate(outs_of.get(i, [])):
            f = os.path.join(md, f"L{i:03d}_{L['op']}_out{j}.mem")
            if os.path.exists(f):
                where[t] = f
    act = lambda t: rd(where[t], abits).reshape(shape[t][1:])
    ok = bad = 0
    for i, L in enumerate(graph["layers"]):
        op = L["op"]; tag = os.path.join(md, f"L{i:03d}_{op}")
        if not os.path.exists(tag + "_out0.mem") or op not in ("Conv", "SiLU", "Add", "Concat"):
            continue
        ins = [t for t in L["inputs"] if t and t not in L.get("const_inputs", [])]
        if not all(t in where for t in ins):
            continue
        exp = rd(tag + "_out0.mem", abits).reshape(shape[L["outputs"][0]][1:])
        if op == "Conv":
            x = act(ins[0])
            W = rd(tag + "_w.mem", 8)
            b = rd(tag + "_b.mem", 32); M = rd(tag + "_M.mem", 32); sh = rd(tag + "_sh.mem", 32)
            O = b.size; g = L["attrs"].get("group", 1); kh, kw = L["attrs"]["kernel_shape"]
            W = W.reshape(O, x.shape[0] // g, kh, kw)
            s0, s1 = L["attrs"].get("strides", [1, 1]); p = L["attrs"].get("pads", [0, 0, 0, 0])
            dh, dw = L["attrs"].get("dilations", [1, 1])
            xp = np.pad(x, ((0, 0), (p[0], p[2]), (p[1], p[3])))
            Ho, Wo = exp.shape[1:]
            acc = np.zeros((O, Ho, Wo), np.int64)
            cg, og = x.shape[0] // g, O // g
            for gi in range(g):
                for c in range(cg):
                    for u in range(kh):
                        for v in range(kw):
                            patch = xp[gi * cg + c, u * dh: u * dh + s0 * (Ho - 1) + 1: s0, v * dw: v * dw + s1 * (Wo - 1) + 1: s1]
                            acc[gi * og:(gi + 1) * og] += W[gi * og:(gi + 1) * og, c, u, v][:, None, None] * patch[None]
            y = requant(acc + b[:, None, None], M[:, None, None], sh[:, None, None], qmax)
        elif op == "SiLU":
            y = rd(tag + "_lut.mem", abits)[act(ins[0]) + qmax + 1]
        elif op == "Add":
            r = [(act(t) * rd(f"{tag}_M{k}.mem", 32)[0] + (1 << (rd(f"{tag}_sh{k}.mem", 32)[0] - 1))) >> rd(f"{tag}_sh{k}.mem", 32)[0]
                 for k, t in enumerate(ins)]
            y = np.clip(r[0] + r[1], -qmax, qmax)
        else:  # Concat
            ax = L["attrs"]["axis"] - 1 if L["attrs"]["axis"] > 0 else L["attrs"]["axis"]     # no batch dim here
            y = np.concatenate([requant(act(t), rd(f"{tag}_M{k}.mem", 32)[0], rd(f"{tag}_sh{k}.mem", 32)[0], qmax)
                                for k, t in enumerate(ins)], axis=ax)
        same = np.array_equal(y, exp)
        ok += same; bad += not same
        print(f"L{i:03d} {op:8s} {'x'.join(map(str, exp.shape)):>14s}  {'OK' if same else 'MISMATCH (%d values)' % int((y != exp).sum())}")
        if a.layers and ok + bad >= a.layers:
            break
    print(f"\n{ok} layers bit-exact, {bad} mismatches")


if __name__ == "__main__":
    main()
