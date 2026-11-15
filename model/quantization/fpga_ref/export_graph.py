"""
export_graph.py - ONNX model -> FPGA-friendly float package (numpy only to read it back).

    python fpga_ref\\export_graph.py                       (all models in models\\)
    python fpga_ref\\export_graph.py models\\yolov8n-seg.onnx

Writes models\\fpga\\<name>_fp32.npz and models\\fpga\\<name>_graph.json.

Compared with the raw ONNX graph, the exported graph is simplified the way hardware
would implement it:
  * BatchNormalization after a Conv is folded into the Conv weights / bias (MiDaS);
  * Constant nodes become stored tensors;
  * Sigmoid(x) * x is fused into one 'SiLU' layer (YOLOv8) -> one 256-entry LUT in int8.
The result is numerically the same network (checked by run_ref.py --check against ONNX Runtime).

The int8 version (what the FPGA runs) is made from this package by calibrate.py.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")
OUT = os.path.join(MODELS, "fpga")


def safe(k):
    return k.replace("/", "_").replace(".", "_").replace(":", "_")


def _attrs(n):
    from onnx import helper
    out = {}
    for a in n.attribute:
        if a.name == "value":
            continue
        v = helper.get_attribute_value(a)
        if isinstance(v, bytes):
            v = v.decode()
        elif hasattr(v, "__len__") and not isinstance(v, str):
            v = [x.decode() if isinstance(x, bytes) else x for x in v]
        out[a.name] = v
    return out


def export(onnx_path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path)
    name = os.path.splitext(os.path.basename(onnx_path))[0].replace("-", "_")
    opset = max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    T = {i.name: numpy_helper.to_array(i).copy() for i in m.graph.initializer}
    nodes = []
    for n in m.graph.node:
        if n.op_type == "Constant":
            T[n.output[0]] = numpy_helper.to_array(n.attribute[0].t).copy()
            continue
        nodes.append(dict(op=n.op_type, name=n.name or n.output[0], inputs=list(n.input),
                          outputs=list(n.output), attrs=_attrs(n)))

    def consumers(t):
        return [x for x in nodes if t in x["inputs"]]

    # ---- fold BatchNormalization into the preceding Conv ----
    n_bn = 0
    for bn in [x for x in nodes if x["op"] == "BatchNormalization"]:
        src = [x for x in nodes if bn["inputs"][0] in x["outputs"]]
        if not src or src[0]["op"] != "Conv" or len(consumers(bn["inputs"][0])) != 1:
            continue
        cv = src[0]
        g, beta, mean, var = (T[k].astype(np.float64) for k in bn["inputs"][1:5])
        k = g / np.sqrt(var + bn["attrs"].get("epsilon", 1e-5))
        W = T[cv["inputs"][1]].astype(np.float64)
        b = T[cv["inputs"][2]].astype(np.float64) if len(cv["inputs"]) > 2 and cv["inputs"][2] else np.zeros(W.shape[0])
        wname = cv["inputs"][1] + "_bnfold"; bname = cv["outputs"][0] + "_bias_bnfold"
        T[wname] = (W * k[:, None, None, None]).astype(np.float32)
        T[bname] = ((b - mean) * k + beta).astype(np.float32)
        cv["inputs"] = [cv["inputs"][0], wname, bname]
        cv["outputs"] = [bn["outputs"][0]]
        nodes.remove(bn); n_bn += 1

    # ---- fuse x * sigmoid(x) -> SiLU ----
    n_silu = 0
    for sg in [x for x in nodes if x["op"] == "Sigmoid"]:
        x_in, s_out = sg["inputs"][0], sg["outputs"][0]
        muls = [x for x in consumers(s_out) if x["op"] == "Mul" and set(x["inputs"]) == {x_in, s_out}]
        if len(muls) == 1 and len(consumers(s_out)) == 1:
            mu = muls[0]
            idx = nodes.index(mu)
            nodes[idx] = dict(op="SiLU", name=mu["name"], inputs=[x_in], outputs=mu["outputs"], attrs={})
            nodes.remove(sg); n_silu += 1

    # ---- keep only tensors that are used, rename to safe keys ----
    used = {i for x in nodes for i in x["inputs"] if i}
    params = {safe(k): v for k, v in T.items() if k in used}
    for x in nodes:
        x["inputs"] = [safe(i) if i else "" for i in x["inputs"]]
        x["outputs"] = [safe(o) for o in x["outputs"]]
        x["const_inputs"] = [i for i in x["inputs"] if i in params]
    for x in nodes:
        for k, v in list(x["attrs"].items()):
            if not isinstance(v, (int, float, str, list)):
                x["attrs"][k] = str(v)
    io = dict(inputs=[(safe(i.name), [d.dim_value for d in i.type.tensor_type.shape.dim])
                      for i in m.graph.input if i.name not in T],
              outputs=[(safe(o.name), [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{name}_fp32.npz"),
             **{k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in params.items()})
    with open(os.path.join(OUT, f"{name}_graph.json"), "w") as f:
        json.dump(dict(model=os.path.basename(onnx_path), opset=opset, io=io, layers=nodes), f, indent=1)
    n_par = int(sum(v.size for k, v in params.items() if v.dtype.kind == "f" and v.ndim >= 1))
    ops = {}
    for x in nodes:
        ops[x["op"]] = ops.get(x["op"], 0) + 1
    print(f"{name}: {len(nodes)} layers ({n_bn} BatchNorm folded, {n_silu} SiLU fused), "
          f"{n_par/1e6:.2f} M parameters\n   ops: {ops}")
    return name


if __name__ == "__main__":
    targets = sys.argv[1:] or [os.path.join(MODELS, f) for f in ("yolov8n-seg.onnx", "model-small.onnx")
                               if os.path.exists(os.path.join(MODELS, f))]
    for t in targets:
        export(t)
    print("written to", OUT)
