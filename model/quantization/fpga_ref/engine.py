"""
engine.py - pure-numpy reference inference for the exported networks (no ONNX Runtime,
no PyTorch). This is the GOLDEN MODEL for the FPGA: every layer can be dumped and compared
with the hardware, bit for bit in int8 mode.

    from engine import Net
    net = Net("models/fpga/yolov8n_seg", "int8")        # or "fp32"
    outputs = net.run(x)                                # x: float32 NCHW, like ONNX Runtime
    outputs, tensors = net.run(x, keep=True)            # + every intermediate tensor

Modes
  fp32 : <name>_fp32.npz  - float maths, same results as ONNX Runtime (checked by run_ref.py --check)
  int8 : <name>_int8.npz  - integer maths exactly as described in INT8_SPEC.md:
         int8 activations (symmetric, zero point 0, one scale per tensor),
         int8 weights (one scale per output channel), int32 bias, int32 accumulator,
         requantisation  q_out = clamp((acc * M + 2^(sh-1)) >> sh, -127, 127)  per output channel,
         SiLU as a 256-entry int8 lookup table, Add / Concat with the same fixed-point rescale.
         Layers that are not integer (YOLO box decoding, softmax, MiDaS bilinear upsampling,
         input normalisation) run in float on dequantised values - they are host / post-processing
         work, or would use a small fixed-point block on the FPGA.
"""
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

QMAX = 127
# ops that pass int8 through unchanged (same scale)
PASS_OPS = {"Relu", "Clip", "MaxPool", "Split", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Flatten"}
INT_OPS = {"Conv", "ConvTranspose", "SiLU", "Add", "Concat"} | PASS_OPS


# ------------------------------------------------------------------ helpers
def adtype(qmax):
    return np.int8 if qmax <= 127 else np.int16


def quantize(x, s, qmax=QMAX):
    """float -> int8 (or int16), symmetric, round half up."""
    return np.clip(np.floor(x / s + 0.5), -qmax, qmax).astype(adtype(qmax))


def mult_shift(m):
    """real multiplier(s) m > 0  ->  (M, sh) with m ~= M * 2^-sh, 2^14 <= M < 2^15 (fits an 18-bit DSP input)."""
    m = np.atleast_1d(np.asarray(m, np.float64))
    e = np.floor(np.log2(np.maximum(m, 1e-30)))
    sh = (14 - e).astype(np.int64)
    M = np.round(m * np.exp2(sh)).astype(np.int64)
    over = M >= (1 << 15)
    M[over] >>= 1; sh[over] -= 1
    if (sh < 1).any() or (sh > 62).any():
        raise ValueError("multiplier out of range")
    return M.astype(np.int32), sh.astype(np.int32)


def rq_raw(acc, M, sh):
    """fixed-point rescale without clamping: (acc*M + 2^(sh-1)) >> sh  (arithmetic shift)."""
    acc = acc.astype(np.int64)
    M = M.astype(np.int64); sh = sh.astype(np.int64)
    return (acc * M + (np.int64(1) << (sh - 1))) >> sh


def rq(acc, M, sh, qmax=QMAX):
    return np.clip(rq_raw(acc, M, sh), -qmax, qmax).astype(adtype(qmax))


def ch(v, nd=4):
    """per-channel vector -> broadcastable over NCHW."""
    return np.asarray(v).reshape((1, -1) + (1,) * (nd - 2))


# ------------------------------------------------------------------ float / integer kernels
def conv2d(x, W, b, a, dtype):
    """x (1,C,H,W), W (O,C/g,kh,kw). Integer inputs give exact integer sums in float64."""
    x = x[0].astype(dtype); W = W.astype(dtype)
    O, Cg, kh, kw = W.shape
    g = a.get("group", 1)
    sh, sw = a.get("strides", [1, 1]); dh, dw = a.get("dilations", [1, 1])
    p = a.get("pads", [0, 0, 0, 0])
    if a.get("auto_pad", "NOTSET") not in ("NOTSET", "VALID"):
        raise NotImplementedError("auto_pad")
    xp = np.pad(x, ((0, 0), (p[0], p[2]), (p[1], p[3])))
    C, Hp, Wp = xp.shape
    Ho = (Hp - (dh * (kh - 1) + 1)) // sh + 1
    Wo = (Wp - (dw * (kw - 1) + 1)) // sw + 1
    if g == C and Cg == 1 and O == C:                                   # depthwise
        y = np.zeros((O, Ho, Wo), dtype)
        for i in range(kh):
            for j in range(kw):
                y += W[:, 0, i, j][:, None, None] * xp[:, i * dh: i * dh + sh * (Ho - 1) + 1: sh,
                                                         j * dw: j * dw + sw * (Wo - 1) + 1: sw]
    else:
        win = swv(xp, (dh * (kh - 1) + 1, dw * (kw - 1) + 1), axis=(1, 2))[:, ::sh, ::sw, ::dh, ::dw][:, :Ho, :Wo]
        y = np.empty((O, Ho, Wo), dtype)
        Og = O // g
        for gi in range(g):
            cols = win[gi * Cg:(gi + 1) * Cg].transpose(1, 2, 0, 3, 4).reshape(Ho * Wo, Cg * kh * kw)
            y[gi * Og:(gi + 1) * Og] = (cols @ W[gi * Og:(gi + 1) * Og].reshape(Og, -1).T).T.reshape(Og, Ho, Wo)
    if b is not None:
        y += b.astype(dtype)[:, None, None]
    return y[None]


def conv_transpose2d(x, W, b, a, dtype):
    """only kernel == stride, no padding, group 1 (YOLOv8 proto upsampler)."""
    k = a.get("kernel_shape", list(W.shape[2:])); s = a.get("strides", [1, 1])
    if list(k) != list(s) or any(a.get("pads", [0, 0, 0, 0])) or a.get("group", 1) != 1:
        raise NotImplementedError("ConvTranspose variant")
    x = x[0].astype(dtype); W = W.astype(dtype)
    C, H, Wd = x.shape; O = W.shape[1]
    y = np.einsum("chw,coab->ohawb", x, W).reshape(O, H * k[0], Wd * k[1])
    if b is not None:
        y += b.astype(dtype)[:, None, None]
    return y[None]


def maxpool(x, a, pad_value):
    k = a["kernel_shape"]; s = a.get("strides", [1, 1]); p = a.get("pads", [0, 0, 0, 0])
    xp = np.pad(x, ((0, 0), (0, 0), (p[0], p[2]), (p[1], p[3])), constant_values=pad_value)
    return swv(xp, k, axis=(2, 3))[:, :, ::s[0], ::s[1]].max((-1, -2))


def resize(x, scales, sizes, a):
    mode = a.get("mode", "nearest")
    ctm = a.get("coordinate_transformation_mode", "asymmetric")
    H, W = x.shape[2:]
    if sizes is not None and len(sizes):
        Ho, Wo = int(sizes[2]), int(sizes[3]); fh, fw = Ho / H, Wo / W
    else:
        fh, fw = float(scales[2]), float(scales[3]); Ho, Wo = int(np.floor(H * fh)), int(np.floor(W * fw))

    def src(n_out, n_in, f):
        o = np.arange(n_out, dtype=np.float64)
        if ctm == "asymmetric":
            return o / f
        if ctm == "half_pixel":
            return (o + 0.5) / f - 0.5
        if ctm == "align_corners":
            return o * (n_in - 1) / max(n_out - 1, 1)
        if ctm == "pytorch_half_pixel":
            return (o + 0.5) / f - 0.5 if n_out > 1 else np.zeros(n_out)
        raise NotImplementedError(ctm)
    sy, sx = src(Ho, H, fh), src(Wo, W, fw)
    if mode == "nearest":
        nm = a.get("nearest_mode", "round_prefer_floor")
        f = {"floor": np.floor, "ceil": np.ceil,
             "round_prefer_floor": lambda v: np.ceil(v - 0.5), "round_prefer_ceil": lambda v: np.floor(v + 0.5)}[nm]
        iy = np.clip(f(sy).astype(int), 0, H - 1); ix = np.clip(f(sx).astype(int), 0, W - 1)
        return x[:, :, iy][:, :, :, ix]
    if mode == "linear":
        sy = np.clip(sy, 0, H - 1); sx = np.clip(sx, 0, W - 1)
        y0 = np.floor(sy).astype(int); x0 = np.floor(sx).astype(int)
        y1 = np.minimum(y0 + 1, H - 1); x1 = np.minimum(x0 + 1, W - 1)
        wy = (sy - y0)[:, None]; wx = (sx - x0)[None, :]
        xf = x.astype(np.float32)
        top = xf[:, :, y0][:, :, :, x0] * (1 - wx) + xf[:, :, y0][:, :, :, x1] * wx
        bot = xf[:, :, y1][:, :, :, x0] * (1 - wx) + xf[:, :, y1][:, :, :, x1] * wx
        return (top * (1 - wy) + bot * wy).astype(np.float32)
    raise NotImplementedError(mode)


def softmax(x, axis, opset):
    if opset < 13:                       # ONNX < 13: flatten to 2-D at axis
        axis = axis % x.ndim
        shp = x.shape
        x2 = x.reshape(int(np.prod(shp[:axis])), -1)
        e = np.exp(x2 - x2.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).reshape(shp)
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


# ------------------------------------------------------------------ plan: which tensors are int8
def plan(layers, inputs):
    """kind[t] = 'q' (int8) or 'f' (float), and src[t] = tensor whose scale t inherits (pass-through ops)."""
    kind = {t: "f" for t in inputs}
    src = {}
    for L in layers:
        op, ins, outs = L["op"], [i for i in L["inputs"] if i], L["outputs"]
        act = [i for i in ins if i not in L.get("const_inputs", [])]
        allq = act and all(kind.get(i) == "q" for i in act)
        if op in ("Conv", "ConvTranspose"):
            k = "q"
        elif op == "SiLU" or op in ("Add", "Concat"):
            k = "q" if allq and not L.get("const_inputs") else "f"
        elif op in PASS_OPS:
            k = "q" if allq else "f"
            if k == "q":
                for o in outs:
                    src[o] = src.get(act[0], act[0])
        elif op == "Resize" and L["attrs"].get("mode", "nearest") == "nearest" and allq:
            k = "q"; src[outs[0]] = src.get(act[0], act[0])
        elif op == "Pad" and allq and L["attrs"].get("mode", "constant") == "constant" and not L["attrs"].get("value", 0):
            k = "q"; src[outs[0]] = src.get(act[0], act[0])
        else:
            k = "f"
        for o in outs:
            kind[o] = k
    return kind, src


# ------------------------------------------------------------------ the network
class Net:
    def __init__(self, base, mode="fp32"):
        """base: path without suffix, e.g. models/fpga/yolov8n_seg"""
        with open(base + "_graph.json") as f:
            g = json.load(f)
        self.layers, self.opset = g["layers"], g.get("opset", 13)
        self.inputs = [i[0] for i in g["io"]["inputs"]]
        self.outputs = [o[0] for o in g["io"]["outputs"]]
        self.input_shapes = [i[1] for i in g["io"]["inputs"]]
        self.mode = mode
        self.P = dict(np.load(f"{base}_{mode}.npz"))
        self.kind, self.src = plan(self.layers, self.inputs) if mode == "int8" else ({}, {})
        self.qmax = int(self.P.get("__qmax", QMAX))         # 127 = 8-bit activations, 32767 = 16-bit

    def scale(self, t):
        return float(self.P[self.src.get(t, t) + "__s"])

    def deq(self, t, v):
        return v.astype(np.float32) * self.scale(t) if self.kind.get(t) == "q" else v

    def run(self, *xs, keep=False, record=None):
        """xs: input arrays (float32 NCHW). Returns list of outputs (float32); keep=True also
        returns the dict of all tensors (int8 tensors stay int8). record(name, array) is called for
        every float tensor (calibration)."""
        P, q8, qm = self.P, self.mode == "int8", self.qmax
        dt = adtype(qm)
        T = dict(zip(self.inputs, [np.asarray(x, np.float32) for x in xs]))
        if record:
            for k, v in T.items():
                record(k, v)
        get = lambda n: (P[n] if n in P else T.get(n)) if n else None
        for L in self.layers:
            op, a, ins, outs = L["op"], L["attrs"], L["inputs"], L["outputs"]
            o0 = outs[0]
            intq = q8 and self.kind.get(o0) == "q"
            if q8 and not intq:                        # float layer inside the int8 net: dequantise inputs
                X = [self.deq(i, T[i]) if i in T else get(i) for i in ins]
            else:
                X = [get(i) for i in ins]
            # ---------------- integer layers ----------------
            if intq and op in ("Conv", "ConvTranspose"):
                xi = ins[0]
                xq = T[xi] if self.kind.get(xi) == "q" else quantize(T[xi], self.scale(xi), qm)
                Wq, B = P[o0 + "__w"], P[o0 + "__b"]
                f = conv2d if op == "Conv" else conv_transpose2d
                acc = np.rint(f(xq, Wq, None, a, np.float64)).astype(np.int64) + ch(B).astype(np.int64)
                Y = [rq(acc, ch(P[o0 + "__M"]), ch(P[o0 + "__sh"]), qm)]
            elif intq and op == "SiLU":
                Y = [P[o0 + "__lut"][T[ins[0]].astype(np.int32) + qm + 1]]
            elif intq and op == "Add":
                s = rq_raw(T[ins[0]], P[o0 + "__M0"], P[o0 + "__sh0"]) + rq_raw(T[ins[1]], P[o0 + "__M1"], P[o0 + "__sh1"])
                Y = [np.clip(s, -qm, qm).astype(dt)]
            elif intq and op == "Concat":
                parts = [rq(T[i].astype(np.int64), P[f"{o0}__M{k}"], P[f"{o0}__sh{k}"], qm) for k, i in enumerate(ins)]
                Y = [np.concatenate(parts, axis=a["axis"])]
            elif intq and op == "Clip":
                s = self.scale(o0)
                lo, hi = (a.get("min"), a.get("max")) if "min" in a or "max" in a else (
                    float(X[1]) if len(X) > 1 and X[1] is not None else None, float(X[2]) if len(X) > 2 and X[2] is not None else None)
                lo = -qm if lo is None else max(-qm, int(np.floor(lo / s + 0.5)))
                hi = qm if hi is None else min(qm, int(np.floor(hi / s + 0.5)))
                Y = [np.clip(X[0], lo, hi).astype(dt)]
            elif intq and op == "Relu":
                Y = [np.maximum(X[0], 0).astype(dt)]
            elif intq and op == "MaxPool":
                Y = [maxpool(X[0], a, -qm)]
            # ---------------- float (or int8 pass-through) layers ----------------
            else:
                Y = self._float_op(op, a, X, L)
            for o, y in zip(outs, Y):
                T[o] = y
                if record and not q8:
                    record(o, y)
        outs = [self.deq(o, T[o]).astype(np.float32) if q8 else T[o] for o in self.outputs]
        return (outs, T) if keep else outs

    def _float_op(self, op, a, X, L):
        x = X[0]
        if op == "Conv":
            return [conv2d(x, X[1], X[2] if len(X) > 2 else None, a, np.float32).astype(np.float32)]
        if op == "ConvTranspose":
            return [conv_transpose2d(x, X[1], X[2] if len(X) > 2 else None, a, np.float32).astype(np.float32)]
        if op == "SiLU":
            return [(x / (1 + np.exp(-np.clip(x, -80, 80)))).astype(np.float32)]
        if op == "Sigmoid":
            return [(1 / (1 + np.exp(-np.clip(x, -80, 80)))).astype(np.float32)]
        if op == "Relu":
            return [np.maximum(x, 0)]
        if op == "Clip":
            lo, hi = (a.get("min", -np.inf), a.get("max", np.inf)) if ("min" in a or "max" in a) else (
                X[1] if len(X) > 1 and X[1] is not None else -np.inf, X[2] if len(X) > 2 and X[2] is not None else np.inf)
            return [np.clip(x, lo, hi)]
        if op in ("Add", "Sub", "Mul", "Div"):
            f = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide}[op]
            return [f(x, X[1]).astype(np.float32)]
        if op == "Concat":
            return [np.concatenate(X, axis=a["axis"])]
        if op == "Split":
            axis = a.get("axis", 0)
            sp = a.get("split") or (list(X[1]) if len(X) > 1 and X[1] is not None else None)
            if sp is None:
                return list(np.split(x, len(L["outputs"]), axis=axis))
            return list(np.split(x, np.cumsum(sp)[:-1], axis=axis))
        if op == "MaxPool":
            return [maxpool(x, a, -np.inf if x.dtype.kind == "f" else -self.qmax)]
        if op == "Resize":
            if len(X) == 2:                       # opset 10: X, scales
                scales, sizes = X[1], None
            else:
                scales = X[2] if len(X) > 2 and X[2] is not None and X[2].size else None
                sizes = X[3] if len(X) > 3 else None
            return [resize(x, scales, sizes, a)]
        if op == "Reshape":
            shp = [int(v) for v in X[1]]
            shp = [x.shape[i] if v == 0 and not a.get("allowzero", 0) else v for i, v in enumerate(shp)]
            return [x.reshape(shp)]
        if op == "Transpose":
            return [x.transpose(a.get("perm"))]
        if op == "Softmax":
            return [softmax(x, a.get("axis", 1 if self.opset < 13 else -1), self.opset).astype(np.float32)]
        if op == "Slice":
            starts, ends = X[1], X[2]
            axes = X[3] if len(X) > 3 and X[3] is not None else range(len(starts))
            steps = X[4] if len(X) > 4 and X[4] is not None else [1] * len(starts)
            sl = [slice(None)] * x.ndim
            for s, e, ax, st in zip(starts, ends, axes, steps):
                n = x.shape[int(ax)]
                s, e = int(max(min(s, n), -n - 1)), int(max(min(e, n), -n - 1))
                sl[int(ax)] = slice(s, e, int(st))
            return [x[tuple(sl)]]
        if op == "Pad":
            pads = a.get("pads") if "pads" in a else [int(v) for v in X[1]]
            val = a.get("value", 0.0) if "pads" in a else (float(X[2]) if len(X) > 2 and X[2] is not None else 0.0)
            n = x.ndim
            mode = a.get("mode", "constant")
            pw = [(pads[i], pads[i + n]) for i in range(n)]
            return [np.pad(x, pw, constant_values=val) if mode == "constant" else np.pad(x, pw, mode={"reflect": "reflect", "edge": "edge"}[mode])]
        if op == "Squeeze":
            axes = a.get("axes") or (list(X[1]) if len(X) > 1 else None)
            return [np.squeeze(x, axis=tuple(axes)) if axes else np.squeeze(x)]
        if op == "Unsqueeze":
            axes = a.get("axes") or list(X[1])
            for ax in sorted(axes):
                x = np.expand_dims(x, ax)
            return [x]
        if op == "Flatten":
            ax = a.get("axis", 1)
            return [x.reshape(int(np.prod(x.shape[:ax])), -1)]
        raise NotImplementedError(op)


class _In:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class Session:
    """Drop-in for onnxruntime.InferenceSession: Session(base, mode).run(None, {name: x})."""
    def __init__(self, base, mode="int8", record=None):
        self.net = Net(base, mode)
        self.record = record
        self.keep = False                 # True: remember every tensor of the last run in .last
        self.last = self.last_input = None

    def get_inputs(self):
        return [_In(n, s) for n, s in zip(self.net.inputs, self.net.input_shapes)]

    def run(self, names, feed):
        xs = [feed[n] for n in self.net.inputs]
        if self.keep:
            outs, self.last = self.net.run(*xs, keep=True, record=self.record)
            self.last_input = xs[0]
            return outs
        return self.net.run(*xs, record=self.record)
