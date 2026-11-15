"""
calibrate.py - make the int8 FPGA package from the float package + real camera frames.

    python fpga_ref\\calibrate.py --camera 40                (grab 40 frames from the ESP32-CAM on COM7)
    python fpga_ref\\calibrate.py --images "recordings\\*\\*.png"
    python fpga_ref\\calibrate.py --images "calib_frames\\*.png" --model yolo

1. runs the float reference model (engine.py, fp32) on the frames, with exactly the same
   pre-processing as the live viewer, and records the range of every tensor
   (99.99th / 99.9th percentile of |value|, see --pct);
2. scale of a tensor = range / 127  (symmetric int8, zero point 0; --abits 16 -> / 32767);
3. writes models\\fpga\\<name>_int8.npz with everything the integer datapath needs
   (see INT8_SPEC.md):
     <layer>__w   int8 weights          <layer>__b   int32 bias
     <layer>__M   int32 multiplier      <layer>__sh  int32 shift      (per output channel)
     <layer>__lut int8[256] SiLU table  <tensor>__s  float32 scale (boundaries / debugging)
     <layer>__M0/__sh0/__M1/__sh1 (Add), <layer>__M<k>/__sh<k> (Concat input k)
   <layer> is the name of the layer's output tensor, as in <name>_graph.json.
Frames used are also saved to fpga_ref\\calib_frames\\ so the run can be repeated offline.
"""
import argparse, glob, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from engine import Net, Session, plan, quantize, mult_shift, QMAX   # noqa: E402

FPGA = os.path.join(ROOT, "models", "fpga")


def grab_camera(n, port, baud, every=0.5):
    sys.path[:0] = [os.path.join(ROOT, "camera")]
    from mp_readers import ProcReader
    cam = ProcReader("camera", port, baud).start()
    out, last, t0 = [], None, time.time()
    if n > 1:
        print(f"[camera] point the camera at the ROOM you will use it in: people, chairs, tables, objects\n"
              f"         (not the ceiling / a blank wall / a lamp) - slowly pan around and walk through the view.")
        for k in (3, 2, 1):
            print(f"\r         starting in {k} s ", end=""); time.sleep(1)
        print()
    print(f"[camera] grabbing {n} frames from {port} ({n * every:.0f} s) ...")
    try:
        while len(out) < n and time.time() - t0 < 60 + n * every * 4:
            with cam.lock:
                it = cam.latest
            if it and it[0] != last:
                last = it[0]; out.append(it[1].copy())
                print(f"\r  {len(out)}/{n}", end="")
                time.sleep(every)
            time.sleep(0.01)
    finally:
        cam.close()
    print()
    if not out:
        raise SystemExit("no camera frames - is COM7 free (viewer / Serial Monitor closed)?")
    return out


class Stats:
    """range of every tensor = max over frames of the pct-th percentile of |x| (pct 100 = plain max).
    Clipping the rare 0.01 % largest values gives much finer steps to everything else."""
    def __init__(self, pct=99.99):
        self.amax, self.pct = {}, pct

    def __call__(self, name, v):
        if v.dtype.kind == "f" and v.size:
            a = np.abs(v).ravel()
            if self.pct >= 100 or a.size < 1000:
                m = float(a.max())
            else:
                k = min(a.size - 1, int(np.ceil(self.pct / 100 * a.size)))
                m = float(np.partition(a, k)[k])
            self.amax[name] = max(self.amax.get(name, 0.0), m)


def build(base, amax, abits=8):
    """float package + activation ranges -> int8 package dict."""
    net = Net(base, "fp32")
    P = net.P
    kind, src = plan(net.layers, net.inputs)
    qa = 2 ** (abits - 1) - 1                                    # activation range +-qa
    # A tensor whose only use is a ReLU / ReLU6 (Clip) shares its int8 scale with it, so it only
    # needs the range AFTER the clip: e.g. MiDaS conv outputs reach +-140 but ReLU6 keeps 0..6;
    # quantising for +-140 would leave only ~6 steps for 0..6. Values above the range saturate,
    # which is exactly what the clip does anyway (bit-exact, no extra hardware).
    amax = dict(amax)
    users = {}
    for L in net.layers:
        for i in L["inputs"]:
            users.setdefault(i, []).append(L)
    for L in net.layers:
        if L["op"] in ("Clip", "Relu") and kind.get(L["outputs"][0]) == "q":
            t = L["inputs"][0]
            if len(users.get(t, [])) == 1 and L["outputs"][0] in amax:
                amax[t] = amax[L["outputs"][0]]
    s = lambda t: max(amax.get(src.get(t, t), 0.0), 1e-8) / qa
    out, weight_keys = {}, set()
    for L in net.layers:
        op, ins, o = L["op"], L["inputs"], L["outputs"][0]
        if kind.get(o) != "q":
            continue
        if op in ("Conv", "ConvTranspose"):
            W = P[ins[1]].astype(np.float64)
            b = P[ins[2]].astype(np.float64) if len(ins) > 2 and ins[2] else np.zeros(W.shape[0 if op == "Conv" else 1])
            weight_keys.update(i for i in ins[1:] if i)
            ax = 0 if op == "Conv" else 1                       # output-channel axis
            Wm = np.moveaxis(W, ax, 0).reshape(W.shape[ax], -1)
            sw = np.abs(Wm).max(1) / QMAX
            si, so = s(ins[0]), s(o)
            sw = np.maximum(sw, np.abs(b) / (si * 2 ** 30))      # channels with ~zero weights but a bias: keep bias in int32
            sw[sw == 0] = 1.0
            Wq = np.clip(np.floor(np.moveaxis(W, ax, 0) / sw.reshape((-1,) + (1,) * (W.ndim - 1)) + 0.5), -QMAX, QMAX)
            out[o + "__w"] = np.moveaxis(Wq, 0, ax).astype(np.int8)
            B = np.floor(b / (si * sw) + 0.5)
            if np.abs(B).max() >= 2 ** 31:
                raise ValueError(f"bias overflow in {o}")
            out[o + "__b"] = B.astype(np.int32)
            out[o + "__M"], out[o + "__sh"] = mult_shift(si * sw / so)
            out[o + "__wscale"] = sw.astype(np.float32)
            out[ins[0] + "__s"] = np.float32(si)
            out[o + "__s"] = np.float32(so)
        elif op == "SiLU":
            si, so = s(ins[0]), s(o)
            v = np.arange(-qa - 1, qa + 1) * si                  # LUT index = q + qa + 1
            out[o + "__lut"] = quantize(v / (1 + np.exp(-v)), so, qa)
            out[o + "__s"] = np.float32(so)
        elif op == "Add":
            so = s(o)
            for k, i in enumerate(ins):
                out[f"{o}__M{k}"], out[f"{o}__sh{k}"] = mult_shift(s(i) / so)
            out[o + "__s"] = np.float32(so)
        elif op == "Concat":
            so = s(o)
            for k, i in enumerate(ins):
                out[f"{o}__M{k}"], out[f"{o}__sh{k}"] = mult_shift(s(i) / so)
            out[o + "__s"] = np.float32(so)
    # scales of every int8 tensor (own scale only; pass-through tensors use their source's)
    for t, k in kind.items():
        if k == "q" and t not in src and t + "__s" not in out:
            out[t + "__s"] = np.float32(s(t))
    # remaining constants (shapes, resize scales, YOLO box-decoding tables, MiDaS normalisation)
    for k, v in P.items():
        if k not in weight_keys:
            out[k] = v
    out["__qmax"] = np.int32(qa)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="grab N frames from the camera")
    ap.add_argument("--images", default=None, help="glob of calibration images")
    ap.add_argument("--port", default="COM7"); ap.add_argument("--baud", type=int, default=921_600)
    ap.add_argument("--model", default="both", choices=["yolo", "midas", "both"])
    ap.add_argument("--pct", type=float, default=None,
                    help="range percentile, 100 = plain max (default: 99.99 YOLO, 99.9 MiDaS - best on test frames)")
    ap.add_argument("--abits", type=int, default=8, choices=[8, 16], help="activation bits (weights are always 8)")
    a = ap.parse_args()

    frames = []
    if a.camera:
        frames = grab_camera(a.camera, a.port, a.baud)
        d = os.path.join(HERE, "calib_frames"); os.makedirs(d, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(d, f"{stamp}_{i:03d}.png"), f)
    if a.images:
        frames += [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in sorted(glob.glob(a.images))]
    if not frames:
        d = os.path.join(HERE, "calib_frames")
        frames = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in sorted(glob.glob(os.path.join(d, "*.png")))]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise SystemExit("no calibration frames: use --camera 40 or --images <glob>")
    print(f"[calib] {len(frames)} frames")

    jobs = []
    if a.model in ("yolo", "both"):
        jobs.append(("yolov8n_seg", "seg"))
    if a.model in ("midas", "both"):
        jobs.append(("model_small", "depth"))
    for name, kind in jobs:
        base = os.path.join(FPGA, name)
        if not os.path.exists(base + "_fp32.npz"):
            raise SystemExit(f"{base}_fp32.npz missing - run: python fpga_ref\\export_graph.py")
        st = Stats(a.pct or (99.99 if kind == "seg" else 99.9))
        if kind == "seg":
            from seg_teacher import SegTeacher
            m = SegTeacher(backend="fp32")
        else:
            from mono_depth import MonoDepth
            m = MonoDepth("midas", backend="fp32")
        m.sess.record = st
        t0 = time.time()
        n_obj = 0
        for i, f in enumerate(frames):
            r = m(f)
            if kind == "seg":
                n_obj += len(r) > 0
            print(f"\r[{name}] float pass {i + 1}/{len(frames)}", end="")
        print(f"  ({(time.time() - t0) / len(frames):.1f} s/frame)")
        pkg = build(base, st.amax, a.abits)
        np.savez(base + "_int8.npz", **pkg)
        nw = sum(v.size for k, v in pkg.items() if k.endswith("__w"))
        print(f"[{name}] written {base}_int8.npz  ({nw/1e6:.2f} M int8 weights)")
        if kind == "seg" and n_obj < 0.5 * len(frames):
            print(f"   WARNING: objects were found in only {n_obj} of {len(frames)} frames. The int8 ranges come from\n"
                  f"   these frames, so frames without people / furniture give a poor int8 model. Re-run with the\n"
                  f"   camera pointed at the room:   python fpga_ref\\calibrate.py --camera 40")


if __name__ == "__main__":
    main()
