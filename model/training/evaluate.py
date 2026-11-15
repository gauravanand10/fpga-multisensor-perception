"""
evaluate.py - how good is the exported U-Net? IoU per class on the validation frames.

    python evaluate.py               the ONNX model (model\\quantization\\models\\tiny_unet.onnx)
    python evaluate.py --int8        also the numpy int8 model (after quantize_fpga.py) = FPGA result

IoU = overlap / union of the predicted and the true pixels of a class (1.0 = perfect).
"own" frames are your ESP32-CAM frames (labelled by YOLO), "coco" the COCO validation photos.
Pictures: runs\\tiny_unet\\eval_<set>.png  (camera | label | prediction).
"""
import argparse, os, time
import numpy as np
import cv2

from config import CLASSES, FUSED, IGNORE, RUNS, INPUT_HW, ensure_fused_on_path
from dataset import build

PALETTE = np.array([[0, 0, 0], [60, 60, 255], [60, 200, 60], [255, 160, 0], [0, 200, 255], [200, 60, 200],
                    [255, 255, 60], [120, 120, 255], [60, 255, 160], [160, 160, 160]], np.uint8)


def predictor(kind):
    ensure_fused_on_path()
    if kind == "onnx":
        import onnxruntime as ort
        s = ort.InferenceSession(os.path.join(FUSED, "models", "tiny_unet.onnx"), providers=["CPUExecutionProvider"])
        return lambda x: s.run(None, {"gray": x})[0]
    from engine import Net
    net = Net(os.path.join(FUSED, "models", "fpga", "tiny_unet"), "int8")
    return lambda x: net.run(x)[0]


def score(pred_fn, ds, limit=None):
    C = len(CLASSES)
    inter = np.zeros(C); union = np.zeros(C); t = 0.0
    n = len(ds) if limit is None else min(limit, len(ds))
    keep = []
    for i in range(n):
        x, y = ds[i]
        t0 = time.perf_counter(); z = pred_fn(x[None].numpy()); t += time.perf_counter() - t0
        p = z[0].argmax(0); y = y.numpy(); v = y != IGNORE
        for c in range(C):
            a, b = (p == c) & v, (y == c) & v
            inter[c] += (a & b).sum(); union[c] += (a | b).sum()
        if i in np.linspace(0, n - 1, min(8, n)).astype(int):
            keep.append((x[0].numpy(), y, p))
    iou = inter / np.maximum(union, 1)
    seen = union > 0
    return iou, seen, t / n * 1000, keep


def picture(keep, fn):
    H, W = INPUT_HW
    rows = []
    for x, y, p in keep:
        cam = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        col = lambda l: cv2.resize(PALETTE[np.where(l == IGNORE, 9, l) % len(PALETTE)], (W, H), interpolation=cv2.INTER_NEAREST)
        rows.append(np.hstack([cam, col(y), col(p)]))
    cv2.imwrite(fn, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--name", default="tiny_unet")
    ap.add_argument("--limit", type=int, default=None, help="frames per set (int8 is slower)")
    a = ap.parse_args()
    _, val = build()
    out = os.path.join(RUNS, a.name); os.makedirs(out, exist_ok=True)
    kinds = ["onnx"] + (["int8"] if a.int8 else [])
    res = {}
    for k in kinds:
        f = predictor(k)
        for name, ds in val.items():
            lim = a.limit if (k == "onnx" or a.limit) else 300
            iou, seen, ms, keep = score(f, ds, lim)
            res[(k, name)] = (iou, seen, ms)
            if k == "onnx":
                picture(keep, os.path.join(out, f"eval_{name}.png"))
    head = f"{'class':12s}" + "".join(f"{k + ' ' + n:>14s}" for (k, n) in res)
    print(head); print("-" * len(head))
    for c, cname in enumerate(CLASSES):
        print(f"{cname:12s}" + "".join(f"{(f'{r[0][c]:.3f}' if r[1][c] else '-'):>14s}" for r in res.values()))
    print("-" * len(head))
    print(f"{'mean (objects)':12s}" + "".join(f"{r[0][1:][r[1][1:]].mean() if r[1][1:].any() else 0:>14.3f}" for r in res.values()))
    print(f"{'ms per frame':12s}" + "".join(f"{r[2]:>14.1f}" for r in res.values()))
    print("pictures:", ", ".join(os.path.join(out, f"eval_{n}.png") for n in val))


if __name__ == "__main__":
    main()
