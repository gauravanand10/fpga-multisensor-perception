"""
export_onnx.py - trained model -> ONNX for the live viewer and for the FPGA tools.

    python export_onnx.py                    uses runs\\tiny_unet\\best_qat.pt (else best.pt)
    python export_onnx.py --ckpt runs\\tiny_unet\\best.pt

Writes
    runs\\<name>\\tiny_unet.onnx
    model\\quantization\\models\\tiny_unet.onnx + tiny_unet.json   (classes, sizes; used by seg_unet.py)
BatchNorm is already folded into the convolutions, so the ONNX graph has only Conv, Relu,
Resize (nearest x2) and Concat - exactly the layers the FPGA conv engine implements.
"""
import argparse, json, os, shutil
import numpy as np
import torch

from config import CLASSES, FUSED, INPUT_HW, OUTPUT_HW, RUNS, WIDTHS
from model import TinyUNet, fuse_bn
from qat import build_qat, enable


def load(ckpt):
    s = torch.load(ckpt, map_location="cpu")
    classes, widths = s.get("classes", CLASSES), tuple(s.get("widths", WIDTHS))
    if s.get("kind") == "qat":
        m = build_qat(len(classes), widths); m.load_state_dict(s["model"]); enable(m, False)
    else:
        m = TinyUNet(len(classes), widths); m.load_state_dict(s["model"]); m = fuse_bn(m)
    return m.eval(), classes, widths, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tiny_unet")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    out = os.path.join(RUNS, a.name)
    ck = a.ckpt or next((p for p in (os.path.join(out, "best_qat.pt"), os.path.join(out, "best.pt")) if os.path.exists(p)), None)
    if not ck:
        raise SystemExit(f"no checkpoint in {out} - run train.py first")
    m, classes, widths, s = load(ck)
    print(f"[export] {ck}  ({s.get('kind', 'float')}, epoch {s.get('epoch')}, val mIoU {s.get('score', 0):.3f})")
    x = torch.rand(1, 1, *INPUT_HW)
    onnx_path = os.path.join(out, "tiny_unet.onnx")
    kw = dict(input_names=["gray"], output_names=["logits"], opset_version=13, do_constant_folding=True)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        torch.onnx.export(m, (x,), onnx_path, dynamo=False, **kw)
    except TypeError:                                   # older PyTorch without the dynamo switch
        torch.onnx.export(m, (x,), onnx_path, **kw)
    import onnx, onnxruntime as ort
    g = onnx.load(onnx_path)
    ops = sorted({n.op_type for n in g.graph.node})
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = m(x).numpy()
    got = sess.run(None, {"gray": x.numpy()})[0]
    diff = float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-9))
    print(f"[export] ops {ops}; output {tuple(got.shape)}; ONNX vs PyTorch max rel. difference {diff:.1e}")
    bad = set(ops) - {"Conv", "Relu", "Resize", "Concat", "Constant"}
    if bad:
        print(f"   WARNING: ops the FPGA engine was not planned for: {sorted(bad)}")
    dst = os.path.join(FUSED, "models"); os.makedirs(dst, exist_ok=True)
    shutil.copy(onnx_path, os.path.join(dst, "tiny_unet.onnx"))
    with open(os.path.join(dst, "tiny_unet.json"), "w") as f:
        json.dump(dict(classes=classes, widths=list(widths), input_hw=list(INPUT_HW), output_hw=list(OUTPUT_HW),
                       checkpoint=os.path.basename(ck), val_miou=s.get("score")), f, indent=1)
    print(f"[export] {onnx_path}\n         copied to {os.path.join(dst, 'tiny_unet.onnx')} (+ tiny_unet.json)")
    print("next:  python evaluate.py     then   python quantize_fpga.py")


if __name__ == "__main__":
    main()
