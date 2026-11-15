"""
verify_int8_vs_golden_vectors.py - the FPGA U-Net model vs the int8 golden vectors

Runs the accelerator model (reference_model.fpga_unet_accelerator) on the golden test frame and
compares all 12 conv-layer outputs with the tensors exported by model/quantization/fpga_ref (the
integer reference of the trained network). Every value must match.

    python verification\\verify_int8_vs_golden_vectors.py
Output: results/verification/int8_vs_golden_vectors.txt / .json
"""
import os
import numpy as np
from _common import save_report, read_hex_bytes
from reference_model import paths, fpga_model as fm
from reference_model.fpga_unet_accelerator import UNetAccelerator


def main():
    acc = UNetAccelerator()
    frame = np.frombuffer(read_hex_bytes(paths.need(os.path.join(paths.RTL_MEM, "tb_frame.mem"),
                                                    "hw/mem")), np.uint8).reshape(fm.IMG_H, fm.IMG_W)
    gold_lab = np.array([int(t, 16) for t in open(os.path.join(paths.RTL_MEM, "tb_labels.mem")).read().split()])
    res, outs, labels = fm.check_against_golden(paths.need(paths.GOLDEN_DIR, "golden_model/data/golden_vectors"),
                                                acc.packed, frame)
    lines = ["int8 U-Net model vs fpga_ref golden vectors (golden test frame)", ""]
    rows = []
    for (name, chk, mism), l in zip(res, acc.layers):
        n = l.cout * l.hout * l.wout
        lines.append(f"  {name:10s} -> {chk:16s} {l.cout:3d} x {l.hout:2d} x {l.wout:3d}  "
                     f"{n:7d} values  mismatches {mism}")
        rows.append(dict(layer=name, tensor=chk, values=n, mismatches=mism))
    lab_err = int((labels.ravel() != gold_lab).sum())
    ok = all(r["mismatches"] == 0 for r in rows) and lab_err == 0
    lines += ["", f"  label map 80x60 vs argmax(golden logits): mismatches {lab_err}", "",
              ("PASS" if ok else "FAIL") + f": {sum(r['mismatches'] == 0 for r in rows)}/{len(rows)} layers bit-exact"]
    save_report("int8_vs_golden_vectors", lines, dict(layers=rows, label_mismatches=lab_err, passed=ok))


if __name__ == "__main__":
    main()
