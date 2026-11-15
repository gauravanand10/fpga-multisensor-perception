"""
evaluate_float_vs_int8.py - how closely the FPGA's int8 arithmetic follows the float PC model

On the COCO 2017 validation images the network never saw in training (dataset/coco_val,
1000 photos, 320x240 grey), each photo goes through the ESP32-CAM model (2x2 average to 160x120),
then through
  * the float32 Tiny U-Net in ONNX Runtime (the original PC system), and
  * the int8 accelerator model (bit-exact with the FPGA RTL).
Reported: pixel agreement between the two label maps, and the IoU of each against the COCO ground
truth (80x60 prediction upscaled x4, 'ignore' pixels left out, IoU over the whole set).

    python experiments\\evaluate_float_vs_int8.py            (all 1000, ~4 min)
    python experiments\\evaluate_float_vs_int8.py --n 200
Output: results/accuracy/ (float_vs_int8_metrics.json/.txt, per_image_metrics.csv).
        The graphs are drawn from these files by docs/make_plots.py.
"""
import argparse, json, os, time
import numpy as np
from _common import ROOT
from reference_model import paths, fpga_model as fm, esp32cam_node
from reference_model.fpga_unet_accelerator import UNetAccelerator
from reference_model.float_reference import FloatUNet

OUT = os.path.join(ROOT, "results", "accuracy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    imgs, gts = paths.load_coco_val()
    n = min(a.n, len(imgs))
    acc, flt = UNetAccelerator(), FloatUNet()
    C = fm.N_CLS
    inter = np.zeros((2, C)); union = np.zeros((2, C))
    agree_px = tot_px = 0
    rows = []
    t0 = time.time()
    for i in range(n):
        small = esp32cam_node.ESP32CamNode.downsample(np.array(imgs[i]))
        li, _ = acc.run(small)
        lf = flt.labels(small)
        gt = np.array(gts[i]); valid = gt != 255
        ag = int((li == lf).sum()); agree_px += ag; tot_px += li.size
        accs = []
        for m, lab in enumerate((lf, li)):
            up = np.repeat(np.repeat(lab, 4, 0), 4, 1)
            accs.append(float((up[valid] == gt[valid]).mean()) if valid.any() else float("nan"))
            for c in range(C):
                p, g = (up == c) & valid, (gt == c) & valid
                inter[m, c] += (p & g).sum(); union[m, c] += (p | g).sum()
        present = [fm.CLASSES[c] for c in range(1, C) if ((gt == c).mean() > 0.01)]
        rows.append((i, ag / li.size, accs[0], accs[1], "+".join(present)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}  agreement so far {100 * agree_px / tot_px:.2f} %  ({time.time() - t0:.0f} s)")
    iou = inter / np.maximum(union, 1)
    res = dict(n_images=n, pixel_agreement_float_int8=agree_px / tot_px,
               iou_float={fm.CLASSES[c]: iou[0, c] for c in range(C)},
               iou_int8={fm.CLASSES[c]: iou[1, c] for c in range(C)},
               miou5_float=float(iou[0, 1:].mean()), miou5_int8=float(iou[1, 1:].mean()),
               pixel_acc_float=float(np.nanmean([r[2] for r in rows])),
               pixel_acc_int8=float(np.nanmean([r[3] for r in rows])))
    json.dump(res, open(os.path.join(OUT, "float_vs_int8_metrics.json"), "w"), indent=1)
    with open(os.path.join(OUT, "per_image_metrics.csv"), "w") as f:
        f.write("index,agreement_float_int8,pixel_acc_float,pixel_acc_int8,classes_present\n")
        f.writelines(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]}\n" for r in rows)
    lines = [f"Float (PC, ONNX) vs int8 (FPGA) Tiny U-Net on {n} COCO 2017 validation images", "",
             f"  pixel agreement float vs int8 label maps : {100 * res['pixel_agreement_float_int8']:.2f} %",
             f"  pixel accuracy vs ground truth           : float {100 * res['pixel_acc_float']:.1f} %, "
             f"int8 {100 * res['pixel_acc_int8']:.1f} %", "",
             f"  {'class':12s} {'IoU float':>10s} {'IoU int8':>10s}"]
    lines += [f"  {fm.CLASSES[c]:12s} {iou[0, c]:10.3f} {iou[1, c]:10.3f}" for c in range(C)]
    lines += [f"  {'mean (5 obj)':12s} {res['miou5_float']:10.3f} {res['miou5_int8']:10.3f}"]
    open(os.path.join(OUT, "float_vs_int8_metrics.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
