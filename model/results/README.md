# Model results

Graphs drawn from the files below by `python docs/make_plots.py`.

| File | Content |
|---|---|
| `training_log.csv` | Per epoch: phase (float / QAT), loss, time, IoU per class on COCO val |
| `float32_vs_int8_metrics.txt` | float32 vs int8 on 1000 COCO validation images |
| `confusion_matrix.csv` | int8 confusion matrix in pixels (rows = ground truth) |

| Training loss and mIoU | IoU per class during training |
|---|---|
| ![Training curve](training_curve.png) | ![IoU per class during training](class_iou_training.png) |

| IoU, float32 vs int8 | Pixel agreement per image |
|---|---|
| ![IoU per class](iou_per_class.png) | ![Pixel agreement](pixel_agreement.png) |

<img src="confusion_matrix.png" width="380" alt="Confusion matrix">

120 float epochs, then 6 quantisation-aware epochs (shaded). The int8 network matches the float32 network
(mean IoU 0.282 for both, 98.17 % of pixels identical). Most errors are object pixels labelled as background,
mainly for the small or rare classes chair and bottle.
