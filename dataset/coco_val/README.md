# COCO validation subset

1000 COCO 2017 validation images in the ESP32-CAM format (320×240 grey) with per-pixel labels for the six
classes. Never used in training; used for the float32 vs int8 evaluation and the end-to-end demonstration.

| File | Content |
|---|---|
| `coco_val_images_part1.npz` … `coco_val_images_part4.npz` | Images 0–249, 250–499, 500–749, 750–999, key `images`, `uint8 [250, 240, 320]` |
| `coco_val_labels.npz` | Label maps, key `labels`, `uint8 [1000, 240, 320]`; 0–5 = class id, 255 = ignore |
| `coco_val_info.json` | Class names, image count, pixel share per class |
| `coco_val_class_statistics.json` | Pixel share and number of images that contain each class |

The images are split in four files of about 15 MB to stay below GitHub's 25 MB browser-upload limit. Loading example and
licence: [`../README.md`](../README.md).
