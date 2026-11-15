# Dataset

The network is trained and evaluated on a subset of **COCO 2017** with the five object classes the system
detects, converted to the format of the ESP32-CAM: greyscale, 320×240.

| Class id | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Name | background | person | chair | tv | laptop | bottle |

## Contents

| File | Description |
|---|---|
| `coco_val/coco_val_images_part1.npz` … `part4.npz` | 1000 validation images, `uint8 [N, 240, 320]` (4 × 250), key `images` |
| `coco_val/coco_val_labels.npz` | Matching label maps, `uint8 [1000, 240, 320]`, class id per pixel, 255 = ignore, key `labels` |
| `coco_val/coco_val_info.json` | Classes, image count, pixel share per class |
| `coco_val/coco_val_class_statistics.json` | Pixel share and number of images containing each class |
| `coco_train_info.json` | Statistics of the 30 000-image training set |

The validation images are never seen during training. They are used by `golden_model/experiments/` for the
float32 vs int8 evaluation and the end-to-end demonstration.

## Statistics

| | Train | Validation |
|---|---|---|
| Images | 30 000 | 1 000 |
| background / person / chair / tv / laptop / bottle (pixel %) | 79.1 / 14.6 / 2.6 / 1.4 / 1.5 / 0.7 | 79.1 / 13.0 / 2.6 / 2.1 / 2.4 / 0.9 |
| Validation images containing person / chair / tv / laptop / bottle | | 591 / 376 / 194 / 171 / 263 |

## Training set

The 30 000 training images (about 10 GB) are not stored in the repository. They are downloaded and prepared by
`model/training/prepare_coco.py` into `dataset/prepared/`: images are selected so that the rare classes (tv,
laptop, bottle) are represented first, and only images in which a target object covers at least 0.4 % of the
frame are kept.

```python
import numpy as np
images = np.concatenate([np.load(f"dataset/coco_val/coco_val_images_part{i}.npz")["images"] for i in (1, 2, 3, 4)])
labels = np.load("dataset/coco_val/coco_val_labels.npz")["labels"]
```

## Licence

COCO 2017 (Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014). Annotations: CC BY 4.0.
Images: the Flickr licence of each image.
