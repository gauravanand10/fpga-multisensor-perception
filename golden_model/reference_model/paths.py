"""paths.py - files used by the reference model.

Model data lives inside this folder (data/). The COCO validation set, the golden vectors of
the int8 network and the RTL simulation output are read from this repository's dataset/, hw/ and data/ folders.
"""
import glob, os

import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)                      # golden_model
PROJECT = os.environ.get("REFMODEL_PROJECT", os.path.dirname(ROOT))   # repository root

DATA = os.path.join(ROOT, "data")
MODEL_INFO = os.path.join(DATA, "model", "tiny_unet_model_info.json")
FLOAT_ONNX = os.path.join(DATA, "model", "tiny_unet_float32.onnx")
INT8_NPZ = os.path.join(DATA, "model", "tiny_unet_int8_quantised.npz")
INT8_PARAMS = os.path.join(DATA, "model", "int8_parameters")
ZONE_TABLES = os.path.join(DATA, "tof_geometry", "tof_zone_tables.npz")

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(PROJECT, "docs", "figures")

# sibling folders (read only)
COCO_DIR = os.path.join(PROJECT, "dataset", "coco_val")
COCO_VAL_IMAGES = os.path.join(COCO_DIR, "coco_val_images_part*.npz")     # 4 parts of (250, 240, 320) uint8 grey
COCO_VAL_LABELS = os.path.join(COCO_DIR, "coco_val_labels.npz")     # (1000, 240, 320) class id, 255 = ignore
GOLDEN_DIR = os.path.join(DATA, "golden_vectors", "mem")
RTL_DIR = PROJECT
RTL_MEM = os.path.join(RTL_DIR, "hw", "mem")
RTL_UART_CAPTURE = os.path.join(DATA, "rtl_simulation", "uart_capture.bin")


def load_coco_val():
    """1000 COCO 2017 validation images (240x320 grey) and labels, stored as compressed parts in dataset/coco_val."""
    imgs = np.concatenate([np.load(p)["images"] for p in sorted(
        glob.glob(os.path.join(COCO_DIR, "coco_val_images_part*.npz")))])
    labels = np.load(need(COCO_VAL_LABELS, "dataset/coco_val"))["labels"]
    return imgs, labels


def need(path, hint=""):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found{(' - ' + hint) if hint else ''}")
    return path
