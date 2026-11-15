"""
config.py - everything the training scripts share. Change things HERE, not in the scripts.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
FUSED = os.path.join(REPO, "model", "quantization")                   # int8 toolkit (fpga_ref) + exported models
DATA = os.path.join(REPO, "dataset", "prepared")                      # prepared datasets (memmaps, not in git)
RUNS = os.path.join(REPO, "model", "runs")                            # checkpoints, logs, ONNX (not in git)
FPGA_MODELS = os.path.join(FUSED, "models", "fpga")                    # where fpga_ref writes packages

# ---- classes the network learns (index 0 = background). Names = COCO / YOLO names (underscores).
# Keep this list SHORT: every class costs accuracy on a network this small.
CLASSES = ["background", "person", "chair", "tv", "laptop", "bottle"]
IGNORE = 255                      # label value ignored by the loss (COCO "crowd" regions)

# ---- image sizes (height, width)
STORE_HW = (240, 320)             # stored dataset frames = the ESP32-CAM resolution
INPUT_HW = (120, 160)             # network input: camera frame resized to half
OUTPUT_HW = (60, 80)              # network output: label map at 1/4 of the camera (the UART label map)

# ---- network size (see model.py): channels at 1/2, 1/4, 1/8 of the input.
# (16, 32, 64) -> 155 K parameters, about 81 % of the FPGA BRAM;
# (16, 32, 80) -> about 205 K parameters, more room for 5 classes, about 90 % of the BRAM (default).
WIDTHS = (16, 32, 80)

# ---- COCO subset
COCO_TRAIN_IMAGES = 30000         # images downloaded for training (each about 160 KB)
COCO_VAL_IMAGES = 1000
MIN_OBJECT_AREA = 0.004           # an image is used only if a target object covers >= 0.4 % of it


def ensure_fused_on_path():
    """make model/quantization and its fpga_ref importable."""
    import sys
    for p in (FUSED, os.path.join(FUSED, "fpga_ref")):
        if p not in sys.path:
            sys.path.insert(0, p)
