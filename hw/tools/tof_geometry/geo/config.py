"""
config.py - settings for the geometry modules ported from
the original PC depth pipeline (tof_zones, calibration, projection, depth_map,
quick_calib), adapted to this setup:
  * ESP32-CAM with GC2145 sensor (1600x1200 UXGA like the OV2640, same AI-Thinker
    lens module) streaming 320x240 over USB serial (COM7)
  * ESP32 + VL53L5CX streaming the 8x8 grid as text over USB serial (COM13)
Calibration files live in tof_geometry\\calibration\\ (written by the viewer's
quick-calibration mode, key k).
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]          # tof_geometry
CALIBRATION_DIR = ROOT_DIR / "calibration"

INTRINSICS_FILES = [CALIBRATION_DIR / "camera_intrinsics_esp32cam.npz"]
EXTRINSIC_FILES = [CALIBRATION_DIR / "extrinsic_refined.npz"]
ORIENTATION_FILE = CALIBRATION_DIR / "tof_orientation.json"
SIM_INTRINSICS_FILE = INTRINSICS_FILES[0]
SIM_EXTRINSIC_FILE = EXTRINSIC_FILES[0]
SIM_ORIENTATION_FILE = ORIENTATION_FILE

# ToF zone model (see tof_zones.py)
TOF_ZONE_MODEL = "st"
TOF_FOV_DEG = 45.0
TOF_FOV_SCALE = 1.0
TOF_FLIP_H = False
TOF_FLIP_V = False
TOF_ROTATE = 0
TOF_TRANSPOSE = False
TOF_DISTANCE_MODE = "radial"
TOF_MIN_MM = 20.0
TOF_MAX_MM = 4000.0

# ToF position relative to the camera lens (x right, y down, z forward), mm.
# Measure yours and pass --offset-mm to the viewer.
NOMINAL_TOF_T_MM = (0.0, 50.0, 0.0)      # this perfboard: ToF 50 mm directly below the camera lens

# depth maps
DEPTH_MIN_MM = 200.0
DEPTH_MAX_MM = 2500.0
DENSE_WORK_SCALE = 0.5          # 320x240 stream -> dense map computed at 160x120
DENSE_FILL_INVALID = True
DENSE_FILL_CONFIDENCE = 0.5
GUIDED_REFINE = False
GUIDED_RADIUS = 6
GUIDED_EPS = 1e-3
GUIDED_SCALE = 0.5
