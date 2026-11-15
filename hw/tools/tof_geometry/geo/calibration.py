"""
calibration.py - load + validate camera intrinsics, camera<->ToF extrinsic,
and ToF orientation, and bundle them with the zone model.

Extrinsic convention used everywhere in depth\\:
    p_cam = R @ p_tof + t          (ToF frame -> camera frame, millimetres)
p_tof is in the ToF frame defined in tof_zones.py (ST convention), p_cam in
the OpenCV camera frame (x right, y down, z forward).

Evidence for the legacy file (extrinsic_camera_tof_final.npz): its solver
(esp32_cam/new_camera_segmentation_clean/calibration_scripts/
solve_extrinsic_final.py) minimises R @ n_tof - n_cam and uses a 25.0 mm
board square and a 35.0 mm baseline, so R_ct maps ToF->camera and t_ct is in
millimetres. The loader still checks this and converts metres if |t| < 1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

import config
from tof_zones import Orientation, ZoneModel, make_zone_model

log = logging.getLogger("calib")


class CalibrationError(RuntimeError):
    pass


# ============================================================
# INTRINSICS
# ============================================================

@dataclass(frozen=True)
class Intrinsics:
    K: np.ndarray                 # 3x3
    dist: np.ndarray              # (n,) OpenCV order k1 k2 p1 p2 [k3 ...]
    size: tuple[int, int]         # (width, height) the calibration was made at
    r_max: float                  # normalised radius where the distortion model stops being monotonic
    source: str = ""
    notes: tuple[str, ...] = ()

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Intrinsics for a different image size of the SAME sensor/lens (same aspect, same crop)."""
        w0, h0 = self.size
        if (width, height) == (w0, h0):
            return self
        sx, sy = width / w0, height / h0
        notes = self.notes
        if abs(sx - sy) > 0.01 * max(sx, sy):
            notes = notes + (f"image {width}x{height} has a different aspect than calibration {w0}x{h0}: "
                             "K scaled per axis, mapping is approximate - recalibrate at this size",)
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return replace(self, K=K, size=(width, height), notes=notes)

    @property
    def fov_deg(self) -> tuple[float, float]:
        w, h = self.size
        return (float(np.degrees(2 * np.arctan(w / 2 / self.K[0, 0]))),
                float(np.degrees(2 * np.arctan(h / 2 / self.K[1, 1]))))


def pixel_rays(intr: "Intrinsics", u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Camera-frame ray (x, y, 1) for each pixel (u, v): undistorted, normalised
    image coordinates, so ray * z is the 3D point at depth z. Shape (..., 3).
    """
    pts = np.stack([np.asarray(u, np.float64), np.asarray(v, np.float64)], -1)
    shape = pts.shape[:-1]
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
    und = cv2.undistortPoints(pts.reshape(-1, 1, 2), intr.K, intr.dist, None, None, None, crit)
    und = und.reshape(shape + (2,))
    return np.concatenate([und, np.ones(shape + (1,))], -1)


def distortion_r_max(dist: np.ndarray, r_limit: float = 3.0) -> float:
    """
    Largest normalised radius r for which r_d = r*(1 + k1 r^2 + k2 r^4 + k3 r^6)
    is still increasing. Beyond it cv2.projectPoints folds points back into
    the image, so projections outside r_max are rejected.
    """
    flat = np.ravel(np.asarray(dist, np.float64))
    d = np.zeros(8)
    d[: min(flat.size, 8)] = flat[:8]
    k1, k2, k3 = d[0], d[1], d[4]
    r = np.linspace(1e-4, r_limit, 30000)
    deriv = 1 + 3 * k1 * r**2 + 5 * k2 * r**4 + 7 * k3 * r**6
    bad = np.nonzero(deriv <= 0.05)[0]
    return float(r[bad[0]] * 0.98) if bad.size else r_limit


def load_intrinsics(path: str | Path) -> Intrinsics:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        K = _first(z, ("camera_matrix", "K", "mtx"))
        dist = _first(z, ("dist_coeffs", "dist", "D"))
        w = int(_first(z, ("image_width", "width"), default=0))
        h = int(_first(z, ("image_height", "height"), default=0))
    if K is None or dist is None:
        raise CalibrationError(f"{path.name}: needs camera_matrix + dist_coeffs, has {sorted(keys)}")
    K = np.asarray(K, np.float64).reshape(3, 3)
    dist = np.asarray(dist, np.float64).ravel()
    notes: list[str] = []
    if w <= 0 or h <= 0:
        w, h = int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))
        notes.append(f"no image size stored; assumed {w}x{h} from principal point")
    if not (K[0, 0] > 0 and K[1, 1] > 0 and abs(K[2, 2] - 1) < 1e-9 and K[1, 0] == 0 and K[2, 0] == 0):
        raise CalibrationError(f"{path.name}: camera_matrix is not a valid pinhole matrix:\n{K}")
    if not (0.25 * w < K[0, 2] < 0.75 * w and 0.25 * h < K[1, 2] < 0.75 * h):
        notes.append(f"principal point ({K[0, 2]:.0f},{K[1, 2]:.0f}) far from image centre")
    if len(dist) not in (4, 5, 8, 12, 14):
        raise CalibrationError(f"{path.name}: unexpected dist_coeffs length {len(dist)}")
    if len(dist) >= 5 and (abs(dist[1]) > 0.5 or abs(dist[4]) > 0.5):
        notes.append(f"large k2/k3 ({dist[1]:.2f}, {dist[4]:.2f}): likely over-fitted, unreliable near image edges")
    r_max = distortion_r_max(dist)
    corner_r = float(np.hypot(max(K[0, 2], w - K[0, 2]) / K[0, 0], max(K[1, 2], h - K[1, 2]) / K[1, 1]))
    if r_max < corner_r:
        notes.append(f"distortion model folds at r={r_max:.3f} < image corner r={corner_r:.3f}")
    return Intrinsics(K, dist, (w, h), r_max, str(path), tuple(notes))


# ============================================================
# EXTRINSIC
# ============================================================

@dataclass(frozen=True)
class Extrinsic:
    R: np.ndarray                 # 3x3, ToF -> camera
    t: np.ndarray                 # (3,), mm, ToF origin in camera coords
    source: str = ""
    notes: tuple[str, ...] = ()
    fov_scale: float | None = None        # stored by calibrate_extrinsic --fit-fov
    distance_mode: str | None = None      # stored by calibrate_extrinsic

    def tof_to_cam(self, p: np.ndarray) -> np.ndarray:
        return p @ self.R.T + self.t

    def cam_to_tof(self, p: np.ndarray) -> np.ndarray:
        return (p - self.t) @ self.R

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(self.R) - 1) / 2, -1, 1))))

    @property
    def tof_axis_in_cam(self) -> np.ndarray:
        return self.R[:, 2]

    def save(self, path: str | Path, **extra) -> None:
        np.savez(path, R_ct=self.R, t_ct=self.t, baseline=np.linalg.norm(self.t),
                 direction="tof_to_camera", t_units="mm", **extra)


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def nominal_extrinsic() -> Extrinsic:
    return Extrinsic(rot_z(180.0), np.asarray(config.NOMINAL_TOF_T_MM, float), "nominal (config)",
                     ("no extrinsic file: using nominal upright mounting",))


def load_extrinsic(path: str | Path) -> Extrinsic:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        R = _first(z, ("R_ct", "R", "R_tof_to_cam"))
        t = _first(z, ("t_ct", "t", "T", "t_tof_to_cam"))
        direction = str(_first(z, ("direction",), default="tof_to_camera"))
        units = str(_first(z, ("t_units",), default=""))
        fov_scale = _first(z, ("fov_scale",), default=None)
        dmode = _first(z, ("distance_mode",), default=None)
        stats = {k: np.asarray(z[k]) for k in ("normal_errors_deg", "plane_errors_mm", "rms_mm") if k in z.files}
    if R is None or t is None:
        raise CalibrationError(f"{path.name}: needs R_ct and t_ct")
    R = np.asarray(R, np.float64).reshape(3, 3)
    t = np.asarray(t, np.float64).ravel()
    notes: list[str] = []
    det = np.linalg.det(R)
    ortho = np.abs(R @ R.T - np.eye(3)).max()
    if ortho > 1e-3:
        raise CalibrationError(f"{path.name}: R is not orthonormal (max |RR^T-I| = {ortho:.2e})")
    if det < 0:
        raise CalibrationError(f"{path.name}: det(R) = {det:.3f} < 0 is a reflection. The zone order is "
                               "mirrored relative to the geometry: fix it with the orientation (TOF_FLIP_H) instead.")
    if direction == "camera_to_tof":
        R, t = R.T, -R.T @ t
        notes.append("file stored camera->ToF; inverted to ToF->camera")
    if units == "m" or (units == "" and 0 < np.linalg.norm(t) < 1.0):
        t = t * 1000.0
        notes.append("translation looked like metres; converted to mm")
    base = float(np.linalg.norm(t))
    if base > 300:
        notes.append(f"baseline {base:.0f} mm is implausibly large for a sensor pair on one board")
    if abs(t[2]) > 0.6 * base and base > 5:
        notes.append(f"{abs(t[2]) / base:.0%} of the translation is along the camera axis - "
                     "poorly constrained by the legacy plane method, treat t as approximate")
    if "normal_errors_deg" in stats:
        ne = stats["normal_errors_deg"]
        notes.append(f"stored fit quality: normal error mean {ne.mean():.1f} deg, max {ne.max():.1f} deg")
    if "plane_errors_mm" in stats:
        pe = np.abs(stats["plane_errors_mm"])
        notes.append(f"stored fit quality: plane error mean {pe.mean():.1f} mm, max {pe.max():.1f} mm")
    return Extrinsic(R, t, str(path), tuple(notes),
                     None if fov_scale is None else float(fov_scale),
                     None if dmode is None else str(dmode))


# ============================================================
# ORIENTATION
# ============================================================

def config_orientation() -> Orientation:
    return Orientation(config.TOF_FLIP_H, config.TOF_FLIP_V, config.TOF_ROTATE, config.TOF_TRANSPOSE)


def load_orientation(path: str | Path | None) -> tuple[Orientation, str]:
    if path is not None and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return Orientation.from_dict(json.load(f)), str(path)
    return config_orientation(), "config.py"


def save_orientation(o: Orientation, path: str | Path, note: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**o.to_dict(), "note": note}, f, indent=2)


# ============================================================
# BUNDLE
# ============================================================

@dataclass(frozen=True)
class CalibrationBundle:
    intrinsics: Intrinsics
    extrinsic: Extrinsic
    orientation: Orientation
    zone_model: ZoneModel
    distance_mode: str
    orientation_source: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def with_orientation(self, o: Orientation) -> "CalibrationBundle":
        return replace(self, orientation=o)

    def summary(self) -> str:
        i, e = self.intrinsics, self.extrinsic
        fx, fy = i.fov_deg
        axis = e.tof_axis_in_cam
        tilt = np.degrees(np.arctan2(np.hypot(axis[0], axis[1]), axis[2]))
        lines = [
            f"intrinsics : {i.source}",
            f"             {i.size[0]}x{i.size[1]}  fx={i.K[0, 0]:.1f} fy={i.K[1, 1]:.1f} "
            f"cx={i.K[0, 2]:.1f} cy={i.K[1, 2]:.1f}  FoV {fx:.1f}x{fy:.1f} deg",
            f"             dist={np.array2string(i.dist, precision=4, suppress_small=True)}  r_max={i.r_max:.3f}",
            f"extrinsic  : {e.source}",
            f"             direction ToF->camera, units mm; rotation {e.rotation_deg:.1f} deg, "
            f"|t|={np.linalg.norm(e.t):.1f} mm, t={np.array2string(e.t, precision=1)}",
            f"             ToF optical axis in camera frame {np.array2string(axis, precision=3)} "
            f"({tilt:.1f} deg from camera axis)",
            f"orientation: {self.orientation.describe()}  (from {self.orientation_source})",
            f"zone model : {self.zone_model.name}, distance mode {self.distance_mode}",
        ]
        for n in i.notes + e.notes + self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)


def load_bundle(sim: bool = False,
                intrinsics_path: str | Path | None = None,
                extrinsic_path: str | Path | None = None,
                orientation_path: str | Path | None = None) -> CalibrationBundle:
    """Pick calibration files per config (sim or real) and validate them."""
    notes: list[str] = []
    if intrinsics_path is None:
        cands = [config.SIM_INTRINSICS_FILE] if sim else config.INTRINSICS_FILES
        intrinsics_path = next((p for p in cands if Path(p).exists()), None)
        if intrinsics_path is None:
            raise CalibrationError("no intrinsics file found in " + ", ".join(str(p) for p in cands))
        if not sim and Path(intrinsics_path).name == "camera_calibration.npz":
            notes.append("using LEGACY intrinsics (calibrated on a different USB camera, not the ESP32-CAM): "
                         "run tools\\calibrate_intrinsics.py")
    intr = load_intrinsics(intrinsics_path)

    if extrinsic_path is None:
        cands = [config.SIM_EXTRINSIC_FILE] if sim else config.EXTRINSIC_FILES
        extrinsic_path = next((p for p in cands if Path(p).exists()), None)
    extr = load_extrinsic(extrinsic_path) if extrinsic_path else nominal_extrinsic()
    if not sim and extrinsic_path and Path(extrinsic_path).name == "extrinsic_camera_tof_final.npz":
        notes.append("using LEGACY extrinsic (solved with a different camera): run tools\\calibrate_extrinsic.py")

    if orientation_path is None:
        orientation_path = config.SIM_ORIENTATION_FILE if sim else config.ORIENTATION_FILE
    orientation, osrc = load_orientation(orientation_path)

    fov_scale = extr.fov_scale if extr.fov_scale is not None else config.TOF_FOV_SCALE
    dmode = extr.distance_mode or config.TOF_DISTANCE_MODE
    if dmode not in ("radial", "perpendicular"):
        raise CalibrationError(f"TOF_DISTANCE_MODE must be radial|perpendicular, got {dmode!r}")
    zm = make_zone_model(config.TOF_ZONE_MODEL, config.TOF_FOV_DEG, float(fov_scale))
    return CalibrationBundle(intr, extr, orientation, zm, dmode, osrc, tuple(notes))


def _first(z, keys, default=None):
    for k in keys:
        if k in z.files:
            v = z[k]
            return v.item() if v.shape == () else v
    return default
