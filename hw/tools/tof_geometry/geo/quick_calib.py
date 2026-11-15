"""
quick_calib.py - chessboard-free camera<->ToF calibration from "click" samples.

Camera intrinsics start from the OV2640 geometry instead of a chessboard:
  * the sensor is 1600x1200 (UXGA); every framesize is the LARGEST window of
    its aspect ratio inside it, scaled down (1:1 modes such as 240x240 use the
    central 1200x1200, 4:3 modes the full 1600x1200, 16:9 1600x900);
  * the stock AI-Thinker lens is sold as ~66 deg diagonal (wide / fisheye
    variants exist: pass --lens-diag-fov), distortion assumed small (zero).
  => fx = fy = (diag_px / 2 / tan(diag_fov / 2)) / crop_scale, principal point = centre.

Samples: the user holds an object (book, box, hand) in front of both sensors,
freezes a synced pair and clicks the object in the image. The ToF side of the
sample is the cluster of nearest zones (within CLUSTER_MM of the closest one):
the mean of their 3D points. Each sample says "this ToF 3D point must project
onto this pixel".

Solve (Levenberg-Marquardt, numpy): rotation ToF->camera (3 params) + focal
length scale (1 param, weak prior towards the nominal lens), for each of the 8
possible zone orientations and 4 starting rolls; the lowest pixel RMS wins.
The translation (sensor offset) is taken from a measurement (--offset-mm):
with an 8x8 sensor it is not observable from a few clicks, and a few cm error
only shifts things by a fraction of a zone beyond ~40 cm.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calibration import Intrinsics, rot_z
from frames import ToFFrame
from tof_zones import Orientation, ZoneModel, all_orientations

SENSOR_W, SENSOR_H = 1600, 1200        # OV2640 UXGA
CLUSTER_MM = 80.0                       # zones within this of the nearest one belong to the object
MIN_SEPARATION_MM = 150.0               # object must be this much closer than the background


def nominal_intrinsics(width: int, height: int, diag_fov_deg: float = 66.0) -> Intrinsics:
    f_sensor = np.hypot(SENSOR_W, SENSOR_H) / 2 / np.tan(np.radians(diag_fov_deg) / 2)
    scale = min(SENSOR_W / width, SENSOR_H / height)          # sensor px per output px
    f = f_sensor / scale
    K = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])
    return Intrinsics(K, np.zeros(5), (width, height), 3.0, f"nominal OV2640 {diag_fov_deg:.0f} deg diag",
                      ("nominal intrinsics (no chessboard): distortion ignored",))


@dataclass
class ClickSample:
    px: np.ndarray            # (2,) clicked pixel (camera image coordinates)
    raw_idx: np.ndarray       # raw zone indices of the object cluster
    dist: np.ndarray          # their distances (mm)


def object_cluster(tof: ToFFrame, cluster_mm: float = CLUSTER_MM,
                   min_sep_mm: float = MIN_SEPARATION_MM) -> tuple[np.ndarray, np.ndarray] | None:
    """Raw indices + distances of the nearest object, or None if nothing stands out."""
    d = tof.dist_mm.ravel()
    v = tof.valid.ravel()
    if v.sum() < 4:
        return None
    dmin = d[v].min()
    idx = np.nonzero(v & (d <= dmin + cluster_mm))[0]
    background = np.median(d[v])
    if background - dmin < min_sep_mm or len(idx) > 40:
        return None
    return idx, d[idx].astype(np.float64)


def sample_point_tof(s: ClickSample, o: Orientation, zm: ZoneModel, radial: bool = True) -> np.ndarray:
    rays = zm.center_rays(unit=radial)
    pts = [rays[o.geom_pos_of_raw(int(i))] * di for i, di in zip(s.raw_idx, s.dist)]
    return np.mean(pts, axis=0)


@dataclass
class QuickResult:
    R: np.ndarray
    fx: float
    orientation: Orientation
    rms_px: float
    residuals_px: np.ndarray
    ranking: list[tuple[float, str]]


def _project(P: np.ndarray, R: np.ndarray, t: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    Pc = P @ R.T + t
    z = np.where(Pc[:, 2] > 1.0, Pc[:, 2], np.nan)
    return np.stack([f * Pc[:, 0] / z + cx, f * Pc[:, 1] / z + cy], -1)


def _lm(P, px, R0, t, f0, cx, cy, fit_f, prior_px, iters=80):
    def resid(x):
        dR, _ = cv2.Rodrigues(x[:3].reshape(3, 1))
        f = f0 * np.exp(x[3]) if fit_f else f0
        r = (_project(P, dR @ R0, t, f, cx, cy) - px).ravel()
        r = np.where(np.isfinite(r), r, 1e4)
        return np.append(r, prior_px * x[3] / 0.25) if fit_f else r
    x = np.zeros(4)
    f_ = resid(x)
    lam = 1e-2
    for _ in range(iters):
        J = np.empty((f_.size, 4))
        for k in range(4):
            h = 1e-6
            xp = x.copy()
            xp[k] += h
            J[:, k] = (resid(xp) - f_) / h
        A = J.T @ J + 1e-12 * np.eye(4)
        g = J.T @ f_
        improved = False
        for _ in range(8):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            fn = resid(x + step)
            if fn @ fn < f_ @ f_:
                x, f_, lam, improved = x + step, fn, max(lam / 3, 1e-9), True
                break
            lam *= 5
        if not improved or np.linalg.norm(step) < 1e-9:
            break
    dR, _ = cv2.Rodrigues(x[:3].reshape(3, 1))
    return dR @ R0, (f0 * np.exp(x[3]) if fit_f else f0), f_


def solve_quick(samples: list[ClickSample], intr: Intrinsics, t: np.ndarray, zm: ZoneModel,
                radial: bool = True, fit_fx: bool = True, prior_px: float = 3.0,
                orientations: list[Orientation] | None = None) -> QuickResult:
    if len(samples) < 4:
        raise ValueError("need at least 4 samples")
    px = np.array([s.px for s in samples], float)
    f0, cx, cy = intr.K[0, 0], intr.K[0, 2], intr.K[1, 2]
    best = None
    ranking = []
    for o in orientations or all_orientations():
        P = np.array([sample_point_tof(s, o, zm, radial) for s in samples])
        best_o = None
        for roll in (0, 90, 180, 270):
            R, f, res = _lm(P, px, rot_z(roll), t, f0, cx, cy, fit_fx, prior_px)
            r = res[: 2 * len(samples)].reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum(r ** 2, 1))))
            if best_o is None or rms < best_o[0]:
                best_o = (rms, R, f, np.linalg.norm(r, axis=1))
        ranking.append((best_o[0], o.describe()))
        if best is None or best_o[0] < best[0]:
            best = (best_o[0], best_o[1], best_o[2], o, best_o[3])
    ranking.sort()
    return QuickResult(best[1], float(best[2]), best[3], best[0], best[4], ranking)
