"""
projection.py - ToF zones -> 3D points -> camera pixels.

For each zone (GEOMETRY grid order, see tof_zones.py):
    ray      : zone-centre direction in the ToF frame (zone model)
    point    : radial        -> unit_ray * d
               perpendicular -> (tx, ty, 1) * d        (config TOF_DISTANCE_MODE)
    camera   : p_cam = R p_tof + t                     (calibration.Extrinsic)
    pixel    : cv2.projectPoints(K, dist), rejected beyond the distortion
               model's monotonic radius (calibration.Intrinsics.r_max)
The zone FOOTPRINT is the four zone-corner rays evaluated at the zone's own
measured distance, so each zone becomes a quad whose position depends on
depth (parallax between the two sensors is handled exactly).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calibration import CalibrationBundle, Intrinsics
from frames import ToFFrame

N = 8


@dataclass(frozen=True)
class ZoneProjection:
    """All arrays are in GEOMETRY grid order (8, 8, ...)."""
    dist_mm: np.ndarray          # (8,8) measured distance
    valid: np.ndarray            # (8,8) sensor-valid AND projectable
    confidence: np.ndarray       # (8,8) sensor confidence (0 where sensor-invalid)
    raw_index: np.ndarray        # (8,8) raw firmware index at each position
    points_cam: np.ndarray       # (8,8,3) zone centre point in camera frame, mm
    depth_cam: np.ndarray        # (8,8) z of points_cam (depth-image convention), mm
    centers_px: np.ndarray       # (8,8,2) NaN where not projectable
    quads_px: np.ndarray         # (8,8,4,2) NaN where not projectable
    image_size: tuple[int, int]  # (w, h)
    sensor_valid: np.ndarray | None = None   # (8,8) sensor-valid, projectable or not


class Projector:
    def __init__(self, bundle: CalibrationBundle, image_size: tuple[int, int] | None = None):
        self.bundle = bundle
        self.intr: Intrinsics = bundle.intrinsics if image_size is None else bundle.intrinsics.scaled_to(*image_size)
        self.R = bundle.extrinsic.R
        self.t = bundle.extrinsic.t
        zm = bundle.zone_model
        radial = bundle.distance_mode == "radial"
        # Per-unit-distance vectors: point = vec * d
        self.center_vec = zm.center_rays(unit=radial)        # (8,8,3)
        self.corner_vec = zm.corner_rays(unit=radial)        # (9,9,3)
        self.raw_index = bundle.orientation.perm             # (8,8)

    @property
    def image_size(self) -> tuple[int, int]:
        return self.intr.size

    # ---------------- generic ----------------

    def project_cam_points(self, p_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame points (...,3) -> pixels (...,2) and ok mask (...)."""
        shape = p_cam.shape[:-1]
        p = p_cam.reshape(-1, 3).astype(np.float64)
        z = p[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            rn = np.hypot(p[:, 0] / z, p[:, 1] / z)
        ok = np.isfinite(rn) & (z > 1.0) & (rn < self.intr.r_max)
        px = np.full((len(p), 2), np.nan)
        if ok.any():
            proj, _ = cv2.projectPoints(p[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.intr.K, self.intr.dist)
            px[ok] = proj.reshape(-1, 2)
        return px.reshape(shape + (2,)), ok.reshape(shape)

    def tof_points(self, dist_geom: np.ndarray, vec: np.ndarray) -> np.ndarray:
        return vec * dist_geom[..., None]

    # ---------------- zones ----------------

    def project_frame(self, tof: ToFFrame) -> ZoneProjection:
        o = self.bundle.orientation
        return self.project(o.apply(tof.dist_mm), o.apply(tof.valid), o.apply(tof.confidence))

    def project(self, dist_geom: np.ndarray, valid_geom: np.ndarray,
                conf_geom: np.ndarray | None = None) -> ZoneProjection:
        d = np.where(valid_geom, dist_geom, np.nan).astype(np.float64)
        p_tof = self.tof_points(d, self.center_vec)                  # (8,8,3)
        p_cam = p_tof @ self.R.T + self.t
        centers, ok_c = self.project_cam_points(p_cam)

        # quads: corners (r,c) (r,c+1) (r+1,c+1) (r+1,c) at each zone's own distance
        cv = self.corner_vec
        quad_vec = np.stack([cv[:-1, :-1], cv[:-1, 1:], cv[1:, 1:], cv[1:, :-1]], axis=2)   # (8,8,4,3)
        q_cam = (quad_vec * d[..., None, None]) @ self.R.T + self.t
        quads, ok_q = self.project_cam_points(q_cam)
        ok = valid_geom & ok_c & ok_q.all(axis=-1)
        centers[~ok] = np.nan
        quads[~ok] = np.nan
        conf = np.ones_like(d, dtype=np.float32) if conf_geom is None else conf_geom.astype(np.float32)
        return ZoneProjection(
            dist_mm=dist_geom.astype(np.float32), valid=ok, confidence=np.where(valid_geom, conf, 0).astype(np.float32),
            raw_index=self.raw_index, points_cam=p_cam, depth_cam=np.where(ok, p_cam[..., 2], np.nan),
            centers_px=centers, quads_px=quads, image_size=self.image_size,
            sensor_valid=valid_geom.astype(bool),
        )

    def fov_outline(self, distance_mm: float, samples_per_edge: int = 8) -> np.ndarray | None:
        """Polyline (M,2) of the ToF FoV border projected at a given distance."""
        c = self.corner_vec
        border = np.concatenate([c[0, :], c[1:, -1], c[-1, -2::-1], c[-2:0:-1, 0]], axis=0)   # ring of (9,9)
        # densify in ToF space so distortion bends the outline correctly
        pts = []
        for a, b in zip(border, np.roll(border, -1, axis=0)):
            for s in np.linspace(0, 1, samples_per_edge, endpoint=False):
                pts.append(a * (1 - s) + b * s)
        p_cam = np.asarray(pts) * distance_mm @ self.R.T + self.t
        px, ok = self.project_cam_points(p_cam)
        if ok.sum() < 3:
            return None
        return px[ok]
