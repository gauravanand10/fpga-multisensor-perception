"""
depth_map.py - sparse and dense depth images in the camera view (STEP 8 / 9).

Depth convention: float32 millimetres, z in the CAMERA frame (like a depth
camera), NaN = unknown.

SPARSE (Step 8)
    Every valid zone's projected footprint quad (projection.py) is filled with
    that zone's camera-frame depth. Far zones are drawn first so nearer ones
    win where footprints overlap (parallax). Also returned: confidence (from
    the ToF status) and the zone id per pixel.

DENSE (Step 9)
    Interpolation is done where the samples are regular - in the ToF's own
    angular grid - not over scattered pixels:
      1. readings are converted to inverse ToF-axis depth 1/z_tof (exactly linear
         across the grid for any plane); invalid zones are filled from valid
         neighbours in that domain (confidence x0.5 per step)
      2. for each pixel (at DENSE_WORK_SCALE) its camera ray is intersected with
         the ToF measurement surface: guess z -> 3D point -> ToF angles ->
         fractional zone coords -> bilinear 1/z_tof -> 3D point along the
         ToF ray -> new z, repeated 3x (converges because the baseline is small).
         This handles parallax exactly for a smooth surface.
      3. pixels whose ray leaves the ToF FoV are masked (NaN).
      4. upsampled to camera resolution.
    Optional: RGB-guided edge refinement (guided filter, He et al. 2010),
    implemented with cv2.boxFilter only; off by default (config.GUIDED_REFINE).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config
from calibration import CalibrationBundle, pixel_rays
from projection import Projector, ZoneProjection

N = 8


@dataclass(frozen=True)
class SparseDepth:
    depth: np.ndarray        # (H,W) float32 mm, NaN unknown
    confidence: np.ndarray   # (H,W) float32 0..1
    zone_id: np.ndarray      # (H,W) int16 geometry index r*8+c, -1 none

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.depth)


@dataclass(frozen=True)
class DenseDepth:
    depth: np.ndarray        # (H,W) float32 mm, NaN outside ToF FoV / no data
    confidence: np.ndarray   # (H,W) float32 0..1 (1 measured, <1 filled/extrapolated)
    in_fov: np.ndarray       # (H,W) bool

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.depth)


def fill_invalid(dist: np.ndarray, valid: np.ndarray, conf: np.ndarray,
                 factor: float = config.DENSE_FILL_CONFIDENCE, max_steps: int = 8):
    """Iteratively fill invalid cells with the mean of filled 8-neighbours."""
    d = np.where(valid, dist, 0).astype(np.float64)
    c = np.where(valid, conf, 0).astype(np.float64)
    have = valid.copy()
    k = np.ones((3, 3), np.float64)
    for _ in range(max_steps):
        if have.all():
            break
        w = cv2.filter2D(have.astype(np.float64), -1, k, borderType=cv2.BORDER_CONSTANT)
        sd = cv2.filter2D(d * have, -1, k, borderType=cv2.BORDER_CONSTANT)
        sc = cv2.filter2D(c * have, -1, k, borderType=cv2.BORDER_CONSTANT)
        new = ~have & (w > 0)
        if not new.any():
            break
        d[new] = sd[new] / w[new]
        c[new] = sc[new] / w[new] * factor
        have = have | new
    d[~have] = np.nan
    return d.astype(np.float32), c.astype(np.float32), have


class DepthMapper:
    def __init__(self, bundle: CalibrationBundle, image_size: tuple[int, int],
                 work_scale: float = config.DENSE_WORK_SCALE):
        self.bundle = bundle
        self.projector = Projector(bundle, image_size)
        self.W, self.H = image_size
        self.ws = work_scale
        self.w = max(8, int(round(self.W * work_scale)))
        self.h = max(8, int(round(self.H * work_scale)))
        uw, vw = np.meshgrid(np.arange(self.w), np.arange(self.h))
        u = (uw + 0.5) * self.W / self.w - 0.5
        v = (vw + 0.5) * self.H / self.h - 0.5
        self.rays = pixel_rays(self.projector.intr, u, v)            # (h,w,3), z = 1
        self.R = bundle.extrinsic.R
        self.t = bundle.extrinsic.t
        self.radial = bundle.distance_mode == "radial"
        self.zm = bundle.zone_model
        # Lookup table tangent (tx, ty) -> fractional zone coords (gr, gc), built once:
        # the iterative inverse is exact but too slow per pixel per frame.
        ext = np.abs(self.zm.corners).max() * 1.6
        self._lut_n = 385
        self._lut_t0 = -ext
        self._lut_dt = 2 * ext / (self._lut_n - 1)
        g = self._lut_t0 + np.arange(self._lut_n) * self._lut_dt
        TX, TY = np.meshgrid(g, g)
        gr, gc = self.zm.tangent_to_grid(TX, TY)
        self._lut_gr = gr.astype(np.float32)
        self._lut_gc = gc.astype(np.float32)

    def tangent_to_grid(self, tx: np.ndarray, ty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = ((tx - self._lut_t0) / self._lut_dt).astype(np.float32)
        iy = ((ty - self._lut_t0) / self._lut_dt).astype(np.float32)
        gr = cv2.remap(self._lut_gr, ix, iy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-99.0)
        gc = cv2.remap(self._lut_gc, ix, iy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-99.0)
        return gr, gc

    # ---------------- sparse ----------------

    def sparse(self, zp: ZoneProjection) -> SparseDepth:
        depth = np.full((self.H, self.W), np.nan, np.float32)
        conf = np.zeros((self.H, self.W), np.float32)
        zid = np.full((self.H, self.W), -1, np.int16)
        idx = np.argwhere(zp.valid)
        order = np.argsort(-zp.depth_cam[zp.valid])                 # far first
        for r, c in idx[order]:
            pts = np.round(zp.quads_px[r, c] * 16).astype(np.int32).reshape(-1, 1, 2)
            z = float(zp.depth_cam[r, c])
            cv2.fillPoly(depth, [pts], z, cv2.LINE_8, 4)
            cv2.fillPoly(conf, [pts], float(zp.confidence[r, c]), cv2.LINE_8, 4)
            cv2.fillPoly(zid, [pts], int(r * N + c), cv2.LINE_8, 4)
        return SparseDepth(depth, conf, zid)

    # ---------------- dense ----------------

    def dense(self, zp: ZoneProjection, iterations: int = 3) -> DenseDepth:
        sv = zp.sensor_valid if zp.sensor_valid is not None else np.isfinite(zp.dist_mm)
        if not sv.any():
            nan = np.full((self.H, self.W), np.nan, np.float32)
            return DenseDepth(nan, np.zeros_like(nan), np.zeros((self.H, self.W), bool))
        conf0 = np.where(sv, zp.confidence, 0).astype(np.float32)
        # Interpolate INVERSE ToF-axis depth w = 1/z_tof: for any plane it is exactly linear
        # in the ToF tangent coordinates, so flat surfaces (and neighbour-filled zones) come
        # out exact. Radial readings d along a unit ray have z_tof = d * ray_z.
        vz = self.zm.center_rays(unit=True)[..., 2] if self.radial else np.ones((N, N))
        with np.errstate(divide="ignore", invalid="ignore"):
            w_raw = np.where(sv, 1.0 / (zp.dist_mm * vz), 0).astype(np.float32)
        if config.DENSE_FILL_INVALID:
            w_grid, c_grid, have = fill_invalid(w_raw, sv, conf0)
        else:
            w_grid = np.where(sv, w_raw, np.nan).astype(np.float32)
            c_grid, have = conf0, sv
        z = np.full((self.h, self.w), float(1.0 / np.nanmedian(w_grid)), np.float64)
        for _ in range(iterations):
            p_cam = self.rays * z[..., None]
            p_tof = (p_cam - self.t) @ self.R
            p_tof[..., 2] = np.where(p_tof[..., 2] > 1e-6, p_tof[..., 2], np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                tx = p_tof[..., 0] / p_tof[..., 2]
                ty = p_tof[..., 1] / p_tof[..., 2]
            gr, gc = self.tangent_to_grid(np.nan_to_num(tx, nan=1e3), np.nan_to_num(ty, nan=1e3))
            w = cv2.remap(w_grid, gc.astype(np.float32), gr.astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
            with np.errstate(divide="ignore", invalid="ignore"):
                z_tof = np.where(w > 0, 1.0 / w, np.nan)
            q = np.stack([tx, ty, np.ones_like(tx)], -1) * z_tof[..., None] @ self.R.T + self.t
            z = np.where(np.isfinite(q[..., 2]), q[..., 2], z)
        in_fov = np.isfinite(p_tof[..., 2]) & (gr >= -0.5) & (gr <= N - 0.5) & (gc >= -0.5) & (gc <= N - 0.5)
        conf = cv2.remap(c_grid, gc.astype(np.float32), gr.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        ok = in_fov & np.isfinite(z_tof) & (z > 0)
        z_w = np.where(ok, z, np.nan).astype(np.float32)
        return self._upsample(z_w, np.where(ok, conf, 0).astype(np.float32), ok)

    def _upsample(self, z: np.ndarray, conf: np.ndarray, ok: np.ndarray) -> DenseDepth:
        size = (self.W, self.H)
        m = ok.astype(np.float32)
        num = cv2.resize(np.nan_to_num(z) * m, size, interpolation=cv2.INTER_LINEAR)
        den = cv2.resize(m, size, interpolation=cv2.INTER_LINEAR)
        cf = cv2.resize(conf, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 0.5
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = np.where(mask & (den > 1e-3), num / den, np.nan).astype(np.float32)
        return DenseDepth(depth, np.where(mask, cf, 0).astype(np.float32), mask)

    # ---------------- optional RGB-guided refinement ----------------

    def refine_guided(self, dense: DenseDepth, image_bgr: np.ndarray,
                      radius: int = config.GUIDED_RADIUS, eps: float = config.GUIDED_EPS,
                      scale: float = config.GUIDED_SCALE) -> DenseDepth:
        """Edge-aware refinement: guided filter with the grey image as guide (masked, NaN-aware)."""
        w, h = max(8, int(self.W * scale)), max(8, int(self.H * scale))
        I = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_AREA)
        I = I.astype(np.float32) / 255.0
        m = cv2.resize(dense.valid.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        p = cv2.resize(np.nan_to_num(dense.depth), (w, h), interpolation=cv2.INTER_AREA) / 1000.0

        def box(x):
            return cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1), normalize=True,
                                 borderType=cv2.BORDER_REFLECT)

        n = np.maximum(box(m), 1e-6)
        mI = box(I * m) / n
        mp = box(p * m) / n
        cov = box(I * p * m) / n - mI * mp
        var = box(I * I * m) / n - mI * mI
        a = cov / (var + eps)
        b = mp - a * mI
        q = (box(a * m) / n) * I + box(b * m) / n
        q = cv2.resize(q * 1000.0, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = np.where(dense.valid, q, np.nan).astype(np.float32)
        return DenseDepth(depth, dense.confidence, dense.in_fov)
