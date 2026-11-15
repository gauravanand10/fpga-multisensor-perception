"""
tof_zones.py - VL53L5CX 8x8 zone geometry and orientation.

Coordinate conventions (see README "Conventions"):

RAW grid
    What the firmware sends: index i = row*8 + col, copied straight from the
    ST driver's distance_mm[i] / target_status[i] (Tof_WebServer/tof_sensor.cpp).

GEOMETRY grid
    RAW grid after the orientation transform (config TOF_TRANSPOSE,
    TOF_ROTATE, TOF_FLIP_H, TOF_FLIP_V). Position (r, c) of this grid is the
    one whose viewing direction is given by the zone model below.
    With the defaults (all off) GEOMETRY == RAW, which is the convention the
    legacy extrinsic calibration was solved in.

ToF frame  (ST pitch/yaw convention, right-handed, millimetres)
    +z = optical axis (out of the sensor)
    +x = direction of increasing column
    +y = direction of decreasing row ("row 0 is up")
    Looking out from behind the sensor with row 0 at the top, +x (column 7)
    therefore points to the viewer's LEFT: the raw column order is mirrored
    w.r.t. the scene. This matches SparkFun's examples, which print columns in
    reverse "to reflect reality". With an upright sensor next to an upright
    camera the ToF->camera rotation is ~180 deg about z, which is exactly what
    the legacy calibration found (171.6 deg about ~z).

Zone model
    "st"      : zone centre directions from ST's published pitch/yaw table
                (the same table the legacy code used, with the zone-48 typo
                fixed). Non-separable: corner zones are further out.
    "uniform" : ideal pinhole grid, TOF_FOV_DEG over 8 zones per axis.
    Both are scaled by TOF_FOV_SCALE (a calibration refinement knob).

Everything is expressed in tangent-plane coordinates (tx, ty) = (x/z, y/z).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

N = 8

# ---------------------------------------------------------------------------
# ST pitch / yaw table (degrees), index = raw zone index.
# Source: Depth_Estimation/tof_zone_table.py (ST's published VL53L5CX table).
# FIX: zone 48 yaw was 203.20 (duplicate of zone 40) in the legacy copy; the
# table is mirror-symmetric (yaw(48) = 360 - yaw(8) = 215.40), so 215.40 is used.
# ---------------------------------------------------------------------------
ST_PITCH_DEG = np.array([
    59.00, 64.00, 67.50, 70.00, 70.00, 67.50, 64.00, 59.00,
    64.00, 70.00, 72.90, 74.90, 74.90, 72.90, 70.00, 64.00,
    67.50, 72.90, 77.40, 80.50, 80.50, 77.40, 72.90, 67.50,
    70.00, 74.90, 80.50, 85.75, 85.75, 80.50, 74.90, 70.00,
    70.00, 74.90, 80.50, 85.75, 85.75, 80.50, 74.90, 70.00,
    67.50, 72.90, 77.40, 80.50, 80.50, 77.40, 72.90, 67.50,
    64.00, 70.00, 72.90, 74.90, 74.90, 72.90, 70.00, 64.00,
    59.00, 64.00, 67.50, 70.00, 70.00, 67.50, 64.00, 59.00,
])
ST_YAW_DEG = np.array([
    135.00, 125.40, 113.20, 98.13, 81.87, 66.80, 54.60, 45.00,
    144.60, 135.00, 120.96, 101.31, 78.69, 59.04, 45.00, 35.40,
    156.80, 149.04, 135.00, 108.45, 71.55, 45.00, 30.96, 23.20,
    171.87, 168.69, 161.55, 135.00, 45.00, 18.45, 11.31, 8.13,
    188.13, 191.31, 198.45, 225.00, 315.00, 341.55, 348.69, 351.87,
    203.20, 210.96, 225.00, 251.55, 288.45, 315.00, 329.04, 336.80,
    215.40, 225.00, 239.04, 258.69, 281.31, 300.96, 315.00, 324.60,
    225.00, 234.60, 246.80, 261.87, 278.13, 293.20, 305.40, 315.00,
])


def st_unit_rays() -> np.ndarray:
    """ST formula: unit direction per raw zone index, shape (64, 3)."""
    p = np.radians(ST_PITCH_DEG)
    y = np.radians(ST_YAW_DEG)
    return np.column_stack([np.cos(y) * np.cos(p), np.sin(y) * np.cos(p), np.sin(p)])


# ============================================================
# ORIENTATION
# ============================================================

@dataclass(frozen=True)
class Orientation:
    """
    Maps the RAW grid to the GEOMETRY grid. Operations are applied to the
    raw grid in this order: transpose, rotate (counter-clockwise, multiples
    of 90), flip_h (mirror columns), flip_v (mirror rows).
    """
    flip_h: bool = False
    flip_v: bool = False
    rotate: int = 0
    transpose: bool = False

    def __post_init__(self):
        if self.rotate % 90:
            raise ValueError(f"rotate must be a multiple of 90, got {self.rotate}")

    @property
    def perm(self) -> np.ndarray:
        """perm[r, c] = raw zone index shown at geometry position (r, c)."""
        return _perm(self.flip_h, self.flip_v, self.rotate % 360, self.transpose)

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """RAW (8,8,...) -> GEOMETRY (8,8,...)."""
        flat = raw.reshape(N * N, *raw.shape[2:])
        return flat[self.perm.ravel()].reshape(raw.shape)

    def invert(self, geom: np.ndarray) -> np.ndarray:
        """GEOMETRY (8,8,...) -> RAW (8,8,...)."""
        out = np.empty_like(geom).reshape(N * N, *geom.shape[2:])
        out[self.perm.ravel()] = geom.reshape(N * N, *geom.shape[2:])
        return out.reshape(geom.shape)

    def geom_pos_of_raw(self, raw_index: int) -> tuple[int, int]:
        r, c = np.argwhere(self.perm == raw_index)[0]
        return int(r), int(c)

    def to_dict(self) -> dict:
        return {"flip_h": self.flip_h, "flip_v": self.flip_v,
                "rotate": self.rotate % 360, "transpose": self.transpose}

    @classmethod
    def from_dict(cls, d: dict) -> "Orientation":
        return cls(bool(d.get("flip_h", False)), bool(d.get("flip_v", False)),
                   int(d.get("rotate", 0)), bool(d.get("transpose", False)))

    def describe(self) -> str:
        return (f"flip_h={int(self.flip_h)} flip_v={int(self.flip_v)} "
                f"rotate={self.rotate % 360} transpose={int(self.transpose)}")


@lru_cache(maxsize=32)
def _perm(flip_h: bool, flip_v: bool, rotate: int, transpose: bool) -> np.ndarray:
    a = np.arange(N * N).reshape(N, N)
    if transpose:
        a = a.T
    a = np.rot90(a, k=rotate // 90)
    if flip_h:
        a = np.fliplr(a)
    if flip_v:
        a = np.flipud(a)
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def all_orientations() -> list[Orientation]:
    """The 8 distinct orientations (dihedral group), one representative each."""
    seen: dict[bytes, Orientation] = {}
    for t in (False, True):
        for rot in (0, 90, 180, 270):
            for fh in (False, True):
                o = Orientation(flip_h=fh, rotate=rot, transpose=t)
                seen.setdefault(o.perm.tobytes(), o)
    return list(seen.values())


# ============================================================
# ZONE MODEL
# ============================================================

@dataclass(frozen=True)
class ZoneModel:
    """
    centers[r, c] = (tx, ty) of zone centre in GEOMETRY grid order.
    corners[i, j] = (tx, ty) of grid corner i in 0..8 (rows), j in 0..8 (cols);
    zone (r, c) is the quad corners[r, c], [r, c+1], [r+1, c+1], [r+1, c].
    """
    centers: np.ndarray   # (8, 8, 2)
    corners: np.ndarray   # (9, 9, 2)
    name: str

    def center_rays(self, unit: bool = True) -> np.ndarray:
        return _rays(self.centers, unit)

    def corner_rays(self, unit: bool = True) -> np.ndarray:
        return _rays(self.corners, unit)

    def grid_to_tangent(self, gr: np.ndarray, gc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear interpolation of zone centres at fractional grid coords (clamped extrapolation)."""
        ext = _extend(self.centers)          # (10, 10, 2), index shift +1
        r = np.clip(gr + 1.0, 0.0, N + 1 - 1e-6)
        c = np.clip(gc + 1.0, 0.0, N + 1 - 1e-6)
        r0 = np.floor(r).astype(np.int32)
        c0 = np.floor(c).astype(np.int32)
        r0 = np.minimum(r0, N)
        c0 = np.minimum(c0, N)
        fr = (r - r0)[..., None]
        fc = (c - c0)[..., None]
        v = (ext[r0, c0] * (1 - fr) * (1 - fc) + ext[r0, c0 + 1] * (1 - fr) * fc
             + ext[r0 + 1, c0] * fr * (1 - fc) + ext[r0 + 1, c0 + 1] * fr * fc)
        return v[..., 0], v[..., 1]

    def tangent_to_grid(self, tx: np.ndarray, ty: np.ndarray, iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """
        Inverse of grid_to_tangent: fractional (gr, gc) for tangent coords.
        Separable initial guess + a few Newton-like correction steps (the
        model is only mildly non-separable).
        """
        col_knots = self.centers[..., 0].mean(axis=0)          # increasing with c
        row_knots = self.centers[..., 1].mean(axis=1)          # decreasing with r
        idx = np.arange(N, dtype=np.float64)
        gc = _interp_extrap(tx, col_knots, idx)
        gr = _interp_extrap(-ty, -row_knots, idx)
        dtx = np.gradient(col_knots)                            # d tx / d c (per column)
        dty = np.gradient(row_knots)                            # d ty / d r (per row, negative)
        for _ in range(iters):
            ftx, fty = self.grid_to_tangent(gr, gc)
            sc = np.interp(gc, idx, dtx)
            sr = np.interp(gr, idx, dty)
            gc = gc + (tx - ftx) / sc
            gr = gr + (ty - fty) / sr
        return gr, gc


def _rays(t: np.ndarray, unit: bool) -> np.ndarray:
    v = np.concatenate([t, np.ones(t.shape[:-1] + (1,))], axis=-1)
    if unit:
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return v


def _extend(g: np.ndarray) -> np.ndarray:
    """Pad an (n, n, k) grid by one cell on each side with linear extrapolation."""
    g = np.concatenate([2 * g[:1] - g[1:2], g, 2 * g[-1:] - g[-2:-1]], axis=0)
    g = np.concatenate([2 * g[:, :1] - g[:, 1:2], g, 2 * g[:, -1:] - g[:, -2:-1]], axis=1)
    return g


def _corners_from_centers(centers: np.ndarray) -> np.ndarray:
    ext = _extend(centers)                                       # (10, 10, 2)
    return 0.25 * (ext[:-1, :-1] + ext[1:, :-1] + ext[:-1, 1:] + ext[1:, 1:])   # (9, 9, 2)


def _interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """np.interp with linear extrapolation; xp must be increasing."""
    y = np.interp(x, xp, fp)
    lo = x < xp[0]
    hi = x > xp[-1]
    y = np.where(lo, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
    y = np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)
    return y


def st_centers_raw() -> np.ndarray:
    """ST table centres as (tx, ty) in RAW grid order, (8, 8, 2)."""
    d = st_unit_rays()
    return np.stack([d[:, 0] / d[:, 2], d[:, 1] / d[:, 2]], axis=-1).reshape(N, N, 2)


@lru_cache(maxsize=16)
def make_zone_model(kind: str = "st", fov_deg: float = 45.0, fov_scale: float = 1.0) -> ZoneModel:
    """
    Zone model in GEOMETRY grid order. For "st", geometry position (r, c)
    uses the ST direction of raw index r*8+c (the table is defined for the
    sensor's own index order; the orientation transform decides which raw
    zone is placed there).
    """
    if kind == "st":
        centers = st_centers_raw()
    elif kind == "uniform":
        half = np.radians(fov_deg) / 2.0
        a = (np.arange(N) + 0.5) / N * 2 * half - half            # per-axis angle of centres
        tx = np.tan(a)[None, :].repeat(N, 0)
        ty = -np.tan(a)[:, None].repeat(N, 1)
        centers = np.stack([tx, ty], axis=-1)
    else:
        raise ValueError(f"unknown zone model {kind!r} (use 'st' or 'uniform')")
    centers = centers * fov_scale
    if kind == "uniform":
        half = np.radians(fov_deg) / 2.0
        e = np.linspace(-half, half, N + 1)
        corners = np.stack([np.tan(e)[None, :].repeat(N + 1, 0), -np.tan(e)[:, None].repeat(N + 1, 1)], -1) * fov_scale
    else:
        corners = _corners_from_centers(centers)
    centers = np.ascontiguousarray(centers)
    corners = np.ascontiguousarray(corners)
    centers.setflags(write=False)
    corners.setflags(write=False)
    return ZoneModel(centers, corners, kind)
