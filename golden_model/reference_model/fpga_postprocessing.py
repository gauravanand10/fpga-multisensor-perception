"""
fpga_postprocessing.py - stage 5: objects and their distances on the FPGA (rtl/postproc.sv)

  1. ToF zones on the label grid: each valid zone's footprint (a rectangle on the 80x60 grid that
     depends on the measured distance, 32 bins, because the sensor sits beside the camera) is painted
     with a z-buffer (nearer zone wins).
  2. Objects: 8-connected regions of one class; regions of >= 24 pixels (0.5 % of the frame) get an
     id 1..63 in raster order.
  3. Distance per object: zones covering >= 30 % of their footprint with the object (else the best
     zone if > 5 %), then the coverage-weighted 35th percentile of their camera-axis depths.
All integer, identical to the RTL. Runs in 0.22-0.52 M cycles (2.2-5.2 ms) in the RTL simulation.
"""
from dataclasses import dataclass

import numpy as np

from . import fpga_model as fm
from . import paths


@dataclass
class PostResult:
    objects: list            # fm.Obj (k, cls, area, x0, y0, x1, y1, dist mm (0 = none), nsel)
    object_map: np.ndarray   # (60, 80) object id per pixel
    overflow: bool           # more than 63 objects
    zone_map: np.ndarray     # (60, 80) ToF zone painted on each pixel (127 = none)


class PostProcessor:
    def __init__(self, zone_tables=paths.ZONE_TABLES):
        self.zt = fm.load_zone_tables(paths.need(zone_tables))

    def run(self, labels, dist, status) -> PostResult:
        objs, kmap, flags, aux = fm.postprocess(np.asarray(labels, np.int64), dist, status, self.zt)
        return PostResult(objs, kmap.reshape(fm.LAB_H, fm.LAB_W), bool(flags["overflow"]),
                          aux["zid"].reshape(fm.LAB_H, fm.LAB_W))
