"""
tof_node.py - stage 2: VL53L5CX 8x8 time-of-flight sensor + its ESP32
(firmware: firmware/esp32_tof_spi)

The sensor talks I2C (400 kHz, SDA 21 / SCL 22) to its ESP32 only, ranging 8x8 zones at 10 Hz.
For every ranging frame the ESP32 sends one SPI burst on Pmod JB (mode 0, 4 MHz):
    'T' 'O' 'F' '1' | frame_no u16 | 64 x u16 distance_mm | 64 x u8 target_status | sum16 u16
Zones are in the ST driver's raw index order; the FPGA applies the zone geometry itself.

simulate_tof_frame() makes a plausible ToF frame for a scene whose true object layout is known
(COCO ground-truth masks + a chosen distance per object). It is used because the COCO photos have
no depth; it uses the same zone footprints (tof_zone_tables.npz, made from hw/tools/tof_geometry/geo) the FPGA
uses, so the zones "see" the objects where the camera sees them.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import fpga_model as fm

PACKET_BYTES = 4 + 2 + 128 + 64 + 2          # 200
VALID_STATUS = 5


@dataclass
class TofFrame:
    frame_no: int
    dist_mm: np.ndarray       # (64,) int, raw ST zone order
    status: np.ndarray        # (64,) int
    packet: bytes


class ToFNode:
    def __init__(self):
        self.frame_no = 0

    def measure(self, dist_mm, status) -> TofFrame:
        dist_mm = np.asarray(dist_mm, np.int64); status = np.asarray(status, np.int64)
        pkt = fm.tof_packet(dist_mm, status, self.frame_no)
        fr = TofFrame(self.frame_no, dist_mm, status, pkt)
        self.frame_no = (self.frame_no + 1) & 0xFFFF
        return fr


@dataclass
class SceneTruth:
    instance_map: np.ndarray  # (60, 80) int, 0 = background, i = object instance i
    instance_cls: dict        # instance -> class id
    instance_dist: dict       # instance -> true depth along the camera axis, mm
    background_mm: int        # depth of the background (a wall), mm


def scene_from_labels(gt_labels_qvga, rng, near=(700, 2600), background_mm=3400, min_px=6):
    """COCO ground truth (240x320, 255 = ignore) -> objects on the 80x60 grid, each at a random distance."""
    g = gt_labels_qvga[1::4, 1::4].astype(np.int64)            # 60 x 80
    g[g == 255] = 0
    inst = np.zeros_like(g); cls, dist = {}, {}
    n = 0
    for c in range(1, fm.N_CLS):
        m = (g == c).astype(np.uint8)
        if not m.any():
            continue
        k, comp = cv2.connectedComponents(m, connectivity=8)
        for j in range(1, k):
            sel = comp == j
            if sel.sum() < min_px:
                continue
            n += 1
            inst[sel] = n; cls[n] = c; dist[n] = int(rng.integers(*near))
    return SceneTruth(inst, cls, dist, background_mm)


def simulate_tof_frame(truth: SceneTruth, zt, rng, noise_mm=12, p_invalid=0.03, min_cover=0.25):
    """Per zone: the nearest object covering >= min_cover of the zone footprint, else the background.
    The sensor reports the distance along each zone's ray (= depth / ray_z), like the real VL53L5CX.
    The footprint depends on the distance (parallax between camera and sensor), so it is looked up at
    the background bin first and re-checked at the object's bin."""
    dist = np.zeros(64, np.int64); status = np.full(64, VALID_STATUS, np.int64)
    inst = truth.instance_map
    for i in range(64):
        d = truth.background_mm * 32768.0 / float(zt.rz_q15[i])
        for _ in range(2):
            x0, y0, x1, y1, v = zt.rects[i, fm.depth_bin(d, zt)]
            if not v:
                break
            sub = inst[y0:y1 + 1, x0:x1 + 1]
            ids, cnt = np.unique(sub[sub > 0], return_counts=True)
            cand = [truth.instance_dist[int(k)] for k, c in zip(ids, cnt) if c >= min_cover * sub.size]
            z = min(cand) if cand else truth.background_mm
            d = z * 32768.0 / float(zt.rz_q15[i])            # depth -> distance along this zone's ray
        dist[i] = max(fm.TOF_MIN_MM, int(round(d + rng.normal(0, noise_mm))))
        if rng.random() < p_invalid:
            status[i] = 255                                   # no valid target in this zone
    return dist, status
