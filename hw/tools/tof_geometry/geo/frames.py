"""
frames.py - data containers passed between acquisition threads and consumers.

Timestamps:
    t_recv  : host time.perf_counter() when the last byte of the frame arrived
              (monotonic, seconds). All sync logic uses this clock.
    sensor_ts : the device's own timestamp if it sends one (ESP32-CAM sends
              X-Timestamp = seconds since boot). Not comparable to t_recv
              directly, but useful for measuring on-device frame intervals.

Frames are treated as immutable once published: consumers must not modify
`image` in place (copy first if you draw on it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraFrame:
    seq: int                      # host-side counter, +1 per decoded frame
    t_recv: float                 # perf_counter() when JPEG fully received
    t_decoded: float              # perf_counter() after cv2.imdecode
    image: np.ndarray             # BGR uint8, HxWx3
    jpeg_bytes: int               # compressed size
    sensor_ts: float | None = None  # X-Timestamp from ESP32-CAM, seconds
    jpeg: bytes | None = None       # original compressed bytes (for recording)
    scene_ts: float | None = None   # simulator only: wall-clock time the scene was rendered

    @property
    def decode_ms(self) -> float:
        return (self.t_decoded - self.t_recv) * 1000.0

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width)"""
        return self.image.shape[0], self.image.shape[1]


@dataclass(frozen=True, slots=True)
class ToFFrame:
    seq: int                      # host-side counter, +1 per NEW sensor frame
    t_recv: float                 # perf_counter() when response fully received
    t_request: float              # perf_counter() when the request was sent
    sensor_frame_no: int          # firmware "sequence" (uint16, wraps at 65536)
    dist_mm: np.ndarray           # float32 [8, 8], RAW firmware order
    status: np.ndarray            # uint8   [8, 8], VL53L5CX target_status, RAW order
    valid: np.ndarray             # bool    [8, 8], RAW order
    confidence: np.ndarray        # float32 [8, 8] 0..1, RAW order (0 where invalid)
    raw_body: str = ""            # exact JSON text received (for recording/replay)
    scene_ts: float | None = None # simulator only (X-Scene-Time header)

    @property
    def rtt_ms(self) -> float:
        return (self.t_recv - self.t_request) * 1000.0

    @property
    def t_mid(self) -> float:
        """Midpoint of request/response: best host-side estimate of when the ESP32 read its buffer."""
        return 0.5 * (self.t_request + self.t_recv)

    @property
    def valid_count(self) -> int:
        return int(self.valid.sum())
