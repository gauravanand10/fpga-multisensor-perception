"""
spi_link.py - stage 3: the two SPI links and the FPGA's packet receivers (rtl/spi_rx.sv)

The ESP32s are SPI masters, the FPGA is the slave on both Pmod ports (only CS, MOSI, SCLK used):
    Pmod JA  camera  8 MHz   JA1 CS_N, JA2 MOSI, JA4 SCLK, JA5 GND
    Pmod JB  ToF     4 MHz   JB1 CS_N, JB2 MOSI, JB4 SCLK, JB5 GND
One CS-low burst = one packet. The receiver checks the magic word, the exact length and the sum16
checksum; only a packet that passes everything is used (camera: written into one of two frame
buffers and the U-Net is started; ToF: the 64+64 values are latched). Bad packets are counted
and the matching error LED lights.
"""
import struct
from dataclasses import dataclass

import numpy as np

from . import fpga_model as fm

CAM_SPI_HZ = 8_000_000
TOF_SPI_HZ = 4_000_000


def transfer_time_s(n_bytes, hz):
    return n_bytes * 8 / hz


@dataclass
class RxResult:
    ok: bool
    reason: str
    frame_no: int = -1


class CameraReceiver:
    """cam_rx: 'CAM1' + frame_no + 19200 px + sum16."""
    N = 4 + 2 + fm.IMG_W * fm.IMG_H + 2

    def __init__(self):
        self.n_ok = self.n_err = 0
        self.frame = None
        self.frame_no = -1

    def receive(self, burst: bytes) -> RxResult:
        r = self._check(burst)
        if r.ok:
            self.n_ok += 1
            self.frame_no = r.frame_no
            self.frame = np.frombuffer(burst, np.uint8, fm.IMG_W * fm.IMG_H, 6).reshape(fm.IMG_H, fm.IMG_W).copy()
        else:
            self.n_err += 1
        return r

    def _check(self, b):
        if len(b) != self.N:
            return RxResult(False, f"length {len(b)} != {self.N}")
        if b[:4] != fm.CAM_MAGIC:
            return RxResult(False, "bad magic")
        px = b[6:-2]
        if fm.sum16(px) != struct.unpack_from("<H", b, len(b) - 2)[0]:
            return RxResult(False, "checksum")
        return RxResult(True, "ok", struct.unpack_from("<H", b, 4)[0])


class ToFReceiver:
    """tof_rx: 'TOF1' + frame_no + 64 x u16 + 64 x u8 + sum16."""
    N = 4 + 2 + 128 + 64 + 2

    def __init__(self):
        self.n_ok = self.n_err = 0
        self.valid = False
        self.dist = np.zeros(64, np.int64); self.status = np.zeros(64, np.int64)
        self.frame_no = -1

    def receive(self, burst: bytes) -> RxResult:
        b = burst
        if len(b) != self.N:
            r = RxResult(False, f"length {len(b)} != {self.N}")
        elif b[:4] != fm.TOF_MAGIC:
            r = RxResult(False, "bad magic")
        elif fm.sum16(b[6:-2]) != struct.unpack_from("<H", b, len(b) - 2)[0]:
            r = RxResult(False, "checksum")
        else:
            r = RxResult(True, "ok", struct.unpack_from("<H", b, 4)[0])
        if r.ok:
            self.n_ok += 1; self.valid = True; self.frame_no = r.frame_no
            self.dist = np.array(struct.unpack_from("<64H", b, 6), np.int64)
            self.status = np.frombuffer(b, np.uint8, 64, 134).astype(np.int64)
        else:
            self.n_err += 1
        return r
