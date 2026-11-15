"""
esp32cam_node.py - stage 1: ESP32-CAM data acquisition (firmware: firmware/esp32_camera_spi)

What the firmware does, per frame:
  1. esp_camera_fb_get(): raw GRAYSCALE QVGA frame, 320 x 240 x 8 bit (no JPEG)
  2. 2x2 block average to 160 x 120:  out = (a + b + c + d + 2) >> 2
     (bit-identical to cv2.resize(..., INTER_AREA), the resize the PC system used, so the FPGA
      sees exactly the input the PC network saw)
  3. one SPI burst on Pmod JA (mode 0, MSB first, 8 MHz):
        'C' 'A' 'M' '1' | frame_no u16 | 19 200 pixels row-major | sum16(pixels) u16   (little endian)
"""
from dataclasses import dataclass

import numpy as np

from . import fpga_model as fm

SENSOR_W, SENSOR_H = 320, 240
PACKET_BYTES = 4 + 2 + fm.IMG_W * fm.IMG_H + 2          # 19 208


@dataclass
class CameraFrame:
    frame_no: int
    qvga: np.ndarray          # (240, 320) uint8, what the sensor delivered
    frame: np.ndarray         # (120, 160) uint8, what is sent to the FPGA
    packet: bytes             # the CAM1 SPI burst


class ESP32CamNode:
    def __init__(self):
        self.frame_no = 0

    @staticmethod
    def downsample(qvga):
        assert qvga.shape == (SENSOR_H, SENSOR_W) and qvga.dtype == np.uint8
        return fm.downsample_2x2(qvga)

    def capture(self, qvga) -> CameraFrame:
        """qvga: a 320x240 grey image standing in for the camera sensor output."""
        small = self.downsample(qvga)
        pkt = fm.cam_packet(small, self.frame_no)
        fr = CameraFrame(self.frame_no, qvga, small, pkt)
        self.frame_no = (self.frame_no + 1) & 0xFFFF
        return fr
