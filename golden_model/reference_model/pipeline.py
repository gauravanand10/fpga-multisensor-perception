"""
pipeline.py - the whole system for one frame, stage by stage:

  ESP32-CAM (capture, 2x2 average, CAM1)  ->  SPI JA  ->  FPGA cam_rx
  ToF ESP32 (VL53L5CX 8x8, TOF1)          ->  SPI JB  ->  FPGA tof_rx
  FPGA: int8 U-Net -> label map -> objects + ToF distance -> UART packet
  laptop: parse packet -> Camera | Depth Map
"""
from dataclasses import dataclass

import numpy as np

from . import esp32cam_node, tof_node, spi_link, uart_link, laptop_host
from .fpga_unet_accelerator import UNetAccelerator, CYCLES_PER_FRAME
from .fpga_postprocessing import PostProcessor


@dataclass
class FrameTrace:
    cam: esp32cam_node.CameraFrame
    tof: tof_node.TofFrame
    cam_rx: spi_link.RxResult
    tof_rx: spi_link.RxResult
    labels: np.ndarray
    post: object
    uart_bytes: bytes
    host_packet: dict


class EndToEndSystem:
    def __init__(self):
        self.camera = esp32cam_node.ESP32CamNode()
        self.tof = tof_node.ToFNode()
        self.cam_rx = spi_link.CameraReceiver()
        self.tof_rx = spi_link.ToFReceiver()
        self.unet = UNetAccelerator()
        self.post = PostProcessor()

    def process(self, qvga, tof_dist, tof_status, send_image=True) -> FrameTrace:
        # ESP32 nodes
        tf = self.tof.measure(tof_dist, tof_status)
        cf = self.camera.capture(qvga)
        # SPI links -> FPGA receivers (the ToF frame arrives independently; the newest one is used)
        r_t = self.tof_rx.receive(tf.packet)
        r_c = self.cam_rx.receive(cf.packet)
        if not r_c.ok:
            raise RuntimeError(f"camera packet rejected: {r_c.reason}")
        # FPGA processing
        labels, _ = self.unet.run(self.cam_rx.frame)
        post = self.post.run(labels, self.tof_rx.dist, self.tof_rx.status)
        uart = uart_link.build_packet(self.cam_rx.frame_no, self.tof_rx.frame_no, self.tof_rx.valid,
                                      CYCLES_PER_FRAME, 0, post, self.tof_rx.dist, self.tof_rx.status,
                                      self.cam_rx.frame if send_image else None)
        # laptop
        pkts = laptop_host.find_packets(uart)
        assert len(pkts) == 1
        return FrameTrace(cf, tf, r_c, r_t, labels, post, uart, pkts[0][1])
