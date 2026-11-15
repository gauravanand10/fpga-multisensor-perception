"""
system_timing.py - latency and throughput budget of the whole chain (per camera frame)

Numbers that come from the RTL simulation are marked; the rest follow from link rates. Not
measured on hardware.
"""
from dataclasses import dataclass

from . import esp32cam_node, tof_node, spi_link, uart_link, fpga_unet_accelerator as acc

POST_CYCLES_RANGE = (218_099, 518_000)      # RTL simulation, tb_postproc / tb_nexys4_top
TOF_RATE_HZ = 10


@dataclass
class Budget:
    cam_spi_ms: float
    tof_spi_ms: float
    unet_ms: float
    post_ms_min: float
    post_ms_max: float
    uart_ms_with_image: float
    uart_ms_without_image: float

    def rows(self):
        return [
            ("ESP32-CAM -> FPGA, SPI 8 MHz, 19 208 B", self.cam_spi_ms, "link rate"),
            ("ToF ESP32 -> FPGA, SPI 4 MHz, 200 B (every 100 ms)", self.tof_spi_ms, "link rate"),
            ("U-Net accelerator, 1 327 877 cycles @ 100 MHz", self.unet_ms, "RTL simulation"),
            ("Objects + distance, 0.22-0.52 M cycles", self.post_ms_max, "RTL simulation (max)"),
            ("FPGA -> laptop, UART 2 Mbaud, with image (~24.2 KB)", self.uart_ms_with_image, "link rate"),
            ("FPGA -> laptop, UART 2 Mbaud, no image (~5.0 KB)", self.uart_ms_without_image, "link rate"),
        ]

    def fps_limit(self, with_image):
        u = self.uart_ms_with_image if with_image else self.uart_ms_without_image
        return 1000.0 / max(self.cam_spi_ms, self.unet_ms + self.post_ms_max, u)


def budget(n_obj=3):
    return Budget(
        cam_spi_ms=1e3 * spi_link.transfer_time_s(esp32cam_node.PACKET_BYTES, spi_link.CAM_SPI_HZ),
        tof_spi_ms=1e3 * spi_link.transfer_time_s(tof_node.PACKET_BYTES, spi_link.TOF_SPI_HZ),
        unet_ms=1e3 * acc.UNetAccelerator.latency_s(),
        post_ms_min=1e3 * POST_CYCLES_RANGE[0] / acc.CLOCK_HZ,
        post_ms_max=1e3 * POST_CYCLES_RANGE[1] / acc.CLOCK_HZ,
        uart_ms_with_image=1e3 * uart_link.transfer_time_s(uart_link.packet_bytes(n_obj, True)),
        uart_ms_without_image=1e3 * uart_link.transfer_time_s(uart_link.packet_bytes(n_obj, False)),
    )
