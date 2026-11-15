# ESP32-CAM camera node

`esp32_camera_spi.ino` captures 320×240 grey frames, averages 2×2 pixels to 160×120 and sends each frame to
the FPGA as one `CAM1` packet (magic, frame number, 19 200 pixels, sum16; 19 208 bytes) over SPI at 8 MHz.

| | |
|---|---|
| Board | AI-Thinker ESP32-CAM (Arduino board "AI Thinker ESP32-CAM", PSRAM enabled) |
| Frame rate | 14 frames/s |
| SPI to FPGA | CS GPIO15 → JA1, MOSI GPIO13 → JA2, SCLK GPIO14 → JA4, GND → JA5 |
| Python model | [`golden_model/reference_model/esp32cam_node.py`](../../golden_model/reference_model/esp32cam_node.py) |

Build settings and upload steps: [`../README.md`](../README.md).
