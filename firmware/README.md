# Firmware

Arduino sketches for the two sensor nodes. Both are SPI masters that send one checksummed packet per frame
to the FPGA.

| Sketch | Board | Sensor | Output |
|---|---|---|---|
| `esp32_camera_spi/esp32_camera_spi.ino` | AI-Thinker ESP32-CAM | GC2145 | `CAM1` packet, 160×120 grey, 14 fps, SPI 8 MHz |
| `esp32_tof_spi/esp32_tof_spi.ino` | ESP32 DevKit | VL53L5CX (I2C) | `TOF1` packet, 8×8 distances + status, 10 Hz, SPI 4 MHz |

## Build settings (Arduino IDE)

| | ESP32-CAM | ToF node |
|---|---|---|
| Board | AI Thinker ESP32-CAM | ESP32 Dev Module |
| Options | PSRAM enabled, Flash mode DIO, 40 MHz | default |
| Library | ESP32 camera driver (core) | SparkFun VL53L5CX |
| Upload | GPIO0 to GND, reset, upload; remove GPIO0–GND, reset | normal |

## Wiring

| Node | Signal | ESP32 pin | Nexys 4 DDR |
|---|---|---|---|
| Camera | CS / MOSI / SCLK / GND | GPIO15 / GPIO13 / GPIO14 / GND | JA1 / JA2 / JA4 / JA5 |
| ToF | CS / MOSI / SCLK / GND | GPIO15 / GPIO13 / GPIO14 / GND | JB1 / JB2 / JB4 / JB5 |
| ToF sensor | SDA / SCL / VIN / GND | GPIO21 / GPIO22 / 3V3 / GND | – |

MISO is not used. GPIO12 must stay unconnected (boot strapping pin). Both boards are powered over USB; do not
connect the Pmod 3.3 V supply to them.

The ToF sketch reports readings below 100 mm as invalid (cover-glass and wiring crosstalk).
