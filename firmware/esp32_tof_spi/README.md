# ToF sensor node

`esp32_tof_spi.ino` reads the VL53L5CX over I2C (8×8 zones, 10 Hz continuous ranging) and sends each frame to
the FPGA as one `TOF1` packet (magic, frame number, 64 distances in mm, 64 status bytes, sum16; 200 bytes)
over SPI at 4 MHz. Readings below 100 mm are marked invalid.

| | |
|---|---|
| Board | ESP32 DevKit, SparkFun VL53L5CX library |
| Sensor wiring | SDA GPIO21, SCL GPIO22, VIN 3V3, GND |
| SPI to FPGA | CS GPIO15 → JB1, MOSI GPIO13 → JB2, SCLK GPIO14 → JB4, GND → JB5 |
| Python model | [`golden_model/reference_model/tof_node.py`](../../golden_model/reference_model/tof_node.py) |

Build settings: [`../README.md`](../README.md).
