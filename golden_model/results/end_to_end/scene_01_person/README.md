# Scene 1: Person

COCO 2017 validation image #716 through the whole system (ToF frame simulated,
64/64 valid zones). float32 and int8 label maps agree on
99.9 % of pixels.

![Pipeline overview](pipeline_overview.png)

| Object | Class | Area (px, 80×60) | ToF zones | FPGA (mm) | True (mm) | Error (mm) |
|---|---|---|---|---|---|---|
| 1 | Person | 141 | 2 | 2311 | 2318 | -7 |

| File | Content |
|---|---|
| `pipeline_overview.png` | Sensor frame, SPI frame, ToF zones, label map, host view |
| `spi_camera_packet.bin` | Exact `CAM1` packet (19208 bytes) |
| `spi_tof_packet.bin` | Exact `TOF1` packet (200 bytes) |
| `uart_result_packet.bin` | Exact result packet sent to the host (24225 bytes) |
| `scene_report.json` | All values above |
