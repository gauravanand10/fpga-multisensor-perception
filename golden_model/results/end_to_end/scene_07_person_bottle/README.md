# Scene 7: Person, Bottle

COCO 2017 validation image #576 through the whole system (ToF frame simulated,
62/64 valid zones). float32 and int8 label maps agree on
97.0 % of pixels.

![Pipeline overview](pipeline_overview.png)

| Object | Class | Area (px, 80×60) | ToF zones | FPGA (mm) | True (mm) | Error (mm) |
|---|---|---|---|---|---|---|
| 1 | Person | 2328 | 36 | 1994 | 1995 | -1 |
| 2 | Bottle | 196 | 4 | 1368 | 1376 | -8 |
| 3 | Person | 78 | 1 | 2009 | 1995 | +14 |

| File | Content |
|---|---|
| `pipeline_overview.png` | Sensor frame, SPI frame, ToF zones, label map, host view |
| `spi_camera_packet.bin` | Exact `CAM1` packet (19208 bytes) |
| `spi_tof_packet.bin` | Exact `TOF1` packet (200 bytes) |
| `uart_result_packet.bin` | Exact result packet sent to the host (24249 bytes) |
| `scene_report.json` | All values above |
