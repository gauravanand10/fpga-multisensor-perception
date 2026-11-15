# Reference model

The system as a Python package, one module per hardware stage. FPGA stages use the exact integer arithmetic
of the RTL. Stage table: [`../README.md`](../README.md).

| Module | Stage |
|---|---|
| `esp32cam_node.py` | 1 – camera node: capture, 2×2 average, `CAM1` packet |
| `tof_node.py` | 2 – VL53L5CX node, `TOF1` packet, ToF scene simulator |
| `spi_link.py` | 3 – SPI links and FPGA packet receivers |
| `fpga_unet_accelerator.py` | 4 – int8 U-Net accelerator and cycle count |
| `fpga_postprocessing.py` | 5 – ToF zones on the label map, connected components, distance per object |
| `uart_link.py` | 6 – result packet, UART timing |
| `laptop_host.py` | 7 – host: stream parsing, checksum, display |
| `pipeline.py` | All stages for one frame |
| `fpga_model.py` | Bit-exact core (copy of `hw/tools/fpga_model.py`) |
| `float_reference.py` | float32 network in ONNX Runtime |
| `system_timing.py` | Latency and throughput budget |
| `paths.py` | File locations |
| `__init__.py` | Package description and stage overview |

```python
from reference_model.pipeline import EndToEndSystem
```
