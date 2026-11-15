# Host Software

Python programs for the PC connected to the board's USB-UART (2 Mbaud).

```powershell
pip install numpy opencv-python pyserial pillow matplotlib
python sw/fpga_dashboard.py --port COM3                 # live
python sw/fpga_dashboard.py --port COM3 --theme light   # for projectors
python sw/fpga_dashboard.py --replay golden_model/data/rtl_simulation/uart_capture.bin   # no board needed
```

| Program | Purpose |
|---|---|
| `fpga_dashboard.py` | Presentation dashboard: camera with detections, depth map, FPGA timing, servo state, both IMUs with live plots, link statistics. Renders at the screen's native resolution. |
| `engineering_viewer.py` | Engineering viewer with raw IMU values and CSV logging |
| `result_viewer.py` | Minimal camera / depth-map viewer for result packets only |
| `make_paper_figures.py` | Figures and a LaTeX table from a recorded capture |

## Dashboard keys

| Key | Action |
|---|---|
| F | full screen |
| L | object labels |
| Z | ToF zones on the camera image |
| O | object boxes |
| B | remove gyroscope offsets (hold the IMUs still) |
| R | start / stop raw capture to `sw/captures/` |
| C | start / stop IMU CSV log to `sw/logs/` |
| S | snapshot to `sw/snapshots/` |
| Q | quit |

## Stream format

Result packet (one per frame): `A5 5A 46 50`, header (version, flags, frame numbers, cycle counts, object count),
12 bytes per object, 64 ToF distances and statuses, 80×60 object map, optional 160×120 camera image, sum16.
IMU packet (between result packets): `A5 5A 49 4D`, flags, WHO_AM_I, servo state, n samples of
16-bit sample number + 12 × int16, sum16. Parsers: `hw/tools/fpga_model.py`.
