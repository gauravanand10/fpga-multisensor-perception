# Tools

Python generators for everything the RTL and testbenches read, and a bit-exact model of the FPGA.

| Script | Purpose |
|---|---|
| `fpga_model.py` | Bit-exact Python model of the FPGA (U-Net, post-processing, packets); source of all expected values |
| `unet_pack.py` | int8 network → weight ROM, parameter ROM, input LUT, layer table `rtl/unet_layers_pkg.sv` |
| `gen_zone_tables.py` | ToF zone footprints on the 80×60 grid → `mem/zone_*.mem` |
| `gen_pp_vectors.py` | Post-processing test cases → `mem/tb_pp*.mem` |
| `gen_top_vectors.py` | Whole-FPGA stimulus and expected UART bytes → `mem/tb_cam*`, `tb_tof*`, `tb_uart*` |
| `gen_imu_vectors.py` | IMU FIFO / packer test → `mem/tb_imu_*.mem` |
| `check_capture.py` | Parses a captured UART stream with both host parsers |
| `compare_pc_distance.py` | FPGA integer distance vs the PC float method |
| [`tof_geometry/`](tof_geometry) | Camera / ToF geometry used for the zone tables |

All are run in order by [`../scripts/prepare.ps1`](../scripts/prepare.ps1).
