# Scripts

Run from the repository root in Windows PowerShell (Vivado 2024.1 at `C:\Xilinx\Vivado\2024.1`).

| Script | Purpose |
|---|---|
| `run_sim.ps1` | xsim simulations: `-Test all`, `conv`, `core`, `postproc`, `servo`, `imu`, `packer`, `top` |
| `build.ps1` → `build.tcl` | Non-project build: synthesis, placement, routing, reports, bitstream (only if timing is met) |
| `build_project.ps1` → `build_project.tcl` | Same build as a Vivado project: creates `hw/vivado/project/fpga_perception_system.xpr`, runs synthesis, implementation and bitstream, writes all reports to `hw/vivado/reports/` |
| `create_vivado_project.tcl` | Vivado project with sources only (no runs) in `hw/vivado/project` |
| `open_design.tcl` | Opens `post_route.dcp` (`-tclargs route`) or `post_synth.dcp` (`-tclargs synth`) in the GUI |
| `program_board.tcl` | Loads the bitstream over JTAG (volatile) |
| `flash_qspi.tcl` | Writes the bitstream to the QSPI flash (boots at power-up) |
| `prepare.ps1` | Regenerates the memory images and test vectors after retraining |

```powershell
powershell -ExecutionPolicy Bypass -File hw\scripts\run_sim.ps1 -Test all
powershell -ExecutionPolicy Bypass -File hw\scripts\build.ps1
C:\Xilinx\Vivado\2024.1\bin\vivado.bat -nojournal -nolog -source hw\scripts\open_design.tcl -tclargs route
```
