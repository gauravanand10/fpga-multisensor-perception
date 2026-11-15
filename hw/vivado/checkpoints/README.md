# Design checkpoints

| File | Content |
|---|---|
| `post_synth.dcp` | Synthesized netlist |
| `post_route.dcp` | Placed and routed design (the one the bitstream was written from) |

Open either in the Vivado GUI without rebuilding:

```powershell
C:\Xilinx\Vivado\2024.1\bin\vivado.bat -nojournal -nolog -source hw\scripts\open_design.tcl -tclargs route
C:\Xilinx\Vivado\2024.1\bin\vivado.bat -nojournal -nolog -source hw\scripts\open_design.tcl -tclargs synth
```
