# Vivado project

`fpga_perception_system.xpr`: the design as a Vivado 2024.1 project with completed synthesis and
implementation runs (`fpga_perception_system.runs/synth_1`, `impl_1`). Open it in Vivado and use
**Open Implemented Design** for placement, timing, utilization and power views.

```powershell
cd hw\vivado\project
C:\Xilinx\Vivado\2024.1\bin\vivado.bat fpga_perception_system.xpr
```

Start Vivado from this folder (as above): its session logs and the `.mem` copies it makes when the
elaborated design is opened then stay here and are ignored by git.

Recreated from scratch by `hw\scripts\build_project.ps1`. Reports: [`../reports/`](../reports);
board-tested bitstream and `.mcs`: [`../bitstream/`](../bitstream).
