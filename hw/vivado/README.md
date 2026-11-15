# Vivado results

Everything Vivado 2024.1 produced for the xc7a100tcsg324-1 at 100 MHz, one copy of each file.

| Folder | Content |
|---|---|
| [`bitstream/`](bitstream) | Board-tested bitstream and QSPI flash image (`.bit`, `.mcs`, `.prm`) |
| [`checkpoints/`](checkpoints) | Synthesized and routed design checkpoints (`.dcp`) |
| [`reports/`](reports) | Timing, area, power, I/O, DRC and methodology reports; `summary.txt` first |
| [`logs/`](logs) | Build logs |
| [`project/`](project) | Vivado project `fpga_perception_system.xpr` with completed runs |

| Resource | Used | Available |
|---|---|---|
| LUT | 27 232 (43 %) | 63 400 |
| FF | 22 036 (17 %) | 126 800 |
| BRAM tile | 122.5 (91 %) | 135 |
| DSP | 44 (18 %) | 240 |
| I/O | 41 (20 %) | 210 |

Timing met: WNS +0.128 ns, WHS +0.013 ns, 0 failing endpoints. Total on-chip power 0.225 W
(0.121 W dynamic, 0.103 W static). Graphs: [`../results`](../results); screenshots: [`../screenshots`](../screenshots).

Both builds write here: `hw\scripts\build.ps1` (batch: reports, checkpoints, bitstream) and
`hw\scripts\build_project.ps1` (project: `project/`, same reports). They give identical results.
