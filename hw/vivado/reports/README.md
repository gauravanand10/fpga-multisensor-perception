# Reports

Vivado reports of the routed design (unless marked synthesis). Start with `summary.txt`: timing met
(WNS +0.128 ns, WHS +0.013 ns), 27 232 LUT, 22 036 FF, 122.5 BRAM tiles, 44 DSP, 41 I/O, 0.225 W total
on-chip power, and the worst path of every module.

| File | Content |
|---|---|
| `summary.txt` | Key numbers, power, worst setup slack per module |
| `timing_summary.rpt` | Full timing summary: setup, hold, pulse width, unconstrained paths |
| `timing_per_module.rpt` | 10 worst paths ending in each module |
| `timing_worst_setup_paths.rpt`, `timing_worst_hold_paths.rpt` | 20 worst setup and hold paths |
| `clocks.rpt`, `clock_utilization.rpt` | Clocks and clock resources |
| `utilization.rpt`, `utilization_hierarchical.rpt` | Area: total and per module |
| `synth_utilization.rpt`, `synth_utilization_hierarchical.rpt`, `synth_timing_summary.rpt` | Same after synthesis |
| `power.rpt` | Power estimate by resource type and hierarchy |
| `io.rpt` | Pin assignment and I/O standards |
| `control_sets.rpt` | Clock-enable and reset control sets |
| `design_analysis.rpt` | Logic levels, congestion and complexity |
| `drc.rpt`, `methodology.rpt`, `route_status.rpt` | Design-rule, methodology and routing checks |

Graphs drawn from these reports: [`../../results`](../../results) (`python docs/make_plots.py`).
Screenshots: [`../../screenshots`](../../screenshots).
