# build_project.tcl - Vivado PROJECT build: creates hw/vivado/project/fpga_perception_system.xpr,
# runs synthesis, implementation and bitstream inside the project with the same timing-driven
# settings as build.tcl, and writes every report to hw/vivado/reports/ (same files as build.tcl).
#
#   vivado -mode batch -source hw/scripts/build_project.tcl      (hw\scripts\build_project.ps1 does this)
#
# Afterwards open the .xpr in the GUI: all runs show as complete, "Open Implemented Design" works
# and every view (device, schematic, timing, utilization, power) is available for screenshots.
# The project's own bitstream stays in hw/vivado/project/fpga_perception_system.runs/impl_1/.
# The board-tested bitstream and flash image in hw/vivado/bitstream/ are not touched.

set root [file normalize [file dirname [file dirname [file dirname [info script]]]]]
set jobs [expr {[llength $argv] ? [lindex $argv 0] : 4}]

# 1. project with sources, ROM images, constraints
source $root/hw/scripts/create_vivado_project.tcl

# 2. same strategy as the non-project build (build.tcl)
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE        PerformanceOptimized [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING         true                 [get_runs synth_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE          Explore              [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE        ExtraTimingOpt       [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED         true                 [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE     AggressiveExplore    [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE        AggressiveExplore    [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true              [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

# 3. synthesis
puts "=== synthesis (synth_1) ==="
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synthesis failed - see hw/vivado/project/fpga_perception_system.runs/synth_1/runme.log" }

set rep $root/hw/vivado/reports
file mkdir $rep
open_run synth_1 -name synth_1
report_utilization                       -file $rep/synth_utilization.rpt
report_utilization -hierarchical         -file $rep/synth_utilization_hierarchical.rpt
report_timing_summary -max_paths 20      -file $rep/synth_timing_summary.rpt
close_design

# 4. implementation + bitstream
puts "=== implementation + bitstream (impl_1) ==="
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "implementation failed - see hw/vivado/project/fpga_perception_system.runs/impl_1/runme.log" }

# 5. reports of the routed design
open_run impl_1
report_timing_summary -max_paths 30 -report_unconstrained -file $rep/timing_summary.rpt
report_timing -max_paths 20 -nworst 1 -setup -file $rep/timing_worst_setup_paths.rpt
report_timing -max_paths 20 -nworst 1 -hold  -file $rep/timing_worst_hold_paths.rpt
report_clocks                            -file $rep/clocks.rpt
report_clock_utilization                 -file $rep/clock_utilization.rpt
report_utilization                       -file $rep/utilization.rpt
report_utilization -hierarchical         -file $rep/utilization_hierarchical.rpt
report_power                             -file $rep/power.rpt
report_io                                -file $rep/io.rpt
report_control_sets -verbose             -file $rep/control_sets.rpt
report_design_analysis -congestion -complexity -timing -file $rep/design_analysis.rpt
report_drc                               -file $rep/drc.rpt
report_methodology                       -file $rep/methodology.rpt
report_route_status                      -file $rep/route_status.rpt

# worst path per module (same as build.tcl)
set mods [list "U-Net engine" u_sys/u_core/u_eng "U-Net memories" u_sys/u_core/g_bank* "weight ROM" u_sys/u_core/u_wrom \
    "U-Net sequencer" u_sys/u_core "post-processing" u_sys/u_pp "camera SPI rx" u_sys/u_cam "ToF SPI rx" u_sys/u_tof \
    "UART packer" u_sys/u_pk "UART tx" u_sys/u_tx "IMU FIFO" u_sys/u_ififo "IMU packer" u_sys/u_ipk \
    "UART arbiter" u_sys/u_arb "IMU SPI" u_imu "servo PWM" u_servo "servo control" u_sctl "system control" u_sys]
set fr [open $rep/timing_per_module.rpt w]
set lines {}
foreach {label pat} $mods {
    set cells [get_cells -quiet -hierarchical -filter "NAME =~ $pat/* && IS_PRIMITIVE"]
    if {[llength $cells] == 0} { continue }
    set paths [get_timing_paths -quiet -setup -max_paths 10 -nworst 1 -to $cells]
    if {[llength $paths] == 0} { continue }
    set p0 [lindex $paths 0]
    lappend lines [format "  %-18s worst slack %8.3f ns   %s -> %s" $label [get_property SLACK $p0] \
        [get_property STARTPOINT_PIN $p0] [get_property ENDPOINT_PIN $p0]]
    puts $fr "==================== $label ($pat) ===================="
    puts $fr [report_timing -of_objects $paths -return_string]
}
close $fr

# summary (same format as build.tcl)
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
set nfail [llength [get_timing_paths -quiet -setup -max_paths 100000 -slack_lesser_than 0]]
set fh [open $rep/summary.txt w]
puts $fh "Nexys4 DDR U-Net + IMU + servo build, Vivado project  ([clock format [clock seconds]])"
puts $fh "part xc7a100tcsg324-1, clock 100 MHz (10.000 ns)"
puts $fh [format "setup WNS = %.3f ns   hold WHS = %.3f ns   failing setup endpoints = %d" $wns $whs $nfail]
puts $fh [expr {$wns >= 0 && $whs >= 0 ? "TIMING MET" : "TIMING FAILED"}]
set u [report_utilization -return_string]
foreach pat {"Slice LUTs" "Slice Registers" "Block RAM Tile" "DSPs" "Bonded IOB"} {
    foreach line [split $u "\n"] { if {[string match "| $pat*" $line]} { puts $fh $line; break } }
}
foreach line [split [report_power -return_string] "\n"] {
    if {[string match "*Total On-Chip Power*" $line] || [string match "*Dynamic (W)*" $line] || [string match "*Device Static*" $line]} { puts $fh $line }
}
puts $fh "worst path per module:"
foreach l $lines { puts $fh $l }
puts $fh "bitstream (project): [lindex [glob -nocomplain $root/hw/vivado/project/fpga_perception_system.runs/impl_1/*.bit] 0]"
close $fh
set fh [open $rep/summary.txt r]; puts [read $fh]; close $fh
