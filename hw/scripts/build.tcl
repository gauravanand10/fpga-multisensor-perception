# build.tcl - Vivado non-project build of the U-Net + IMU + servo design for the Nexys4 DDR
#   synthesis -> place -> route -> reports -> bitstream   (nothing is programmed)
#
#   vivado -mode batch -source hw/scripts/build.tcl    (hw\scripts\build.ps1 does this)
#
# Output in hw/vivado:
#   reports/summary.txt                   key numbers, power, worst path of every module
#   reports/timing_summary.rpt            full timing report (must show WNS >= 0)
#   reports/timing_per_module.rpt                 10 worst paths ending in each module
#   reports/utilization.rpt / utilization_hierarchical.rpt  LUT / FF / BRAM / DSP use
#   reports/power.rpt, drc.rpt, methodology.rpt; checkpoints/post_*.dcp
#   bitstream/fpga_perception_system.bit  ONLY written when all timing is met
#
# Old bitstreams are deleted first, so a .bit in hw/vivado/bitstream always belongs to a
# build that met timing.

set root  [file normalize [file dirname [file dirname [file dirname [info script]]]]]
set out   $root/hw/vivado
set rep   $out/reports
set bitd  $out/bitstream
set ckd   $out/checkpoints
set part  xc7a100tcsg324-1
file mkdir $out $rep $bitd $ckd
foreach f [glob -nocomplain $bitd/*.bit] { file delete -force $f }
foreach f [list $rep/summary.txt $rep/TIMING_FAILED.txt] { file delete -force $f }

set_part $part
set rtl [list unet_pkg.sv unet_layers_pkg.sv bram.sv unet_engine.sv unet_core.sv postproc.sv \
              spi_rx.sv uart_tx.sv imu_spi.sv imu_packer.sv servo_pwm.sv unet_system.sv nexys4_top.sv]
foreach f $rtl { read_verilog -sv $root/rtl/$f }
# $readmemh uses plain file names: copy the init files into a work folder and run
# synthesis there (Vivado writes .Xil / temp files into the current folder, so it must
# never be mem\ itself)
set work $out/meminit
file mkdir $work
foreach f [list unet_weights.mem unet_params.mem unet_inlut.mem zone_rects.mem zone_rz.mem zone_binlo.mem] {
    if {![file exists $root/hw/mem/$f]} { error "missing $root/hw/mem/$f - run hw/scripts/prepare.ps1" }
    file copy -force $root/hw/mem/$f $work/$f
    read_mem $work/$f
}
read_xdc $root/hw/constraints/nexys4_ddr.xdc
cd $work

puts "=== synth_design ==="
synth_design -top nexys4_top -part $part -directive PerformanceOptimized -retiming
write_checkpoint -force $ckd/post_synth.dcp
report_utilization -file $rep/synth_utilization.rpt
report_timing_summary -max_paths 20 -file $rep/synth_timing_summary.rpt

puts "=== opt / place / route (timing-driven) ==="
opt_design      -directive Explore
place_design    -directive ExtraTimingOpt
phys_opt_design -directive AggressiveExplore
route_design    -directive AggressiveExplore
# a second physical optimisation after routing fixes small remaining violations
if {[get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]] < 0} {
    phys_opt_design -directive AggressiveExplore
}
write_checkpoint -force $ckd/post_route.dcp

report_timing_summary -max_paths 30 -file $rep/timing_summary.rpt
report_utilization                 -file $rep/utilization.rpt
report_utilization -hierarchical   -file $rep/utilization_hierarchical.rpt
report_power                       -file $rep/power.rpt
report_drc                         -file $rep/drc.rpt
report_methodology                 -file $rep/methodology.rpt

# ---------------- worst path per module ----------------
set mods [list \
    "U-Net engine"      u_sys/u_core/u_eng \
    "U-Net memories"    u_sys/u_core/g_bank* \
    "weight ROM"        u_sys/u_core/u_wrom \
    "U-Net sequencer"   u_sys/u_core \
    "post-processing"   u_sys/u_pp \
    "camera SPI rx"     u_sys/u_cam \
    "ToF SPI rx"        u_sys/u_tof \
    "UART packer"       u_sys/u_pk \
    "UART tx"           u_sys/u_tx \
    "IMU FIFO"          u_sys/u_ififo \
    "IMU packer"        u_sys/u_ipk \
    "UART arbiter"      u_sys/u_arb \
    "IMU SPI"           u_imu \
    "servo PWM"         u_servo \
    "servo control"     u_sctl \
    "system control"    u_sys ]
set fr [open $rep/timing_per_module.rpt w]
set lines {}
foreach {label pat} $mods {
    set cells [get_cells -quiet -hierarchical -filter "NAME =~ $pat/* && IS_PRIMITIVE"]
    if {[llength $cells] == 0} { continue }
    set paths [get_timing_paths -quiet -setup -max_paths 10 -nworst 1 -to $cells]
    if {[llength $paths] == 0} { continue }
    set p0 [lindex $paths 0]
    set s  [get_property SLACK $p0]
    lappend lines [format "  %-18s worst slack %8.3f ns   %s -> %s" $label $s \
        [get_property STARTPOINT_PIN $p0] [get_property ENDPOINT_PIN $p0]]
    puts $fr "==================== $label ($pat) ===================="
    puts $fr [report_timing -of_objects $paths -return_string]
}
close $fr

# ---------------- summary + bitstream ----------------
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
set nfail [llength [get_timing_paths -quiet -setup -max_paths 100000 -slack_lesser_than 0]]
set ok [expr {$wns >= 0 && $whs >= 0}]

set fh [open $rep/summary.txt w]
puts $fh "Nexys4 DDR U-Net + IMU + servo build  ([clock format [clock seconds]])"
puts $fh "part $part, clock 100 MHz (10.000 ns)"
puts $fh [format "setup WNS = %.3f ns   hold WHS = %.3f ns   failing setup endpoints = %d" $wns $whs $nfail]
puts $fh [expr {$ok ? "TIMING MET" : "TIMING FAILED - no bitstream written"}]
foreach {name pat} {LUT "Slice LUTs" FF "Slice Registers" BRAM "Block RAM Tile" DSP "DSPs" IO "Bonded IOB"} {
    set rpt [report_utilization -return_string]
    foreach line [split $rpt "\n"] {
        if {[string match "| $pat*" $line]} { puts $fh $line; break }
    }
}
foreach line [split [report_power -return_string] "\n"] {
    if {[string match "*Total On-Chip Power*" $line] || [string match "*Dynamic (W)*" $line] || [string match "*Device Static*" $line]} { puts $fh $line }
}
puts $fh "worst path per module:"
foreach l $lines { puts $fh $l }
close $fh

if {$ok} {
    puts "=== write_bitstream ==="
    write_bitstream -force $bitd/fpga_perception_system.bit
    set fh [open $rep/summary.txt a]
    puts $fh "bitstream: hw/vivado/bitstream/fpga_perception_system.bit"
    close $fh
} else {
    set fh [open $rep/TIMING_FAILED.txt w]
    puts $fh "Timing not met (WNS $wns ns, WHS $whs ns). No bitstream was written."
    puts $fh "See summary.txt and timing_per_module.rpt."
    close $fh
}

set fh [open $rep/summary.txt r]
puts [read $fh]
close $fh
if {!$ok} { exit 2 }
