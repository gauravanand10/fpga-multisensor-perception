# open_design.tcl - open the synthesized or implemented design in the Vivado GUI (for schematics,
# device view, timing and utilization screenshots). Nothing is rebuilt.
#
#   vivado -source hw/scripts/open_design.tcl -tclargs route      implemented (placed + routed) design
#   vivado -source hw/scripts/open_design.tcl -tclargs synth      synthesized design
#
# Useful GUI views afterwards: Window > Device, Schematic (F4), Reports > Timing Summary / Utilization.
set root  [file normalize [file dirname [file dirname [file dirname [info script]]]]]
set which [expr {[llength $argv] ? [lindex $argv 0] : "route"}]
set dcp   [expr {$which eq "synth" ? "$root/hw/vivado/checkpoints/post_synth.dcp" : "$root/hw/vivado/checkpoints/post_route.dcp"}]
if {![file exists $dcp]} { error "checkpoint not found: $dcp" }
open_checkpoint $dcp
start_gui
puts "opened $dcp"
