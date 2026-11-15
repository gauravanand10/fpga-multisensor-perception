# program_board.tcl - ONLY for later, when the Nexys4 DDR is connected and you decide to program it.
# Not run by any other script. Loads hw/vivado/bitstream/fpga_perception_system.bit into the FPGA (volatile:
# it is gone after power-off).
#
#   vivado -mode batch -source hw/scripts/program_board.tcl
set root [file normalize [file dirname [file dirname [file dirname [info script]]]]]
set bit  $root/hw/vivado/bitstream/fpga_perception_system.bit
if {![file exists $bit]} { error "bitstream not found: $bit (run hw/scripts/build.ps1 first)" }
open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices xc7a100t_0] 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
puts "programmed $bit"
close_hw_target
disconnect_hw_server
close_hw_manager
