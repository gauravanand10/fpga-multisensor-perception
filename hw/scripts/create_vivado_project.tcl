# create_vivado_project.tcl - a Vivado GUI project for the U-Net + IMU + servo design (Nexys4 DDR)
# in hw/vivado/project.
# Board: Nexys4 DDR -> part xc7a100tcsg324-1.  Language: SystemVerilog.
#
# Vivado GUI: Tools > Run Tcl Script...  or in the Tcl console:
#   cd <repository root>
#   source hw/scripts/create_vivado_project.tcl
# Then Flow Navigator: Run Synthesis / Run Implementation / Generate Bitstream.
# (Simulations: use hw\scripts\run_sim.ps1 - same xsim, set up for the golden-vector paths.)

set root [file normalize [file dirname [file dirname [file dirname [info script]]]]]

create_project fpga_perception_system $root/hw/vivado/project -part xc7a100tcsg324-1 -force
if {[llength [get_board_parts -quiet *nexys4_ddr*]]} {
    set_property board_part [lindex [get_board_parts *nexys4_ddr*] end] [current_project]
}
set rtl [list unet_pkg.sv unet_layers_pkg.sv bram.sv unet_engine.sv unet_core.sv postproc.sv \
              spi_rx.sv uart_tx.sv imu_spi.sv imu_packer.sv servo_pwm.sv unet_system.sv nexys4_top.sv]
foreach f $rtl { add_files -norecurse $root/rtl/$f }
foreach f [list unet_weights.mem unet_params.mem unet_inlut.mem zone_rects.mem zone_rz.mem zone_binlo.mem] {
    add_files -norecurse $root/hw/mem/$f
}
set_property file_type SystemVerilog [get_files *.sv]
set_property file_type {Memory Initialization Files} [get_files *.mem]
add_files -fileset constrs_1 -norecurse $root/hw/constraints/nexys4_ddr.xdc
set_property top nexys4_top [current_fileset]
update_compile_order -fileset sources_1
puts "Project created: $root/hw/vivado/project/fpga_perception_system.xpr  (top nexys4_top)"
