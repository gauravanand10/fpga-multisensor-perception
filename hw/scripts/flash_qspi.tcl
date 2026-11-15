# flash_qspi.tcl - write fpga_perception_system.bit PERMANENTLY into the Nexys4 DDR QSPI flash
#   (Spansion/Infineon S25FL128S, 16 MB). Run by the user, not by any other script.
#
#   Board: jumper JP1 (MODE) can stay on JTAG while flashing; USB cable on PROG/UART.
#   cd <repository root>
#   C:\Xilinx\Vivado\2024.1\bin\vivado.bat -mode batch -source hw\scripts\flash_qspi.tcl
#   Then: power off, set JP1 to QSPI, power on -> the design loads by itself.
#
# 1) hw\vivado\bitstream\fpga_perception_system.mcs is made from the .bit (write_cfgmem, SPIx1 = matches the
#    bitstream SPI_BUSWIDTH 1; for SPIx4 set BITSTREAM.CONFIG.SPI_BUSWIDTH 4 in the XDC and rebuild)
# 2) the flash is erased, programmed and verified through JTAG
set root [file normalize [file dirname [file dirname [file dirname [info script]]]]]
set bit  $root/hw/vivado/bitstream/fpga_perception_system.bit
set mcs  $root/hw/vivado/bitstream/fpga_perception_system.mcs
if {![file exists $bit]} { error "bitstream not found: $bit (run hw/scripts/build.ps1 first)" }

puts "=== write_cfgmem: $mcs ==="
write_cfgmem -force -format mcs -size 16 -interface SPIx1 -loadbit "up 0x00000000 $bit" -file $mcs

puts "=== open hardware ==="
open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices xc7a100t_0] 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev

set parts [get_cfgmem_parts {s25fl128sxxxxxx0-spi-x1_x2_x4}]
if {[llength $parts] == 0} {
    puts "available flash parts containing 128:"
    puts [get_cfgmem_parts *128*]
    error "flash part s25fl128sxxxxxx0-spi-x1_x2_x4 not found - send this output"
}
create_hw_cfgmem -hw_device $dev [lindex $parts 0]
set cfg [get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.ADDRESS_RANGE  {use_file} $cfg
set_property PROGRAM.FILES          [list $mcs] $cfg
set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} $cfg
set_property PROGRAM.BLANK_CHECK 0 $cfg
set_property PROGRAM.ERASE       1 $cfg
set_property PROGRAM.CFG_PROGRAM 1 $cfg
set_property PROGRAM.VERIFY      1 $cfg
set_property PROGRAM.CHECKSUM    0 $cfg

puts "=== loading the flash helper into the FPGA ==="
create_hw_bitstream -hw_device $dev [get_property PROGRAM.HW_CFGMEM_BITFILE $dev]
program_hw_devices $dev
refresh_hw_device $dev

puts "=== erase + program + verify flash (about 1-3 min) ==="
program_hw_cfgmem -hw_cfgmem $cfg
puts "FLASH PROGRAMMED AND VERIFIED: $mcs"
puts "Now: power off, set JP1 to QSPI, power on."
close_hw_target
disconnect_hw_server
close_hw_manager
