# Bitstream

The design tested on the board on 14 Nov 2026 and stored in its QSPI flash.

| File | Use |
|---|---|
| `fpga_perception_system.bit` | Loads the design over JTAG (lost at power-off): `hw\scripts\program_board.tcl` |
| `fpga_perception_system.mcs`, `.prm` | QSPI flash image, 16 MB, SPIx1: stored permanently, loaded at power-up with jumper JP1 on QSPI (`hw\scripts\flash_qspi.tcl`) |

Constraints used: [`hw/constraints/nexys4_ddr.xdc`](../../constraints/nexys4_ddr.xdc).
