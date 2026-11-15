# RTL simulation output

`uart_capture.bin` holds the UART bytes produced by the whole-FPGA testbench `tb/tb_nexys4_top.sv` in Vivado
xsim (`hw\scripts\run_sim.ps1 -Test top`): two result packets and the IMU packets
between them (the corrupted SPI packet in the stimulus is rejected and produces no output). `verification/verify_rtl_simulation_output.py` compares it byte by byte with the golden
model (0 differing bytes). It can also be replayed in the dashboard:
`python sw/fpga_dashboard.py --replay golden_model/data/rtl_simulation/uart_capture.bin`.
