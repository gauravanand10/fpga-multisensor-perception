# Testbenches

Self-checking SystemVerilog testbenches for Vivado xsim. Each prints a final `PASS` or `FAIL` line.
Expected values come from the golden model (`golden_model/`, `hw/tools/`).

| Testbench | Device under test | Checks | Result |
|---|---|---|---|
| `tb_conv_engine.sv` | `conv_engine` | 5 golden layers, every output value | 5/5 bit-exact |
| `tb_unet_core.sv` | `unet_core` | all 12 layers and the label map against golden tensors | 12/12 + label map bit-exact |
| `tb_postproc.sv` | `postproc` | 6 scenes: object table and 80×60 object map | 6/6 bit-exact |
| `tb_servo_pwm.sv` | `servo_pwm`, `servo_ctrl` | period, pulse width at 0/45/90/135/180°, limits, slew, glitch-free updates, sweep | pass |
| `tb_imu_spi.sv` | `imu_spi` + `mpu6500_model.sv` | init sequence, chip selects, 12 values, missing IMU, hot-plug recovery | pass |
| `tb_imu_packer.sv` | `imu_fifo`, `imu_packer`, `uart_arbiter` | IMU packets byte-identical to the model, overflow flag | 4/4 packets |
| `tb_nexys4_top.sv` | whole FPGA, pin to pin | SPI in, UART out: result packets byte-identical, corrupted packet rejected, IMU packets between results | 2/2 + 88 IMU packets |

`mpu6500_model.sv` is a behavioural MPU-6500 SPI slave (register file, burst read, WHO_AM_I).

## Run

```powershell
powershell -ExecutionPolicy Bypass -File hw\scripts\run_sim.ps1 -Test all     # or conv, core, postproc, servo, imu, packer, top
```

Work folders are created in `build/xsim_<test>/`. The top-level test also writes `uart_capture.bin`, which
`sw/fpga_dashboard.py --replay` can display and `hw/tools/check_capture.py` parses.
