# run_sim.ps1 - Vivado xsim simulations of the U-Net + IMU + servo design (Nexys4 DDR, SystemVerilog)
#
#   cd <repository root>
#   powershell -ExecutionPolicy Bypass -File hw\scripts\run_sim.ps1 -Test all
#
#   -Test conv      Stage-1 single-lane conv engine, 5 golden layers          (~2 min)
#   -Test core      full U-Net, 12 layers + label map vs golden vectors        (~5-15 min)
#   -Test postproc  connected components + ToF zones + distances, 6 cases       (~2-5 min)
#   -Test servo     servo PWM: period, pulse widths, clamp, slew, glitch, sweep (real timing)
#   -Test imu       IMU SPI master vs an MPU-6500 SPI slave model (init, 12 values,
#                   IMU absent, IMU appearing later)
#   -Test packer    IMU FIFO + IMU packet builder + UART arbiter vs the Python model
#   -Test top       whole FPGA pin-to-pin: SPI in -> UART out (result + IMU packets),
#                   then the legacy result-packet reader and the full stream parser over the capture
#   -Test new       servo, imu, packer
#   -Test all       everything
#
# Each test runs in build\xsim_<test>. Nothing is programmed onto the board.
param([ValidateSet("conv", "core", "postproc", "servo", "imu", "packer", "top", "new", "all")][string]$Test = "new")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repository root
$py   = if ($env:PYTHON) { $env:PYTHON } else { "python" }   # set $env:PYTHON to use a specific interpreter

$xvlog = Get-ChildItem -Path "C:\Xilinx" -Filter "xvlog.bat" -Recurse -Depth 4 -ErrorAction SilentlyContinue |
         Where-Object { $_.DirectoryName -like "*Vivado*\bin" } |
         Sort-Object FullName -Descending | Select-Object -First 1
if (-not $xvlog) { throw "xvlog.bat not found under C:\Xilinx" }
$bin = $xvlog.DirectoryName
Write-Host "Vivado bin : $bin"

$pk   = @("rtl\unet_pkg.sv", "rtl\unet_layers_pkg.sv", "rtl\bram.sv")
$sets = @{
    "conv"     = @{ top = "tb_conv_engine"; files = @("rtl\unet_pkg.sv", "rtl\conv_engine.sv", "tb\tb_conv_engine.sv") }
    "core"     = @{ top = "tb_unet_core";   files = $pk + @("rtl\unet_engine.sv", "rtl\unet_core.sv", "tb\tb_unet_core.sv") }
    "postproc" = @{ top = "tb_postproc";    files = $pk + @("rtl\postproc.sv", "tb\tb_postproc.sv") }
    "servo"    = @{ top = "tb_servo_pwm";   files = @("rtl\servo_pwm.sv", "tb\tb_servo_pwm.sv") }
    "imu"      = @{ top = "tb_imu_spi";     files = @("rtl\imu_spi.sv", "tb\mpu6500_model.sv", "tb\tb_imu_spi.sv") }
    "packer"   = @{ top = "tb_imu_packer";  files = @("rtl\uart_tx.sv", "rtl\imu_packer.sv", "tb\tb_imu_packer.sv") }
    "top"      = @{ top = "tb_nexys4_top";  files = $pk + @("rtl\unet_engine.sv", "rtl\unet_core.sv", "rtl\postproc.sv",
                                                             "rtl\spi_rx.sv", "rtl\uart_tx.sv", "rtl\imu_spi.sv",
                                                             "rtl\imu_packer.sv", "rtl\servo_pwm.sv", "rtl\unet_system.sv",
                                                             "rtl\nexys4_top.sv", "tb\mpu6500_model.sv", "tb\tb_nexys4_top.sv") }
}
# uart_tx.sv holds uart_packer, which imports unet_pkg
$sets["packer"].files = @("rtl\unet_pkg.sv") + $sets["packer"].files

$tests = switch ($Test) {
    "all"   { @("conv", "core", "postproc", "servo", "imu", "packer", "top") }
    "new"   { @("servo", "imu", "packer") }
    default { @($Test) }
}

$summary = @()
foreach ($t in $tests) {
    $s = $sets[$t]
    $work = Join-Path $root "build\xsim_$t"
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    Push-Location $work
    try {
        Write-Host ""
        Write-Host "=== $t : $($s.top) ===" -ForegroundColor Cyan
        $src = $s.files | ForEach-Object { Join-Path $root $_ }
        & "$bin\xvlog.bat" -sv @src
        if ($LASTEXITCODE) { throw "xvlog failed ($t)" }
        & "$bin\xelab.bat" $s.top -s "snap_$t" -timescale 1ns/1ps
        if ($LASTEXITCODE) { throw "xelab failed ($t)" }
        $t0 = Get-Date
        & "$bin\xsim.bat" "snap_$t" -R -log "xsim_$t.log"
        if ($LASTEXITCODE) { throw "xsim failed ($t)" }
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        $res = Select-String -Path "xsim_$t.log" -Pattern "^(PASS|FAIL|TIMEOUT|ERROR)" | Select-Object -Last 1
        $line = if ($res) { $res.Line } else { "no result line - see build\xsim_$t\xsim_$t.log" }
        $summary += "{0,-9} {1}   ({2} min)" -f $t, $line, $mins
    } finally { Pop-Location }

    if ($t -eq "top") {
        # legacy result-packet reader + full stream parser over the capture
        Write-Host ""
        Write-Host "=== top : parsers over build\xsim_top\uart_capture.bin ===" -ForegroundColor Cyan
        $out = & $py (Join-Path $root "hw\tools\check_capture.py") (Join-Path $root "build\xsim_top\uart_capture.bin")
        $code = $LASTEXITCODE
        $out | ForEach-Object { Write-Host $_ }
        $last = ($out | Select-Object -Last 1)
        $summary += "{0,-9} {1}" -f "parsers", $last
        if ($code -ne 0) { Write-Host "parser check failed (exit $code)" -ForegroundColor Red }
    }
}
Write-Host ""
Write-Host "================ summary ================" -ForegroundColor Cyan
$summary | ForEach-Object { Write-Host $_ }
if ($tests -contains "top") {
    Write-Host ""
    Write-Host "UART stream of the top test: build\xsim_top\uart_capture.bin"
    Write-Host "  (replay with sw\engineering_viewer.py --replay build\xsim_top\uart_capture.bin)"
}
