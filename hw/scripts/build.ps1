# build.ps1 - run the Vivado build (synthesis -> bitstream) in batch mode
#
#   cd <repository root>
#   powershell -ExecutionPolicy Bypass -File hw\scripts\build.ps1
#
# Takes roughly 20-60 minutes (timing-driven settings). Results in hw\vivado: reports\summary.txt
# first, bitstream\, checkpoints\, logs\vivado_build.log. The bitstream is written ONLY if all timing is met.
# This does NOT program the board.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repository root
$viv = Get-ChildItem -Path "C:\Xilinx" -Filter "vivado.bat" -Recurse -Depth 4 -ErrorAction SilentlyContinue |
       Where-Object { $_.DirectoryName -like "*Vivado*\bin" } |
       Sort-Object FullName -Descending | Select-Object -First 1
if (-not $viv) { throw "vivado.bat not found under C:\Xilinx" }
$out = Join-Path $root "hw\vivado"
$logd = Join-Path $out "logs"
New-Item -ItemType Directory -Force -Path $logd | Out-Null
$t0 = Get-Date
Push-Location $logd
try {
    & $viv.FullName -mode batch -source (Join-Path $root "hw\scripts\build.tcl") -log vivado_build.log -nojournal -notrace
    $code = $LASTEXITCODE
} finally { Pop-Location }
$mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Write-Host ""
Write-Host "================ build summary ($mins min) ================" -ForegroundColor Cyan
$sum = Join-Path $out "reports\summary.txt"
if (Test-Path $sum) { Get-Content $sum } else { Write-Host "no summary.txt - see hw\vivado\logs\vivado_build.log" }
if ($code -eq 2) {
    Write-Host "TIMING FAILED: no bitstream written. See hw\vivado\reports\summary.txt and timing_per_module.rpt." -ForegroundColor Red
    exit 2
} elseif ($code -ne 0) {
    throw "Vivado build failed - see hw\vivado\logs\vivado_build.log"
}
Write-Host "TIMING MET - bitstream: hw\vivado\bitstream\fpga_perception_system.bit" -ForegroundColor Green
