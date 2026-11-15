# build_project.ps1 - create the Vivado project and run synthesis, implementation and bitstream in it
#
#   cd <repository root>
#   powershell -ExecutionPolicy Bypass -File hw\scripts\build_project.ps1
#
# About 20-60 minutes. Result: hw\vivado\project\fpga_perception_system.xpr with completed runs, all
# reports in hw\vivado\reports\ (summary.txt first), log in hw\vivado\logs\. Does NOT program the board.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$viv  = "C:\Xilinx\Vivado\2024.1\bin\vivado.bat"
if (-not (Test-Path $viv)) { throw "vivado.bat not found at $viv" }
$log = Join-Path $root "hw\vivado\logs"
New-Item -ItemType Directory -Force -Path $log | Out-Null
$t0 = Get-Date
Push-Location $log
try {
    & $viv -mode batch -source (Join-Path $root "hw\scripts\build_project.tcl") -log vivado_project_build.log -nojournal -notrace
    $code = $LASTEXITCODE
} finally { Pop-Location }
$mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Write-Host "`n================ project build ($mins min) ================" -ForegroundColor Cyan
if ($code -ne 0) { throw "Vivado project build failed - see hw\vivado\logs\vivado_project_build.log" }
Get-Content (Join-Path $root "hw\vivado\reports\summary.txt")
Write-Host "`nOpen in the GUI:  cd hw\vivado\project; $viv fpga_perception_system.xpr" -ForegroundColor Green
