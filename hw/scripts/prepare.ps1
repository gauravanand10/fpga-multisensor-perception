# prepare.ps1 - regenerate every generated file from your trained model and geometry
#
#   cd <repository root>
#   powershell -ExecutionPolicy Bypass -File hw\scripts\prepare.ps1
#
# Only needed after retraining the U-Net (new golden vectors in golden_model\data\golden_vectors) or
# after changing the ToF mounting / orientation_auto.json. The files are already in mem\.
#   1. unet_pack.py           weights/params/LUT .mem + rtl\unet_layers_pkg.sv, checks 12/12 layers
#   2. gen_zone_tables.py     ToF zone rectangles for the 80x60 grid (your geo\ code)
#   3. gen_pp_vectors.py      post-processing test cases
#   4. gen_top_vectors.py     SPI packets + expected UART packets for the top-level test
#   5. compare_pc_distance.py FPGA distance vs the original PC float method (report only)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repository root
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }   # set $env:PYTHON to use a specific interpreter
Push-Location $root
try {
    foreach ($s in @("unet_pack.py", "gen_zone_tables.py", "gen_pp_vectors.py", "gen_top_vectors.py", "gen_imu_vectors.py", "compare_pc_distance.py")) {
        Write-Host ""
        Write-Host "=== $s ===" -ForegroundColor Cyan
        & $py "hw\tools\$s"
        if ($LASTEXITCODE) { throw "$s failed" }
    }
} finally { Pop-Location }
