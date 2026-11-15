# run_all.ps1 - regenerate every result of the Python reference model
#
#   cd <repository root>\golden_model
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
#
# About 5 minutes. Apart from docs\figures\ and the graphs drawn by docs\make_plots.py (model\results\,
# hw\results\), nothing outside this folder is changed; the COCO set, the golden vectors and the
# RTL simulation output are only read.
# "Continue": in Windows PowerShell 5.1, "Stop" turns any stderr text of python.exe into a fatal
# error; failures are detected through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$here = $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }   # set $env:PYTHON to use a specific interpreter
$has = & $py -c "import importlib.util; print(importlib.util.find_spec('matplotlib') is not None)"
if ("$has".Trim() -ne "True") {
    Write-Host "installing matplotlib into the venv (needed for the figures)" -ForegroundColor Yellow
    & $py -m pip install matplotlib
    if ($LASTEXITCODE -ne 0) { throw "pip install matplotlib failed" }
}
Push-Location $here
try {
    $steps = @(
        @("1/6 int8 model vs golden vectors",         "verification\verify_int8_vs_golden_vectors.py"),
        @("2/6 model vs RTL simulation (xsim bytes)", "verification\verify_rtl_simulation_output.py"),
        @("3/6 end-to-end system demo",               "experiments\run_end_to_end_demo.py"),
        @("4/6 float vs int8 on 1000 COCO images",    "experiments\evaluate_float_vs_int8.py"),
        @("5/6 architecture figures",                 "..\docs\make_figures.py"),
        @("6/6 result graphs from stored data",       "..\docs\make_plots.py"))
    foreach ($s in $steps) {
        Write-Host "`n=== $($s[0]) ===" -ForegroundColor Cyan
        & $py $s[1]
        if ($LASTEXITCODE -ne 0) { throw "$($s[1]) failed" }
    }
} finally { Pop-Location }
Write-Host "`nDone. Results in results\, figures in docs\figures\, graphs in model\results\, hw\results\ and results\end_to_end\." -ForegroundColor Green
