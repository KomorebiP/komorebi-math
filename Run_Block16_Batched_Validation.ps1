param(
    [string]$Python = "python",
    [string]$Output = ".\release\BLOCK16_ISOLATED_SOURCE_VALIDATION.json",
    [int]$Timeout = 900
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

& $Python (Join-Path $Root "verification\run_isolated_test_batches.py") --timeout $Timeout --output $Output
if ($LASTEXITCODE -ne 0) { throw "Isolated Block 16 validation failed" }
