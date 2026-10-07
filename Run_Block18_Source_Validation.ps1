$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $Root "src"
python -m compileall -q (Join-Path $Root "src")
python (Join-Path $Root "verification\block18_math_verifier.py") --output (Join-Path $Root "release\BLOCK18_MATH_VERIFICATION.json")
python -c "import pytest" 2>$null
if ($LASTEXITCODE -eq 0) {
    python -m pytest -q (Join-Path $Root "tests\test_block18_frame_algebra.py")
} else {
    Write-Host "pytest not installed; independent Block 18 verifier completed"
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
