param(
    [string]$Python = "python",
    [string]$Environment = ".venv-block19",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentPath = Join-Path $Root $Environment
$Dist = Join-Path $Root "release\dist"
$Wheels = @(Get-ChildItem -Path $Dist -Filter "komorebi_math-1.0.0b3-*.whl" -File)
if ($Wheels.Count -ne 1) { throw "Expected exactly one Block 19 wheel under release\dist" }
$Wheel = $Wheels[0]
$ExpectedMembers = @("komorebi_math-1.0.0b3-py3-none-any.whl", "komorebi_math-1.0.0b3.tar.gz")
$ObservedMembers = @(Get-ChildItem -Path $Dist -File | ForEach-Object { $_.Name } | Sort-Object)
if ((Compare-Object ($ExpectedMembers | Sort-Object) $ObservedMembers).Count -ne 0) {
    throw "release\dist must contain only the Block 19 wheel and sdist"
}
if ($Recreate -and (Test-Path $EnvironmentPath)) { Remove-Item -Recurse -Force $EnvironmentPath }
if (-not (Test-Path $EnvironmentPath)) {
    & $Python -m venv --system-site-packages $EnvironmentPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Block 19 environment" }
}
$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
if (-not (Test-Path $EnvironmentPython)) { $EnvironmentPython = Join-Path $EnvironmentPath "bin\python" }
& $EnvironmentPython -m pip install --no-index --no-deps --force-reinstall $Wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "Wheel installation failed" }

$SavedPath = $env:PYTHONPATH
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Push-Location $env:TEMP
try {
    & $EnvironmentPython -c "from pathlib import Path; import komorebi_math; assert komorebi_math.__version__ == '1.0.0b3'; assert any(p in {'site-packages','dist-packages'} for p in Path(komorebi_math.__file__).resolve().parts); print(komorebi_math.__version__, komorebi_math.__file__)"
    if ($LASTEXITCODE -ne 0) { throw "Installed import failed" }
    & $EnvironmentPython -m komorebi_math verify integration
    if ($LASTEXITCODE -ne 0) { throw "Installed integration verification failed" }
    & $EnvironmentPython -m komorebi_math verify block19
    if ($LASTEXITCODE -ne 0) { throw "Installed Block 19 verification failed" }
    & $EnvironmentPython -m komorebi_math verify stable
    if ($LASTEXITCODE -ne 0) { throw "Installed stable verification failed" }
    & $EnvironmentPython -m komorebi_math verify block15
    if ($LASTEXITCODE -ne 0) { throw "Installed cumulative Block 15 verification failed" }
    & $EnvironmentPython -m komorebi_math verify documentation
    if ($LASTEXITCODE -ne 0) { throw "Installed stable documentation verification failed" }
@'
from komorebi_math.verify import (
    run_block15_verification,
    run_block19_verification,
    run_documentation_verification,
    run_integration_verification,
    run_stable_verification,
)
expected = (
    ("integration", run_integration_verification(), 12),
    ("block19", run_block19_verification(), 24),
    ("stable", run_stable_verification(), 6),
    ("block15", run_block15_verification(), 174),
    ("documentation", run_documentation_verification(), 1),
)
for name, report, count in expected:
    if not report.passed or len(report.checks) != count:
        raise SystemExit(f"{name} installed verification count changed")
print("installed CLI counts: integration=12 block19=24 stable=6 block15=174 documentation=1")
'@ | & $EnvironmentPython -
    if ($LASTEXITCODE -ne 0) { throw "Installed Block 19 verification counts changed" }
    & $EnvironmentPython (Join-Path $Root "verification\block19_math_verifier.py") --quiet --output (Join-Path $Root "release\BLOCK19_INSTALLED_MATH_VERIFICATION.json")
    if ($LASTEXITCODE -ne 0) { throw "Installed independent Block 19 verification failed" }
    & $EnvironmentPython (Join-Path $Root "verification\block19_adversarial_runner.py") --output (Join-Path $Root "release\BLOCK19_INSTALLED_ADVERSARIAL_TESTS.json")
    if ($LASTEXITCODE -ne 0) { throw "Installed adversarial Block 19 verification failed" }
    & $EnvironmentPython (Join-Path $Root "examples\block19_measure_integration_worked_example.py") --json | Set-Content -Encoding utf8 (Join-Path $Root "release\BLOCK19_INSTALLED_WORKED_EXAMPLE.json")
    if ($LASTEXITCODE -ne 0) { throw "Installed worked example failed" }
@'
import json
import sys
from pathlib import Path
math_report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
adversarial = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
worked = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8-sig"))
if not math_report.get("success") or math_report.get("total") != 16:
    raise SystemExit("installed independent verification is not 16/16")
if not adversarial.get("passed") or adversarial.get("total_count") != 47:
    raise SystemExit("installed adversarial verification is not 47/47")
if not worked.get("passed") or worked.get("exact_integral") != "106/5":
    raise SystemExit("installed worked example did not return 106/5")
'@ | & $EnvironmentPython - `
    (Join-Path $Root "release\BLOCK19_INSTALLED_MATH_VERIFICATION.json") `
    (Join-Path $Root "release\BLOCK19_INSTALLED_ADVERSARIAL_TESTS.json") `
    (Join-Path $Root "release\BLOCK19_INSTALLED_WORKED_EXAMPLE.json")
    if ($LASTEXITCODE -ne 0) { throw "Installed Block 19 record counts changed" }
}
finally {
    Pop-Location
    if ($null -ne $SavedPath) { $env:PYTHONPATH = $SavedPath }
}
