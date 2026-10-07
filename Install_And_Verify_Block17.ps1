param(
    [string]$Python = "python",
    [string]$Environment = ".venv-block17",
    [switch]$Recreate,
    [switch]$OnlineDependencies
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentPath = Join-Path $Root $Environment
$Wheel = Get-ChildItem -Path (Join-Path $Root "release\dist") -Filter "komorebi_math-1.0.0b1-*.whl" -File | Select-Object -First 1
if (-not $Wheel) { throw "Block 17 wheel not found under release\dist" }

if ($Recreate -and (Test-Path $EnvironmentPath)) { Remove-Item -Recurse -Force $EnvironmentPath }
if (-not (Test-Path $EnvironmentPath)) {
    if ($OnlineDependencies) {
        & $Python -m venv $EnvironmentPath
    } else {
        & $Python -m venv --system-site-packages $EnvironmentPath
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Block 17 environment" }
}

$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
if (-not (Test-Path $EnvironmentPython)) { $EnvironmentPython = Join-Path $EnvironmentPath "bin\python" }
if ($OnlineDependencies) {
    & $EnvironmentPython -m pip install --upgrade pip "numpy>=1.26" 
    if ($LASTEXITCODE -ne 0) { throw "Base dependency installation failed" }
}
& $EnvironmentPython -m pip install --no-deps --force-reinstall $Wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "Wheel installation failed" }

$SavedPath = $env:PYTHONPATH
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Push-Location $env:TEMP
try {
    $Check = @'
from pathlib import Path
import komorebi_math
from komorebi_math.stable import check_stable_contract
assert komorebi_math.__version__ == "1.0.0b1"
location = Path(komorebi_math.__file__).resolve()
assert any(part in {"site-packages", "dist-packages"} for part in location.parts)
check_stable_contract().certificate.require()
print(f"verified {komorebi_math.__version__} from {location}")
'@
    $Check | & $EnvironmentPython -
    if ($LASTEXITCODE -ne 0) { throw "Installed import and contract check failed" }
    & $EnvironmentPython -m komorebi_math api check --stable
    if ($LASTEXITCODE -ne 0) { throw "Installed stable API check failed" }
    & $EnvironmentPython -m komorebi_math release contract
    if ($LASTEXITCODE -ne 0) { throw "Installed release contract check failed" }
    & $EnvironmentPython -m komorebi_math verify stable
    if ($LASTEXITCODE -ne 0) { throw "Installed stable verification failed" }
    & $EnvironmentPython -m komorebi_math verify release
    if ($LASTEXITCODE -ne 0) { throw "Installed release verification failed" }
    & $EnvironmentPython -m komorebi_math release upgrade
    if ($LASTEXITCODE -ne 0) { throw "Installed upgrade verification failed" }
    & $EnvironmentPython -m komorebi_math release evidence
    if ($LASTEXITCODE -ne 0) { throw "Installed evidence verification failed" }
    & $EnvironmentPython -m komorebi_math release documentation
    if ($LASTEXITCODE -ne 0) { throw "Installed documentation verification failed" }
    & $EnvironmentPython -m komorebi_math release readiness --output (Join-Path $Root "release\BLOCK17_INSTALLED_BETA_READINESS.json") --force
    if ($LASTEXITCODE -ne 0) { throw "Installed beta readiness failed" }
    & $EnvironmentPython -m komorebi_math verify block17
    if ($LASTEXITCODE -ne 0) { throw "Installed cumulative Block 17 verification failed" }
    & $EnvironmentPython -m komorebi_math release qualify --output (Join-Path $Root "release\BLOCK17_INSTALLED_QUALIFICATION.json") --force
    if ($LASTEXITCODE -ne 0) { throw "Installed release qualification failed" }
    $PipCheck = & $EnvironmentPython -m pip check 2>&1
    $PipCheckExit = $LASTEXITCODE
    $PipCheck | ForEach-Object { Write-Host $_ }
    if ($PipCheckExit -ne 0) {
        $KomorebiFailure = $PipCheck | Where-Object { $_ -match '^komorebi[-_]math ' }
        if ($KomorebiFailure) { throw "Installed Komorebi Math dependency consistency check failed" }
        Write-Warning "pip check reported only unrelated inherited system-site-package issues; Komorebi Math requirements are consistent"
    }
}
finally {
    Pop-Location
    if ($null -ne $SavedPath) { $env:PYTHONPATH = $SavedPath }
}
