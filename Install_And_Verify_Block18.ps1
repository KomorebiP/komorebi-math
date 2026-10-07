param(
    [string]$Python = "python",
    [string]$Environment = ".venv-block18",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentPath = Join-Path $Root $Environment
$Wheel = Get-ChildItem -Path (Join-Path $Root "release\dist") -Filter "komorebi_math-1.0.0b2-*.whl" -File | Select-Object -First 1
if (-not $Wheel) { throw "Block 18 wheel not found under release\dist" }
if ($Recreate -and (Test-Path $EnvironmentPath)) { Remove-Item -Recurse -Force $EnvironmentPath }
if (-not (Test-Path $EnvironmentPath)) {
    & $Python -m venv --system-site-packages $EnvironmentPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Block 18 environment" }
}
$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
if (-not (Test-Path $EnvironmentPython)) { $EnvironmentPython = Join-Path $EnvironmentPath "bin\python" }
& $EnvironmentPython -m pip install --no-deps --force-reinstall $Wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "Wheel installation failed" }
$SavedPath = $env:PYTHONPATH
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Push-Location $env:TEMP
try {
    & $EnvironmentPython -c "import komorebi_math; assert komorebi_math.__version__ == '1.0.0b2'; print(komorebi_math.__version__, komorebi_math.__file__)"
    if ($LASTEXITCODE -ne 0) { throw "Installed import failed" }
    & $EnvironmentPython -m komorebi_math verify frame-algebra
    if ($LASTEXITCODE -ne 0) { throw "Installed frame-algebra verification failed" }
    & $EnvironmentPython -m komorebi_math verify block18
    if ($LASTEXITCODE -ne 0) { throw "Installed Block 18 verification failed" }
    & $EnvironmentPython -m komorebi_math verify stable
    if ($LASTEXITCODE -ne 0) { throw "Installed stable verification failed" }
}
finally {
    Pop-Location
    if ($null -ne $SavedPath) { $env:PYTHONPATH = $SavedPath }
}
