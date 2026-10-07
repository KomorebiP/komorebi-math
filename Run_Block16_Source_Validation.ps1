param(
    [string]$Python = "python",
    [switch]$Focused,
    [switch]$Monolithic,
    [switch]$Extended,
    [int]$BatchTimeout = 900
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

& $Python -c "import platform,sys,pytest,numpy,komorebi_math; print('python',platform.python_version(),sys.executable); print('pytest',pytest.__version__,'numpy',numpy.__version__); print('komorebi',komorebi_math.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Python, pytest, NumPy, and the source package are required" }

Write-Host "Optional automatic-differentiation backend preflight"
& $Python (Join-Path $Root "verification\optional_ad_backend_status.py")
if ($LASTEXITCODE -ne 0) { throw "Optional backend preflight could not run" }

if ($Focused) {
    & $Python -m pytest -q --tb=short `
        tests\test_block15_polyphase_refinement.py `
        tests\test_block15_persistence.py `
        tests\test_block15_stable_api.py `
        tests\test_block16_analysis_synthesis.py `
        tests\test_block16_contracts.py `
        tests\test_block16_release_qualification.py `
        tests\test_block16_cli.py `
        tests\test_block16_release_contract.py `
        tests\test_block16_gpu_qualification.py `
        tests\test_block16_companion_crosswalk.py `
        tests\test_block16_release_layout.py `
        tests\test_block16_static_audit.py `
        tests\test_block16_package_audit.py `
        tests\test_optional_backend_probe.py
} elseif ($Monolithic) {
    & $Python -m pytest -q --tb=short tests
} else {
    & $Python (Join-Path $Root "verification\run_isolated_test_batches.py") `
        --timeout $BatchTimeout `
        --output (Join-Path $Root "release\BLOCK16_ISOLATED_SOURCE_VALIDATION.json")
}
if ($LASTEXITCODE -ne 0) { throw "Block 16 source validation failed" }

& $Python -m komorebi_math api check --stable
if ($LASTEXITCODE -ne 0) { throw "Stable API validation failed" }
& $Python -m komorebi_math release contract
if ($LASTEXITCODE -ne 0) { throw "Release contract validation failed" }
& $Python -m komorebi_math verify stable
if ($LASTEXITCODE -ne 0) { throw "Stable contract validation failed" }
& $Python -m komorebi_math verify release
if ($LASTEXITCODE -ne 0) { throw "Release policy validation failed" }
& $Python -m komorebi_math release qualify --output (Join-Path $Root "release\BLOCK16_SOURCE_QUALIFICATION.json") --force
if ($LASTEXITCODE -ne 0) { throw "Release qualification failed" }

if ($Extended) {
    & $Python (Join-Path $Root "verification\block16_static_audit.py") --output (Join-Path $Root "release\BLOCK16_STATIC_AUDIT.json")
    if ($LASTEXITCODE -ne 0) { throw "Static release audit failed" }
    & $Python -m ruff check src tests
    if ($LASTEXITCODE -ne 0) { throw "Ruff validation failed" }
    & $Python -m mypy src
    if ($LASTEXITCODE -ne 0) { throw "Mypy validation failed" }
    & $Python -m mkdocs build --strict
    if ($LASTEXITCODE -ne 0) { throw "Documentation validation failed" }
}
