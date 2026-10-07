param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

& $Python (Join-Path $Root "verification\build_block16_distributions.py") --output (Join-Path $Root "release\BLOCK16_DISTRIBUTION_BUILD.json")
if ($LASTEXITCODE -ne 0) { throw "Distribution build failed" }

& $Python (Join-Path $Root "verification\block16_distribution_audit.py") --output (Join-Path $Root "release\BLOCK16_DISTRIBUTION_VALIDATION.json")
if ($LASTEXITCODE -ne 0) { throw "Distribution audit failed" }
