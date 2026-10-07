param(
    [string]$Python = "python",
    [string]$OutputDirectory = "",
    [switch]$SkipMkDocs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $Root "release\handoff"
}
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

$GateArguments = @(
    (Join-Path $Root "verification\block21_publication_gate.py"),
    "--root", $Root,
    "--full-tests",
    "--installed-artifacts",
    "--output", (Join-Path $Root "release\REPAIR_R2_PUBLICATION_GATE.json"),
    "--markdown", (Join-Path $Root "release\REPAIR_R2_PUBLICATION_GATE.md")
)
if ($SkipMkDocs) { $GateArguments += "--skip-mkdocs" } else { $GateArguments += "--require-mkdocs" }
$Dist = Join-Path $Root "release\dist"
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
Get-ChildItem -LiteralPath $Dist -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "komorebi_math-1.0.0b4*" } |
    Remove-Item -Force
& $Python (Join-Path $Root "verification\build_block21_distributions.py") --root $Root --output-directory $Dist --report (Join-Path $Root "release\REPAIR_R2_DISTRIBUTION_BUILD.json")
if ($LASTEXITCODE -ne 0) { throw "Distribution build failed" }

& $Python @GateArguments
if ($LASTEXITCODE -ne 0) { throw "Block 21 publication gate failed" }

& $Python (Join-Path $Root "verification\build_block21_handoff.py") `
    --root $Root `
    --output-directory $OutputDirectory `
    --artifact-directory $Dist `
    --gate (Join-Path $Root "release\REPAIR_R2_PUBLICATION_GATE.json") `
    --python $Python
if ($LASTEXITCODE -ne 0) { throw "Block 21 handoff build failed" }

Write-Host "Block 21 publication handoff passed and was written to $OutputDirectory"
