param(
    [string]$Python = "python",
    [switch]$SkipMkDocs,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

$Arguments = @(
    (Join-Path $Root "verification\block21_publication_gate.py"),
    "--root", $Root,
    "--output", (Join-Path $Root "release\BLOCK21_PUBLICATION_GATE.json"),
    "--markdown", (Join-Path $Root "release\BLOCK21_PUBLICATION_GATE.md")
)
if (-not $Quick) { $Arguments += "--full-tests" }
if ($SkipMkDocs) { $Arguments += "--skip-mkdocs" } else { $Arguments += "--require-mkdocs" }

& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Block 21 publication gate failed" }

& $Python (Join-Path $Root "examples\block21_public_capability_tour.py") --output (Join-Path $Root "release\BLOCK21_PUBLIC_CAPABILITY_TOUR.json")
if ($LASTEXITCODE -ne 0) { throw "Block 21 capability tour failed" }

Write-Host "Block 21 publication gate passed"
