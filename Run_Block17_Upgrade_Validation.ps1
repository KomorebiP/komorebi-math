param(
    [string]$Python = "python",
    [string]$Output = "release\BLOCK17_UPGRADE_VALIDATION.json"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Destination = if ([System.IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $Root $Output }

& $Python (Join-Path $Root "verification\run_block17_upgrade_validation.py") `
    --python $Python `
    --output $Destination
if ($LASTEXITCODE -ne 0) { throw "Block 17 predecessor upgrade validation failed" }
Write-Host "Upgrade validation record: $Destination"
