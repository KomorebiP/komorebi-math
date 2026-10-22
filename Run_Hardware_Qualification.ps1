param(
    [string]$Python = "python",
    [string]$Candidate = "qualification\candidate.json",
    [string]$Output = "qualification\hardware.json",
    [int]$Timeout = 660
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$CandidatePath = if ([System.IO.Path]::IsPathRooted($Candidate)) { $Candidate } else { Join-Path $Root $Candidate }
$OutputPath = if ([System.IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $Root $Output }

& $Python -I -B (Join-Path $Root "verification\hardware_qualification.py") `
    --root $Root --candidate $CandidatePath --output $OutputPath --timeout $Timeout
if ($LASTEXITCODE -ne 0) {
    throw "Hardware qualification failed. Inspect the JSON report and its command diagnostics: $OutputPath"
}
Write-Host "Installed-wheel CUDA qualification passed: $OutputPath"
