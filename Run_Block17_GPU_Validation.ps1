param(
    [string]$Python = "python",
    [string]$Output = "release\BLOCK17_GPU_QUALIFICATION.json"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
$Destination = if ([System.IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $Root $Output }

& $Python -m komorebi_math release qualify --require-gpu --output $Destination --force
if ($LASTEXITCODE -ne 0) { throw "CUDA release qualification failed" }
& $Python -m komorebi_math release readiness --gpu --require-gpu --output (Join-Path $Root "release\BLOCK17_GPU_READINESS.json") --force
if ($LASTEXITCODE -ne 0) { throw "CUDA beta readiness failed" }
& $Python -m komorebi_math verify gpu
if ($LASTEXITCODE -ne 0) { throw "CUDA verification suite failed" }
Write-Host "GPU qualification record: $Destination"
