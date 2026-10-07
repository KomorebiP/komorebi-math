param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
& $Python -m pip wheel . --no-deps --no-build-isolation --wheel-dir (Join-Path $Root "release\dist")
if ($LASTEXITCODE -ne 0) { throw "Block 18 wheel build failed" }
@'
from setuptools.build_meta import build_sdist
print(build_sdist("release/dist"))
'@ | & $Python -
if ($LASTEXITCODE -ne 0) { throw "Block 18 source distribution build failed" }
