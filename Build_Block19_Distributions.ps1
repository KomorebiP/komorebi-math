param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dist = Join-Path $Root "release\dist"
$Predecessors = Join-Path $Root "release\predecessors"
$Temporary = Join-Path ([System.IO.Path]::GetTempPath()) ("komorebi-block19-build-" + [guid]::NewGuid())
Set-Location $Root
New-Item -ItemType Directory -Force $Temporary, $Dist, $Predecessors | Out-Null

try {
@'
import json
from pathlib import Path
root = Path.cwd()
requirements = {
    "release/BLOCK19_SOURCE_QUALIFICATION.json": ("passed", True, None, None),
    "release/BLOCK19_MATH_VERIFICATION.json": ("success", True, "total", 16),
    "release/BLOCK19_ADVERSARIAL_TESTS.json": ("passed", True, "total_count", 47),
    "release/BLOCK19_WORKED_EXAMPLE.json": ("passed", True, "exact_integral", "106/5"),
    "release/BLOCK19_API_ADDITIVITY.json": ("passed", True, None, None),
}
for relative, (flag, expected_flag, count, expected_count) in requirements.items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"required source qualification record is missing: {relative}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get(flag) != expected_flag:
        raise SystemExit(f"source qualification record did not pass: {relative}")
    if count is not None and payload.get(count) != expected_count:
        raise SystemExit(f"source qualification count mismatch in {relative}")
source_qualification = json.loads(
    (root / "release/BLOCK19_SOURCE_QUALIFICATION.json").read_text(encoding="utf-8")
)
documentation = source_qualification.get("suites", {}).get("documentation", {})
if not (
    documentation.get("passed") is True
    and documentation.get("checks") == 1
    and documentation.get("expected") == 1
):
    raise SystemExit("source documentation verification is not exactly 1/1")
'@ | & $Python -
if ($LASTEXITCODE -ne 0) { throw "Block 19 source qualification gate failed" }

@'
from hashlib import sha256
from pathlib import Path
root = Path.cwd()
dist = root / "release" / "dist"
predecessors = root / "release" / "predecessors"
def digest(path):
    return sha256(path.read_bytes()).hexdigest()
for source in sorted(dist.glob("komorebi_math-*")):
    if source.name.startswith("komorebi_math-1.0.0b3"):
        continue
    target = predecessors / source.name
    if target.exists():
        if digest(source) != digest(target):
            raise SystemExit(f"refusing to overwrite a different predecessor: {target}")
        source.unlink()
    else:
        source.replace(target)
'@ | & $Python -
if ($LASTEXITCODE -ne 0) { throw "Could not preserve predecessor distributions" }

foreach ($Path in @((Join-Path $Root "build"), (Join-Path $Root "src\komorebi_math.egg-info"))) {
    if (Test-Path $Path) { Remove-Item -Recurse -Force $Path }
}

@'
from pathlib import Path
import sys
from setuptools.build_meta import build_sdist, build_wheel
destination = Path(sys.argv[1]).resolve()
print(build_wheel(str(destination)))
print(build_sdist(str(destination)))
expected = {"komorebi_math-1.0.0b3-py3-none-any.whl", "komorebi_math-1.0.0b3.tar.gz"}
observed = {path.name for path in destination.iterdir() if path.is_file()}
if observed != expected:
    raise SystemExit(f"unexpected Block 19 build outputs: {sorted(observed)}")
'@ | & $Python - $Temporary
if ($LASTEXITCODE -ne 0) { throw "Block 19 distribution build failed" }

Get-ChildItem -Path $Dist -Force | Remove-Item -Recurse -Force
Copy-Item (Join-Path $Temporary "komorebi_math-1.0.0b3-py3-none-any.whl") $Dist
Copy-Item (Join-Path $Temporary "komorebi_math-1.0.0b3.tar.gz") $Dist

@'
import json
from hashlib import sha256
from pathlib import Path
root = Path.cwd()
dist = root / "release" / "dist"
expected = ("komorebi_math-1.0.0b3-py3-none-any.whl", "komorebi_math-1.0.0b3.tar.gz")
observed = tuple(sorted(path.name for path in dist.iterdir()))
if observed != tuple(sorted(expected)):
    raise SystemExit(f"release/dist is not Block-19-only: {observed}")
def record(path):
    return {"name": path.name, "size": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()}
payload = {
    "schema": "komorebi-math.block19-distribution-build",
    "schema_version": 1,
    "package_version": "1.0.0b3",
    "passed": True,
    "artifacts": [record(dist / name) for name in expected],
    "dist_members": list(observed),
}
(root / "release" / "BLOCK19_DISTRIBUTION_BUILD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
'@ | & $Python -
if ($LASTEXITCODE -ne 0) { throw "Block 19 distribution inventory failed" }
}
finally {
    if (Test-Path $Temporary) { Remove-Item -Recurse -Force $Temporary }
}
