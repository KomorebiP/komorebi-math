param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dist = Join-Path $Root "release\dist"
$Predecessors = Join-Path $Root "release\predecessors"
$Temporary = Join-Path ([System.IO.Path]::GetTempPath()) ("komorebi-block20-build-" + [guid]::NewGuid())
$Primary = Join-Path $Temporary "primary"
Set-Location $Root
New-Item -ItemType Directory -Force $Temporary, $Primary, $Dist, $Predecessors | Out-Null

try {
& $Python (Join-Path $Root "verification\block20_source_audit.py") --root $Root --quiet --output (Join-Path $Temporary "fresh-source-audit.json")
if ($LASTEXITCODE -ne 0) { throw "Fresh Block 20 source audit failed" }
@'
import json
import sys
from pathlib import Path
root = Path.cwd()
requirements = {
    "release/BLOCK20_SOURCE_QUALIFICATION.json": ("passed", True, None, None),
    "release/BLOCK20_MATH_VERIFICATION.json": ("success", True, "total", 15),
    "release/BLOCK20_ADVERSARIAL_TESTS.json": ("passed", True, "total_count", 30),
    "release/BLOCK20_SOURCE_AUDIT.json": ("passed", True, None, None),
    "release/BLOCK20_WORKED_EXAMPLE.json": ("pipeline_is_I4", True, "result", "5/4"),
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
    (root / "release/BLOCK20_SOURCE_QUALIFICATION.json").read_text(encoding="utf-8")
)
fresh_audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
if fresh_audit.get("passed") is not True or fresh_audit.get("source_digest") != source_qualification.get("source_digest"):
    raise SystemExit("source changed after Block 20 qualification")
for suite, expected in (("expression", 22), ("block20", 46), ("stable", 6), ("block15", 174)):
    observed = source_qualification.get("suites", {}).get(suite, {})
    if not (observed.get("passed") is True and observed.get("checks") == expected and observed.get("expected") == expected):
        raise SystemExit(f"source {suite} verification is not exactly {expected}/{expected}")
documentation = source_qualification.get("suites", {}).get("documentation", {})
if not (
    documentation.get("passed") is True
    and documentation.get("checks") == 1
    and documentation.get("expected") == 1
):
    raise SystemExit("source documentation verification is not exactly 1/1")
'@ | & $Python - (Join-Path $Temporary "fresh-source-audit.json")
if ($LASTEXITCODE -ne 0) { throw "Block 20 source qualification gate failed" }

@'
from hashlib import sha256
from pathlib import Path
root = Path.cwd()
dist = root / "release" / "dist"
predecessors = root / "release" / "predecessors"
def digest(path):
    return sha256(path.read_bytes()).hexdigest()
for source in sorted(dist.glob("komorebi_math-*")):
    if source.name.startswith("komorebi_math-1.0.0b4"):
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
expected = {"komorebi_math-1.0.0b4-py3-none-any.whl", "komorebi_math-1.0.0b4.tar.gz"}
observed = {path.name for path in destination.iterdir() if path.is_file()}
if observed != expected:
    raise SystemExit(f"unexpected Block 20 build outputs: {sorted(observed)}")
'@ | & $Python - $Primary
if ($LASTEXITCODE -ne 0) { throw "Block 20 distribution build failed" }
& $Python (Join-Path $Root "verification\canonicalize_block20_sdist.py") `
    (Join-Path $Primary "komorebi_math-1.0.0b4.tar.gz") `
    --epoch 1787270400
if ($LASTEXITCODE -ne 0) { throw "Block 20 sdist canonicalization failed" }

Get-ChildItem -Path $Dist -Force | Remove-Item -Recurse -Force
Copy-Item (Join-Path $Primary "komorebi_math-1.0.0b4-py3-none-any.whl") $Dist
Copy-Item (Join-Path $Primary "komorebi_math-1.0.0b4.tar.gz") $Dist

@'
import json
from hashlib import sha256
from pathlib import Path
root = Path.cwd()
dist = root / "release" / "dist"
expected = ("komorebi_math-1.0.0b4-py3-none-any.whl", "komorebi_math-1.0.0b4.tar.gz")
observed = tuple(sorted(path.name for path in dist.iterdir()))
if observed != tuple(sorted(expected)):
    raise SystemExit(f"release/dist is not Block-20-only: {observed}")
def record(path):
    return {"name": path.name, "size": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()}
payload = {
    "schema": "komorebi-math.block20-distribution-build",
    "schema_version": 1,
    "package_version": "1.0.0b4",
    "passed": True,
    "source_digest": json.loads(
        (root / "release/BLOCK20_SOURCE_QUALIFICATION.json").read_text(encoding="utf-8")
    )["source_digest"],
    "artifacts": [record(dist / name) for name in expected],
    "dist_members": list(observed),
}
(root / "release" / "BLOCK20_DISTRIBUTION_BUILD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
'@ | & $Python -
if ($LASTEXITCODE -ne 0) { throw "Block 20 distribution inventory failed" }
& $Python (Join-Path $Root "verification\block20_distribution_audit.py") `
    --root $Root `
    --wheel (Join-Path $Dist "komorebi_math-1.0.0b4-py3-none-any.whl") `
    --sdist (Join-Path $Dist "komorebi_math-1.0.0b4.tar.gz") `
    --output (Join-Path $Root "release\BLOCK20_DISTRIBUTION_AUDIT.json")
if ($LASTEXITCODE -ne 0) { throw "Block 20 distribution audit failed" }
}
finally {
    if (Test-Path $Temporary) { Remove-Item -Recurse -Force $Temporary }
}
