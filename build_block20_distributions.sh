#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
DIST="$ROOT/release/dist"
PREDECESSORS="$ROOT/release/predecessors"
TEMPORARY=$(mktemp -d "${TMPDIR:-/tmp}/komorebi-block20-build.XXXXXX")
trap 'rm -rf "$TEMPORARY"' EXIT HUP INT TERM
SOURCE_DATE_EPOCH=1787270400
export SOURCE_DATE_EPOCH

cd "$ROOT"

mkdir -p "$DIST" "$PREDECESSORS"
# Relocate the exact predecessor before freezing or comparing the source digest.
"$PYTHON" - <<'PY'
from hashlib import sha256
from pathlib import Path

root = Path.cwd()
dist = root / "release" / "dist"
predecessors = root / "release" / "predecessors"
for name in (
    "komorebi_math-1.0.0b3-py3-none-any.whl",
    "komorebi_math-1.0.0b3.tar.gz",
):
    source = dist / name
    target = predecessors / name
    if source.exists() and target.exists():
        if sha256(source.read_bytes()).digest() != sha256(target.read_bytes()).digest():
            raise SystemExit(f"refusing conflicting predecessor artifacts: {source} and {target}")
        source.unlink()
    elif source.exists():
        source.replace(target)
    if not target.is_file():
        raise SystemExit(f"required Block 19 predecessor artifact is missing: {target}")
PY

# Refuse to package a partial source qualification run.
"$PYTHON" "$ROOT/verification/block20_source_audit.py" \
  --root "$ROOT" --quiet --output "$TEMPORARY/fresh-source-audit.json"
"$PYTHON" - "$TEMPORARY/fresh-source-audit.json" <<'PY'
from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

root = Path.cwd()
requirements = {
    "release/BLOCK20_SOURCE_QUALIFICATION.json": ("passed", True, None, None),
    "release/BLOCK20_MATH_VERIFICATION.json": ("success", True, "total", 15),
    "release/BLOCK20_ADVERSARIAL_TESTS.json": ("passed", True, "total_count", 30),
    "release/BLOCK20_PYTEST.json": ("passed", True, "passed_count", 32),
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
        raise SystemExit(
            f"source qualification count mismatch in {relative}: "
            f"expected {expected_count}, received {payload.get(count)!r}"
        )
source_qualification = json.loads(
    (root / "release/BLOCK20_SOURCE_QUALIFICATION.json").read_text(encoding="utf-8")
)
evidence_paths = (
    root / "release/BLOCK20_MATH_VERIFICATION.json",
    root / "release/BLOCK20_ADVERSARIAL_TESTS.json",
    root / "release/BLOCK20_PYTEST.json",
    root / "release/BLOCK20_WORKED_EXAMPLE.json",
    root / "release/BLOCK20_SOURCE_AUDIT.json",
)
observed_evidence = {
    path.name: sha256(path.read_bytes()).hexdigest() for path in evidence_paths
}
if source_qualification.get("evidence_sha256") != observed_evidence:
    raise SystemExit("source qualification evidence hashes do not match current records")
focused = source_qualification.get("focused_pytest", {})
if not (
    focused.get("passed") is True
    and focused.get("collected") == 32
    and focused.get("passed_count") == 32
    and focused.get("failed") == 0
    and focused.get("skipped") == 0
):
    raise SystemExit("focused pytest qualification is not exactly 32/32")
fresh_audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if fresh_audit.get("passed") is not True:
    raise SystemExit("fresh Block 20 source audit failed")
if source_qualification.get("source_digest") != fresh_audit.get("source_digest"):
    raise SystemExit("source changed after Block 20 qualification; rerun source validation")
for suite, expected in (("expression", 22), ("block20", 46), ("stable", 6), ("block15", 174)):
    observed = source_qualification.get("suites", {}).get(suite, {})
    if not (
        observed.get("passed") is True
        and observed.get("checks") == expected
        and observed.get("expected") == expected
    ):
        raise SystemExit(f"source {suite} verification is not exactly {expected}/{expected}")
documentation = source_qualification.get("suites", {}).get("documentation", {})
if not (
    documentation.get("passed") is True
    and documentation.get("checks") == 1
    and documentation.get("expected") == 1
):
    raise SystemExit("source documentation verification is not exactly 1/1")
PY

mkdir -p "$DIST" "$PREDECESSORS"

# Preserve prior beta artifacts outside the current distribution directory.
"$PYTHON" - <<'PY'
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

root = Path.cwd()
dist = root / "release" / "dist"
predecessors = root / "release" / "predecessors"

def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

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
PY

# Setuptools may otherwise reuse an earlier build/lib tree. Clean only these
# project-owned build caches, then build both formats into a private directory.
"$PYTHON" - <<'PY'
from pathlib import Path
import shutil

root = Path.cwd().resolve()
for relative in ("build", "src/komorebi_math.egg-info"):
    target = (root / relative).resolve()
    if root not in target.parents:
        raise SystemExit(f"unsafe build-cache target: {target}")
    if target.exists():
        shutil.rmtree(target)
PY

mkdir -p "$TEMPORARY/primary"
"$PYTHON" - "$TEMPORARY/primary" <<'PY'
from pathlib import Path
import sys
from setuptools.build_meta import build_sdist, build_wheel

destination = Path(sys.argv[1]).resolve()
wheel = build_wheel(str(destination))
sdist = build_sdist(str(destination))
expected = {
    "komorebi_math-1.0.0b4-py3-none-any.whl",
    "komorebi_math-1.0.0b4.tar.gz",
}
observed = {path.name for path in destination.iterdir() if path.is_file()}
if observed != expected:
    raise SystemExit(f"unexpected Block 20 build outputs: {sorted(observed)}")
print(wheel)
print(sdist)
PY
"$PYTHON" "$ROOT/verification/canonicalize_block20_sdist.py" \
  "$TEMPORARY/primary/komorebi_math-1.0.0b4.tar.gz" \
  --epoch "$SOURCE_DATE_EPOCH"

# Rebuild from the same frozen source and toolchain, then demand byte-identical
# wheel and sdist artifacts under the fixed SOURCE_DATE_EPOCH.
"$PYTHON" - <<'PY'
from pathlib import Path
import shutil

root = Path.cwd().resolve()
for relative in ("build", "src/komorebi_math.egg-info"):
    target = (root / relative).resolve()
    if root not in target.parents:
        raise SystemExit(f"unsafe build-cache target: {target}")
    if target.exists():
        shutil.rmtree(target)
PY
mkdir -p "$TEMPORARY/replay"
"$PYTHON" - "$TEMPORARY/replay" <<'PY'
from pathlib import Path
import sys
from setuptools.build_meta import build_sdist, build_wheel

destination = Path(sys.argv[1]).resolve()
wheel = build_wheel(str(destination))
sdist = build_sdist(str(destination))
expected = {
    "komorebi_math-1.0.0b4-py3-none-any.whl",
    "komorebi_math-1.0.0b4.tar.gz",
}
observed = {path.name for path in destination.iterdir() if path.is_file()}
if observed != expected:
    raise SystemExit(f"unexpected Block 20 replay build outputs: {sorted(observed)}")
print(wheel)
print(sdist)
PY
"$PYTHON" "$ROOT/verification/canonicalize_block20_sdist.py" \
  "$TEMPORARY/replay/komorebi_math-1.0.0b4.tar.gz" \
  --epoch "$SOURCE_DATE_EPOCH"
"$PYTHON" - "$TEMPORARY" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
for name in (
    "komorebi_math-1.0.0b4-py3-none-any.whl",
    "komorebi_math-1.0.0b4.tar.gz",
):
    first = sha256((root / "primary" / name).read_bytes()).hexdigest()
    replay = sha256((root / "replay" / name).read_bytes()).hexdigest()
    if first != replay:
        raise SystemExit(f"Block 20 build is not byte reproducible: {name}")
print("byte-reproducible wheel and sdist build replay passed")
PY

"$PYTHON" - "$TEMPORARY/primary" <<'PY'
from __future__ import annotations

import json
import importlib.metadata
import platform
import shutil
import sys
from hashlib import sha256
from pathlib import Path

root = Path.cwd()
source = Path(sys.argv[1]).resolve()
dist = (root / "release" / "dist").resolve()
if dist != (root.resolve() / "release" / "dist") or root.resolve() not in dist.parents:
    raise SystemExit(f"unsafe distribution target: {dist}")
expected = (
    "komorebi_math-1.0.0b4-py3-none-any.whl",
    "komorebi_math-1.0.0b4.tar.gz",
)

for path in tuple(dist.iterdir()):
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
for name in expected:
    shutil.copy2(source / name, dist / name)

observed = tuple(sorted(path.name for path in dist.iterdir()))
if observed != tuple(sorted(expected)):
    raise SystemExit(f"release/dist is not Block-20-only: {observed}")

def record(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }

payload = {
    "schema": "komorebi-math.block20-distribution-build",
    "schema_version": 1,
    "package_version": "1.0.0b4",
    "passed": True,
    "byte_reproducible_rebuild": True,
    "source_date_epoch": 1787270400,
    "build_toolchain": {
        "python": platform.python_version(),
        "setuptools": importlib.metadata.version("setuptools"),
        "wheel": importlib.metadata.version("wheel"),
    },
    "source_digest": json.loads(
        (root / "release" / "BLOCK20_SOURCE_QUALIFICATION.json").read_text(encoding="utf-8")
    )["source_digest"],
    "artifacts": [record(dist / name) for name in expected],
    "dist_members": list(observed),
}
(root / "release" / "BLOCK20_DISTRIBUTION_BUILD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

"$PYTHON" "$ROOT/verification/block20_distribution_audit.py" \
  --root "$ROOT" \
  --wheel "$DIST/komorebi_math-1.0.0b4-py3-none-any.whl" \
  --sdist "$DIST/komorebi_math-1.0.0b4.tar.gz" \
  --output "$ROOT/release/BLOCK20_DISTRIBUTION_AUDIT.json"
