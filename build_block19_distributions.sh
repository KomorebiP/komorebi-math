#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
DIST="$ROOT/release/dist"
PREDECESSORS="$ROOT/release/predecessors"
TEMPORARY=$(mktemp -d "${TMPDIR:-/tmp}/komorebi-block19-build.XXXXXX")
trap 'rm -rf "$TEMPORARY"' EXIT HUP INT TERM

cd "$ROOT"

# Refuse to package a partial source qualification run.
"$PYTHON" - <<'PY'
from __future__ import annotations

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
        raise SystemExit(
            f"source qualification count mismatch in {relative}: "
            f"expected {expected_count}, received {payload.get(count)!r}"
        )
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
    if source.name.startswith("komorebi_math-1.0.0b3"):
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

"$PYTHON" - "$TEMPORARY" <<'PY'
from pathlib import Path
import sys
from setuptools.build_meta import build_sdist, build_wheel

destination = Path(sys.argv[1]).resolve()
wheel = build_wheel(str(destination))
sdist = build_sdist(str(destination))
expected = {
    "komorebi_math-1.0.0b3-py3-none-any.whl",
    "komorebi_math-1.0.0b3.tar.gz",
}
observed = {path.name for path in destination.iterdir() if path.is_file()}
if observed != expected:
    raise SystemExit(f"unexpected Block 19 build outputs: {sorted(observed)}")
print(wheel)
print(sdist)
PY

"$PYTHON" - "$TEMPORARY" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from hashlib import sha256
from pathlib import Path

root = Path.cwd()
source = Path(sys.argv[1]).resolve()
dist = root / "release" / "dist"
expected = (
    "komorebi_math-1.0.0b3-py3-none-any.whl",
    "komorebi_math-1.0.0b3.tar.gz",
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
    raise SystemExit(f"release/dist is not Block-19-only: {observed}")

def record(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }

payload = {
    "schema": "komorebi-math.block19-distribution-build",
    "schema_version": 1,
    "package_version": "1.0.0b3",
    "passed": True,
    "artifacts": [record(dist / name) for name in expected],
    "dist_members": list(observed),
}
(root / "release" / "BLOCK19_DISTRIBUTION_BUILD.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
