#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=python3
cd "$ROOT"
"$PYTHON" -m pip wheel . --no-deps --no-build-isolation --wheel-dir release/dist
"$PYTHON" - <<'PY'
from setuptools.build_meta import build_sdist
print(build_sdist("release/dist"))
PY
