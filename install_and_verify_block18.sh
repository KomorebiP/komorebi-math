#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=python3
ENVIRONMENT="$ROOT/.venv-block18"
WHEEL=$(find "$ROOT/release/dist" -maxdepth 1 -name 'komorebi_math-1.0.0b2-*.whl' -print | head -n 1)
[ -n "$WHEEL" ] || { echo "Block 18 wheel not found" >&2; exit 1; }
[ -d "$ENVIRONMENT" ] || "$PYTHON" -m venv --system-site-packages "$ENVIRONMENT"
"$ENVIRONMENT/bin/python" -m pip install --no-deps --force-reinstall "$WHEEL"
cd /tmp
PYTHONPATH= "$ENVIRONMENT/bin/python" - <<'PY'
from pathlib import Path
import komorebi_math
assert komorebi_math.__version__ == "1.0.0b2"
location = Path(komorebi_math.__file__).resolve()
assert any(part in {"site-packages", "dist-packages"} for part in location.parts)
print("verified", komorebi_math.__version__, "from", location)
PY
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math verify frame-algebra
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math verify block18
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math verify stable
