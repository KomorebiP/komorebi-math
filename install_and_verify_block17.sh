#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
ENVIRONMENT=${ENVIRONMENT:-"$ROOT/.venv-block17"}
WHEEL=$(find "$ROOT/release/dist" -maxdepth 1 -name 'komorebi_math-1.0.0b1-*.whl' -print | head -n 1)
[ -n "$WHEEL" ] || { echo "Block 17 wheel not found" >&2; exit 1; }
[ -d "$ENVIRONMENT" ] || "$PYTHON" -m venv --system-site-packages "$ENVIRONMENT"
"$ENVIRONMENT/bin/python" -m pip install --no-deps --force-reinstall "$WHEEL"
cd /tmp
PYTHONPATH= "$ENVIRONMENT/bin/python" - <<'PY'
from pathlib import Path
import komorebi_math
from komorebi_math.stable import check_stable_contract
assert komorebi_math.__version__ == "1.0.0b1"
check_stable_contract().certificate.require()
print("verified", komorebi_math.__version__, "from", Path(komorebi_math.__file__).resolve())
PY
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math api check --stable
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math release contract
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math verify stable
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math verify release
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math release upgrade
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math release evidence
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math release documentation
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math release readiness --output "$ROOT/release/BLOCK17_INSTALLED_BETA_READINESS.json" --force
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math verify block17
PYTHONPATH= "$ENVIRONMENT/bin/python" -m komorebi_math release qualify --output "$ROOT/release/BLOCK17_INSTALLED_QUALIFICATION.json" --force
CHECK_OUTPUT=$("$ENVIRONMENT/bin/python" -m pip check 2>&1) || CHECK_STATUS=$?
CHECK_STATUS=${CHECK_STATUS:-0}
printf '%s\n' "$CHECK_OUTPUT"
if [ "$CHECK_STATUS" -ne 0 ]; then
    if printf '%s\n' "$CHECK_OUTPUT" | grep -Eiq '^komorebi[-_]math '; then
        echo "Komorebi Math dependency consistency check failed" >&2
        exit "$CHECK_STATUS"
    fi
    echo "pip check reported only unrelated inherited system-site-package issues; Komorebi Math requirements are consistent" >&2
fi
