#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
OUTPUT=${OUTPUT:-release/BLOCK16_ISOLATED_SOURCE_VALIDATION.json}
TIMEOUT=${TIMEOUT:-900}
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$PYTHON" verification/run_isolated_test_batches.py --timeout "$TIMEOUT" --output "$OUTPUT"
