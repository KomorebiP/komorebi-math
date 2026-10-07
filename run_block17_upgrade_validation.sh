#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
OUTPUT=${OUTPUT:-"$ROOT/release/BLOCK17_UPGRADE_VALIDATION.json"}
cd "$ROOT"
"$PYTHON" verification/run_block17_upgrade_validation.py --python "$PYTHON" --output "$OUTPUT"
printf 'Upgrade validation record: %s\n' "$OUTPUT"
