#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
OUTPUT=${OUTPUT:-"$ROOT/release/BLOCK17_GPU_QUALIFICATION.json"}
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
"$PYTHON" -m komorebi_math release qualify --require-gpu --output "$OUTPUT" --force
"$PYTHON" -m komorebi_math release readiness --gpu --require-gpu --output "$ROOT/release/BLOCK17_GPU_READINESS.json" --force
"$PYTHON" -m komorebi_math verify gpu
printf 'GPU qualification record: %s\n' "$OUTPUT"
