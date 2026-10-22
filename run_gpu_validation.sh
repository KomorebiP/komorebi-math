#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
OUTPUT=${OUTPUT:-"$ROOT/qualification/gpu.json"}
cd "$ROOT"
"$PYTHON" -m komorebi_math release check --require-gpu --output "$OUTPUT" --force
"$PYTHON" -m komorebi_math verify gpu
printf 'GPU qualification record: %s\n' "$OUTPUT"
