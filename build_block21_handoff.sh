#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY:-"$ROOT/release/handoff"}
export PYTHONPATH="$ROOT/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
GATE_ARGS=(
  "$ROOT/verification/block21_publication_gate.py"
  --root "$ROOT"
  --full-tests
  --installed-artifacts
  --output "$ROOT/release/REPAIR_R2_PUBLICATION_GATE.json"
  --markdown "$ROOT/release/REPAIR_R2_PUBLICATION_GATE.md"
)
if "$PYTHON" -c 'import mkdocs' >/dev/null 2>&1; then
  GATE_ARGS+=(--require-mkdocs)
else
  GATE_ARGS+=(--skip-mkdocs)
fi
mkdir -p "$ROOT/release/dist"
find "$ROOT/release/dist" -maxdepth 1 -type f -name 'komorebi_math-1.0.0b4*' -delete
"$PYTHON" "$ROOT/verification/build_block21_distributions.py" --root "$ROOT" --output-directory "$ROOT/release/dist" --report "$ROOT/release/REPAIR_R2_DISTRIBUTION_BUILD.json"
"$PYTHON" "${GATE_ARGS[@]}"
"$PYTHON" "$ROOT/verification/build_block21_handoff.py" \
  --root "$ROOT" \
  --output-directory "$OUTPUT_DIRECTORY" \
  --artifact-directory "$ROOT/release/dist" \
  --gate "$ROOT/release/REPAIR_R2_PUBLICATION_GATE.json" \
  --python "$PYTHON"
echo "Block 21 publication handoff passed and was written to $OUTPUT_DIRECTORY"
