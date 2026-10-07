#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
export PYTHONPATH="$ROOT/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
ARGS=(
  "$ROOT/verification/block21_publication_gate.py"
  --root "$ROOT"
  --output "$ROOT/release/BLOCK21_PUBLICATION_GATE.json"
  --markdown "$ROOT/release/BLOCK21_PUBLICATION_GATE.md"
  --full-tests
)
if "$PYTHON" -c 'import mkdocs' >/dev/null 2>&1; then
  ARGS+=(--require-mkdocs)
else
  ARGS+=(--skip-mkdocs)
fi
"$PYTHON" "${ARGS[@]}"
"$PYTHON" "$ROOT/examples/block21_public_capability_tour.py" --output "$ROOT/release/BLOCK21_PUBLIC_CAPABILITY_TOUR.json"
echo "Block 21 publication gate passed"
