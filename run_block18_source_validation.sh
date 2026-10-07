#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$ROOT/src"
python -m compileall -q "$ROOT/src"
python "$ROOT/verification/block18_math_verifier.py" --output "$ROOT/release/BLOCK18_MATH_VERIFICATION.json"
if python -c "import pytest" >/dev/null 2>&1; then
  python -m pytest -q "$ROOT/tests/test_block18_frame_algebra.py"
else
  echo "pytest not installed; independent Block 18 verifier completed"
fi
