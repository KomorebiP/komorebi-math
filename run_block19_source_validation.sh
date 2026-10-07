#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$ROOT/src"
PYTHON=${PYTHON:-python3}
cd "$ROOT"

"$PYTHON" -m compileall -q "$ROOT/src" "$ROOT/examples" "$ROOT/verification"
"$PYTHON" -m komorebi_math verify integration
"$PYTHON" -m komorebi_math verify block19
"$PYTHON" -m komorebi_math verify stable
"$PYTHON" -m komorebi_math verify block15
"$PYTHON" -m komorebi_math verify documentation
"$PYTHON" - <<'PY'
import json
from pathlib import Path

from komorebi_math.verify import (
    run_block15_verification,
    run_block19_verification,
    run_documentation_verification,
    run_integration_verification,
    run_stable_verification,
)

expected = (
    ("integration", run_integration_verification(), 12),
    ("block19", run_block19_verification(), 24),
    ("stable", run_stable_verification(), 6),
    ("block15", run_block15_verification(), 174),
    ("documentation", run_documentation_verification(), 1),
)
suite_payload = {}
for name, report, count in expected:
    suite_payload[name] = {
        "passed": report.passed,
        "checks": len(report.checks),
        "expected": count,
    }
    if not report.passed or len(report.checks) != count:
        raise SystemExit(
            f"{name} verification mismatch: passed={report.passed}, "
            f"checks={len(report.checks)}, expected={count}"
        )
payload = {
    "schema": "komorebi-math.block19-source-qualification",
    "schema_version": 1,
    "package_version": "1.0.0b3",
    "passed": True,
    "suites": suite_payload,
}
Path("release/BLOCK19_SOURCE_QUALIFICATION.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("source CLI counts: integration=12 block19=24 stable=6 block15=174 documentation=1")
PY
"$PYTHON" "$ROOT/verification/block19_math_verifier.py" \
  --output "$ROOT/release/BLOCK19_MATH_VERIFICATION.json"
"$PYTHON" "$ROOT/verification/block19_adversarial_runner.py" \
  --output "$ROOT/release/BLOCK19_ADVERSARIAL_TESTS.json"
"$PYTHON" "$ROOT/examples/block19_measure_integration_worked_example.py" --json \
  > "$ROOT/release/BLOCK19_WORKED_EXAMPLE.json"
"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
requirements = (
    ("release/BLOCK19_MATH_VERIFICATION.json", "success", True, "total", 16),
    ("release/BLOCK19_ADVERSARIAL_TESTS.json", "passed", True, "total_count", 47),
    ("release/BLOCK19_WORKED_EXAMPLE.json", "passed", True, "exact_integral", "106/5"),
)
for relative, flag, expected_flag, field, expected in requirements:
    payload = json.loads((root / relative).read_text(encoding="utf-8"))
    if payload.get(flag) != expected_flag or payload.get(field) != expected:
        raise SystemExit(f"Block 19 source record mismatch: {relative}")
print("source records: independent=16 adversarial=47 worked=106/5")
PY

if "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
  "$PYTHON" -m pytest -q "$ROOT/tests/test_block19_integration.py"
else
  echo "pytest not installed; the fixture-free Block 19 adversarial runner completed"
fi
