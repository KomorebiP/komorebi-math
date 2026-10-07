#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$ROOT/src"
PYTHON=${PYTHON:-python3}
cd "$ROOT"

mkdir -p "$ROOT/release/dist" "$ROOT/release/predecessors"
"$PYTHON" - <<'PY'
from hashlib import sha256
from pathlib import Path

root = Path.cwd()
dist = root / "release" / "dist"
predecessors = root / "release" / "predecessors"
names = (
    "komorebi_math-1.0.0b3-py3-none-any.whl",
    "komorebi_math-1.0.0b3.tar.gz",
)
for name in names:
    source = dist / name
    target = predecessors / name
    if source.exists() and target.exists():
        if sha256(source.read_bytes()).digest() != sha256(target.read_bytes()).digest():
            raise SystemExit(f"conflicting predecessor artifacts: {source} and {target}")
        source.unlink()
    elif source.exists():
        source.replace(target)
    if not target.is_file():
        raise SystemExit(f"required Block 19 predecessor artifact is missing: {target}")
PY

"$PYTHON" -m compileall -q "$ROOT/src" "$ROOT/examples" "$ROOT/verification"
"$PYTHON" -m komorebi_math verify expression
"$PYTHON" -m komorebi_math verify integration
"$PYTHON" -m komorebi_math verify block20
"$PYTHON" -m komorebi_math verify stable
"$PYTHON" -m komorebi_math verify block15
"$PYTHON" -m komorebi_math verify documentation
"$PYTHON" - <<'PY'
import json
from pathlib import Path

from komorebi_math.verify import (
    run_block15_verification,
    run_block19_verification,
    run_block20_verification,
    run_documentation_verification,
    run_integration_verification,
    run_expression_verification,
    run_stable_verification,
)

expected = (
    ("expression", run_expression_verification(), 22),
    ("integration", run_integration_verification(), 12),
    ("block19", run_block19_verification(), 24),
    ("block20", run_block20_verification(), 46),
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
    "schema": "komorebi-math.block20-source-qualification",
    "schema_version": 1,
    "package_version": "1.0.0b4",
    "passed": True,
    "suites": suite_payload,
}
Path("release/BLOCK20_SOURCE_QUALIFICATION.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("source CLI counts: expression=22 integration=12 block19=24 block20=46 stable=6 block15=174 documentation=1")
PY
"$PYTHON" "$ROOT/verification/block20_math_verifier.py" \
  --output "$ROOT/release/BLOCK20_MATH_VERIFICATION.json"
"$PYTHON" "$ROOT/verification/block20_adversarial_runner.py" \
  --output "$ROOT/release/BLOCK20_ADVERSARIAL_TESTS.json"
"$PYTHON" "$ROOT/examples/block20_typed_expression_worked_example.py" --json \
  > "$ROOT/release/BLOCK20_WORKED_EXAMPLE.json"
"$PYTHON" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

try:
    import pytest
except ImportError as exc:
    raise SystemExit("pytest is required for Block 20 source qualification") from exc


class Results:
    collected = 0
    passed = 0
    failed = 0
    skipped = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        if report.passed:
            self.passed += 1
        elif report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1


results = Results()
exit_code = int(
    pytest.main(
        ["-q", "tests/test_block20_expression_calculus.py"],
        plugins=[results],
    )
)
passed = (
    exit_code == 0
    and results.collected == 32
    and results.passed == 32
    and results.failed == 0
    and results.skipped == 0
)
payload = {
    "schema": "komorebi-math.block20-focused-pytest",
    "schema_version": 1,
    "package_version": "1.0.0b4",
    "passed": passed,
    "collected_count": results.collected,
    "passed_count": results.passed,
    "failed_count": results.failed,
    "skipped_count": results.skipped,
    "exit_code": exit_code,
}
Path("release/BLOCK20_PYTEST.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not passed:
    raise SystemExit("focused Block 20 pytest result is not exactly 32/32")
PY
"$PYTHON" "$ROOT/verification/block20_source_audit.py" \
  --root "$ROOT" --output "$ROOT/release/BLOCK20_SOURCE_AUDIT.json"
"$PYTHON" - <<'PY'
import json
from hashlib import sha256
from pathlib import Path

root = Path.cwd()
requirements = (
    ("release/BLOCK20_MATH_VERIFICATION.json", "success", True, "total", 15),
    ("release/BLOCK20_ADVERSARIAL_TESTS.json", "passed", True, "total_count", 30),
    ("release/BLOCK20_PYTEST.json", "passed", True, "passed_count", 32),
    ("release/BLOCK20_SOURCE_AUDIT.json", "passed", True, "package_version", "1.0.0b4"),
    ("release/BLOCK20_WORKED_EXAMPLE.json", "pipeline_is_I4", True, "result", "5/4"),
)
for relative, flag, expected_flag, field, expected in requirements:
    payload = json.loads((root / relative).read_text(encoding="utf-8"))
    if payload.get(flag) != expected_flag or payload.get(field) != expected:
        raise SystemExit(f"Block 20 source record mismatch: {relative}")
worked = json.loads((root / "release/BLOCK20_WORKED_EXAMPLE.json").read_text(encoding="utf-8"))
if worked.get("affine_regression", {}).get("base_integral") != "106/5":
    raise SystemExit("Block 20 affine worked regression is not 106/5")
audit = json.loads((root / "release/BLOCK20_SOURCE_AUDIT.json").read_text(encoding="utf-8"))
qualification_path = root / "release/BLOCK20_SOURCE_QUALIFICATION.json"
qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
qualification["source_digest"] = audit["source_digest"]
qualification["source_file_count"] = audit["source_file_count"]
qualification["focused_pytest"] = {
    "passed": True,
    "collected": 32,
    "passed_count": 32,
    "failed": 0,
    "skipped": 0,
}
qualification["evidence_sha256"] = {
    path.name: sha256(path.read_bytes()).hexdigest()
    for path in (
        root / "release/BLOCK20_MATH_VERIFICATION.json",
        root / "release/BLOCK20_ADVERSARIAL_TESTS.json",
        root / "release/BLOCK20_PYTEST.json",
        root / "release/BLOCK20_WORKED_EXAMPLE.json",
        root / "release/BLOCK20_SOURCE_AUDIT.json",
    )
}
qualification_path.write_text(
    json.dumps(qualification, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("source records: independent=15 adversarial=30 worked=5/4 affine=106/5")
PY
