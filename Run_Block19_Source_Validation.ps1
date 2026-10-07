param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $Root "src"

& $Python -m compileall -q (Join-Path $Root "src") (Join-Path $Root "examples") (Join-Path $Root "verification")
if ($LASTEXITCODE -ne 0) { throw "Block 19 compilation failed" }
& $Python -m komorebi_math verify integration
if ($LASTEXITCODE -ne 0) { throw "Installed integration verification failed" }
& $Python -m komorebi_math verify block19
if ($LASTEXITCODE -ne 0) { throw "Installed Block 19 verification failed" }
& $Python -m komorebi_math verify stable
if ($LASTEXITCODE -ne 0) { throw "Frozen stable verification failed" }
& $Python -m komorebi_math verify block15
if ($LASTEXITCODE -ne 0) { throw "Cumulative Block 15 verification failed" }
& $Python -m komorebi_math verify documentation
if ($LASTEXITCODE -ne 0) { throw "Stable documentation verification failed" }
@'
import json
import sys
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
    suite_payload[name] = {"passed": report.passed, "checks": len(report.checks), "expected": count}
    if not report.passed or len(report.checks) != count:
        raise SystemExit(f"{name} verification mismatch: passed={report.passed}, checks={len(report.checks)}, expected={count}")
payload = {
    "schema": "komorebi-math.block19-source-qualification",
    "schema_version": 1,
    "package_version": "1.0.0b3",
    "passed": True,
    "suites": suite_payload,
}
(Path(sys.argv[1]) / "release/BLOCK19_SOURCE_QUALIFICATION.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("source CLI counts: integration=12 block19=24 stable=6 block15=174 documentation=1")
'@ | & $Python - $Root
if ($LASTEXITCODE -ne 0) { throw "Block 19 source verification counts changed" }
& $Python (Join-Path $Root "verification\block19_math_verifier.py") --output (Join-Path $Root "release\BLOCK19_MATH_VERIFICATION.json")
if ($LASTEXITCODE -ne 0) { throw "Independent Block 19 verification failed" }
& $Python (Join-Path $Root "verification\block19_adversarial_runner.py") --output (Join-Path $Root "release\BLOCK19_ADVERSARIAL_TESTS.json")
if ($LASTEXITCODE -ne 0) { throw "Block 19 adversarial tests failed" }
& $Python (Join-Path $Root "examples\block19_measure_integration_worked_example.py") --json | Set-Content -Encoding utf8 (Join-Path $Root "release\BLOCK19_WORKED_EXAMPLE.json")
if ($LASTEXITCODE -ne 0) { throw "Block 19 worked example failed" }
@'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
requirements = (
    ("release/BLOCK19_MATH_VERIFICATION.json", "success", True, "total", 16),
    ("release/BLOCK19_ADVERSARIAL_TESTS.json", "passed", True, "total_count", 47),
    ("release/BLOCK19_WORKED_EXAMPLE.json", "passed", True, "exact_integral", "106/5"),
)
for relative, flag, expected_flag, field, expected in requirements:
    payload = json.loads((root / relative).read_text(encoding="utf-8-sig"))
    if payload.get(flag) != expected_flag or payload.get(field) != expected:
        raise SystemExit(f"Block 19 source record mismatch: {relative}")
print("source records: independent=16 adversarial=47 worked=106/5")
'@ | & $Python - $Root
if ($LASTEXITCODE -ne 0) { throw "Block 19 source record counts changed" }

& $Python -c "import pytest" 2>$null
if ($LASTEXITCODE -eq 0) {
    & $Python -m pytest -q (Join-Path $Root "tests\test_block19_integration.py")
    if ($LASTEXITCODE -ne 0) { throw "Block 19 pytest suite failed" }
} else {
    Write-Host "pytest not installed; the fixture-free Block 19 adversarial runner completed"
}
