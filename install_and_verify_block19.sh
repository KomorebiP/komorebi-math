#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
DIST="$ROOT/release/dist"
WHEEL="$DIST/komorebi_math-1.0.0b3-py3-none-any.whl"
SDIST="$DIST/komorebi_math-1.0.0b3.tar.gz"
TEMPORARY=$(mktemp -d "${TMPDIR:-/tmp}/komorebi-block19-install.XXXXXX")
trap 'rm -rf "$TEMPORARY"' EXIT HUP INT TERM

[ -f "$WHEEL" ] || { echo "Block 19 wheel not found: $WHEEL" >&2; exit 1; }
[ -f "$SDIST" ] || { echo "Block 19 sdist not found: $SDIST" >&2; exit 1; }

"$PYTHON" - "$DIST" <<'PY'
from pathlib import Path
import sys

dist = Path(sys.argv[1])
expected = {
    "komorebi_math-1.0.0b3-py3-none-any.whl",
    "komorebi_math-1.0.0b3.tar.gz",
}
observed = {path.name for path in dist.iterdir()}
if observed != expected:
    raise SystemExit(f"release/dist is not Block-19-only: {sorted(observed)}")
PY

verify_installed() {
  ENVIRONMENT_PYTHON=$1
  SOURCE_ROOT=$2
  PREFIX=$3
  LABEL=$4
  RUN_DIRECTORY="$TEMPORARY/${PREFIX}-run"
  QUALIFICATION="$ROOT/release/${PREFIX}_QUALIFICATION.json"
  MATH_REPORT="$ROOT/release/${PREFIX}_MATH_VERIFICATION.json"
  ADVERSARIAL_REPORT="$ROOT/release/${PREFIX}_ADVERSARIAL_TESTS.json"
  WORKED_REPORT="$ROOT/release/${PREFIX}_WORKED_EXAMPLE.json"
  mkdir -p "$RUN_DIRECTORY"
  cd "$RUN_DIRECTORY"

  PYTHONPATH= "$ENVIRONMENT_PYTHON" - "$QUALIFICATION" "$LABEL" "$SOURCE_ROOT/src" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import komorebi_math
from komorebi_math.integration import ExactPolynomial, IntervalMeasure, integrate
from komorebi_math.verify import (
    run_block15_verification,
    run_block19_verification,
    run_documentation_verification,
    run_integration_verification,
    run_stable_verification,
)

output = Path(sys.argv[1])
label = sys.argv[2]
source = Path(sys.argv[3]).resolve()
location = Path(komorebi_math.__file__).resolve()
suites = {
    "integration": (run_integration_verification(), 12),
    "block19": (run_block19_verification(), 24),
    "stable": (run_stable_verification(), 6),
    "block15": (run_block15_verification(), 174),
    "documentation": (run_documentation_verification(), 1),
}
suite_payload = {
    name: {"passed": report.passed, "checks": len(report.checks), "expected": expected}
    for name, (report, expected) in suites.items()
}
passed = (
    komorebi_math.__version__ == "1.0.0b3"
    and any(part in {"site-packages", "dist-packages"} for part in location.parts)
    and source not in location.parents
    and integrate(ExactPolynomial.univariate((4, -2, 1)), IntervalMeasure(-1, 2)).require() == 12
    and all(item["passed"] and item["checks"] == item["expected"] for item in suite_payload.values())
)
payload = {
    "schema": "komorebi-math.block19-installed-qualification",
    "schema_version": 1,
    "package_version": komorebi_math.__version__,
    "label": label,
    "origin": str(location),
    "passed": passed,
    "suites": suite_payload,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if not passed:
    raise SystemExit(1)
PY

  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify integration
  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify block19
  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify stable
  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify block15
  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify documentation
  PYTHONPATH= "$ENVIRONMENT_PYTHON" "$SOURCE_ROOT/verification/block19_math_verifier.py" \
    --quiet --output "$MATH_REPORT"
  PYTHONPATH= "$ENVIRONMENT_PYTHON" "$SOURCE_ROOT/verification/block19_adversarial_runner.py" \
    --output "$ADVERSARIAL_REPORT"
  PYTHONPATH= "$ENVIRONMENT_PYTHON" "$SOURCE_ROOT/examples/block19_measure_integration_worked_example.py" --json \
    > "$WORKED_REPORT"

  PYTHONPATH= "$ENVIRONMENT_PYTHON" - "$MATH_REPORT" "$ADVERSARIAL_REPORT" "$WORKED_REPORT" <<'PY'
import json
import sys
from pathlib import Path

math_report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
adversarial = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
worked = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if not math_report.get("success") or math_report.get("total") != 16:
    raise SystemExit("installed independent verification is not 16/16")
if not adversarial.get("passed") or adversarial.get("total_count") != 47:
    raise SystemExit("installed adversarial verification is not 47/47")
if not worked.get("passed") or worked.get("exact_integral") != "106/5":
    raise SystemExit("installed worked example did not return 106/5")
PY
}

PRIMARY_ENV="$TEMPORARY/primary-env"
"$PYTHON" -m venv --system-site-packages "$PRIMARY_ENV"
"$PRIMARY_ENV/bin/python" -m pip install --no-index --no-deps --force-reinstall "$WHEEL"
verify_installed "$PRIMARY_ENV/bin/python" "$ROOT" "BLOCK19_INSTALLED" "primary wheel"

EXTRACT_DIRECTORY="$TEMPORARY/sdist-extract"
mkdir -p "$EXTRACT_DIRECTORY"
"$PYTHON" - "$SDIST" "$EXTRACT_DIRECTORY" <<'PY'
from pathlib import Path, PurePosixPath
import sys
import tarfile

sdist = Path(sys.argv[1])
destination = Path(sys.argv[2])
with tarfile.open(sdist, "r:gz") as archive:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe sdist member: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported sdist member type: {member.name}")
    archive.extractall(destination)
roots = [path for path in destination.iterdir() if path.is_dir()]
expected = destination / "komorebi_math-1.0.0b3"
if roots != [expected]:
    raise SystemExit(f"unexpected extracted sdist roots: {roots}")
for name in (
    "PACKAGE_AUDIT.json",
    "PACKAGE_AUDIT.md",
    "STATIC_AUDIT.json",
    "STATIC_AUDIT.md",
    "SHA256SUMS.txt",
):
    if (expected / name).exists():
        raise SystemExit(f"historical generic metadata leaked into Block 19 sdist: {name}")
PY

EXTRACTED_ROOT="$EXTRACT_DIRECTORY/komorebi_math-1.0.0b3"
REBUILT_DIST="$TEMPORARY/rebuilt-dist"
mkdir -p "$REBUILT_DIST"
"$PYTHON" - "$EXTRACTED_ROOT" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
for relative in ("build", "src/komorebi_math.egg-info"):
    target = (root / relative).resolve()
    if root not in target.parents:
        raise SystemExit(f"unsafe extracted build-cache target: {target}")
    if target.exists():
        shutil.rmtree(target)
PY
(
  cd "$EXTRACTED_ROOT"
  "$PYTHON" - "$REBUILT_DIST" <<'PY'
from pathlib import Path
import sys
from setuptools.build_meta import build_wheel

destination = Path(sys.argv[1]).resolve()
name = build_wheel(str(destination))
if name != "komorebi_math-1.0.0b3-py3-none-any.whl":
    raise SystemExit(f"unexpected rebuilt wheel: {name}")
print(name)
PY
)
REBUILT_WHEEL="$REBUILT_DIST/komorebi_math-1.0.0b3-py3-none-any.whl"

"$PYTHON" "$ROOT/verification/block19_distribution_audit.py" \
  --project "$ROOT" \
  --dist "$DIST" \
  --rebuilt-wheel "$REBUILT_WHEEL" \
  --output "$ROOT/release/BLOCK19_DISTRIBUTION_AUDIT.json"

SDIST_ENV="$TEMPORARY/sdist-env"
"$PYTHON" -m venv --system-site-packages "$SDIST_ENV"
"$SDIST_ENV/bin/python" -m pip install --no-index --no-deps --force-reinstall "$REBUILT_WHEEL"
verify_installed "$SDIST_ENV/bin/python" "$EXTRACTED_ROOT" "BLOCK19_SDIST_INSTALLED" "fresh sdist rebuilt wheel"

"$PYTHON" - "$ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

root = Path(sys.argv[1])
records = {
    "distribution_audit": root / "release/BLOCK19_DISTRIBUTION_AUDIT.json",
    "primary_qualification": root / "release/BLOCK19_INSTALLED_QUALIFICATION.json",
    "primary_math": root / "release/BLOCK19_INSTALLED_MATH_VERIFICATION.json",
    "primary_adversarial": root / "release/BLOCK19_INSTALLED_ADVERSARIAL_TESTS.json",
    "primary_worked": root / "release/BLOCK19_INSTALLED_WORKED_EXAMPLE.json",
    "sdist_qualification": root / "release/BLOCK19_SDIST_INSTALLED_QUALIFICATION.json",
    "sdist_math": root / "release/BLOCK19_SDIST_INSTALLED_MATH_VERIFICATION.json",
    "sdist_adversarial": root / "release/BLOCK19_SDIST_INSTALLED_ADVERSARIAL_TESTS.json",
    "sdist_worked": root / "release/BLOCK19_SDIST_INSTALLED_WORKED_EXAMPLE.json",
}
payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in records.items()}
checks = {
    "distribution_audit": payloads["distribution_audit"].get("passed") is True,
    "primary_qualification": payloads["primary_qualification"].get("passed") is True,
    "primary_documentation_1": (
        payloads["primary_qualification"].get("suites", {}).get("documentation", {}).get("passed") is True
        and payloads["primary_qualification"].get("suites", {}).get("documentation", {}).get("checks") == 1
        and payloads["primary_qualification"].get("suites", {}).get("documentation", {}).get("expected") == 1
    ),
    "primary_math_16": payloads["primary_math"].get("success") is True and payloads["primary_math"].get("total") == 16,
    "primary_adversarial_47": payloads["primary_adversarial"].get("passed") is True and payloads["primary_adversarial"].get("total_count") == 47,
    "primary_worked_106_5": payloads["primary_worked"].get("passed") is True and payloads["primary_worked"].get("exact_integral") == "106/5",
    "sdist_qualification": payloads["sdist_qualification"].get("passed") is True,
    "sdist_documentation_1": (
        payloads["sdist_qualification"].get("suites", {}).get("documentation", {}).get("passed") is True
        and payloads["sdist_qualification"].get("suites", {}).get("documentation", {}).get("checks") == 1
        and payloads["sdist_qualification"].get("suites", {}).get("documentation", {}).get("expected") == 1
    ),
    "sdist_math_16": payloads["sdist_math"].get("success") is True and payloads["sdist_math"].get("total") == 16,
    "sdist_adversarial_47": payloads["sdist_adversarial"].get("passed") is True and payloads["sdist_adversarial"].get("total_count") == 47,
    "sdist_worked_106_5": payloads["sdist_worked"].get("passed") is True and payloads["sdist_worked"].get("exact_integral") == "106/5",
}
artifacts = []
for path in (
    root / "release/dist/komorebi_math-1.0.0b3-py3-none-any.whl",
    root / "release/dist/komorebi_math-1.0.0b3.tar.gz",
):
    artifacts.append({"name": path.name, "size": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()})
payload = {
    "schema": "komorebi-math.block19-distribution-validation",
    "schema_version": 1,
    "package_version": "1.0.0b3",
    "passed": all(checks.values()),
    "checks": checks,
    "artifacts": artifacts,
}
(root / "release/BLOCK19_DISTRIBUTION_VALIDATION.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["passed"]:
    raise SystemExit(1)
PY
