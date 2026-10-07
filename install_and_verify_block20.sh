#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
DIST="$ROOT/release/dist"
WHEEL="$DIST/komorebi_math-1.0.0b4-py3-none-any.whl"
SDIST="$DIST/komorebi_math-1.0.0b4.tar.gz"
TEMPORARY=$(mktemp -d "${TMPDIR:-/tmp}/komorebi-block20-install.XXXXXX")
trap 'rm -rf "$TEMPORARY"' EXIT HUP INT TERM

[ -f "$WHEEL" ] || { echo "Block 20 wheel not found: $WHEEL" >&2; exit 1; }
[ -f "$SDIST" ] || { echo "Block 20 sdist not found: $SDIST" >&2; exit 1; }

"$PYTHON" - "$DIST" "$ROOT" <<'PY'
import json
from hashlib import sha256
from pathlib import Path
import sys

dist = Path(sys.argv[1])
root = Path(sys.argv[2])
expected = {
    "komorebi_math-1.0.0b4-py3-none-any.whl",
    "komorebi_math-1.0.0b4.tar.gz",
}
observed = {path.name for path in dist.iterdir()}
if observed != expected:
    raise SystemExit(f"release/dist is not Block-20-only: {sorted(observed)}")
build = json.loads((root / "release/BLOCK20_DISTRIBUTION_BUILD.json").read_text(encoding="utf-8"))
audit = json.loads((root / "release/BLOCK20_DISTRIBUTION_AUDIT.json").read_text(encoding="utf-8"))
qualification = json.loads((root / "release/BLOCK20_SOURCE_QUALIFICATION.json").read_text(encoding="utf-8"))
if build.get("passed") is not True or audit.get("passed") is not True:
    raise SystemExit("Block 20 build or distribution audit did not pass")
if build.get("byte_reproducible_rebuild") is not True:
    raise SystemExit("Block 20 build did not pass its byte-reproducibility replay")
if build.get("source_digest") != qualification.get("source_digest"):
    raise SystemExit("Block 20 build is not bound to the qualified source digest")
evidence_paths = (
    root / "release/BLOCK20_MATH_VERIFICATION.json",
    root / "release/BLOCK20_ADVERSARIAL_TESTS.json",
    root / "release/BLOCK20_PYTEST.json",
    root / "release/BLOCK20_WORKED_EXAMPLE.json",
    root / "release/BLOCK20_SOURCE_AUDIT.json",
)
observed_evidence = {
    path.name: sha256(path.read_bytes()).hexdigest() for path in evidence_paths
}
if qualification.get("evidence_sha256") != observed_evidence:
    raise SystemExit("source qualification evidence hashes do not match current records")
recorded = {item["name"]: item for item in build.get("artifacts", [])}
for name in expected:
    path = dist / name
    item = recorded.get(name, {})
    if item.get("size") != path.stat().st_size or item.get("sha256") != sha256(path.read_bytes()).hexdigest():
        raise SystemExit(f"distribution artifact changed after build: {name}")
if audit.get("wheel", {}).get("sha256") != recorded["komorebi_math-1.0.0b4-py3-none-any.whl"]["sha256"]:
    raise SystemExit("distribution audit wheel digest is not bound to the build")
if audit.get("sdist", {}).get("sha256") != recorded["komorebi_math-1.0.0b4.tar.gz"]["sha256"]:
    raise SystemExit("distribution audit sdist digest is not bound to the build")
PY

verify_installed() {
  ENVIRONMENT_PYTHON=$1
  SOURCE_ROOT=$2
  PREFIX=$3
  LABEL=$4
  ARTIFACT=$5
  RUN_DIRECTORY="$TEMPORARY/${PREFIX}-run"
  QUALIFICATION="$ROOT/release/${PREFIX}_QUALIFICATION.json"
  MATH_REPORT="$ROOT/release/${PREFIX}_MATH_VERIFICATION.json"
  ADVERSARIAL_REPORT="$ROOT/release/${PREFIX}_ADVERSARIAL_TESTS.json"
  WORKED_REPORT="$ROOT/release/${PREFIX}_WORKED_EXAMPLE.json"
  mkdir -p "$RUN_DIRECTORY"
  cd "$RUN_DIRECTORY"

  PYTHONPATH= "$ENVIRONMENT_PYTHON" - \
    "$QUALIFICATION" "$LABEL" "$SOURCE_ROOT/src" "$ARTIFACT" \
    "$ROOT/release/BLOCK20_SOURCE_QUALIFICATION.json" <<'PY'
from __future__ import annotations

import json
import sys
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import komorebi_math
from komorebi_math.expression import (
    PolynomialCoordinateSpace,
    apply,
    derivative,
    evaluate_expression,
    exact_value,
)
from komorebi_math.integration import ExactPolynomial, IntervalMeasure, integrate
from komorebi_math.verify import (
    run_block15_verification,
    run_block19_verification,
    run_block20_verification,
    run_documentation_verification,
    run_integration_verification,
    run_expression_verification,
    run_stable_verification,
)

output = Path(sys.argv[1])
label = sys.argv[2]
source = Path(sys.argv[3]).resolve()
artifact = Path(sys.argv[4]).resolve()
source_qualification = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
location = Path(komorebi_math.__file__).resolve()
try:
    site_index = next(
        index
        for index, part in enumerate(location.parts)
        if part in {"site-packages", "dist-packages"}
    )
except StopIteration:
    stable_origin = "outside-site-packages"
else:
    stable_origin = "/".join(location.parts[site_index:])
suites = {
    "expression": (run_expression_verification(), 22),
    "integration": (run_integration_verification(), 12),
    "block19": (run_block19_verification(), 24),
    "block20": (run_block20_verification(), 46),
    "stable": (run_stable_verification(), 6),
    "block15": (run_block15_verification(), 174),
    "documentation": (run_documentation_verification(), 1),
}
suite_payload = {
    name: {"passed": report.passed, "checks": len(report.checks), "expected": expected}
    for name, (report, expected) in suites.items()
}
monomial2 = PolynomialCoordinateSpace(
    "x",
    2,
    ((Fraction(1),), (Fraction(0), Fraction(1)), (Fraction(0), Fraction(0), Fraction(1))),
    "installed smoke",
)
expression_smoke = evaluate_expression(
    apply(derivative(monomial2), exact_value(monomial2, (1, 2, 3)))
).value
passed = (
    komorebi_math.__version__ == "1.0.0b4"
    and any(part in {"site-packages", "dist-packages"} for part in location.parts)
    and source not in location.parents
    and integrate(ExactPolynomial.univariate((4, -2, 1)), IntervalMeasure(-1, 2)).require() == 12
    and expression_smoke == (Fraction(2), Fraction(6))
    and all(item["passed"] and item["checks"] == item["expected"] for item in suite_payload.values())
)
payload = {
    "schema": "komorebi-math.block20-installed-qualification",
    "schema_version": 1,
    "package_version": komorebi_math.__version__,
    "label": label,
    "origin": stable_origin,
    "source_digest": source_qualification.get("source_digest"),
    "tested_artifact": {
        "name": artifact.name,
        "size": artifact.stat().st_size,
        "sha256": sha256(artifact.read_bytes()).hexdigest(),
    },
    "passed": passed,
    "suites": suite_payload,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
if not passed:
    raise SystemExit(1)
PY

  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify expression
  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify integration
  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify block20
  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify stable
  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify block15
  PYTHONPATH= "$ENVIRONMENT_PYTHON" -m komorebi_math verify documentation
  PYTHONPATH= "$ENVIRONMENT_PYTHON" "$SOURCE_ROOT/verification/block20_math_verifier.py" \
    --quiet --output "$MATH_REPORT"
  PYTHONPATH= "$ENVIRONMENT_PYTHON" "$SOURCE_ROOT/verification/block20_adversarial_runner.py" \
    --output "$ADVERSARIAL_REPORT"
  PYTHONPATH= "$ENVIRONMENT_PYTHON" "$SOURCE_ROOT/examples/block20_typed_expression_worked_example.py" --json \
    > "$WORKED_REPORT"

  PYTHONPATH= "$ENVIRONMENT_PYTHON" - "$MATH_REPORT" "$ADVERSARIAL_REPORT" "$WORKED_REPORT" <<'PY'
import json
import sys
from pathlib import Path

math_report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
adversarial = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
worked = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if not math_report.get("success") or math_report.get("total") != 15:
    raise SystemExit("installed independent verification is not 15/15")
if not adversarial.get("passed") or adversarial.get("total_count") != 30:
    raise SystemExit("installed adversarial verification is not 30/30")
if not worked.get("pipeline_is_I4") or worked.get("result") != "5/4":
    raise SystemExit("installed worked example did not return I4 and 5/4")
if worked.get("affine_regression", {}).get("base_integral") != "106/5":
    raise SystemExit("installed worked example did not retain 106/5")
PY
}

PRIMARY_ENV="$TEMPORARY/primary-env"
"$PYTHON" -m venv --system-site-packages "$PRIMARY_ENV"
"$PRIMARY_ENV/bin/python" -m pip install --no-index --no-deps --force-reinstall "$WHEEL"
verify_installed "$PRIMARY_ENV/bin/python" "$ROOT" "BLOCK20_INSTALLED" "primary wheel" "$WHEEL"

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
    normalized = set()
    folded = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            not member.name
            or "\x00" in member.name
            or "\\" in member.name
            or path.is_absolute()
            or ".." in path.parts
            or (path.parts and ":" in path.parts[0])
        ):
            raise SystemExit(f"unsafe sdist member: {member.name}")
        name = path.as_posix().rstrip("/")
        if name in normalized or name.casefold() in folded:
            raise SystemExit(f"duplicate or case-colliding sdist member: {member.name}")
        normalized.add(name)
        folded.add(name.casefold())
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported sdist member type: {member.name}")
    archive.extractall(destination)
roots = [path for path in destination.iterdir() if path.is_dir()]
expected = destination / "komorebi_math-1.0.0b4"
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
        raise SystemExit(f"historical generic metadata leaked into Block 20 sdist: {name}")
PY

EXTRACTED_ROOT="$EXTRACT_DIRECTORY/komorebi_math-1.0.0b4"
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
if name != "komorebi_math-1.0.0b4-py3-none-any.whl":
    raise SystemExit(f"unexpected rebuilt wheel: {name}")
print(name)
PY
)
REBUILT_WHEEL="$REBUILT_DIST/komorebi_math-1.0.0b4-py3-none-any.whl"

"$PYTHON" "$ROOT/verification/block20_distribution_audit.py" \
  --root "$ROOT" \
  --wheel "$WHEEL" \
  --sdist "$SDIST" \
  --rebuilt-wheel "$REBUILT_WHEEL" \
  --output "$ROOT/release/BLOCK20_DISTRIBUTION_AUDIT.json"

SDIST_ENV="$TEMPORARY/sdist-env"
"$PYTHON" -m venv --system-site-packages "$SDIST_ENV"
"$SDIST_ENV/bin/python" -m pip install --no-index --no-deps --force-reinstall "$REBUILT_WHEEL"
verify_installed "$SDIST_ENV/bin/python" "$EXTRACTED_ROOT" "BLOCK20_SDIST_INSTALLED" "fresh sdist rebuilt wheel" "$REBUILT_WHEEL"

"$PYTHON" - "$ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

root = Path(sys.argv[1])
records = {
    "distribution_audit": root / "release/BLOCK20_DISTRIBUTION_AUDIT.json",
    "primary_qualification": root / "release/BLOCK20_INSTALLED_QUALIFICATION.json",
    "primary_math": root / "release/BLOCK20_INSTALLED_MATH_VERIFICATION.json",
    "primary_adversarial": root / "release/BLOCK20_INSTALLED_ADVERSARIAL_TESTS.json",
    "primary_worked": root / "release/BLOCK20_INSTALLED_WORKED_EXAMPLE.json",
    "sdist_qualification": root / "release/BLOCK20_SDIST_INSTALLED_QUALIFICATION.json",
    "sdist_math": root / "release/BLOCK20_SDIST_INSTALLED_MATH_VERIFICATION.json",
    "sdist_adversarial": root / "release/BLOCK20_SDIST_INSTALLED_ADVERSARIAL_TESTS.json",
    "sdist_worked": root / "release/BLOCK20_SDIST_INSTALLED_WORKED_EXAMPLE.json",
}
payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in records.items()}
source_qualification = json.loads(
    (root / "release/BLOCK20_SOURCE_QUALIFICATION.json").read_text(encoding="utf-8")
)
distribution_build_path = root / "release/BLOCK20_DISTRIBUTION_BUILD.json"
distribution_build = json.loads(distribution_build_path.read_text(encoding="utf-8"))
source_digest = source_qualification.get("source_digest")
expected_report_schemas = {
    "primary_math": "komorebi-math.block20-independent-math-verification",
    "primary_adversarial": "komorebi-math.block20-adversarial-tests",
    "primary_worked": "komorebi-math.block20-worked-example",
    "sdist_math": "komorebi-math.block20-independent-math-verification",
    "sdist_adversarial": "komorebi-math.block20-adversarial-tests",
    "sdist_worked": "komorebi-math.block20-worked-example",
}
checks = {
    "distribution_audit": payloads["distribution_audit"].get("passed") is True,
    "primary_qualification": payloads["primary_qualification"].get("passed") is True,
    "primary_source_digest": payloads["primary_qualification"].get("source_digest") == source_digest,
    "primary_documentation_1": (
        payloads["primary_qualification"].get("suites", {}).get("documentation", {}).get("passed") is True
        and payloads["primary_qualification"].get("suites", {}).get("documentation", {}).get("checks") == 1
        and payloads["primary_qualification"].get("suites", {}).get("documentation", {}).get("expected") == 1
    ),
    "primary_math_15": payloads["primary_math"].get("success") is True and payloads["primary_math"].get("total") == 15,
    "primary_adversarial_30": payloads["primary_adversarial"].get("passed") is True and payloads["primary_adversarial"].get("total_count") == 30,
    "primary_worked_5_4": payloads["primary_worked"].get("pipeline_is_I4") is True and payloads["primary_worked"].get("result") == "5/4",
    "primary_affine_106_5": payloads["primary_worked"].get("affine_regression", {}).get("base_integral") == "106/5",
    "sdist_qualification": payloads["sdist_qualification"].get("passed") is True,
    "sdist_source_digest": payloads["sdist_qualification"].get("source_digest") == source_digest,
    "sdist_documentation_1": (
        payloads["sdist_qualification"].get("suites", {}).get("documentation", {}).get("passed") is True
        and payloads["sdist_qualification"].get("suites", {}).get("documentation", {}).get("checks") == 1
        and payloads["sdist_qualification"].get("suites", {}).get("documentation", {}).get("expected") == 1
    ),
    "sdist_math_15": payloads["sdist_math"].get("success") is True and payloads["sdist_math"].get("total") == 15,
    "sdist_adversarial_30": payloads["sdist_adversarial"].get("passed") is True and payloads["sdist_adversarial"].get("total_count") == 30,
    "sdist_worked_5_4": payloads["sdist_worked"].get("pipeline_is_I4") is True and payloads["sdist_worked"].get("result") == "5/4",
    "sdist_affine_106_5": payloads["sdist_worked"].get("affine_regression", {}).get("base_integral") == "106/5",
}
expected_suites = {
    "expression": 22,
    "integration": 12,
    "block19": 24,
    "block20": 46,
    "stable": 6,
    "block15": 174,
    "documentation": 1,
}
for prefix in ("primary", "sdist"):
    suite_payload = payloads[f"{prefix}_qualification"].get("suites", {})
    for suite_name, expected in expected_suites.items():
        suite = suite_payload.get(suite_name, {})
        checks[f"{prefix}_{suite_name}_{expected}"] = (
            suite.get("passed") is True
            and suite.get("checks") == expected
            and suite.get("expected") == expected
        )
for name, schema in expected_report_schemas.items():
    report = payloads[name]
    checks[f"{name}_metadata"] = (
        report.get("schema") == schema
        and report.get("schema_version") == 1
        and report.get("package_version") == "1.0.0b4"
    )
artifacts = []
for path in (
    root / "release/dist/komorebi_math-1.0.0b4-py3-none-any.whl",
    root / "release/dist/komorebi_math-1.0.0b4.tar.gz",
):
    artifacts.append({"name": path.name, "size": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()})
build_artifacts = {item.get("name"): item for item in distribution_build.get("artifacts", [])}
checks["artifact_inventory_matches_build"] = all(
    build_artifacts.get(item["name"]) == item for item in artifacts
)
evidence_sha256 = {
    path.name: sha256(path.read_bytes()).hexdigest()
    for path in records.values()
}
payload = {
    "schema": "komorebi-math.block20-distribution-validation",
    "schema_version": 1,
    "package_version": "1.0.0b4",
    "passed": all(checks.values()),
    "checks": checks,
    "source_digest": source_digest,
    "distribution_build_sha256": sha256(distribution_build_path.read_bytes()).hexdigest(),
    "evidence_sha256": evidence_sha256,
    "artifacts": artifacts,
}
(root / "release/BLOCK20_DISTRIBUTION_VALIDATION.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["passed"]:
    raise SystemExit(1)
PY
