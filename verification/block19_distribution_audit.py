"""Audit the frozen Block 19 wheel, sdist, and optional rebuilt wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import tarfile
import zipfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any


VERSION = "1.0.0b3"
WHEEL_NAME = f"komorebi_math-{VERSION}-py3-none-any.whl"
SDIST_NAME = f"komorebi_math-{VERSION}.tar.gz"


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _safe_archive_path(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _wheel_record(archive: zipfile.ZipFile) -> tuple[bool, list[str]]:
    problems: list[str] = []
    record_names = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
    if len(record_names) != 1:
        return False, [f"expected one RECORD, received {len(record_names)}"]
    record_name = record_names[0]
    rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
    row_names: set[str] = set()
    for row in rows:
        if len(row) != 3:
            problems.append(f"malformed RECORD row: {row!r}")
            continue
        name, declared_hash, declared_size = row
        row_names.add(name)
        if name not in archive.namelist():
            problems.append(f"RECORD names a missing member: {name}")
            continue
        if name == record_name:
            if declared_hash or declared_size:
                problems.append("the RECORD self-row must have empty hash and size")
            continue
        payload = archive.read(name)
        if declared_size != str(len(payload)):
            problems.append(f"RECORD size mismatch: {name}")
        expected_hash = "sha256=" + base64.urlsafe_b64encode(sha256(payload).digest()).rstrip(b"=").decode("ascii")
        if declared_hash != expected_hash:
            problems.append(f"RECORD hash mismatch: {name}")
    missing_rows = sorted(set(archive.namelist()) - row_names)
    if missing_rows:
        problems.extend(f"member missing from RECORD: {name}" for name in missing_rows)
    return not problems, problems


def _source_payload(project: Path) -> dict[str, bytes]:
    package = project / "src" / "komorebi_math"
    payload: dict[str, bytes] = {}
    for path in package.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(package)
        included = (
            path.suffix == ".py"
            or relative.as_posix() == "py.typed"
            or (relative.parts[:1] == ("data",) and path.suffix == ".json")
        )
        if included:
            payload[f"komorebi_math/{relative.as_posix()}"] = path.read_bytes()
    return payload


def _wheel_package_payload(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("komorebi_math/") and not name.endswith("/")
        }


def _normalized_wheel_payload(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith(".dist-info/RECORD") and not name.endswith("/")
        }


def audit(project: Path, dist: Path, rebuilt_wheel: Path | None = None) -> dict[str, Any]:
    project = project.resolve()
    dist = dist.resolve()
    wheel = dist / WHEEL_NAME
    sdist = dist / SDIST_NAME
    observed_dist = sorted(path.name for path in dist.iterdir()) if dist.is_dir() else []
    expected_dist = sorted((WHEEL_NAME, SDIST_NAME))

    checks: dict[str, bool] = {
        "dist_contains_only_block19_wheel_and_sdist": observed_dist == expected_dist,
        "wheel_exists": wheel.is_file(),
        "sdist_exists": sdist.is_file(),
    }
    wheel_detail: dict[str, Any] = {}
    sdist_detail: dict[str, Any] = {}

    if wheel.is_file():
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            bad_member = archive.testzip()
            record_valid, record_problems = _wheel_record(archive)
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            metadata = archive.read(metadata_names[0]).decode("utf-8") if len(metadata_names) == 1 else ""
            forbidden = sorted(
                name
                for name in names
                if name.startswith(("build/", "docs/", "examples/", "release/", "tests/", "verification/"))
                or "__pycache__" in PurePosixPath(name).parts
                or name.endswith(".pyc")
            )
            required = {
                "komorebi_math/integration/__init__.py",
                "komorebi_math/integration/adapters.py",
                "komorebi_math/integration/coordinates.py",
                "komorebi_math/integration/identities.py",
                "komorebi_math/integration/measures.py",
                "komorebi_math/integration/polynomial.py",
                "komorebi_math/integration/quadrature.py",
                "komorebi_math/integration/rules.py",
                "komorebi_math/verify/integration.py",
                "komorebi_math/data/public_api_1_0.json",
                "komorebi_math/data/public_api_stable_1_0.json",
                "komorebi_math/data/public_api_beta_1_0.json",
                "komorebi_math/data/portable_codec_contract_1_0.json",
                "komorebi_math/data/stable_documentation_1_0.json",
                "komorebi_math/data/stable_semantics_1_0.json",
            }
            missing_required = sorted(required - set(names))
            source_payload = _source_payload(project)
            wheel_payload = _wheel_package_payload(wheel)
            payload_differences = sorted(
                set(source_payload).symmetric_difference(wheel_payload)
                | {
                    name
                    for name in set(source_payload).intersection(wheel_payload)
                    if source_payload[name] != wheel_payload[name]
                }
            )
            checks.update(
                {
                    "wheel_zip_integrity": bad_member is None,
                    "wheel_safe_paths": all(_safe_archive_path(name) for name in names),
                    "wheel_complete_record": record_valid,
                    "wheel_metadata_version": f"Version: {VERSION}\n" in metadata,
                    "wheel_required_payload": not missing_required,
                    "wheel_forbidden_payload_absent": not forbidden,
                    "wheel_payload_matches_frozen_source": not payload_differences,
                }
            )
            wheel_detail = {
                "name": wheel.name,
                "size": wheel.stat().st_size,
                "sha256": _file_digest(wheel),
                "member_count": len(names),
                "bad_member": bad_member,
                "record_problems": record_problems,
                "missing_required": missing_required,
                "forbidden": forbidden,
                "source_payload_differences": payload_differences,
            }

    if sdist.is_file():
        with tarfile.open(sdist, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            roots = sorted({PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts})
            present = {
                "/".join(PurePosixPath(name).parts[1:])
                for name in names
                if len(PurePosixPath(name).parts) > 1
            }
            required = {
                "README_BLOCK19.md",
                "docs/BLOCK_19_RELEASE.md",
                "docs/BLOCK_19_WORKED_EXAMPLE.md",
                "docs/FINITE_MEASURE_INTEGRATION.md",
                "examples/block19_measure_integration_worked_example.py",
                "tests/test_block19_integration.py",
                "verification/block19_math_verifier.py",
                "verification/block19_adversarial_runner.py",
                "verification/block19_distribution_audit.py",
                "run_block19_source_validation.sh",
                "install_and_verify_block19.sh",
                "build_block19_distributions.sh",
                "Run_Block19_Source_Validation.ps1",
                "Install_And_Verify_Block19.ps1",
                "Build_Block19_Distributions.ps1",
                "release/predecessors/komorebi_math-0.15.0a2-py3-none-any.whl",
                "release/predecessors/komorebi_math-0.16.0a2-py3-none-any.whl",
                "release/predecessors/komorebi_math-1.0.0b1-py3-none-any.whl",
                "release/predecessors/komorebi_math-1.0.0b2-py3-none-any.whl",
            }
            historical_root_audits = {
                "PACKAGE_AUDIT.json",
                "PACKAGE_AUDIT.md",
                "STATIC_AUDIT.json",
                "STATIC_AUDIT.md",
            }
            historical_checksum = "SHA256SUMS.txt"
            forbidden_generated = sorted(
                relative
                for relative in present
                if relative.startswith(("build/", "release/dist/"))
                or "__pycache__" in PurePosixPath(relative).parts
                or relative.endswith(".pyc")
            )
            missing_required = sorted(required - present)
            shipped_historical_audits = sorted(historical_root_audits.intersection(present))
            shipped_historical_checksum = historical_checksum if historical_checksum in present else None
            safe_types = all(member.isfile() or member.isdir() for member in members)
            checks.update(
                {
                    "sdist_safe_paths": all(_safe_archive_path(name) for name in names),
                    "sdist_regular_files_and_directories_only": safe_types,
                    "sdist_single_expected_root": roots == [f"komorebi_math-{VERSION}"],
                    "sdist_required_payload": not missing_required,
                    "sdist_generated_payload_absent": not forbidden_generated,
                    "sdist_historical_generic_audits_absent": not shipped_historical_audits,
                    "sdist_historical_generic_checksum_absent": shipped_historical_checksum is None,
                }
            )
            sdist_detail = {
                "name": sdist.name,
                "size": sdist.stat().st_size,
                "sha256": _file_digest(sdist),
                "member_count": len(names),
                "roots": roots,
                "missing_required": missing_required,
                "forbidden_generated": forbidden_generated,
                "historical_generic_audits": shipped_historical_audits,
                "historical_generic_checksum": shipped_historical_checksum,
            }

    rebuild_detail: dict[str, Any] = {"provided": False}
    if rebuilt_wheel is not None:
        rebuilt_wheel = rebuilt_wheel.resolve()
        differences: list[str] = []
        if wheel.is_file() and rebuilt_wheel.is_file():
            primary = _normalized_wheel_payload(wheel)
            rebuilt = _normalized_wheel_payload(rebuilt_wheel)
            differences = sorted(
                set(primary).symmetric_difference(rebuilt)
                | {
                    name
                    for name in set(primary).intersection(rebuilt)
                    if primary[name] != rebuilt[name]
                }
            )
        checks["fresh_sdist_rebuilt_wheel_exists"] = rebuilt_wheel.is_file()
        checks["fresh_sdist_rebuilt_wheel_payload_matches"] = rebuilt_wheel.is_file() and not differences
        rebuild_detail = {
            "provided": True,
            "name": rebuilt_wheel.name,
            "size": rebuilt_wheel.stat().st_size if rebuilt_wheel.is_file() else None,
            "sha256": _file_digest(rebuilt_wheel) if rebuilt_wheel.is_file() else None,
            "normalized_payload_differences": differences,
        }

    passed = bool(checks) and all(checks.values())
    payload: dict[str, Any] = {
        "schema": "komorebi-math.block19-distribution-audit",
        "schema_version": 1,
        "package_version": VERSION,
        "passed": passed,
        "checks": checks,
        "dist_members": observed_dist,
        "wheel": wheel_detail,
        "sdist": sdist_detail,
        "fresh_sdist_rebuild": rebuild_detail,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["report_sha256"] = _digest(canonical.encode("utf-8"))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--rebuilt-wheel", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    project = arguments.project.resolve()
    dist = arguments.dist.resolve() if arguments.dist else project / "release" / "dist"
    payload = audit(project, dist, arguments.rebuilt_wheel)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(text, end="")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
