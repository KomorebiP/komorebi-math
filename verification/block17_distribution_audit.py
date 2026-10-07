"""Inspect and independently rebuild the Block 17 wheel and source distribution."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any


VERSION = "1.0.0b1"
WHEEL_PREFIX = f"komorebi_math-{VERSION}"
SDIST_NAME = f"komorebi_math-{VERSION}.tar.gz"


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _wheel_payload(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/RECORD")
        }


def _wheel_record_valid(path: Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        record_names = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            return False
        rows = list(csv.reader(io.StringIO(archive.read(record_names[0]).decode("utf-8"))))
        return bool(rows) and all(len(row) == 3 for row in rows)


def audit(project: Path, dist: Path, *, rebuild: bool = True) -> dict[str, Any]:
    wheels = sorted(dist.glob(f"{WHEEL_PREFIX}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one Block 17 wheel, found {len(wheels)}")
    wheel = wheels[0]
    sdist = dist / SDIST_NAME
    if not sdist.is_file():
        raise RuntimeError(f"source distribution is missing: {sdist}")

    with zipfile.ZipFile(wheel) as archive:
        bad_zip = archive.testzip()
        wheel_names = archive.namelist()
        wheel_safe = all(_safe(name) for name in wheel_names)
        required_wheel = {
            "komorebi_math/release/contract.py",
            "komorebi_math/release/upgrade.py",
            "komorebi_math/release/evidence.py",
            "komorebi_math/release/documentation.py",
            "komorebi_math/release/readiness.py",
            "komorebi_math/release/gpu.py",
            "komorebi_math/data/release_contract_1_0.json",
            "komorebi_math/data/upgrade_contract_1_0.json",
            "komorebi_math/data/validation_evidence_1_0.json",
            "komorebi_math/data/stable_documentation_1_0.json",
            "komorebi_math/data/public_api_stable_1_0.json",
            "komorebi_math/data/stable_semantics_1_0.json",
            "komorebi_math/data/portable_codec_contract_1_0.json",
        }
        wheel_required_missing = sorted(required_wheel - set(wheel_names))
        wheel_forbidden = sorted(
            name for name in wheel_names if name.startswith(("tests/", "docs/", "verification/", "release/"))
        )
        metadata_name = next(name for name in wheel_names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        version_ok = f"Version: {VERSION}\n" in metadata

    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        sdist_names = [member.name for member in members]
        sdist_safe = all(_safe(name) for name in sdist_names)
        roots = {PurePosixPath(name).parts[0] for name in sdist_names if PurePosixPath(name).parts}
        required_suffixes = {
            "README_BLOCK17.md",
            "Run_Block17_Source_Validation.ps1",
            "Install_And_Verify_Block17.ps1",
            "Run_Block17_GPU_Validation.ps1",
            "Run_Block17_Upgrade_Validation.ps1",
            "Build_Block17_Distributions.ps1",
            "run_block17_source_validation.sh",
            "install_and_verify_block17.sh",
            "run_block17_gpu_validation.sh",
            "run_block17_upgrade_validation.sh",
            "build_block17_distributions.sh",
            "docs/BLOCK_17_RELEASE.md",
            "docs/UPGRADE_CONTRACT.md",
            "docs/BETA_TO_RC.md",
            "docs/STABLE_API.md",
            "constraints/base-py313.txt",
            "environment/windows-cuda126.yml",
            ".github/workflows/ci.yml",
            "tests/test_block17_release_contract.py",
            "tests/test_block17_upgrade_contract.py",
            "tests/test_block17_evidence.py",
            "tests/test_block17_documentation.py",
            "verification/block17_distribution_audit.py",
            "verification/build_block17_distributions.py",
            "verification/run_block17_upgrade_validation.py",
            "release/predecessors/komorebi_math-0.15.0a2-py3-none-any.whl",
            "release/predecessors/komorebi_math-0.16.0a2-py3-none-any.whl",
        }
        present_suffixes = {
            "/".join(PurePosixPath(name).parts[1:])
            for name in sdist_names
            if len(PurePosixPath(name).parts) > 1
        }
        sdist_required_missing = sorted(required_suffixes - present_suffixes)

    rebuild_payload: dict[str, Any] = {"executed": False}
    if rebuild:
        with tempfile.TemporaryDirectory(prefix="komorebi-block17-sdist-") as temporary:
            temporary_path = Path(temporary)
            with tarfile.open(sdist, "r:gz") as archive:
                archive.extractall(temporary_path, filter="data")
            extracted_root = next(path for path in temporary_path.iterdir() if path.is_dir())
            rebuilt_dist = temporary_path / "rebuilt-dist"
            code = (
                "from pathlib import Path; import json; "
                "from verification.build_block17_distributions import build; "
                "p=build(Path('.'), Path(r'" + str(rebuilt_dist) + "'), clean=True); "
                "print(json.dumps(p, sort_keys=True))"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(extracted_root)
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=extracted_root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=240,
                check=False,
            )
            rebuilt_wheels = sorted(rebuilt_dist.glob(f"{WHEEL_PREFIX}-*.whl"))
            payload_equal = False
            differences: list[str] = []
            rebuilt_wheel_hash = None
            if completed.returncode == 0 and len(rebuilt_wheels) == 1:
                rebuilt_wheel = rebuilt_wheels[0]
                rebuilt_wheel_hash = _hash(rebuilt_wheel)
                left = _wheel_payload(wheel)
                right = _wheel_payload(rebuilt_wheel)
                differences = sorted(
                    set(left).symmetric_difference(right)
                    | {name for name in set(left).intersection(right) if left[name] != right[name]}
                )
                payload_equal = not differences
            rebuild_payload = {
                "executed": True,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
                "wheel_count": len(rebuilt_wheels),
                "rebuilt_wheel_sha256": rebuilt_wheel_hash,
                "normalized_payload_equal": payload_equal,
                "differences": differences,
            }

    checks = {
        "wheel_zip_integrity": bad_zip is None,
        "wheel_safe_paths": wheel_safe,
        "wheel_required_payload": not wheel_required_missing,
        "wheel_forbidden_payload": not wheel_forbidden,
        "wheel_metadata_version": version_ok,
        "wheel_record_shape": _wheel_record_valid(wheel),
        "sdist_tar_integrity": True,
        "sdist_safe_paths": sdist_safe,
        "sdist_single_root": len(roots) == 1,
        "sdist_required_payload": not sdist_required_missing,
        "sdist_rebuilt_wheel_payload": (not rebuild) or bool(rebuild_payload["normalized_payload_equal"]),
    }
    payload: dict[str, Any] = {
        "schema": "komorebi-math.distribution-audit",
        "schema_version": 1,
        "package_version": VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "wheel": {
            "name": wheel.name,
            "size": wheel.stat().st_size,
            "sha256": _hash(wheel),
            "members": len(wheel_names),
            "bad_member": bad_zip,
            "required_missing": wheel_required_missing,
            "forbidden": wheel_forbidden,
        },
        "sdist": {
            "name": sdist.name,
            "size": sdist.stat().st_size,
            "sha256": _hash(sdist),
            "members": len(sdist_names),
            "roots": sorted(roots),
            "required_missing": sdist_required_missing,
        },
        "rebuild": rebuild_payload,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dist")
    parser.add_argument("--output")
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args(argv)
    project = Path(args.project).resolve()
    dist = Path(args.dist).resolve() if args.dist else project / "release" / "dist"
    payload = audit(project, dist, rebuild=not args.no_rebuild)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
