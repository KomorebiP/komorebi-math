"""Build and replay-audit the deterministic Block 20 handoff ZIP."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from unicodedata import normalize

VERSION = "1.0.0b4"
ARCHIVE_ROOT = f"Komorebi_Math_Complete_Handoff_{VERSION}"
ARCHIVE_NAME = f"{ARCHIVE_ROOT}.zip"
FIXED_TIMESTAMP = (2026, 8, 21, 0, 0, 0)

_EXCLUDED_DIRECTORIES = {
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".temp",
    ".tmp",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "env",
    "node_modules",
    "temp",
    "temporary",
    "tmp",
    "venv",
    "virtualenv",
}
_EXCLUDED_RELEASE_FILES = {
    "BLOCK20_HANDOFF_BUILD.json",
}
_EXCLUDED_EVIDENCE_FILES = {
    "BLOCK20_HANDOFF_BUILD.json",
    "BLOCK20_HANDOFF_MANIFEST.json",
}
_EXCLUDED_DIRECTORY_PREFIXES = (
    ".temp-",
    ".tmp-",
    ".venv-",
    "temp-",
    "temporary-",
    "tmp-",
    "venv-",
    "virtualenv-",
)
_EXPECTED_SUITES = {
    "expression": 22,
    "integration": 12,
    "block19": 24,
    "block20": 46,
    "stable": 6,
    "block15": 174,
    "documentation": 1,
}


def _digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace one builder-owned regular output without following symlinks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SystemExit(f"refusing unsafe builder output path: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".temporary",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not load required record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"required record is not an object: {path}")
    return value


def _require_suites(record: dict[str, Any], *, label: str) -> None:
    suites = record.get("suites")
    if not isinstance(suites, dict):
        raise SystemExit(f"{label} has no suite inventory")
    for name, expected in _EXPECTED_SUITES.items():
        suite = suites.get(name)
        if not isinstance(suite, dict) or not (
            suite.get("passed") is True
            and suite.get("checks") == expected
            and suite.get("expected") == expected
        ):
            raise SystemExit(f"{label} suite is not exactly {expected}/{expected}: {name}")


def _safe_relative(path: PurePosixPath) -> None:
    text = path.as_posix()
    if (
        path.is_absolute()
        or not path.parts
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
        or normalize("NFC", text) != text
    ):
        raise SystemExit(f"unsafe handoff path: {text!r}")


def _excluded(relative: PurePosixPath) -> bool:
    if any(
        part.casefold() in _EXCLUDED_DIRECTORIES or part.casefold().endswith(".egg-info")
        for part in relative.parts[:-1]
    ):
        return True
    name = relative.name
    if name.casefold() in {".git", ".hg"}:
        return True
    if name.endswith((".pyc", ".pyo")):
        return True
    if name == ARCHIVE_NAME or name == f"{ARCHIVE_NAME}.sha256":
        return True
    if "handoff" in name.casefold() and name.casefold().endswith(
        (".zip", ".tar", ".tar.gz", ".tgz")
    ):
        return True
    if (
        len(relative.parts) >= 2
        and relative.parts[0] == "release"
        and name in _EXCLUDED_RELEASE_FILES
    ):
        return True
    return False


def _excluded_directory_name(name: str) -> bool:
    folded = name.casefold()
    return (
        folded in _EXCLUDED_DIRECTORIES
        or folded.endswith(".egg-info")
        or folded.startswith(_EXCLUDED_DIRECTORY_PREFIXES)
    )


def _collect(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    folded: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            child = directory_path / name
            relative = PurePosixPath(child.relative_to(root).as_posix())
            if name.casefold() == "dist" and relative.as_posix() != "release/dist":
                continue
            if _excluded_directory_name(name) or _excluded(relative):
                continue
            if child.is_symlink():
                raise SystemExit(f"handoff source contains a symlink: {relative}")
            mode = child.stat(follow_symlinks=False).st_mode
            if not stat.S_ISDIR(mode):
                raise SystemExit(f"handoff source contains a special directory entry: {relative}")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = directory_path / name
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if _excluded(relative):
                continue
            _safe_relative(relative)
            if path.is_symlink():
                raise SystemExit(f"handoff source contains a symlink: {relative}")
            mode = path.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                raise SystemExit(f"handoff source contains a special file: {relative}")
            key = normalize("NFC", relative.as_posix()).casefold()
            if key in folded and folded[key] != relative.as_posix():
                raise SystemExit(
                    f"casefold-colliding handoff paths: {folded[key]!r}, {relative.as_posix()!r}"
                )
            folded[key] = relative.as_posix()
            files.append(path)
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def _fresh_source_audit(root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="komorebi-block20-handoff-audit-") as temporary:
        output = Path(temporary) / "source-audit.json"
        process = subprocess.run(
            (
                sys.executable,
                str(root / "verification/block20_source_audit.py"),
                "--root",
                str(root),
                "--quiet",
                "--output",
                str(output),
            ),
            cwd=root,
            check=False,
        )
        if process.returncode != 0:
            raise SystemExit("fresh source audit failed before handoff construction")
        return _load(output)


def _requirements(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    required = {
        "source_qualification": "BLOCK20_SOURCE_QUALIFICATION.json",
        "source_audit": "BLOCK20_SOURCE_AUDIT.json",
        "math": "BLOCK20_MATH_VERIFICATION.json",
        "adversarial": "BLOCK20_ADVERSARIAL_TESTS.json",
        "worked": "BLOCK20_WORKED_EXAMPLE.json",
        "distribution_build": "BLOCK20_DISTRIBUTION_BUILD.json",
        "distribution_audit": "BLOCK20_DISTRIBUTION_AUDIT.json",
        "distribution_validation": "BLOCK20_DISTRIBUTION_VALIDATION.json",
        "installed": "BLOCK20_INSTALLED_QUALIFICATION.json",
        "sdist_installed": "BLOCK20_SDIST_INSTALLED_QUALIFICATION.json",
        "pytest": "BLOCK20_PYTEST.json",
    }
    records = {key: _load(root / "release" / name) for key, name in required.items()}
    expected_schemas = {
        "source_qualification": "komorebi-math.block20-source-qualification",
        "source_audit": "komorebi-math.block20-source-audit",
        "math": "komorebi-math.block20-independent-math-verification",
        "adversarial": "komorebi-math.block20-adversarial-tests",
        "worked": "komorebi-math.block20-worked-example",
        "distribution_build": "komorebi-math.block20-distribution-build",
        "distribution_audit": "komorebi-math.block20-distribution-audit",
        "distribution_validation": "komorebi-math.block20-distribution-validation",
        "installed": "komorebi-math.block20-installed-qualification",
        "sdist_installed": "komorebi-math.block20-installed-qualification",
        "pytest": "komorebi-math.block20-focused-pytest",
    }
    for key, expected_schema in expected_schemas.items():
        if records[key].get("schema") != expected_schema:
            raise SystemExit(f"required Block 20 record has the wrong schema: {required[key]}")
        if records[key].get("schema_version") != 1:
            raise SystemExit(f"required Block 20 record has the wrong schema version: {required[key]}")
        if records[key].get("package_version") != VERSION:
            raise SystemExit(f"required Block 20 record has the wrong package version: {required[key]}")
    for key in (
        "source_qualification",
        "source_audit",
        "distribution_build",
        "distribution_audit",
        "distribution_validation",
        "installed",
        "sdist_installed",
        "pytest",
    ):
        if records[key].get("passed") is not True:
            raise SystemExit(f"required Block 20 record did not pass: {required[key]}")
    if records["math"].get("success") is not True or records["math"].get("total") != 15:
        raise SystemExit("independent mathematical record is not 15/15")
    if (
        records["adversarial"].get("passed") is not True
        or records["adversarial"].get("total_count") != 30
    ):
        raise SystemExit("adversarial record is not 30/30")
    if (
        records["pytest"].get("passed") is not True
        or records["pytest"].get("collected_count") != 32
        or records["pytest"].get("passed_count") != 32
        or records["pytest"].get("failed_count") != 0
        or records["pytest"].get("skipped_count") != 0
    ):
        raise SystemExit("focused pytest record is not exactly 32/32")
    if (
        records["worked"].get("pipeline_is_I4") is not True
        or records["worked"].get("result") != "5/4"
        or records["worked"].get("affine_regression", {}).get("base_integral") != "106/5"
    ):
        raise SystemExit("worked record does not contain I4, 5/4, and 106/5")
    _require_suites(records["source_qualification"], label="source qualification")
    fresh = _fresh_source_audit(root)
    qualified_digest = records["source_qualification"].get("source_digest")
    if (
        fresh.get("passed") is not True
        or not qualified_digest
        or fresh.get("source_digest") != qualified_digest
        or records["source_audit"].get("source_digest") != qualified_digest
        or records["distribution_build"].get("source_digest") != qualified_digest
    ):
        raise SystemExit("handoff source digest does not match qualification and build records")

    source_evidence_paths = (
        root / "release/BLOCK20_MATH_VERIFICATION.json",
        root / "release/BLOCK20_ADVERSARIAL_TESTS.json",
        root / "release/BLOCK20_PYTEST.json",
        root / "release/BLOCK20_WORKED_EXAMPLE.json",
        root / "release/BLOCK20_SOURCE_AUDIT.json",
    )
    source_evidence = {
        path.name: _digest(path) for path in source_evidence_paths
    }
    if records["source_qualification"].get("evidence_sha256") != source_evidence:
        raise SystemExit("source qualification evidence hashes failed replay")
    if records["distribution_build"].get("byte_reproducible_rebuild") is not True:
        raise SystemExit("distribution build lacks its byte-reproducibility replay")

    distribution_validation = records["distribution_validation"]
    build_path = root / "release/BLOCK20_DISTRIBUTION_BUILD.json"
    if (
        distribution_validation.get("source_digest") != qualified_digest
        or distribution_validation.get("distribution_build_sha256") != _digest(build_path)
    ):
        raise SystemExit("distribution validation is not bound to source and build records")
    validation_evidence_names = {
        "BLOCK20_DISTRIBUTION_AUDIT.json",
        "BLOCK20_INSTALLED_QUALIFICATION.json",
        "BLOCK20_INSTALLED_MATH_VERIFICATION.json",
        "BLOCK20_INSTALLED_ADVERSARIAL_TESTS.json",
        "BLOCK20_INSTALLED_WORKED_EXAMPLE.json",
        "BLOCK20_SDIST_INSTALLED_QUALIFICATION.json",
        "BLOCK20_SDIST_INSTALLED_MATH_VERIFICATION.json",
        "BLOCK20_SDIST_INSTALLED_ADVERSARIAL_TESTS.json",
        "BLOCK20_SDIST_INSTALLED_WORKED_EXAMPLE.json",
    }
    validation_evidence = {
        name: _digest(root / "release" / name)
        for name in sorted(validation_evidence_names)
    }
    if distribution_validation.get("evidence_sha256") != validation_evidence:
        raise SystemExit("distribution validation evidence hashes failed replay")
    installed_report_schemas = {
        "BLOCK20_INSTALLED_MATH_VERIFICATION.json": (
            "komorebi-math.block20-independent-math-verification",
            "success",
            "total",
            15,
        ),
        "BLOCK20_INSTALLED_ADVERSARIAL_TESTS.json": (
            "komorebi-math.block20-adversarial-tests",
            "passed",
            "total_count",
            30,
        ),
        "BLOCK20_INSTALLED_WORKED_EXAMPLE.json": (
            "komorebi-math.block20-worked-example",
            "pipeline_is_I4",
            "result",
            "5/4",
        ),
        "BLOCK20_SDIST_INSTALLED_MATH_VERIFICATION.json": (
            "komorebi-math.block20-independent-math-verification",
            "success",
            "total",
            15,
        ),
        "BLOCK20_SDIST_INSTALLED_ADVERSARIAL_TESTS.json": (
            "komorebi-math.block20-adversarial-tests",
            "passed",
            "total_count",
            30,
        ),
        "BLOCK20_SDIST_INSTALLED_WORKED_EXAMPLE.json": (
            "komorebi-math.block20-worked-example",
            "pipeline_is_I4",
            "result",
            "5/4",
        ),
    }
    for name, (schema, flag, field, expected) in installed_report_schemas.items():
        report = _load(root / "release" / name)
        if not (
            report.get("schema") == schema
            and report.get("schema_version") == 1
            and report.get("package_version") == VERSION
            and report.get(flag) is True
            and report.get(field) == expected
        ):
            raise SystemExit(f"installed subordinate evidence is invalid: {name}")
    for key in ("installed", "sdist_installed"):
        if records[key].get("source_digest") != qualified_digest:
            raise SystemExit(f"{key} evidence is not bound to the qualified source")
        _require_suites(records[key], label=key)
    return fresh, records


def _write_manifest(
    root: Path,
    fresh: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> Path:
    dist = root / "release" / "dist"
    expected_dist_names = {
        f"komorebi_math-{VERSION}-py3-none-any.whl",
        f"komorebi_math-{VERSION}.tar.gz",
    }
    try:
        dist_entries = tuple(dist.iterdir())
    except OSError as exc:
        raise SystemExit(f"could not enumerate release/dist: {exc}") from exc
    if {path.name for path in dist_entries} != expected_dist_names or any(
        path.is_symlink() or not path.is_file() for path in dist_entries
    ):
        raise SystemExit(
            "release/dist must contain exactly the regular Block 20 wheel and sdist"
        )
    artifacts = []
    for name in (
        f"komorebi_math-{VERSION}-py3-none-any.whl",
        f"komorebi_math-{VERSION}.tar.gz",
    ):
        path = dist / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"handoff distribution is missing: {path}")
        artifacts.append(
            {"name": name, "size": path.stat().st_size, "sha256": _digest(path)}
        )
    build_artifacts = {
        item.get("name"): item for item in records["distribution_build"].get("artifacts", [])
    }
    if any(build_artifacts.get(item["name"]) != item for item in artifacts):
        raise SystemExit("handoff distribution artifacts differ from the build inventory")
    if records["distribution_validation"].get("artifacts") != artifacts:
        raise SystemExit("handoff distribution artifacts differ from validation inventory")
    wheel_item = next(
        item for item in artifacts if item["name"].endswith(".whl")
    )
    sdist_item = next(
        item for item in artifacts if item["name"].endswith(".tar.gz")
    )
    if records["installed"].get("tested_artifact") != wheel_item:
        raise SystemExit("primary installed evidence is not bound to the release wheel")
    audit = records["distribution_audit"]
    if (
        audit.get("wheel", {}).get("sha256") != wheel_item["sha256"]
        or audit.get("wheel", {}).get("size") != wheel_item["size"]
        or audit.get("sdist", {}).get("sha256") != sdist_item["sha256"]
        or audit.get("sdist", {}).get("size") != sdist_item["size"]
    ):
        raise SystemExit("distribution audit is not bound to the release artifacts")
    rebuilt_item = records["sdist_installed"].get("tested_artifact")
    rebuild_audit = audit.get("fresh_sdist_rebuild", {})
    if not isinstance(rebuilt_item, dict) or (
        rebuilt_item.get("name") != f"komorebi_math-{VERSION}-py3-none-any.whl"
        or rebuilt_item.get("sha256") != rebuild_audit.get("sha256")
        or rebuilt_item.get("size") != rebuild_audit.get("size")
    ):
        raise SystemExit("sdist installed evidence is not bound to the audited rebuilt wheel")

    evidence_names = sorted(
        path.name
        for path in (root / "release").glob("BLOCK20_*.json")
        if path.name not in _EXCLUDED_EVIDENCE_FILES
    )
    evidence = [
        {
            "name": name,
            "size": (root / "release" / name).stat().st_size,
            "sha256": _digest(root / "release" / name),
        }
        for name in evidence_names
    ]
    manifest_path = root / "release" / "BLOCK20_HANDOFF_MANIFEST.json"
    payload = {
        "schema": "komorebi-math.block20-handoff-manifest",
        "schema_version": 1,
        "package_version": VERSION,
        "archive_root": ARCHIVE_ROOT,
        "source_digest": fresh["source_digest"],
        "source_file_count": fresh["source_file_count"],
        "stable_symbols": 131,
        "beta_symbols": 83,
        "portable_codecs": 51,
        "stable_semantic_cases": 10,
        "expression_checks": 22,
        "block20_checks": 46,
        "independent_math_checks": 15,
        "adversarial_cases": 30,
        "worked_results": {"pipeline": "I4", "integral": "5/4", "affine": "106/5"},
        "artifacts": artifacts,
        "evidence": evidence,
        "platform_claims": {
            "source_validation_current_platform": True,
            "fresh_venv_wheel_current_platform_with_inherited_dependencies": True,
            "fresh_sdist_rebuilt_wheel_current_platform_with_inherited_dependencies": True,
            "dependency_isolated_current_platform": False,
            "operating_system_attested_in_evidence": False,
            "pytest_focused": True,
            "windows_current": False,
            "cuda_current": False,
            "mkdocs_strict_current": False,
            "authenticated_git_tag": False,
        },
    }
    _atomic_write(
        manifest_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest_path


def _write_zip(root: Path, destination: Path, files: tuple[Path, ...]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.temporary")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for path in files:
            relative = PurePosixPath(path.relative_to(root).as_posix())
            member = f"{ARCHIVE_ROOT}/{relative.as_posix()}"
            info = zipfile.ZipInfo(member, FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            executable = path.suffix == ".sh"
            mode = 0o755 if executable else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.flag_bits |= 0x800
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.replace(destination)


def _audit_zip(root: Path, destination: Path, files: tuple[Path, ...]) -> dict[str, Any]:
    expected = {
        f"{ARCHIVE_ROOT}/{path.relative_to(root).as_posix()}": path for path in files
    }
    with zipfile.ZipFile(destination, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        normalized: set[str] = set()
        folded: set[str] = set()
        for info in infos:
            path = PurePosixPath(info.filename)
            _safe_relative(path)
            if not path.parts or path.parts[0] != ARCHIVE_ROOT:
                raise SystemExit(f"handoff member is outside the single root: {info.filename}")
            if info.filename in normalized or info.filename.casefold() in folded:
                raise SystemExit(f"duplicate or case-colliding handoff member: {info.filename}")
            normalized.add(info.filename)
            folded.add(info.filename.casefold())
            mode = (info.external_attr >> 16) & 0o170000
            if mode != stat.S_IFREG or info.flag_bits & 0x1:
                raise SystemExit(f"handoff member is not a regular unencrypted file: {info.filename}")
            if info.date_time != FIXED_TIMESTAMP:
                raise SystemExit(f"handoff member timestamp is not deterministic: {info.filename}")
        if set(names) != set(expected) or len(names) != len(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise SystemExit(f"handoff inventory mismatch; missing={missing}, extra={extra}")
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise SystemExit(f"handoff CRC failure: {bad_crc}")
        for name, source in expected.items():
            if archive.read(name) != source.read_bytes():
                raise SystemExit(f"handoff byte mismatch: {name}")
    return {
        "member_count": len(expected),
        "single_root": ARCHIVE_ROOT,
        "safe_paths": True,
        "casefold_unique": True,
        "crc_valid": True,
        "source_bytes_equal": True,
        "fixed_timestamp": list(FIXED_TIMESTAMP),
    }


def build(root: Path, output_directory: Path) -> dict[str, Any]:
    root = root.resolve()
    output_directory = output_directory.resolve()
    if not root.is_dir():
        raise SystemExit(f"project root is missing: {root}")
    fresh, records = _requirements(root)
    manifest_path = _write_manifest(root, fresh, records)
    manifest_snapshot = manifest_path.read_bytes()
    files = _collect(root)
    destination = output_directory / ARCHIVE_NAME
    _write_zip(root, destination, files)
    archive_audit = _audit_zip(root, destination, files)
    # Evidence JSON and release artifacts are intentionally outside the source
    # digest.  Replay their entire linkage after ZIP construction, regenerate
    # the deterministic manifest, and require byte identity with the manifest
    # that was actually archived.
    post_evidence_source, post_evidence_records = _requirements(root)
    post_manifest = _write_manifest(root, post_evidence_source, post_evidence_records)
    if post_manifest.read_bytes() != manifest_snapshot:
        raise SystemExit("Block 20 evidence or distribution changed during handoff construction")
    post_build_source = _fresh_source_audit(root)
    if (
        post_build_source.get("passed") is not True
        or post_build_source.get("source_digest") != fresh.get("source_digest")
    ):
        raise SystemExit("source changed during Block 20 handoff construction")
    archive_digest = _digest(destination)
    checksum_path = output_directory / f"{ARCHIVE_NAME}.sha256"
    _atomic_write(
        checksum_path,
        f"{archive_digest}  {ARCHIVE_NAME}\n".encode("ascii"),
    )
    payload = {
        "schema": "komorebi-math.block20-handoff-build",
        "schema_version": 1,
        "package_version": VERSION,
        "passed": True,
        "source_digest": fresh["source_digest"],
        "archive": {
            "path": str(destination),
            "name": destination.name,
            "size": destination.stat().st_size,
            "sha256": archive_digest,
        },
        "checksum_file": str(checksum_path),
        "audit": archive_audit,
        "post_build_source_digest": post_build_source["source_digest"],
    }
    _atomic_write(
        root / "release" / "BLOCK20_HANDOFF_BUILD.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project root",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="directory for the ZIP and checksum",
    )
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)
    report = build(arguments.root, arguments.output_directory)
    if not arguments.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
