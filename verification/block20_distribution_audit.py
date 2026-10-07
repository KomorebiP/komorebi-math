"""Strict, standalone distribution audit for Komorebi Math Block 20.

The audit deliberately uses only the Python standard library and never imports
the package under inspection.  Archive members are read, not extracted.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import stat
import tarfile
import unicodedata
import zipfile
from collections.abc import Sequence
from email import policy
from email.parser import BytesParser
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

VERSION = "1.0.0b4"
VERSIONED_DATA_SUFFIX = "1_0_0b4"
CANONICAL_PROJECT_NAME = "komorebi-math"
WHEEL_NAME = f"komorebi_math-{VERSION}-py3-none-any.whl"
SDIST_NAME = f"komorebi_math-{VERSION}.tar.gz"
SDIST_ROOT = f"komorebi_math-{VERSION}"
DIST_INFO_ROOT = f"komorebi_math-{VERSION}.dist-info"

PREDECESSOR_VERSION = "1.0.0b3"
PREDECESSOR_WHEEL_NAME = (
    f"komorebi_math-{PREDECESSOR_VERSION}-py3-none-any.whl"
)
PREDECESSOR_SDIST_NAME = f"komorebi_math-{PREDECESSOR_VERSION}.tar.gz"


WHEEL_REQUIRED_FILES = frozenset(
    {
        "komorebi_math/expression/__init__.py",
        "komorebi_math/expression/certification.py",
        "komorebi_math/expression/engine.py",
        "komorebi_math/expression/model.py",
        "komorebi_math/expression/planning.py",
        "komorebi_math/expression/polynomial.py",
        "komorebi_math/expression/portable.py",
        "komorebi_math/verify/expression.py",
        "komorebi_math/data/public_api_1_0.json",
        "komorebi_math/data/public_api_stable_1_0.json",
        "komorebi_math/data/public_api_beta_1_0.json",
        "komorebi_math/data/portable_codec_contract_1_0.json",
        "komorebi_math/data/stable_semantics_1_0.json",
        "komorebi_math/data/stable_documentation_1_0.json",
        f"komorebi_math/data/public_api_{VERSIONED_DATA_SUFFIX}.json",
        f"komorebi_math/data/public_api_stable_{VERSIONED_DATA_SUFFIX}.json",
        f"komorebi_math/data/public_api_beta_{VERSIONED_DATA_SUFFIX}.json",
        f"komorebi_math/data/portable_codec_contract_{VERSIONED_DATA_SUFFIX}.json",
        f"komorebi_math/data/stable_semantics_{VERSIONED_DATA_SUFFIX}.json",
        f"komorebi_math/data/stable_documentation_{VERSIONED_DATA_SUFFIX}.json",
    }
)


SDIST_REQUIRED_FILES = frozenset(
    {
        "pyproject.toml",
        "MANIFEST.in",
        "README_BLOCK20.md",
        "docs/TYPED_FINITE_EXPRESSION_CALCULUS.md",
        "docs/BLOCK_20_WORKED_EXAMPLE.md",
        "docs/BLOCK_20_RELEASE.md",
        "examples/block20_typed_expression_worked_example.py",
        "tests/test_block20_expression_calculus.py",
        "verification/block20_math_verifier.py",
        "verification/block20_adversarial_runner.py",
        "verification/block20_source_audit.py",
        "verification/block20_distribution_audit.py",
        "verification/canonicalize_block20_sdist.py",
        "verification/build_block20_handoff.py",
        "run_block20_source_validation.sh",
        "install_and_verify_block20.sh",
        "build_block20_distributions.sh",
        "Run_Block20_Source_Validation.ps1",
        "Install_And_Verify_Block20.ps1",
        "Build_Block20_Distributions.ps1",
        "src/komorebi_math/expression/__init__.py",
        "src/komorebi_math/expression/certification.py",
        "src/komorebi_math/expression/engine.py",
        "src/komorebi_math/expression/model.py",
        "src/komorebi_math/expression/planning.py",
        "src/komorebi_math/expression/polynomial.py",
        "src/komorebi_math/expression/portable.py",
        "src/komorebi_math/verify/expression.py",
        "src/komorebi_math/data/public_api_1_0.json",
        "src/komorebi_math/data/public_api_stable_1_0.json",
        "src/komorebi_math/data/public_api_beta_1_0.json",
        "src/komorebi_math/data/portable_codec_contract_1_0.json",
        "src/komorebi_math/data/stable_semantics_1_0.json",
        "src/komorebi_math/data/stable_documentation_1_0.json",
        f"src/komorebi_math/data/public_api_{VERSIONED_DATA_SUFFIX}.json",
        f"src/komorebi_math/data/public_api_stable_{VERSIONED_DATA_SUFFIX}.json",
        f"src/komorebi_math/data/public_api_beta_{VERSIONED_DATA_SUFFIX}.json",
        f"src/komorebi_math/data/portable_codec_contract_{VERSIONED_DATA_SUFFIX}.json",
        f"src/komorebi_math/data/stable_semantics_{VERSIONED_DATA_SUFFIX}.json",
        f"src/komorebi_math/data/stable_documentation_{VERSIONED_DATA_SUFFIX}.json",
        f"release/predecessors/{PREDECESSOR_WHEEL_NAME}",
        f"release/predecessors/{PREDECESSOR_SDIST_NAME}",
    }
)


BASE_CHECK_NAMES = (
    "project_root_exists",
    "release_dist_exists",
    "release_dist_contains_only_b4_wheel_and_sdist",
    "b3_predecessor_wheel_present",
    "b3_predecessor_sdist_present",
    "b3_predecessors_absent_from_dist",
    "wheel_exists",
    "wheel_archive_readable",
    "wheel_zip_integrity",
    "wheel_safe_member_paths",
    "wheel_unique_normalized_names",
    "wheel_no_casefold_collisions",
    "wheel_regular_unencrypted_members_only",
    "wheel_metadata_unique",
    "wheel_metadata_name",
    "wheel_metadata_version",
    "wheel_expected_dist_info_root",
    "wheel_record_unique",
    "wheel_record_complete",
    "wheel_record_sha256_and_sizes_valid",
    "wheel_required_block20_files",
    "wheel_generated_payload_absent",
    "source_package_payload_readable",
    "wheel_package_payload_matches_source",
    "sdist_exists",
    "sdist_is_gzip",
    "sdist_archive_readable",
    "sdist_archive_integrity",
    "sdist_safe_member_paths",
    "sdist_unique_normalized_names",
    "sdist_no_casefold_collisions",
    "sdist_regular_files_and_directories_only",
    "sdist_single_expected_root",
    "sdist_metadata_unique",
    "sdist_metadata_name",
    "sdist_metadata_version",
    "sdist_required_block20_files",
    "sdist_generated_payload_absent",
)


_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_NAME_NORMALIZER = re.compile(r"[-_.]+")


class ArchivePathError(ValueError):
    """Raised when an archive member name is unsafe or ambiguous."""


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_facts(path: Path, exists: bool) -> dict[str, Any]:
    if not exists:
        return {"size": None, "sha256": None, "error": "not a regular file"}
    try:
        size = path.stat().st_size
        digest = _file_digest(path)
    except OSError as exc:
        return {
            "size": None,
            "sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"size": size, "sha256": digest, "error": None}


def _canonical_project_name(value: str) -> str:
    return _NAME_NORMALIZER.sub("-", value).lower()


def _normalize_archive_name(name: str) -> str:
    """Return one canonical POSIX member name or reject it.

    A terminal slash is accepted for a directory but does not participate in
    duplicate detection.  No other empty, dot, or dot-dot component is
    accepted.  NFC normalization makes canonically equivalent Unicode names
    collide instead of becoming platform-dependent extraction aliases.
    """

    if not isinstance(name, str):
        raise ArchivePathError("member name is not text")
    if not name:
        raise ArchivePathError("empty member name")
    if "\x00" in name:
        raise ArchivePathError("NUL in member name")
    if "\\" in name:
        raise ArchivePathError("backslash in member name")
    if name.startswith("/") or PurePosixPath(name).is_absolute():
        raise ArchivePathError("absolute member name")
    if _DRIVE_PREFIX.match(name):
        raise ArchivePathError("drive-qualified member name")

    core = name[:-1] if name.endswith("/") else name
    if not core:
        raise ArchivePathError("archive-root member name")
    raw_parts = core.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ArchivePathError("empty, dot, or traversal component")
    normalized_parts = [unicodedata.normalize("NFC", part) for part in raw_parts]
    if any(not part or part in {".", ".."} for part in normalized_parts):
        raise ArchivePathError("invalid normalized component")
    normalized = "/".join(normalized_parts)
    if _DRIVE_PREFIX.match(normalized):
        raise ArchivePathError("normalized drive-qualified member name")
    return normalized


def _name_audit(names: Sequence[str]) -> dict[str, Any]:
    normalized: list[str | None] = []
    unsafe: list[str] = []
    by_normalized: dict[str, list[str]] = {}
    for name in names:
        try:
            canonical = _normalize_archive_name(name)
        except (ArchivePathError, ValueError) as exc:
            normalized.append(None)
            unsafe.append(f"{name!r}: {exc}")
            continue
        normalized.append(canonical)
        by_normalized.setdefault(canonical, []).append(name)

    duplicates = {
        canonical: originals
        for canonical, originals in by_normalized.items()
        if len(originals) > 1
    }
    casefold_groups: dict[str, list[str]] = {}
    for canonical in by_normalized:
        casefold_groups.setdefault(canonical.casefold(), []).append(canonical)
    casefold_collisions = {
        folded: sorted(values)
        for folded, values in casefold_groups.items()
        if len(values) > 1
    }
    return {
        "normalized": normalized,
        "unsafe": unsafe,
        "duplicates": duplicates,
        "casefold_collisions": casefold_collisions,
        "safe": not unsafe,
        "unique": not duplicates,
        "casefold_unique": not casefold_collisions,
    }


def _metadata_values(payload: bytes) -> tuple[str | None, str | None, list[str]]:
    issues: list[str] = []
    try:
        message = BytesParser(policy=policy.compat32).parsebytes(payload)
    except Exception as exc:  # malformed email-style metadata
        return None, None, [f"metadata parse failed: {type(exc).__name__}: {exc}"]
    names = message.get_all("Name", [])
    versions = message.get_all("Version", [])
    if len(names) != 1:
        issues.append(f"expected one Name field, received {len(names)}")
    if len(versions) != 1:
        issues.append(f"expected one Version field, received {len(versions)}")
    name = str(names[0]).strip() if len(names) == 1 else None
    version = str(versions[0]).strip() if len(versions) == 1 else None
    return name, version, issues


def _record_audit(
    record_name: str,
    record_payload: bytes,
    members: dict[str, tuple[str, bytes]],
) -> dict[str, Any]:
    structural_issues: list[str] = []
    digest_issues: list[str] = []
    row_names: dict[str, str] = {}
    try:
        text = record_payload.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        return {
            "complete": False,
            "hashes_and_sizes_valid": False,
            "structural_issues": [f"RECORD parse failed: {type(exc).__name__}: {exc}"],
            "digest_issues": [],
        }

    for index, row in enumerate(rows, start=1):
        if len(row) != 3:
            structural_issues.append(
                f"RECORD row {index} has {len(row)} fields instead of 3"
            )
            continue
        raw_name, declared_hash, declared_size = row
        try:
            canonical = _normalize_archive_name(raw_name)
        except ArchivePathError as exc:
            structural_issues.append(f"unsafe RECORD row {index} path: {exc}")
            continue
        if canonical in row_names:
            structural_issues.append(
                f"duplicate normalized RECORD path: {canonical!r}"
            )
            continue
        row_names[canonical] = raw_name
        member = members.get(canonical)
        if member is None:
            structural_issues.append(f"RECORD names a missing member: {raw_name!r}")
            continue
        archive_name, payload = member
        if raw_name != archive_name:
            structural_issues.append(
                f"RECORD spelling differs from archive member: {raw_name!r} != {archive_name!r}"
            )
        if canonical == record_name:
            if declared_hash or declared_size:
                digest_issues.append(
                    "the RECORD self-row must have empty hash and size"
                )
            continue
        expected_hash = "sha256=" + base64.urlsafe_b64encode(
            sha256(payload).digest()
        ).rstrip(b"=").decode("ascii")
        if declared_hash != expected_hash:
            digest_issues.append(f"RECORD SHA256 mismatch: {archive_name}")
        if declared_size != str(len(payload)):
            digest_issues.append(f"RECORD size mismatch: {archive_name}")

    missing = sorted(set(members) - set(row_names))
    for name in missing:
        structural_issues.append(f"archive member missing from RECORD: {name}")
    extra = sorted(set(row_names) - set(members))
    for name in extra:
        structural_issues.append(f"RECORD row has no archive member: {name}")
    if record_name not in row_names:
        structural_issues.append("RECORD has no self-row")
    complete = not structural_issues
    return {
        "complete": complete,
        "hashes_and_sizes_valid": complete and not digest_issues,
        "structural_issues": structural_issues,
        "digest_issues": digest_issues,
    }


def _zip_member_is_regular_or_directory(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        return False
    if file_type == stat.S_IFDIR and not info.is_dir():
        return False
    if file_type == stat.S_IFREG and info.is_dir():
        return False
    return True


def _inspect_wheel(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "readable": False,
        "integrity": False,
        "safe_paths": False,
        "unique_names": False,
        "casefold_unique": False,
        "regular_unencrypted": False,
        "metadata_unique": False,
        "metadata_name": None,
        "metadata_version": None,
        "metadata_issues": [],
        "dist_info_roots": [],
        "record_unique": False,
        "record": {
            "complete": False,
            "hashes_and_sizes_valid": False,
            "structural_issues": [],
            "digest_issues": [],
        },
        "names": [],
        "file_names": [],
        "normalized_names": [],
        "unsafe_names": [],
        "duplicate_names": {},
        "casefold_collisions": {},
        "member_type_issues": [],
        "read_issues": [],
        "bad_crc_member": None,
        "payload": {},
        "package_payload": {},
        "normalized_metadata_payload": {},
        "exception": None,
    }
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            infos = archive.infolist()
            # ZipInfo preserves the untruncated central-directory spelling in
            # ``orig_filename``.  Auditing only ``filename`` would miss a NUL
            # suffix because zipfile truncates that convenience attribute.
            names = [info.orig_filename for info in infos]
            result["names"] = names
            name_audit = _name_audit(names)
            result["normalized_names"] = name_audit["normalized"]
            result["unsafe_names"] = name_audit["unsafe"]
            result["duplicate_names"] = name_audit["duplicates"]
            result["casefold_collisions"] = name_audit["casefold_collisions"]
            result["safe_paths"] = name_audit["safe"]
            result["unique_names"] = name_audit["unique"]
            result["casefold_unique"] = name_audit["casefold_unique"]

            type_issues: list[str] = []
            for info in infos:
                if info.flag_bits & 0x1:
                    type_issues.append(f"encrypted ZIP member: {info.filename!r}")
                if not _zip_member_is_regular_or_directory(info):
                    type_issues.append(
                        f"nonregular ZIP member: {info.filename!r} "
                        f"mode={((info.external_attr >> 16) & 0xFFFF):#o}"
                    )
            result["member_type_issues"] = type_issues
            result["regular_unencrypted"] = not type_issues

            read_issues: list[str] = []
            entry_payloads: list[bytes | None] = []
            try:
                result["bad_crc_member"] = archive.testzip()
            except Exception as exc:
                result["bad_crc_member"] = f"{type(exc).__name__}: {exc}"
            for info in infos:
                try:
                    entry_payloads.append(archive.read(info))
                except Exception as exc:
                    entry_payloads.append(None)
                    read_issues.append(
                        f"{info.filename!r}: {type(exc).__name__}: {exc}"
                    )
            result["read_issues"] = read_issues
            result["integrity"] = (
                result["bad_crc_member"] is None and not read_issues
            )
            result["readable"] = True

            payload: dict[str, tuple[str, bytes]] = {}
            file_names: set[str] = set()
            if name_audit["safe"] and name_audit["unique"]:
                for info, canonical, data in zip(
                    infos, name_audit["normalized"], entry_payloads, strict=True
                ):
                    if canonical is not None and data is not None:
                        payload[canonical] = (info.orig_filename, data)
                        if not info.is_dir():
                            file_names.add(canonical)
            result["payload"] = payload
            result["file_names"] = sorted(file_names)

            metadata_members = [
                name
                for name in file_names
                if name.endswith(".dist-info/METADATA")
            ]
            result["metadata_unique"] = len(metadata_members) == 1
            if len(metadata_members) == 1:
                metadata_name = metadata_members[0]
                name, version, issues = _metadata_values(
                    payload[metadata_name][1]
                )
                result["metadata_name"] = name
                result["metadata_version"] = version
                result["metadata_issues"] = issues

            dist_info_roots = sorted(
                {
                    name.split("/", 1)[0]
                    for name in payload
                    if "/" in name and name.split("/", 1)[0].endswith(".dist-info")
                }
            )
            result["dist_info_roots"] = dist_info_roots

            record_members = [
                name for name in file_names if name.endswith(".dist-info/RECORD")
            ]
            result["record_unique"] = len(record_members) == 1
            if len(record_members) == 1:
                record_name = record_members[0]
                if name_audit["safe"] and name_audit["unique"]:
                    result["record"] = _record_audit(
                        record_name, payload[record_name][1], payload
                    )

            package_payload: dict[str, bytes] = {}
            normalized_metadata: dict[str, bytes] = {}
            for canonical, (raw_name, data) in payload.items():
                if canonical.startswith("komorebi_math/") and not raw_name.endswith("/"):
                    package_payload[canonical] = data
                    continue
                first, separator, rest = canonical.partition("/")
                if (
                    separator
                    and first.endswith(".dist-info")
                    and rest != "RECORD"
                    and not raw_name.endswith("/")
                ):
                    normalized_metadata[f"{{dist-info}}/{rest}"] = data
            result["package_payload"] = package_payload
            result["normalized_metadata_payload"] = normalized_metadata
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    return result


def _source_package_payload(root: Path) -> tuple[dict[str, bytes], list[str]]:
    package = root / "src" / "komorebi_math"
    payload: dict[str, bytes] = {}
    issues: list[str] = []
    if not package.is_dir():
        return payload, [f"missing package source directory: {package}"]
    try:
        paths = sorted(package.rglob("*"))
    except OSError as exc:
        return payload, [f"source traversal failed: {type(exc).__name__}: {exc}"]
    for path in paths:
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        try:
            if path.is_symlink():
                issues.append(f"source package symlink is not allowed: {relative.as_posix()}")
                continue
            mode = path.stat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                issues.append(
                    f"source package entry is not regular: {relative.as_posix()}"
                )
                continue
            payload[f"komorebi_math/{relative.as_posix()}"] = path.read_bytes()
        except OSError as exc:
            issues.append(
                f"cannot read source package entry {relative.as_posix()!r}: "
                f"{type(exc).__name__}: {exc}"
            )
    return payload, issues


def _inspect_sdist(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "is_gzip": False,
        "readable": False,
        "integrity": False,
        "safe_paths": False,
        "unique_names": False,
        "casefold_unique": False,
        "regular_only": False,
        "single_expected_root": False,
        "metadata_unique": False,
        "metadata_name": None,
        "metadata_version": None,
        "metadata_issues": [],
        "names": [],
        "file_names": [],
        "normalized_names": [],
        "unsafe_names": [],
        "duplicate_names": {},
        "casefold_collisions": {},
        "member_type_issues": [],
        "read_issues": [],
        "roots": [],
        "payload": {},
        "exception": None,
    }
    try:
        with path.open("rb") as handle:
            result["is_gzip"] = handle.read(2) == b"\x1f\x8b"
    except OSError as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            result["names"] = names
            name_audit = _name_audit(names)
            result["normalized_names"] = name_audit["normalized"]
            result["unsafe_names"] = name_audit["unsafe"]
            result["duplicate_names"] = name_audit["duplicates"]
            result["casefold_collisions"] = name_audit["casefold_collisions"]
            result["safe_paths"] = name_audit["safe"]
            result["unique_names"] = name_audit["unique"]
            result["casefold_unique"] = name_audit["casefold_unique"]

            type_issues: list[str] = []
            allowed_types = {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}
            for member in members:
                if member.type not in allowed_types:
                    kind = (
                        "symlink" if member.issym() else
                        "hardlink" if member.islnk() else
                        "character-device" if member.ischr() else
                        "block-device" if member.isblk() else
                        "fifo" if member.isfifo() else
                        "nonregular"
                    )
                    type_issues.append(f"{kind} tar member: {member.name!r}")
                elif not (member.isreg() or member.isdir()):
                    type_issues.append(f"nonregular tar member: {member.name!r}")
            result["member_type_issues"] = type_issues
            result["regular_only"] = not type_issues

            read_issues: list[str] = []
            entry_payloads: list[bytes | None] = []
            for member in members:
                if member.type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                    try:
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise tarfile.ExtractError("regular member has no data stream")
                        entry_payloads.append(extracted.read())
                    except Exception as exc:
                        entry_payloads.append(None)
                        read_issues.append(
                            f"{member.name!r}: {type(exc).__name__}: {exc}"
                        )
                else:
                    entry_payloads.append(b"" if member.isdir() else None)
            result["read_issues"] = read_issues
            result["integrity"] = not read_issues
            result["readable"] = True

            payload: dict[str, tuple[str, bytes]] = {}
            file_names: set[str] = set()
            if name_audit["safe"] and name_audit["unique"]:
                for member, canonical, data in zip(
                    members, name_audit["normalized"], entry_payloads, strict=True
                ):
                    if canonical is not None and data is not None:
                        payload[canonical] = (member.name, data)
                        if member.isreg():
                            file_names.add(canonical)
            result["payload"] = payload
            result["file_names"] = sorted(file_names)

            roots = sorted(
                {
                    canonical.split("/", 1)[0]
                    for canonical in name_audit["normalized"]
                    if canonical is not None
                }
            )
            result["roots"] = roots
            result["single_expected_root"] = roots == [SDIST_ROOT]

            root_metadata = f"{SDIST_ROOT}/PKG-INFO"
            metadata_members = [name for name in file_names if name == root_metadata]
            result["metadata_unique"] = metadata_members == [root_metadata]
            if result["metadata_unique"]:
                name, version, issues = _metadata_values(
                    payload[metadata_members[0]][1]
                )
                result["metadata_name"] = name
                result["metadata_version"] = version
                result["metadata_issues"] = issues
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    return result


def _differences(left: dict[str, bytes], right: dict[str, bytes]) -> list[str]:
    missing = {f"missing:{name}" for name in set(left) - set(right)}
    extra = {f"extra:{name}" for name in set(right) - set(left)}
    changed = {
        f"changed:{name}"
        for name in set(left).intersection(right)
        if left[name] != right[name]
    }
    return sorted(missing | extra | changed)


def _public_wheel_detail(result: dict[str, Any]) -> dict[str, Any]:
    hidden = {
        "payload",
        "package_payload",
        "normalized_metadata_payload",
        "names",
        "file_names",
        "normalized_names",
    }
    detail = {key: value for key, value in result.items() if key not in hidden}
    detail["member_count"] = len(result.get("names", []))
    detail["file_count"] = len(result.get("file_names", []))
    return detail


def _public_sdist_detail(result: dict[str, Any]) -> dict[str, Any]:
    hidden = {"payload", "names", "file_names", "normalized_names"}
    detail = {key: value for key, value in result.items() if key not in hidden}
    detail["member_count"] = len(result.get("names", []))
    detail["file_count"] = len(result.get("file_names", []))
    return detail


def _reported_path(path: Path, root: Path) -> str:
    """Return a checkout-independent path label for deterministic evidence."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def audit(
    root: Path,
    wheel: Path | None = None,
    sdist: Path | None = None,
    rebuilt_wheel: Path | None = None,
) -> dict[str, Any]:
    """Audit source, release layout, wheel, sdist, and an optional rebuild."""

    root = root.resolve()
    dist = root / "release" / "dist"
    wheel = (wheel if wheel is not None else dist / WHEEL_NAME).resolve()
    sdist = (sdist if sdist is not None else dist / SDIST_NAME).resolve()
    rebuilt_wheel = rebuilt_wheel.resolve() if rebuilt_wheel is not None else None

    checks = {name: False for name in BASE_CHECK_NAMES}
    check_details: dict[str, Any] = {}

    def record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = bool(passed)
        check_details[name] = detail

    record("project_root_exists", root.is_dir(), ".")

    dist_members: list[str] = []
    dist_error: str | None = None
    if dist.is_dir():
        try:
            dist_members = sorted(path.name for path in dist.iterdir())
        except OSError as exc:
            dist_error = f"{type(exc).__name__}: {exc}"
    else:
        dist_error = "release/dist is missing or is not a directory"
    expected_dist_members = sorted([WHEEL_NAME, SDIST_NAME])
    record(
        "release_dist_exists",
        dist.is_dir() and not dist.is_symlink() and dist_error is None,
        dist_error,
    )
    dist_files_are_regular = all(
        (dist / name).is_file() and not (dist / name).is_symlink()
        for name in expected_dist_members
    )
    record(
        "release_dist_contains_only_b4_wheel_and_sdist",
        dist.is_dir()
        and not dist.is_symlink()
        and dist_error is None
        and dist_members == expected_dist_members
        and dist_files_are_regular,
        {"expected": expected_dist_members, "observed": dist_members, "error": dist_error},
    )

    predecessor_dir = root / "release" / "predecessors"
    predecessor_wheel = predecessor_dir / PREDECESSOR_WHEEL_NAME
    predecessor_sdist = predecessor_dir / PREDECESSOR_SDIST_NAME
    record(
        "b3_predecessor_wheel_present",
        predecessor_wheel.is_file() and not predecessor_wheel.is_symlink(),
        _reported_path(predecessor_wheel, root),
    )
    record(
        "b3_predecessor_sdist_present",
        predecessor_sdist.is_file() and not predecessor_sdist.is_symlink(),
        _reported_path(predecessor_sdist, root),
    )
    b3_in_dist = sorted(
        name
        for name in dist_members
        if PREDECESSOR_VERSION.casefold() in name.casefold()
    )
    record(
        "b3_predecessors_absent_from_dist",
        dist.is_dir() and dist_error is None and not b3_in_dist,
        b3_in_dist,
    )

    wheel_exists = wheel.is_file() and not wheel.is_symlink()
    wheel_facts = _file_facts(wheel, wheel_exists)
    record("wheel_exists", wheel_exists, _reported_path(wheel, root))
    wheel_result = _inspect_wheel(wheel) if wheel_exists else {
        "exception": "wheel missing",
        "readable": False,
        "integrity": False,
        "safe_paths": False,
        "unique_names": False,
        "casefold_unique": False,
        "regular_unencrypted": False,
        "metadata_unique": False,
        "metadata_name": None,
        "metadata_version": None,
        "metadata_issues": [],
        "dist_info_roots": [],
        "record_unique": False,
        "record": {"complete": False, "hashes_and_sizes_valid": False},
        "names": [],
        "file_names": [],
        "payload": {},
        "package_payload": {},
        "normalized_metadata_payload": {},
    }
    record("wheel_archive_readable", wheel_result["readable"], wheel_result.get("exception"))
    record("wheel_zip_integrity", wheel_result["integrity"], {
        "bad_crc_member": wheel_result.get("bad_crc_member"),
        "read_issues": wheel_result.get("read_issues", []),
    })
    record("wheel_safe_member_paths", wheel_result["safe_paths"], wheel_result.get("unsafe_names", []))
    record("wheel_unique_normalized_names", wheel_result["unique_names"], wheel_result.get("duplicate_names", {}))
    record("wheel_no_casefold_collisions", wheel_result["casefold_unique"], wheel_result.get("casefold_collisions", {}))
    record("wheel_regular_unencrypted_members_only", wheel_result["regular_unencrypted"], wheel_result.get("member_type_issues", []))
    record("wheel_metadata_unique", wheel_result["metadata_unique"], wheel_result.get("metadata_issues", []))
    wheel_metadata_name_ok = (
        wheel_result.get("metadata_name") is not None
        and _canonical_project_name(wheel_result["metadata_name"])
        == CANONICAL_PROJECT_NAME
        and not wheel_result.get("metadata_issues")
    )
    record("wheel_metadata_name", wheel_metadata_name_ok, wheel_result.get("metadata_name"))
    record("wheel_metadata_version", wheel_result.get("metadata_version") == VERSION and not wheel_result.get("metadata_issues"), wheel_result.get("metadata_version"))
    record("wheel_expected_dist_info_root", wheel_result.get("dist_info_roots") == [DIST_INFO_ROOT], wheel_result.get("dist_info_roots", []))
    record("wheel_record_unique", wheel_result["record_unique"], wheel_result.get("record", {}).get("structural_issues", []))
    record("wheel_record_complete", wheel_result.get("record", {}).get("complete", False), wheel_result.get("record", {}).get("structural_issues", []))
    record("wheel_record_sha256_and_sizes_valid", wheel_result.get("record", {}).get("hashes_and_sizes_valid", False), wheel_result.get("record", {}).get("digest_issues", []))

    wheel_names = set(wheel_result.get("file_names", []))
    missing_wheel_required = sorted(WHEEL_REQUIRED_FILES - wheel_names)
    record("wheel_required_block20_files", not missing_wheel_required and wheel_result["readable"], missing_wheel_required)
    forbidden_wheel = sorted(
        name
        for name in wheel_names
        if "__pycache__" in PurePosixPath(name).parts
        or name.endswith((".pyc", ".pyo"))
        or name.startswith(("build/", "docs/", "examples/", "release/", "tests/", "verification/"))
    )
    record("wheel_generated_payload_absent", wheel_result["readable"] and not forbidden_wheel, forbidden_wheel)

    source_payload, source_issues = _source_package_payload(root)
    record("source_package_payload_readable", bool(source_payload) and not source_issues, source_issues)
    source_differences = _differences(source_payload, wheel_result.get("package_payload", {}))
    record(
        "wheel_package_payload_matches_source",
        bool(source_payload)
        and not source_issues
        and wheel_result["readable"]
        and wheel_result["safe_paths"]
        and wheel_result["unique_names"]
        and not source_differences,
        source_differences,
    )

    sdist_exists = sdist.is_file() and not sdist.is_symlink()
    sdist_facts = _file_facts(sdist, sdist_exists)
    record("sdist_exists", sdist_exists, _reported_path(sdist, root))
    sdist_result = _inspect_sdist(sdist) if sdist_exists else {
        "exception": "sdist missing",
        "is_gzip": False,
        "readable": False,
        "integrity": False,
        "safe_paths": False,
        "unique_names": False,
        "casefold_unique": False,
        "regular_only": False,
        "single_expected_root": False,
        "metadata_unique": False,
        "metadata_name": None,
        "metadata_version": None,
        "metadata_issues": [],
        "names": [],
        "file_names": [],
        "payload": {},
    }
    record("sdist_is_gzip", sdist_result["is_gzip"], None)
    record("sdist_archive_readable", sdist_result["readable"], sdist_result.get("exception"))
    record("sdist_archive_integrity", sdist_result["integrity"], sdist_result.get("read_issues", []))
    record("sdist_safe_member_paths", sdist_result["safe_paths"], sdist_result.get("unsafe_names", []))
    record("sdist_unique_normalized_names", sdist_result["unique_names"], sdist_result.get("duplicate_names", {}))
    record("sdist_no_casefold_collisions", sdist_result["casefold_unique"], sdist_result.get("casefold_collisions", {}))
    record("sdist_regular_files_and_directories_only", sdist_result["regular_only"], sdist_result.get("member_type_issues", []))
    record("sdist_single_expected_root", sdist_result["single_expected_root"], sdist_result.get("roots", []))
    record("sdist_metadata_unique", sdist_result["metadata_unique"], sdist_result.get("metadata_issues", []))
    sdist_metadata_name_ok = (
        sdist_result.get("metadata_name") is not None
        and _canonical_project_name(sdist_result["metadata_name"])
        == CANONICAL_PROJECT_NAME
        and not sdist_result.get("metadata_issues")
    )
    record("sdist_metadata_name", sdist_metadata_name_ok, sdist_result.get("metadata_name"))
    record("sdist_metadata_version", sdist_result.get("metadata_version") == VERSION and not sdist_result.get("metadata_issues"), sdist_result.get("metadata_version"))

    sdist_names = set(sdist_result.get("file_names", []))
    required_sdist_names = {f"{SDIST_ROOT}/{name}" for name in SDIST_REQUIRED_FILES}
    missing_sdist_required = sorted(required_sdist_names - sdist_names)
    record("sdist_required_block20_files", not missing_sdist_required and sdist_result["readable"], missing_sdist_required)
    forbidden_sdist = sorted(
        name
        for name in sdist_names
        if name.startswith(f"{SDIST_ROOT}/build/")
        or name.startswith(f"{SDIST_ROOT}/release/dist/")
        or "__pycache__" in PurePosixPath(name).parts
        or name.endswith((".pyc", ".pyo"))
    )
    record("sdist_generated_payload_absent", sdist_result["readable"] and not forbidden_sdist, forbidden_sdist)

    rebuild_detail: dict[str, Any] = {"provided": rebuilt_wheel is not None}
    if rebuilt_wheel is not None:
        optional_names = (
            "rebuilt_wheel_exists",
            "rebuilt_wheel_archive_readable",
            "rebuilt_wheel_zip_integrity",
            "rebuilt_wheel_safe_member_paths",
            "rebuilt_wheel_unique_normalized_names",
            "rebuilt_wheel_no_casefold_collisions",
            "rebuilt_wheel_regular_unencrypted_members_only",
            "rebuilt_wheel_metadata_unique",
            "rebuilt_wheel_metadata_name",
            "rebuilt_wheel_metadata_version",
            "rebuilt_wheel_expected_dist_info_root",
            "rebuilt_wheel_record_unique",
            "rebuilt_wheel_record_complete",
            "rebuilt_wheel_record_sha256_and_sizes_valid",
            "rebuilt_wheel_package_payload_matches",
            "rebuilt_wheel_metadata_payload_matches",
            "rebuilt_wheel_normalized_payload_matches",
        )
        for name in optional_names:
            checks[name] = False
        rebuilt_exists = rebuilt_wheel.is_file() and not rebuilt_wheel.is_symlink()
        rebuilt_facts = _file_facts(rebuilt_wheel, rebuilt_exists)
        record(
            "rebuilt_wheel_exists",
            rebuilt_exists,
            _reported_path(rebuilt_wheel, root),
        )
        rebuilt_result = _inspect_wheel(rebuilt_wheel) if rebuilt_exists else {
            "readable": False,
            "integrity": False,
            "safe_paths": False,
            "unique_names": False,
            "casefold_unique": False,
            "regular_unencrypted": False,
            "metadata_unique": False,
            "metadata_name": None,
            "metadata_version": None,
            "metadata_issues": [],
            "dist_info_roots": [],
            "record_unique": False,
            "record": {"complete": False, "hashes_and_sizes_valid": False},
            "package_payload": {},
            "normalized_metadata_payload": {},
            "exception": "rebuilt wheel missing",
        }
        record("rebuilt_wheel_archive_readable", rebuilt_result["readable"], rebuilt_result.get("exception"))
        record("rebuilt_wheel_zip_integrity", rebuilt_result["integrity"], rebuilt_result.get("read_issues", []))
        record("rebuilt_wheel_safe_member_paths", rebuilt_result["safe_paths"], rebuilt_result.get("unsafe_names", []))
        record("rebuilt_wheel_unique_normalized_names", rebuilt_result["unique_names"], rebuilt_result.get("duplicate_names", {}))
        record("rebuilt_wheel_no_casefold_collisions", rebuilt_result["casefold_unique"], rebuilt_result.get("casefold_collisions", {}))
        record(
            "rebuilt_wheel_regular_unencrypted_members_only",
            rebuilt_result["regular_unencrypted"],
            rebuilt_result.get("member_type_issues", []),
        )
        record(
            "rebuilt_wheel_metadata_unique",
            rebuilt_result["metadata_unique"],
            rebuilt_result.get("metadata_issues", []),
        )
        rebuilt_name_ok = (
            rebuilt_result.get("metadata_name") is not None
            and _canonical_project_name(rebuilt_result["metadata_name"])
            == CANONICAL_PROJECT_NAME
            and not rebuilt_result.get("metadata_issues")
        )
        record("rebuilt_wheel_metadata_name", rebuilt_name_ok, rebuilt_result.get("metadata_name"))
        record("rebuilt_wheel_metadata_version", rebuilt_result.get("metadata_version") == VERSION and not rebuilt_result.get("metadata_issues"), rebuilt_result.get("metadata_version"))
        record(
            "rebuilt_wheel_expected_dist_info_root",
            rebuilt_result.get("dist_info_roots") == [DIST_INFO_ROOT],
            rebuilt_result.get("dist_info_roots", []),
        )
        record(
            "rebuilt_wheel_record_unique",
            rebuilt_result["record_unique"],
            rebuilt_result.get("record", {}).get("structural_issues", []),
        )
        record("rebuilt_wheel_record_complete", rebuilt_result.get("record", {}).get("complete", False), rebuilt_result.get("record", {}).get("structural_issues", []))
        record("rebuilt_wheel_record_sha256_and_sizes_valid", rebuilt_result.get("record", {}).get("hashes_and_sizes_valid", False), rebuilt_result.get("record", {}).get("digest_issues", []))

        rebuilt_package_differences = _differences(
            wheel_result.get("package_payload", {}),
            rebuilt_result.get("package_payload", {}),
        )
        rebuilt_metadata_differences = _differences(
            wheel_result.get("normalized_metadata_payload", {}),
            rebuilt_result.get("normalized_metadata_payload", {}),
        )
        comparison_ready = (
            wheel_result["readable"]
            and wheel_result["safe_paths"]
            and wheel_result["unique_names"]
            and wheel_result["regular_unencrypted"]
            and wheel_result["metadata_unique"]
            and wheel_result.get("dist_info_roots") == [DIST_INFO_ROOT]
            and rebuilt_result["readable"]
            and rebuilt_result["safe_paths"]
            and rebuilt_result["unique_names"]
            and rebuilt_result["regular_unencrypted"]
            and rebuilt_result["metadata_unique"]
            and rebuilt_result.get("dist_info_roots") == [DIST_INFO_ROOT]
        )
        record("rebuilt_wheel_package_payload_matches", comparison_ready and not rebuilt_package_differences, rebuilt_package_differences)
        record("rebuilt_wheel_metadata_payload_matches", comparison_ready and not rebuilt_metadata_differences, rebuilt_metadata_differences)
        record("rebuilt_wheel_normalized_payload_matches", comparison_ready and not rebuilt_package_differences and not rebuilt_metadata_differences, {
            "package_differences": rebuilt_package_differences,
            "metadata_differences": rebuilt_metadata_differences,
        })
        rebuild_detail = {
            "provided": True,
            "path": _reported_path(rebuilt_wheel, root),
            **rebuilt_facts,
            "audit": _public_wheel_detail(rebuilt_result),
            "package_differences": rebuilt_package_differences,
            "metadata_differences": rebuilt_metadata_differences,
        }

    passed = bool(checks) and all(checks.values())
    report: dict[str, Any] = {
        "schema": "komorebi-math.block20-distribution-audit",
        "schema_version": 1,
        "package_version": VERSION,
        "passed": passed,
        "checks": checks,
        "check_details": check_details,
        "paths": {
            "root": ".",
            "dist": _reported_path(dist, root),
            "wheel": _reported_path(wheel, root),
            "sdist": _reported_path(sdist, root),
            "rebuilt_wheel": (
                _reported_path(rebuilt_wheel, root)
                if rebuilt_wheel is not None
                else None
            ),
        },
        "dist_members": dist_members,
        "wheel": {
            "path": _reported_path(wheel, root),
            **wheel_facts,
            "audit": _public_wheel_detail(wheel_result),
            "missing_required": missing_wheel_required,
            "source_payload_differences": source_differences,
        },
        "sdist": {
            "path": _reported_path(sdist, root),
            **sdist_facts,
            "audit": _public_sdist_detail(sdist_result),
            "missing_required": missing_sdist_required,
        },
        "fresh_sdist_rebuild": rebuild_detail,
    }
    canonical = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    report["report_sha256"] = _digest(canonical)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project root (default: parent of verification/)",
    )
    parser.add_argument("--wheel", type=Path, help="wheel to inspect")
    parser.add_argument("--sdist", type=Path, help="source distribution to inspect")
    parser.add_argument(
        "--rebuilt-wheel",
        type=Path,
        help="optional fresh wheel rebuilt from the audited sdist",
    )
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the JSON report on stdout",
    )
    arguments = parser.parse_args(argv)
    report = audit(
        arguments.root,
        arguments.wheel,
        arguments.sdist,
        arguments.rebuilt_wheel,
    )
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not arguments.quiet:
        print(text, end="")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
