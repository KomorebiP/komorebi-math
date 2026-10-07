"""Independent frozen-source and contract audit for Block 20.

This script intentionally uses only the Python standard library and never
imports the package under audit.  Every check is isolated so one malformed
artifact cannot prevent the remaining source and compatibility checks from
running.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import stat
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

AUDIT_SCHEMA = "komorebi-math.block20-source-audit"
AUDIT_SCHEMA_VERSION = 1
CURRENT_VERSION = "1.0.0b4"
BASELINE_VERSION = "1.0.0b3"
EXPRESSION_SCHEMA = "komorebi_math.expression.v1"

_PUBLIC_SCHEMA = "komorebi-math.public-api"
_CODEC_SCHEMA = "komorebi-math.portable-codec-contract"
_SEMANTICS_SCHEMA = "komorebi-math.stable-semantics"
_DOCUMENTATION_SCHEMA = "komorebi-math.stable-documentation"

_ARTIFACTS = {
    "research_baseline": "public_api_1_0_0b3.json",
    "research_current": "public_api_1_0_0b4.json",
    "stable_baseline": "public_api_stable_1_0_0b3.json",
    "stable_current": "public_api_stable_1_0_0b4.json",
    "beta_baseline": "public_api_beta_1_0_0b3.json",
    "beta_current": "public_api_beta_1_0_0b4.json",
    "codec_baseline": "portable_codec_contract_1_0_0b3.json",
    "codec_current": "portable_codec_contract_1_0_0b4.json",
    "semantics_baseline": "stable_semantics_1_0_0b3.json",
    "semantics_current": "stable_semantics_1_0_0b4.json",
    "documentation_baseline": "stable_documentation_1_0_0b3.json",
    "documentation_current": "stable_documentation_1_0_0b4.json",
}

# The b3 documentation snapshot inherited runtime type docstrings for data
# exports.  Those strings were not API descriptions and the two Path entries
# varied between PosixPath and WindowsPath.  Freeze the exact correction
# boundary: only these twelve historical-to-semantic summary replacements are
# accepted, while every identity, kind, implementation, signature, reference,
# and all other summaries remain exact.
_APPROVED_DOCUMENTATION_SUMMARY_CORRECTIONS = {
    "DEFAULT_PORTABLE_CODEC_CONTRACT": (
        "Path subclass for non-Windows systems.",
        "Path to the committed portable-codec compatibility contract.",
    ),
    "DEFAULT_STABLE_SEMANTICS_FIXTURE": (
        "Path subclass for non-Windows systems.",
        "Path to the committed stable-semantics fixture.",
    ),
    "PORTABLE_CODEC_CONTRACT_SCHEMA": (
        "str(object='') -> str",
        "Schema identifier for portable-codec compatibility contracts.",
    ),
    "PORTABLE_CODEC_CONTRACT_SCHEMA_VERSION": (
        "int([x]) -> integer",
        "Current portable-codec compatibility-contract schema version.",
    ),
    "PORTABLE_SCHEMA": (
        "str(object='') -> str",
        "Schema identifier for portable persistence envelopes.",
    ),
    "PORTABLE_SCHEMA_VERSION": (
        "int([x]) -> integer",
        "Current portable persistence schema version.",
    ),
    "PolyphaseCoordinateRecord": (
        "Represent a PEP 604 union type",
        "Union of the stable one-dimensional polyphase coordinate record types.",
    ),
    "STABLE_API_STABILITY_LEVEL": (
        "str(object='') -> str",
        "Stability-policy identifier for the frozen stable namespace.",
    ),
    "STABLE_PUBLIC_NAMESPACES": (
        "Built-in immutable sequence.",
        "Canonical namespace tuple recorded by the frozen stable manifest.",
    ),
    "STABLE_SEMANTICS_SCHEMA": (
        "str(object='') -> str",
        "Schema identifier for stable-semantics fixtures.",
    ),
    "STABLE_SEMANTICS_SCHEMA_VERSION": (
        "int([x]) -> integer",
        "Current stable-semantics fixture schema version.",
    ),
    "__version__": (
        "str(object='') -> str",
        "Installed Komorebi Math package version.",
    ),
}

_VERSIONED_ARTIFACTS = {
    "research_baseline": BASELINE_VERSION,
    "research_current": CURRENT_VERSION,
    "stable_baseline": BASELINE_VERSION,
    "stable_current": CURRENT_VERSION,
    "beta_baseline": BASELINE_VERSION,
    "beta_current": CURRENT_VERSION,
    "codec_baseline": BASELINE_VERSION,
    "codec_current": CURRENT_VERSION,
    "documentation_baseline": BASELINE_VERSION,
    "documentation_current": CURRENT_VERSION,
}

_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
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
        "dist",
        "env",
        "node_modules",
        "temp",
        "temporary",
        "tmp",
        "venv",
        "virtualenv",
    }
)

_TEXT_SUFFIXES = frozenset(
    {".cff", ".in", ".md", ".ps1", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)
_PLACEHOLDER_WORDS = ("TO" + "DO", "TB" + "D", "FIX" + "ME")
_PLACEHOLDER = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _PLACEHOLDER_WORDS) + r")\b",
    re.IGNORECASE,
)


class AuditFailure(AssertionError):
    """One expected audit assertion failure."""


@dataclass(frozen=True, slots=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SourceInventory:
    digest: str
    file_count: int
    total_bytes: int
    excluded_entries: int
    symlinks: tuple[str, ...]
    unsafe_paths: tuple[str, ...]
    casefold_collisions: tuple[str, ...]
    special_paths: tuple[str, ...]
    hash_errors: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _capture(name: str, action: Callable[[], str]) -> AuditCheck:
    try:
        detail = action()
    except Exception as exc:  # Continue through every independent gate.
        return AuditCheck(name, False, f"{type(exc).__name__}: {exc}")
    return AuditCheck(name, True, detail)


def _duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditFailure(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> Any:
    raise AuditFailure(f"nonfinite JSON token {token!r}")


class ArtifactStore:
    """Load each committed artifact once with duplicate-key rejection."""

    def __init__(self, data_directory: Path) -> None:
        self.data_directory = data_directory
        self.cache: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, Exception] = {}

    def load(self, role: str) -> dict[str, Any]:
        if role in self.cache:
            return self.cache[role]
        if role in self.errors:
            raise self.errors[role]
        filename = _ARTIFACTS[role]
        path = self.data_directory / filename
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(
                text,
                object_pairs_hook=_duplicate_rejector,
                parse_constant=_reject_nonfinite,
            )
            if not isinstance(value, dict):
                raise AuditFailure(f"{filename} root must be an object")
        except Exception as exc:
            self.errors[role] = exc
            raise
        self.cache[role] = value
        return value


def _exact_fields(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    observed = set(value)
    if observed != fields:
        missing = sorted(fields - observed)
        extra = sorted(observed - fields)
        raise AuditFailure(f"{name} fields differ; missing={missing}, extra={extra}")


def _strict_int(value: Any, name: str, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditFailure(f"{name} must be an integer")
    if expected is not None and value != expected:
        raise AuditFailure(f"{name} must be {expected}, received {value}")
    return value


def _strict_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditFailure(f"{name} must be a nonempty string")
    return value


def _public_manifest(
    value: Mapping[str, Any],
    name: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], tuple[str, ...]]:
    _exact_fields(
        value,
        {"namespaces", "package_version", "schema", "schema_version", "stability_level", "symbols"},
        name,
    )
    _require(value["schema"] == _PUBLIC_SCHEMA, f"{name} has the wrong schema")
    _strict_int(value["schema_version"], f"{name} schema version", 1)
    _strict_text(value["package_version"], f"{name} package version")
    _strict_text(value["stability_level"], f"{name} stability level")
    namespaces_value = value["namespaces"]
    symbols_value = value["symbols"]
    _require(isinstance(namespaces_value, list), f"{name} namespaces must be an array")
    _require(isinstance(symbols_value, list), f"{name} symbols must be an array")
    namespaces = tuple(
        _strict_text(item, f"{name} namespace") for item in namespaces_value
    )
    _require(bool(namespaces), f"{name} must declare at least one namespace")
    _require(len(namespaces) == len(set(namespaces)), f"{name} contains duplicate namespaces")

    symbol_fields = {
        "implementation_module",
        "kind",
        "name",
        "namespace",
        "qualname",
        "signature",
    }
    symbols: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for index, raw_symbol in enumerate(symbols_value):
        _require(isinstance(raw_symbol, dict), f"{name} symbol {index} must be an object")
        _exact_fields(raw_symbol, symbol_fields, f"{name} symbol {index}")
        namespace = _strict_text(raw_symbol["namespace"], f"{name} symbol namespace")
        exported_name = _strict_text(raw_symbol["name"], f"{name} symbol name")
        _strict_text(raw_symbol["kind"], f"{name} symbol kind")
        _strict_text(
            raw_symbol["implementation_module"],
            f"{name} implementation module",
        )
        _strict_text(raw_symbol["qualname"], f"{name} symbol qualname")
        signature = raw_symbol["signature"]
        _require(
            signature is None or isinstance(signature, str),
            f"{name} symbol signature must be a string or null",
        )
        _require(namespace in namespaces, f"{name} symbol lies outside its namespaces")
        key = (namespace, exported_name)
        _require(key not in symbols, f"{name} contains duplicate symbol key {key!r}")
        symbols[key] = dict(raw_symbol)
        ordered_keys.append(key)
    _require(ordered_keys == sorted(ordered_keys), f"{name} symbols are not deterministically ordered")
    return symbols, namespaces


def _codec_contract(value: Mapping[str, Any], name: str) -> dict[str, dict[str, Any]]:
    _exact_fields(value, {"entries", "package_version", "schema", "schema_version"}, name)
    _require(value["schema"] == _CODEC_SCHEMA, f"{name} has the wrong schema")
    _strict_int(value["schema_version"], f"{name} schema version", 1)
    _strict_text(value["package_version"], f"{name} package version")
    entries_value = value["entries"]
    _require(isinstance(entries_value, list), f"{name} entries must be an array")
    entries: dict[str, dict[str, Any]] = {}
    ordered_tags: list[str] = []
    for index, raw_entry in enumerate(entries_value):
        _require(isinstance(raw_entry, dict), f"{name} entry {index} must be an object")
        _exact_fields(raw_entry, {"implementation_type", "tag", "version"}, f"{name} entry {index}")
        tag = _strict_text(raw_entry["tag"], f"{name} codec tag")
        _strict_text(raw_entry["implementation_type"], f"{name} implementation type")
        version = _strict_int(raw_entry["version"], f"{name} codec version")
        _require(version >= 1, f"{name} codec versions must be positive")
        _require(tag not in entries, f"{name} contains duplicate codec tag {tag!r}")
        entries[tag] = dict(raw_entry)
        ordered_tags.append(tag)
    _require(ordered_tags == sorted(ordered_tags), f"{name} codecs are not deterministically ordered")
    return entries


def _semantic_fixture(value: Mapping[str, Any], name: str) -> dict[str, dict[str, Any]]:
    _exact_fields(value, {"cases", "schema", "schema_version"}, name)
    _require(value["schema"] == _SEMANTICS_SCHEMA, f"{name} has the wrong schema")
    _strict_int(value["schema_version"], f"{name} schema version", 1)
    cases_value = value["cases"]
    _require(isinstance(cases_value, list), f"{name} cases must be an array")
    cases: dict[str, dict[str, Any]] = {}
    for index, raw_case in enumerate(cases_value):
        _require(isinstance(raw_case, dict), f"{name} case {index} must be an object")
        _require(
            {"expected", "input", "kind", "name"} <= set(raw_case),
            f"{name} case {index} is missing required fields",
        )
        case_name = _strict_text(raw_case["name"], f"{name} case name")
        _strict_text(raw_case["kind"], f"{name} case kind")
        _require(case_name not in cases, f"{name} contains duplicate case {case_name!r}")
        cases[case_name] = dict(raw_case)
    return cases


def _documentation_contract(
    value: Mapping[str, Any],
    name: str,
) -> dict[str, dict[str, Any]]:
    _exact_fields(value, {"package_version", "schema", "schema_version", "symbols"}, name)
    _require(value["schema"] == _DOCUMENTATION_SCHEMA, f"{name} has the wrong schema")
    _strict_int(value["schema_version"], f"{name} schema version", 1)
    _strict_text(value["package_version"], f"{name} package version")
    symbols_value = value["symbols"]
    _require(isinstance(symbols_value, list), f"{name} symbols must be an array")
    fields = {
        "implementation_module",
        "kind",
        "name",
        "qualname",
        "reference",
        "signature",
        "summary",
    }
    symbols: dict[str, dict[str, Any]] = {}
    ordered_names: list[str] = []
    for index, raw_symbol in enumerate(symbols_value):
        _require(isinstance(raw_symbol, dict), f"{name} symbol {index} must be an object")
        _exact_fields(raw_symbol, fields, f"{name} symbol {index}")
        symbol_name = _strict_text(raw_symbol["name"], f"{name} symbol name")
        for field in ("implementation_module", "kind", "qualname", "reference", "summary"):
            _strict_text(raw_symbol[field], f"{name} symbol {field}")
        signature = raw_symbol["signature"]
        _require(
            signature is None or isinstance(signature, str),
            f"{name} symbol signature must be a string or null",
        )
        _require(symbol_name not in symbols, f"{name} contains duplicate symbol {symbol_name!r}")
        symbols[symbol_name] = dict(raw_symbol)
        ordered_names.append(symbol_name)
    _require(ordered_names == sorted(ordered_names), f"{name} symbols are not deterministically ordered")
    return symbols


def _without_package_version(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "package_version"}


def _runtime_version(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values.append(node.value.value)
    _require(len(values) == 1, "runtime version file must contain one literal __version__ assignment")
    return values[0]


def _expression_schema_literal(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[str] = []
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Name)
            and target.id == "EXPRESSION_SCHEMA"
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            values.append(value.value)
    _require(len(values) == 1, "portable expression module must declare one literal schema")
    return values[0]


def _contains_exact_string(value: Any, target: str) -> bool:
    if isinstance(value, str):
        return value == target
    if isinstance(value, list):
        return any(_contains_exact_string(item, target) for item in value)
    if isinstance(value, dict):
        return any(
            key == target or _contains_exact_string(item, target)
            for key, item in value.items()
        )
    return False


def _excluded_directory(relative: PurePosixPath) -> bool:
    name = relative.name.casefold()
    temporary_prefixes = (
        ".temp-",
        ".tmp-",
        ".venv-",
        "temp-",
        "temporary-",
        "tmp-",
        "venv-",
        "virtualenv-",
    )
    return (
        name in _EXCLUDED_DIRECTORY_NAMES
        or name.endswith(".egg-info")
        or name.startswith(temporary_prefixes)
    )


def _final_handoff_archive(relative: PurePosixPath) -> bool:
    name = relative.name.casefold()
    archive = name.endswith(
        (".7z", ".rar", ".tar", ".tar.bz2", ".tar.gz", ".tar.xz", ".tar.zst", ".tgz", ".zip")
    )
    return archive and "handoff" in name and ("komorebi" in name or "block20" in name)


def _excluded_file(relative: PurePosixPath) -> bool:
    if relative.suffix.casefold() == ".pyc":
        return True
    if (
        len(relative.parts) >= 2
        and relative.parts[0].casefold() == "release"
        and relative.name.casefold().startswith("block20_")
    ):
        return True
    return _final_handoff_archive(relative)


def _unsafe_path(relative: PurePosixPath) -> bool:
    text = relative.as_posix()
    if relative.is_absolute() or not relative.parts:
        return True
    if "\\" in text or any(part in {"", ".", ".."} for part in relative.parts):
        return True
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        return True
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def _hash_regular_file(path: Path) -> tuple[int, bytes]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        mode = os.fstat(handle.fileno()).st_mode
        if not stat.S_ISREG(mode):
            raise AuditFailure("path ceased to be a regular file during hashing")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.digest()


def _source_inventory(project: Path) -> SourceInventory:
    records: list[tuple[str, int, bytes]] = []
    symlinks: list[str] = []
    unsafe_paths: list[str] = []
    collisions: list[str] = []
    special_paths: list[str] = []
    hash_errors: list[str] = []
    casefold_paths: dict[str, str] = {}
    excluded_entries = 0
    pending: list[tuple[Path, PurePosixPath]] = [(project, PurePosixPath())]

    while pending:
        directory, relative_directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            hash_errors.append(f"{relative_directory.as_posix() or '.'}: {type(exc).__name__}: {exc}")
            continue
        for entry in entries:
            relative = relative_directory / entry.name
            if _excluded_directory(relative):
                excluded_entries += 1
                continue
            if _excluded_file(relative):
                excluded_entries += 1
                continue
            relative_text = relative.as_posix()
            folded = relative_text.casefold()
            previous = casefold_paths.get(folded)
            if previous is None:
                casefold_paths[folded] = relative_text
            elif previous != relative_text:
                collisions.append(f"{previous} <> {relative_text}")
            if _unsafe_path(relative):
                unsafe_paths.append(relative_text)
            try:
                if entry.is_symlink():
                    symlinks.append(relative_text)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append((Path(entry.path), relative))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    special_paths.append(relative_text)
                    continue
                size, file_digest = _hash_regular_file(Path(entry.path))
                records.append((relative_text, size, file_digest))
            except (OSError, AuditFailure) as exc:
                hash_errors.append(f"{relative_text}: {type(exc).__name__}: {exc}")

    records.sort(key=lambda item: item[0])
    aggregate = sha256()
    aggregate.update(b"komorebi-math.block20-frozen-source\x00")
    total_bytes = 0
    safe_record_count = 0
    for relative_text, size, file_digest in records:
        try:
            path_bytes = relative_text.encode("utf-8")
            aggregate.update(len(path_bytes).to_bytes(8, "big"))
            aggregate.update(path_bytes)
            aggregate.update(size.to_bytes(8, "big"))
            aggregate.update(file_digest)
        except (OverflowError, UnicodeEncodeError) as exc:
            hash_errors.append(f"{relative_text}: cannot bind source record: {exc}")
            continue
        total_bytes += size
        safe_record_count += 1

    return SourceInventory(
        digest=aggregate.hexdigest(),
        file_count=safe_record_count,
        total_bytes=total_bytes,
        excluded_entries=excluded_entries,
        symlinks=tuple(sorted(set(symlinks))),
        unsafe_paths=tuple(sorted(set(unsafe_paths))),
        casefold_collisions=tuple(sorted(set(collisions))),
        special_paths=tuple(sorted(set(special_paths))),
        hash_errors=tuple(sorted(set(hash_errors))),
    )


def _block20_text_files(project: Path) -> tuple[Path, ...]:
    candidates: set[Path] = set()
    expression = project / "src" / "komorebi_math" / "expression"
    if expression.is_dir():
        candidates.update(
            path
            for path in expression.rglob("*.py")
            if path.is_file() and not path.is_symlink()
        )

    explicit = (
        "CHANGELOG.md",
        "MANIFEST.in",
        "README.md",
        "README_BLOCK20.md",
        "pyproject.toml",
        "docs/IMPLEMENTATION_ROADMAP.md",
        "docs/PUBLIC_API.md",
        "docs/RELEASE_PROCESS.md",
        "docs/index.md",
        "src/komorebi_math/__init__.py",
        "src/komorebi_math/_features.py",
        "src/komorebi_math/cli/contracts.py",
        "src/komorebi_math/cli/main.py",
        "src/komorebi_math/verify/__init__.py",
        "src/komorebi_math/verify/expression.py",
    )
    candidates.update(
        project / relative
        for relative in explicit
        if (project / relative).is_file() and not (project / relative).is_symlink()
    )

    for directory_name in ("docs", "examples", "tests", "verification"):
        directory = project / directory_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.suffix.casefold() not in _TEXT_SUFFIXES
            ):
                continue
            normalized = re.sub(r"[_-]", "", path.name.casefold())
            if "block20" in normalized or "typed_finite_expression" in path.name.casefold():
                candidates.add(path)

    for path in project.iterdir() if project.is_dir() else ():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.suffix.casefold() not in _TEXT_SUFFIXES
        ):
            continue
        normalized = re.sub(r"[_-]", "", path.name.casefold())
        if "block20" in normalized:
            candidates.add(path)
    return tuple(sorted(candidates, key=lambda path: path.relative_to(project).as_posix()))


def _placeholder_findings(project: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    findings: list[str] = []
    read_errors: list[str] = []
    files = _block20_text_files(project)
    for path in files:
        relative = path.relative_to(project).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            read_errors.append(f"{relative}: {type(exc).__name__}: {exc}")
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in _PLACEHOLDER.finditer(line):
                findings.append(f"{relative}:{line_number}:{match.group(0).upper()}")
    return tuple(sorted(findings)), tuple(sorted(read_errors))


def run_audit(root: str | Path | None = None) -> dict[str, Any]:
    """Run every independent Block 20 source gate and return JSON data."""

    project = (
        Path(__file__).resolve().parents[1]
        if root is None
        else Path(root).expanduser().resolve()
    )
    data_directory = project / "src" / "komorebi_math" / "data"
    store = ArtifactStore(data_directory)
    counts: dict[str, int] = {
        "artifact_files_expected": len(_ARTIFACTS),
        "artifact_files_loaded": 0,
        "research_baseline_symbols": 0,
        "research_current_symbols": 0,
        "research_added_symbols": 0,
        "stable_symbols": 0,
        "beta_symbols": 0,
        "portable_codec_entries": 0,
        "stable_semantic_cases": 0,
        "stable_documentation_symbols": 0,
        "stable_documentation_summary_corrections": 0,
        "source_files": 0,
        "source_bytes": 0,
        "source_excluded_entries": 0,
        "source_symlinks": 0,
        "source_unsafe_paths": 0,
        "source_casefold_collisions": 0,
        "source_special_paths": 0,
        "source_hash_errors": 0,
        "block20_text_files": 0,
        "unresolved_placeholders": 0,
    }
    inventory_box: dict[str, SourceInventory] = {}
    placeholder_box: dict[str, tuple[str, ...]] = {}

    def root_layout() -> str:
        _require(project.is_dir(), f"project root is not a directory: {project}")
        required = (
            project / "pyproject.toml",
            project / "src" / "komorebi_math" / "_version.py",
            data_directory,
        )
        missing = [path.relative_to(project).as_posix() for path in required if not path.exists()]
        _require(not missing, f"project root is missing required paths: {missing}")
        return "project metadata, runtime version, and data directory are present"

    def json_artifacts() -> str:
        failures: list[str] = []
        for role in _ARTIFACTS:
            try:
                store.load(role)
            except Exception as exc:
                failures.append(f"{_ARTIFACTS[role]}: {type(exc).__name__}: {exc}")
        counts["artifact_files_loaded"] = len(store.cache)
        _require(not failures, f"invalid committed JSON artifacts: {failures}")
        return f"all {len(_ARTIFACTS)} committed artifacts parsed without duplicate object keys"

    def versions() -> str:
        for role, expected in _VERSIONED_ARTIFACTS.items():
            artifact = store.load(role)
            _require(
                artifact.get("package_version") == expected,
                f"{_ARTIFACTS[role]} package_version must be {expected}",
            )
        pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
        observed_build = pyproject.get("project", {}).get("version")
        _require(observed_build == CURRENT_VERSION, f"build version must be {CURRENT_VERSION}")
        observed_runtime = _runtime_version(project / "src" / "komorebi_math" / "_version.py")
        _require(observed_runtime == CURRENT_VERSION, f"runtime version must be {CURRENT_VERSION}")
        return "current artifacts/build/runtime are 1.0.0b4 and saved baselines are 1.0.0b3"

    def research_additivity() -> str:
        baseline = store.load("research_baseline")
        current = store.load("research_current")
        baseline_symbols, _ = _public_manifest(baseline, "research baseline")
        current_symbols, _ = _public_manifest(current, "research current")
        for field in ("schema", "schema_version", "stability_level", "namespaces"):
            _require(current[field] == baseline[field], f"research {field} changed across the boundary")
        missing = sorted(set(baseline_symbols) - set(current_symbols))
        changed = sorted(
            key
            for key in set(baseline_symbols).intersection(current_symbols)
            if baseline_symbols[key] != current_symbols[key]
        )
        _require(not missing, f"research symbols were removed: {missing[:20]}")
        _require(not changed, f"research symbols changed: {changed[:20]}")
        added = sorted(set(current_symbols) - set(baseline_symbols))
        counts["research_baseline_symbols"] = len(baseline_symbols)
        counts["research_current_symbols"] = len(current_symbols)
        counts["research_added_symbols"] = len(added)
        return (
            f"{len(baseline_symbols)} baseline symbols are exact; "
            f"{len(added)} additive research symbols observed"
        )

    def frozen_public(role_prefix: str, count_name: str) -> str:
        baseline = store.load(f"{role_prefix}_baseline")
        current = store.load(f"{role_prefix}_current")
        baseline_symbols, baseline_namespaces = _public_manifest(
            baseline,
            f"{role_prefix} baseline",
        )
        current_symbols, current_namespaces = _public_manifest(
            current,
            f"{role_prefix} current",
        )
        _require(
            _without_package_version(current) == _without_package_version(baseline),
            f"{role_prefix} manifest changed beyond package_version",
        )
        _require(current_namespaces == baseline_namespaces, f"{role_prefix} namespaces changed")
        _require(
            current["stability_level"] == baseline["stability_level"],
            f"{role_prefix} stability changed",
        )
        _require(current_symbols == baseline_symbols, f"{role_prefix} symbols changed")
        counts[count_name] = len(current_symbols)
        return f"{len(current_symbols)} {role_prefix} symbols, namespaces, and stability are exact"

    def codecs() -> str:
        baseline = store.load("codec_baseline")
        current = store.load("codec_current")
        baseline_entries = _codec_contract(baseline, "codec baseline")
        current_entries = _codec_contract(current, "codec current")
        _require(len(baseline_entries) == 51, "baseline codec inventory must contain exactly 51 entries")
        _require(len(current_entries) == 51, "current codec inventory must contain exactly 51 entries")
        _require(
            _without_package_version(current) == _without_package_version(baseline),
            "portable codec contract changed beyond package_version",
        )
        _require(current_entries == baseline_entries, "portable codec entries changed")
        counts["portable_codec_entries"] = len(current_entries)
        return "all 51 portable codec entries are exact across the saved boundary"

    def semantics() -> str:
        baseline = store.load("semantics_baseline")
        current = store.load("semantics_current")
        baseline_cases = _semantic_fixture(baseline, "semantic baseline")
        current_cases = _semantic_fixture(current, "semantic current")
        _require(len(baseline_cases) == 10, "baseline semantics must contain exactly 10 cases")
        _require(len(current_cases) == 10, "current semantics must contain exactly 10 cases")
        _require(current == baseline, "stable semantic fixture changed")
        counts["stable_semantic_cases"] = len(current_cases)
        return "all 10 stable semantic cases are byte-structure exact"

    def documentation() -> str:
        baseline = store.load("documentation_baseline")
        current = store.load("documentation_current")
        baseline_symbols = _documentation_contract(baseline, "documentation baseline")
        current_symbols = _documentation_contract(current, "documentation current")
        _require(len(baseline_symbols) == 131, "baseline documentation must contain 131 symbols")
        _require(len(current_symbols) == 131, "current documentation must contain 131 symbols")
        _require(
            {
                key: value
                for key, value in current.items()
                if key not in {"package_version", "symbols"}
            }
            == {
                key: value
                for key, value in baseline.items()
                if key not in {"package_version", "symbols"}
            },
            "stable documentation top-level contract changed",
        )
        _require(
            set(current_symbols) == set(baseline_symbols),
            "stable documentation symbol names changed",
        )
        for name in sorted(baseline_symbols):
            baseline_entry = baseline_symbols[name]
            current_entry = current_symbols[name]
            _require(
                {key: value for key, value in current_entry.items() if key != "summary"}
                == {
                    key: value
                    for key, value in baseline_entry.items()
                    if key != "summary"
                },
                f"stable documentation identity or contract fields changed for {name}",
            )
        summary_corrections = {
            name: (baseline_symbols[name]["summary"], current_symbols[name]["summary"])
            for name in sorted(baseline_symbols)
            if baseline_symbols[name]["summary"] != current_symbols[name]["summary"]
        }
        _require(
            summary_corrections == _APPROVED_DOCUMENTATION_SUMMARY_CORRECTIONS,
            "stable documentation summary corrections differ from the approved boundary",
        )
        for name in _APPROVED_DOCUMENTATION_SUMMARY_CORRECTIONS:
            _require(
                baseline_symbols[name]["kind"] == current_symbols[name]["kind"] == "data",
                f"approved documentation correction is no longer a data export: {name}",
            )
        counts["stable_documentation_symbols"] = len(current_symbols)
        counts["stable_documentation_summary_corrections"] = len(summary_corrections)
        return (
            "all 131 stable documentation identities are exact; "
            "12 approved data summaries are platform neutral"
        )

    def expression_schema_isolation() -> str:
        declared = _expression_schema_literal(
            project / "src" / "komorebi_math" / "expression" / "portable.py"
        )
        _require(declared == EXPRESSION_SCHEMA, f"expression schema must be {EXPRESSION_SCHEMA!r}")
        current = store.load("codec_current")
        entries = _codec_contract(current, "codec current")
        expression_entries = sorted(
            tag
            for tag, entry in entries.items()
            if tag == EXPRESSION_SCHEMA
            or entry["implementation_type"].startswith("komorebi_math.expression.")
        )
        _require(
            not _contains_exact_string(current, EXPRESSION_SCHEMA),
            "dedicated expression schema appears in the frozen codec contract",
        )
        _require(
            not expression_entries,
            f"expression codecs entered the stable contract: {expression_entries}",
        )
        return "dedicated research expression schema is absent from the frozen codec registry"

    def get_inventory() -> SourceInventory:
        if "value" not in inventory_box:
            inventory_box["value"] = _source_inventory(project)
            inventory = inventory_box["value"]
            counts["source_files"] = inventory.file_count
            counts["source_bytes"] = inventory.total_bytes
            counts["source_excluded_entries"] = inventory.excluded_entries
            counts["source_symlinks"] = len(inventory.symlinks)
            counts["source_unsafe_paths"] = len(inventory.unsafe_paths)
            counts["source_casefold_collisions"] = len(inventory.casefold_collisions)
            counts["source_special_paths"] = len(inventory.special_paths)
            counts["source_hash_errors"] = len(inventory.hash_errors)
        return inventory_box["value"]

    def source_safety() -> str:
        inventory = get_inventory()
        problems = {
            "symlinks": inventory.symlinks,
            "unsafe_paths": inventory.unsafe_paths,
            "casefold_collisions": inventory.casefold_collisions,
            "special_paths": inventory.special_paths,
        }
        observed = {key: values for key, values in problems.items() if values}
        _require(not observed, f"unsafe source-tree entries: {observed}")
        return "no symlinks, unsafe/backslash paths, casefold collisions, or special files"

    def source_digest() -> str:
        inventory = get_inventory()
        _require(not inventory.hash_errors, f"source hashing errors: {inventory.hash_errors}")
        _require(inventory.file_count > 0, "frozen source set is empty")
        return f"{inventory.file_count} files bind to SHA-256 {inventory.digest}"

    def placeholders() -> str:
        findings, read_errors = _placeholder_findings(project)
        files = _block20_text_files(project)
        counts["block20_text_files"] = len(files)
        counts["unresolved_placeholders"] = len(findings)
        placeholder_box["findings"] = findings
        placeholder_box["read_errors"] = read_errors
        _require(not read_errors, f"could not read Block 20 text files: {read_errors}")
        _require(not findings, f"unresolved placeholder markers: {findings}")
        return f"{len(files)} Block 20 production, documentation, script, and test files are resolved"

    checks = [
        _capture("project root layout", root_layout),
        _capture("committed JSON syntax and duplicate keys", json_artifacts),
        _capture("current and baseline versions", versions),
        _capture("research public API additivity", research_additivity),
        _capture("stable public API freeze", lambda: frozen_public("stable", "stable_symbols")),
        _capture("beta public API freeze", lambda: frozen_public("beta", "beta_symbols")),
        _capture("portable codec freeze", codecs),
        _capture("stable semantics freeze", semantics),
        _capture("stable documentation freeze", documentation),
        _capture("research expression schema isolation", expression_schema_isolation),
        _capture("frozen source path safety", source_safety),
        _capture("frozen source digest", source_digest),
        _capture("Block 20 unresolved placeholders", placeholders),
    ]

    inventory = inventory_box.get("value")
    source = {
        "binding": (
            "SHA-256(domain || uint64be(path_size) || path_utf8 || "
            "uint64be(file_size) || file_sha256)"
        ),
        "sha256": "" if inventory is None else inventory.digest,
        "file_count": 0 if inventory is None else inventory.file_count,
        "total_bytes": 0 if inventory is None else inventory.total_bytes,
    }
    diagnostics = {
        "artifact_errors": {
            _ARTIFACTS[role]: f"{type(error).__name__}: {error}"
            for role, error in sorted(store.errors.items())
        },
        "symlinks": [] if inventory is None else list(inventory.symlinks),
        "unsafe_paths": [] if inventory is None else list(inventory.unsafe_paths),
        "casefold_collisions": [] if inventory is None else list(inventory.casefold_collisions),
        "special_paths": [] if inventory is None else list(inventory.special_paths),
        "source_hash_errors": [] if inventory is None else list(inventory.hash_errors),
        "unresolved_placeholders": list(placeholder_box.get("findings", ())),
        "placeholder_read_errors": list(placeholder_box.get("read_errors", ())),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "package_version": CURRENT_VERSION,
        "baseline_version": BASELINE_VERSION,
        "root": ".",
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "counts": counts,
        "source_digest": source["sha256"],
        "source_file_count": source["file_count"],
        "source": source,
        "diagnostics": diagnostics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="project root to audit",
    )
    parser.add_argument("--output", help="optional JSON report path")
    parser.add_argument("--quiet", action="store_true", help="suppress stdout")
    args = parser.parse_args(argv)
    report = run_audit(args.root)
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
    if not args.quiet:
        print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
