"""Validate a source-to-API inventory without treating it as a proof.

Valid means that the inventory is internally consistent and its requested
links resolve. Partial, missing and separately maintained mechanisms remain
visible even when this metadata validation passes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "komorebi.paper-machinery-coverage.v1"
STATUSES = frozenset({"implemented", "partial", "missing", "theory_only", "external", "separate_reference"})


def load_coverage_catalog() -> dict[str, Any]:
    """Return an owned copy of the installed, versioned coverage inventory."""
    text = (
        resources.files("komorebi_math")
        .joinpath("data", "paper_machinery_coverage.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    valid: bool
    source_count: int
    mechanism_count: int
    status_counts: tuple[tuple[str, int], ...]
    errors: tuple[str, ...]
    unresolved: tuple[str, ...]
    catalog_sha256: str
    checked_apis: int
    source_bytes_checked: bool
    test_paths_checked: bool
    api_links_checked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "interpretation": "Metadata and requested links validated; no theorem or complete-runtime approval.",
            "source_count": self.source_count,
            "mechanism_count": self.mechanism_count,
            "status_counts": dict(self.status_counts),
            "errors": list(self.errors),
            "unresolved_mechanisms": list(self.unresolved),
            "all_catalogued_mechanisms_implemented": not self.unresolved
            and self.valid
            and self.api_links_checked,
            "scope": "Only the mechanisms explicitly inventoried in this catalog.",
            "catalog_sha256": self.catalog_sha256,
            "checked_api_references": self.checked_apis,
            "source_bytes_checked": self.source_bytes_checked,
            "test_paths_checked": self.test_paths_checked,
            "api_links_checked": self.api_links_checked,
        }


def _relative(path: Any) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("a nonempty relative path is required")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in path or ":" in path or "\x00" in path:
        raise ValueError("source and test paths must be safe relative paths")
    return str(parsed)


def _source_bytes(root: Path, source: Mapping[str, Any]) -> bytes:
    path = _relative(source.get("path"))
    archive = source.get("archive")
    if archive is None:
        return root.joinpath(path).read_bytes()
    archive_path = root.joinpath(_relative(archive))
    with zipfile.ZipFile(archive_path) as handle:
        info = handle.getinfo(path)
        if info.file_size > 64 * 1024 * 1024:
            raise ValueError("individual source member exceeds the 64 MiB verification cap")
        return handle.read(info)


def _api_name(name: str) -> list[str]:
    if not isinstance(name, str) or not name.startswith("komorebi_math."):
        raise ValueError("API references must belong to komorebi_math")
    parts = name.split(".")
    if any(not part.isidentifier() for part in parts):
        raise ValueError("invalid qualified API reference")
    return parts


def _resolve_api(name: str) -> Any:
    parts = _api_name(name)
    for cut in range(len(parts), 1, -1):
        module_name = ".".join(parts[:cut])
        try:
            value = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and not (module_name == exc.name or module_name.startswith(exc.name + ".")):
                raise ImportError(f"optional or transitive dependency unavailable: {exc.name}") from exc
            continue
        for part in parts[cut:]:
            value = getattr(value, part)
        return value
    raise ImportError(f"API module does not resolve: {name}")


def validate_coverage_catalog(
    catalog: Mapping[str, Any] | None = None,
    *,
    resolve_apis: bool = True,
    repository_root: str | Path | None = None,
    paper_source_root: str | Path | None = None,
) -> CoverageReport:
    """Check metadata, APIs and optional original source/test bytes.

    ``paper_source_root`` contains the original named PDFs, Markdown and ZIP
    archives. Archive member hashes are verified without executing their code.
    ``repository_root`` enables test-path validation. An installed wheel does
    not claim to contain the original papers or the repository test files.
    """
    errors: list[str] = []
    checked: set[str] = set()
    counts: Counter[str] = Counter()
    unresolved: list[str] = []
    sources: list[Any] = []
    mechanisms: list[Any] = []
    digest = ""
    try:
        if type(resolve_apis) is not bool:
            raise ValueError("resolve_apis must be a Boolean")
        raw = load_coverage_catalog() if catalog is None else catalog
        # JSON ownership also rejects nonfinite and opaque, mutable metadata.
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        data = json.loads(encoded)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ValueError("unsupported coverage schema")
        sources = data.get("sources")
        mechanisms = data.get("mechanisms")
        if not isinstance(sources, list) or not sources:
            raise ValueError("a nonempty source inventory is required")
        if not isinstance(mechanisms, list) or not mechanisms:
            raise ValueError("a nonempty mechanism inventory is required")
    except (ValueError, TypeError, OSError) as exc:
        return CoverageReport(False, 0, 0, (), (str(exc),), (), digest, 0, False, False, False)
    source_ids: set[str] = set()
    for source in sources:
        try:
            if not isinstance(source, dict):
                raise ValueError("source must be a mapping")
            identifier = source.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in source_ids:
                raise ValueError("source identifiers must be nonempty and unique")
            source_ids.add(identifier)
            if not isinstance(source.get("title"), str) or not source["title"].strip():
                raise ValueError("source title is required")
            _relative(source.get("path"))
            if not isinstance(source.get("sha256"), str) or not re.fullmatch(
                "[0-9a-f]{64}", source["sha256"]
            ):
                raise ValueError("source SHA-256 must be a lowercase 64-digit digest")
            if source.get("archive") is not None:
                _relative(source["archive"])
            if paper_source_root is not None:
                actual = hashlib.sha256(_source_bytes(Path(paper_source_root), source)).hexdigest()
                if actual != source["sha256"]:
                    raise ValueError("original source hash differs")
        except (ValueError, TypeError, OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            errors.append(f"source {source.get('id', '?') if isinstance(source, dict) else '?'}: {exc}")
    mechanism_ids: set[str] = set()
    for entry in mechanisms:
        key = entry.get("key", "?") if isinstance(entry, dict) else "?"
        try:
            if not isinstance(entry, dict):
                raise ValueError("mechanism must be a mapping")
            if not isinstance(key, str) or not key or key in mechanism_ids:
                raise ValueError("mechanism keys must be nonempty and unique")
            mechanism_ids.add(key)
            if entry.get("source_id") not in source_ids:
                raise ValueError("mechanism refers to an unknown source")
            status = entry.get("status")
            if status not in STATUSES:
                raise ValueError("unknown mechanism status")
            counts[status] += 1
            if status != "implemented":
                unresolved.append(key)
            for field in ("section", "summary"):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    raise ValueError(f"{field} is required")
            for field in ("api", "tests", "boundaries"):
                values = entry.get(field)
                if not isinstance(values, list) or any(
                    not isinstance(x, str) or not x.strip() for x in values
                ):
                    raise ValueError(f"{field} must be a list of nonempty text")
            if not entry["boundaries"]:
                raise ValueError("a supported mathematical boundary is required")
            if status == "implemented" and (not entry["api"] or not entry["tests"]):
                raise ValueError("implemented mechanisms require API and test references")
            for name in entry["api"]:
                _api_name(name)
            for path in entry["tests"]:
                _relative(path)
            if repository_root is not None:
                for path in entry["tests"]:
                    if not Path(repository_root).joinpath(_relative(path)).is_file():
                        raise ValueError(f"test path does not resolve: {path}")
            if resolve_apis:
                for name in entry["api"]:
                    if name not in checked:
                        _resolve_api(name)
                        checked.add(name)
        except (ValueError, TypeError, AttributeError, ImportError) as exc:
            errors.append(f"mechanism {key}: {exc}")
    return CoverageReport(
        not errors,
        len(sources),
        len(mechanisms),
        tuple(sorted(counts.items())),
        tuple(errors),
        tuple(sorted(unresolved)),
        digest,
        len(checked),
        paper_source_root is not None and not any(x.startswith("source ") for x in errors),
        repository_root is not None and not any(x.startswith("mechanism ") for x in errors),
        resolve_apis and not any(x.startswith("mechanism ") for x in errors),
    )


__all__ = ["CoverageReport", "load_coverage_catalog", "validate_coverage_catalog"]
