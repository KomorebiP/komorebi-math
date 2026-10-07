"""Exact documentation coverage for the frozen stable namespace."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

from .._version import __version__
from ..certificates import Certificate, Exactness
from ..compatibility import create_stable_api_manifest
from ..exceptions import ValidationError

STABLE_DOCUMENTATION_SCHEMA = "komorebi-math.stable-documentation"
STABLE_DOCUMENTATION_SCHEMA_VERSION = 1
DEFAULT_STABLE_DOCUMENTATION = (
    Path(__file__).resolve().parents[1] / "data" / "stable_documentation_1_0.json"
)

# Data exports do not own useful runtime docstrings: ``inspect.getdoc`` sees
# the documentation of their value type instead.  That was especially harmful
# for the two package-owned paths, whose summaries changed between
# ``PosixPath`` and ``WindowsPath``.  Keep the documentation contract semantic,
# platform neutral, and closed over the frozen data inventory.
_STABLE_DATA_SUMMARIES = {
    "DEFAULT_PORTABLE_CODEC_CONTRACT": (
        "Path to the committed portable-codec compatibility contract."
    ),
    "DEFAULT_STABLE_SEMANTICS_FIXTURE": (
        "Path to the committed stable-semantics fixture."
    ),
    "PORTABLE_CODEC_CONTRACT_SCHEMA": (
        "Schema identifier for portable-codec compatibility contracts."
    ),
    "PORTABLE_CODEC_CONTRACT_SCHEMA_VERSION": (
        "Current portable-codec compatibility-contract schema version."
    ),
    "PORTABLE_SCHEMA": "Schema identifier for portable persistence envelopes.",
    "PORTABLE_SCHEMA_VERSION": "Current portable persistence schema version.",
    "PolyphaseCoordinateRecord": (
        "Union of the stable one-dimensional polyphase coordinate record types."
    ),
    "STABLE_API_STABILITY_LEVEL": (
        "Stability-policy identifier for the frozen stable namespace."
    ),
    "STABLE_PUBLIC_NAMESPACES": (
        "Canonical namespace tuple recorded by the frozen stable manifest."
    ),
    "STABLE_SEMANTICS_SCHEMA": (
        "Schema identifier for stable-semantics fixtures."
    ),
    "STABLE_SEMANTICS_SCHEMA_VERSION": (
        "Current stable-semantics fixture schema version."
    ),
    "__version__": "Installed Komorebi Math package version.",
}


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class StableDocumentationEntry:
    """One exact documentation record for a stable export."""

    name: str
    kind: str
    implementation_module: str
    qualname: str
    signature: str | None
    summary: str
    reference: str

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (
                self.name,
                self.kind,
                self.implementation_module,
                self.qualname,
                self.summary,
                self.reference,
            )
        ):
            raise ValidationError("stable documentation fields must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "implementation_module": self.implementation_module,
            "qualname": self.qualname,
            "signature": self.signature,
            "summary": self.summary,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StableDocumentationEntry":
        required = {
            "name",
            "kind",
            "implementation_module",
            "qualname",
            "signature",
            "summary",
            "reference",
        }
        if set(value) != required:
            raise ValidationError("stable documentation entry fields do not match the schema")
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            implementation_module=str(value["implementation_module"]),
            qualname=str(value["qualname"]),
            signature=None if value["signature"] is None else str(value["signature"]),
            summary=str(value["summary"]),
            reference=str(value["reference"]),
        )


@dataclass(frozen=True, slots=True)
class StableDocumentationContract:
    """Closed documentation inventory for the stable 1.0 namespace."""

    package_version: str
    symbols: tuple[StableDocumentationEntry, ...]
    schema: str = STABLE_DOCUMENTATION_SCHEMA
    schema_version: int = STABLE_DOCUMENTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != STABLE_DOCUMENTATION_SCHEMA:
            raise ValidationError("unknown stable documentation schema")
        if self.schema_version != STABLE_DOCUMENTATION_SCHEMA_VERSION:
            raise ValidationError("unsupported stable documentation schema version")
        names = [entry.name for entry in self.symbols]
        if len(names) != len(set(names)):
            raise ValidationError("stable documentation names must be unique")

    @property
    def digest(self) -> str:
        return _canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "symbols": [entry.to_dict() for entry in self.symbols],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StableDocumentationContract":
        required = {"schema", "schema_version", "package_version", "symbols"}
        if set(value) != required:
            raise ValidationError("stable documentation fields do not match the schema")
        symbols = value["symbols"]
        if not isinstance(symbols, list):
            raise ValidationError("stable documentation symbols must be an array")
        if any(not isinstance(entry, Mapping) for entry in symbols):
            raise ValidationError("stable documentation symbols must contain objects")
        return cls(
            package_version=str(value["package_version"]),
            symbols=tuple(
                sorted(StableDocumentationEntry.from_dict(entry) for entry in symbols)
            ),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class StableDocumentationReport:
    """Comparison between the committed and observed stable documentation."""

    reference_digest: str
    current_digest: str
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    changed: tuple[str, ...]
    deprecated: tuple[str, ...]
    certificate: Certificate

    @property
    def passed(self) -> bool:
        return self.certificate.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reference_digest": self.reference_digest,
            "current_digest": self.current_digest,
            "missing": list(self.missing),
            "extra": list(self.extra),
            "changed": list(self.changed),
            "deprecated": list(self.deprecated),
            "certificate": self.certificate.to_dict(),
        }


def create_stable_documentation_contract(
    *, package_version: str | None = None
) -> StableDocumentationContract:
    """Create deterministic coverage from the live stable namespace."""

    stable = import_module("komorebi_math.stable")
    manifest = create_stable_api_manifest(package_version=package_version)
    entries: list[StableDocumentationEntry] = []
    for symbol in manifest.symbols:
        value = getattr(stable, symbol.name)
        if symbol.kind == "data":
            try:
                summary = _STABLE_DATA_SUMMARIES[symbol.name]
            except KeyError as exc:
                raise ValidationError(
                    f"stable data export has no platform-neutral summary: {symbol.name}"
                ) from exc
        else:
            docstring = inspect.getdoc(value) or symbol.name
            summary = docstring.strip().splitlines()[0]
        entries.append(
            StableDocumentationEntry(
                name=symbol.name,
                kind=symbol.kind,
                implementation_module=symbol.implementation_module,
                qualname=symbol.qualname,
                signature=symbol.signature,
                summary=summary,
                reference="docs/STABLE_API.md",
            )
        )
    return StableDocumentationContract(
        package_version=__version__ if package_version is None else str(package_version),
        symbols=tuple(sorted(entries)),
    )


def load_stable_documentation(
    path: str | Path | None = None,
) -> StableDocumentationContract:
    """Load the committed stable documentation inventory."""

    source = DEFAULT_STABLE_DOCUMENTATION if path is None else Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"could not load stable documentation {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError("stable documentation is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError("stable documentation root must be an object")
    return StableDocumentationContract.from_dict(raw)


def check_stable_documentation(
    path: str | Path | None = None,
) -> StableDocumentationReport:
    """Require exact one-to-one documentation of every stable export."""

    reference = load_stable_documentation(path)
    current = create_stable_documentation_contract()
    expected = {entry.name: entry for entry in reference.symbols}
    observed = {entry.name: entry for entry in current.symbols}
    missing = tuple(sorted(set(expected) - set(observed)))
    extra = tuple(sorted(set(observed) - set(expected)))
    changed = tuple(
        sorted(
            name
            for name in set(expected) & set(observed)
            if expected[name] != observed[name]
        )
    )
    stable = import_module("komorebi_math.stable")
    deprecated = tuple(
        sorted(
            name
            for name in stable.__all__
            if getattr(getattr(stable, name), "__komorebi_deprecation__", None) is not None
        )
    )
    valid = (
        reference.package_version == __version__
        and not missing
        and not extra
        and not changed
        and not deprecated
        and reference.digest == current.digest
    )
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={
            "missing": float(len(missing)),
            "extra": float(len(extra)),
            "changed": float(len(changed)),
            "deprecated": float(len(deprecated)),
        },
        assumptions_checked=(
            "one documentation record per stable export",
            "exact callable signatures and first-line summaries",
            "no deprecated stable export",
        ),
        assumptions_unchecked=(
            "narrative completeness beyond the committed stable reference",
        ),
        diagnostics={"symbols": len(current.symbols)},
        theorem_id="KM-DOCS-017",
        message=(
            "stable documentation contract passed"
            if valid
            else "stable documentation contract failed"
        ),
    )
    return StableDocumentationReport(
        reference_digest=reference.digest,
        current_digest=current.digest,
        missing=missing,
        extra=extra,
        changed=changed,
        deprecated=deprecated,
        certificate=certificate,
    )


__all__ = [
    "DEFAULT_STABLE_DOCUMENTATION",
    "STABLE_DOCUMENTATION_SCHEMA",
    "STABLE_DOCUMENTATION_SCHEMA_VERSION",
    "StableDocumentationContract",
    "StableDocumentationEntry",
    "StableDocumentationReport",
    "check_stable_documentation",
    "create_stable_documentation_contract",
    "load_stable_documentation",
]
