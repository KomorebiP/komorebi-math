"""Frozen portable-codec inventory and migration-path compatibility checks.

The ordinary portable persistence registry is dynamic so extensions can add
codecs.  The 1.0 candidate contract records the codecs shipped by Komorebi Math
itself and verifies that a release has not silently removed a tag, changed the
Python record represented by a tag, reduced a codec version, or advanced a
codec version without every adjacent migration step.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__
from .certificates import Certificate, Exactness
from .exceptions import ValidationError
from .export.persistence import portable_codecs, portable_migrations

PORTABLE_CODEC_CONTRACT_SCHEMA = "komorebi-math.portable-codec-contract"
PORTABLE_CODEC_CONTRACT_SCHEMA_VERSION = 1
DEFAULT_PORTABLE_CODEC_CONTRACT = (
    Path(__file__).with_name("data") / "portable_codec_contract_1_0.json"
)


@dataclass(frozen=True, slots=True, order=True)
class PortableCodecContractEntry:
    """One codec identity frozen by the persistence compatibility contract."""

    tag: str
    version: int
    implementation_type: str

    def __post_init__(self) -> None:
        tag = str(self.tag).strip()
        if not tag or any(character.isspace() for character in tag):
            raise ValidationError("portable codec contract tags must be nonempty and contain no whitespace")
        version = self.version
        if type(version) is not int or version < 1:
            raise ValidationError("portable codec contract versions must be positive integers")
        implementation = str(self.implementation_type).strip()
        if not implementation or "." not in implementation:
            raise ValidationError("portable codec implementation types must be qualified names")
        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "implementation_type", implementation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "version": self.version,
            "implementation_type": self.implementation_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PortableCodecContractEntry":
        if set(value) != {"tag", "version", "implementation_type"}:
            raise ValidationError("portable codec contract entry fields do not match the schema")
        return cls(
            tag=str(value["tag"]),
            version=value["version"],
            implementation_type=str(value["implementation_type"]),
        )


@dataclass(frozen=True, slots=True)
class PortableCodecContract:
    """Deterministic package-owned portable codec inventory."""

    package_version: str
    entries: tuple[PortableCodecContractEntry, ...]
    schema: str = PORTABLE_CODEC_CONTRACT_SCHEMA
    schema_version: int = PORTABLE_CODEC_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != PORTABLE_CODEC_CONTRACT_SCHEMA:
            raise ValidationError("unknown portable codec contract schema")
        if (type(self.schema_version) is not int
                or self.schema_version != PORTABLE_CODEC_CONTRACT_SCHEMA_VERSION):
            raise ValidationError("unsupported portable codec contract schema version")
        entries = tuple(self.entries)
        if tuple(sorted(entries)) != entries:
            raise ValidationError("portable codec contract entries must use deterministic ordering")
        tags = tuple(entry.tag for entry in entries)
        if len(set(tags)) != len(tags):
            raise ValidationError("portable codec contract contains duplicate tags")
        object.__setattr__(self, "package_version", str(self.package_version))
        object.__setattr__(self, "entries", entries)

    def entry_map(self) -> dict[str, PortableCodecContractEntry]:
        return {entry.tag: entry for entry in self.entries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PortableCodecContract":
        if set(value) != {"schema", "schema_version", "package_version", "entries"}:
            raise ValidationError("portable codec contract fields do not match the schema")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise ValidationError("portable codec contract entries must be a list")
        if any(not isinstance(entry, Mapping) for entry in raw_entries):
            raise ValidationError("portable codec contract entries must be objects")
        return cls(
            schema=str(value["schema"]),
            schema_version=value["schema_version"],
            package_version=str(value["package_version"]),
            entries=tuple(PortableCodecContractEntry.from_dict(entry) for entry in raw_entries),
        )


@dataclass(frozen=True, slots=True)
class PortableCodecCompatibilityReport:
    """Comparison of a committed codec contract with the active registry."""

    reference_version: str
    current_version: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    type_changed: tuple[tuple[str, str, str], ...]
    version_decreased: tuple[tuple[str, int, int], ...]
    version_increased: tuple[tuple[str, int, int], ...]
    missing_migrations: tuple[tuple[str, int, int], ...]
    certificate: Certificate

    @property
    def compatible(self) -> bool:
        return self.certificate.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_version": self.reference_version,
            "current_version": self.current_version,
            "compatible": self.compatible,
            "added": list(self.added),
            "removed": list(self.removed),
            "type_changed": [list(value) for value in self.type_changed],
            "version_decreased": [list(value) for value in self.version_decreased],
            "version_increased": [list(value) for value in self.version_increased],
            "missing_migrations": [list(value) for value in self.missing_migrations],
            "certificate": self.certificate.to_dict(),
        }


def _qualified_type(type_: type[Any]) -> str:
    return f"{type_.__module__}.{type_.__qualname__}"


def create_portable_codec_contract(
    *,
    package_version: str | None = None,
) -> PortableCodecContract:
    """Snapshot every package-owned codec in deterministic tag order."""

    entries = tuple(
        sorted(
            PortableCodecContractEntry(
                codec.tag,
                codec.version,
                _qualified_type(codec.type_),
            )
            for codec in portable_codecs()
            if codec.type_.__module__.startswith("komorebi_math.")
        )
    )
    return PortableCodecContract(
        package_version=__version__ if package_version is None else str(package_version),
        entries=entries,
    )


def load_portable_codec_contract(
    path: str | Path | None = None,
) -> PortableCodecContract:
    """Load and validate the committed codec compatibility contract."""

    source = DEFAULT_PORTABLE_CODEC_CONTRACT if path is None else Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"could not read portable codec contract {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError("portable codec contract is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError("portable codec contract root must be an object")
    return PortableCodecContract.from_dict(raw)


def save_portable_codec_contract(
    contract: PortableCodecContract,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a canonical portable-codec contract."""

    if not isinstance(contract, PortableCodecContract):
        raise ValidationError("save_portable_codec_contract requires a PortableCodecContract")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise ValidationError(f"refusing to overwrite portable codec contract {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(contract.to_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def compare_portable_codec_contract(
    reference: PortableCodecContract,
    current: PortableCodecContract | None = None,
) -> PortableCodecCompatibilityReport:
    """Check tag, type, version, and adjacent migration compatibility."""

    if not isinstance(reference, PortableCodecContract):
        raise ValidationError("reference must be a PortableCodecContract")
    observed = create_portable_codec_contract() if current is None else current
    if not isinstance(observed, PortableCodecContract):
        raise ValidationError("current must be a PortableCodecContract")

    expected = reference.entry_map()
    actual = observed.entry_map()
    expected_tags = set(expected)
    actual_tags = set(actual)
    added = tuple(sorted(actual_tags - expected_tags))
    removed = tuple(sorted(expected_tags - actual_tags))
    common = sorted(expected_tags & actual_tags)
    type_changed = tuple(
        (tag, expected[tag].implementation_type, actual[tag].implementation_type)
        for tag in common
        if expected[tag].implementation_type != actual[tag].implementation_type
    )
    version_decreased = tuple(
        (tag, expected[tag].version, actual[tag].version)
        for tag in common
        if actual[tag].version < expected[tag].version
    )
    version_increased = tuple(
        (tag, expected[tag].version, actual[tag].version)
        for tag in common
        if actual[tag].version > expected[tag].version
    )

    migration_edges = {
        (migration.tag, migration.from_version, migration.to_version)
        for migration in portable_migrations()
    }
    missing: list[tuple[str, int, int]] = []
    for tag, old, new in version_increased:
        for source in range(old, new):
            edge = (tag, source, source + 1)
            if edge not in migration_edges:
                missing.append(edge)
    missing_migrations = tuple(sorted(missing))

    valid = not removed and not type_changed and not version_decreased and not missing_migrations
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={
            "removed_tags": float(len(removed)),
            "type_changes": float(len(type_changed)),
            "version_decreases": float(len(version_decreased)),
            "missing_migrations": float(len(missing_migrations)),
        },
        assumptions_checked=(
            "package-owned portable codec tags",
            "codec implementation type identity",
            "monotone codec versions",
            "adjacent migration path for each version increase",
        ),
        assumptions_unchecked=(
            "semantic equivalence of extension-owned codecs",
            "compatibility of undocumented serialized implementation details",
        ),
        diagnostics={
            "reference_entries": len(reference.entries),
            "current_entries": len(observed.entries),
            "added": len(added),
            "version_increased": len(version_increased),
        },
        theorem_id="KM-PERSIST-016",
        message=(
            "portable codec registry is compatible with the committed 1.0 candidate contract"
            if valid
            else "portable codec registry is incompatible with the committed 1.0 candidate contract"
        ),
    )
    return PortableCodecCompatibilityReport(
        reference_version=reference.package_version,
        current_version=observed.package_version,
        added=added,
        removed=removed,
        type_changed=type_changed,
        version_decreased=version_decreased,
        version_increased=version_increased,
        missing_migrations=missing_migrations,
        certificate=certificate,
    )


def check_portable_codec_contract(
    path: str | Path | None = None,
) -> PortableCodecCompatibilityReport:
    """Compare the active registry with the committed codec contract."""

    return compare_portable_codec_contract(load_portable_codec_contract(path))


__all__ = [
    "DEFAULT_PORTABLE_CODEC_CONTRACT",
    "PORTABLE_CODEC_CONTRACT_SCHEMA",
    "PORTABLE_CODEC_CONTRACT_SCHEMA_VERSION",
    "PortableCodecCompatibilityReport",
    "PortableCodecContract",
    "PortableCodecContractEntry",
    "check_portable_codec_contract",
    "compare_portable_codec_contract",
    "create_portable_codec_contract",
    "load_portable_codec_contract",
    "save_portable_codec_contract",
]
