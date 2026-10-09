"""Versioned, digest-checked persistence for finite Komorebi objects.

The ordinary export layer is intentionally permissive and presentation focused.
This module provides a stricter round-trip format for executable finite records.
Every nonprimitive type uses an explicit codec tag and version.  The top-level
payload is hashed in canonical JSON form, so accidental modification is detected
before reconstruction.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import isfinite, prod
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from .._version import __version__
from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError

T = TypeVar("T")
Encoder = Callable[[Any], Any]
Decoder = Callable[[Any], Any]
Migrator = Callable[[Any], Any]

PORTABLE_SCHEMA = "komorebi-math.portable"
PORTABLE_SCHEMA_VERSION = 1
_TYPE_KEY = "__komorebi_type__"
_VERSION_KEY = "__version__"
_DATA_KEY = "data"


@dataclass(frozen=True, slots=True)
class PortableCodec:
    """One explicit type codec in the portable persistence registry."""

    tag: str
    version: int
    type_: type[Any]
    encoder: Encoder
    decoder: Decoder


@dataclass(frozen=True, slots=True, order=True)
class PortableMigration:
    """One explicit, adjacent codec-data migration."""

    tag: str
    from_version: int
    to_version: int
    migrator: Migrator


_CODECS_BY_TYPE: dict[type[Any], PortableCodec] = {}
_CODECS_BY_TAG: dict[str, PortableCodec] = {}
_MIGRATIONS_BY_TAG: dict[str, dict[int, PortableMigration]] = {}
_BUILTINS_REGISTERED = False


def _validate_tag(tag: str) -> str:
    normalized = str(tag).strip()
    if not normalized or any(character.isspace() for character in normalized):
        raise ValidationError("portable codec tags must be nonempty and contain no whitespace")
    return normalized


def _validate_version(version: int) -> int:
    if type(version) is not int or version < 1:
        raise ValidationError("portable codec versions must be positive integers")
    return version


def _portable_integer(value: Any, *, label: str) -> int:
    """Accept the JSON integer type without truncation or boolean coercion."""
    if type(value) is not int:
        raise ValidationError(f"{label} must be an integer")
    return value


def _check_json_nesting(text: str, max_depth: int) -> None:
    """Reject excessive syntax nesting before entering the recursive JSON parser.

    The generous syntax bound allows the tagged wrappers used by the portable
    representation. The semantic depth check still runs during decoding. This
    scanner is not a JSON validator or a byte/allocation resource policy.
    """
    bound = 3 * (max_depth + 1) + 4
    nesting = 0
    quoted = False
    escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            nesting += 1
            if nesting > bound:
                raise ValidationError("maximum portable JSON nesting depth exceeded")
        elif character in "]}":
            nesting -= 1


def register_portable_codec(
    type_: type[T],
    *,
    tag: str,
    version: int,
    encoder: Callable[[T], Any],
    decoder: Callable[[Any], T],
    replace: bool = False,
) -> None:
    """Register one stable round-trip codec."""

    if not isinstance(type_, type):
        raise ValidationError("portable codec type must be a Python type")
    if not callable(encoder) or not callable(decoder):
        raise ValidationError("portable codec encoder and decoder must be callable")
    normalized_tag = _validate_tag(tag)
    normalized_version = _validate_version(version)
    existing_type = _CODECS_BY_TYPE.get(type_)
    existing_tag = _CODECS_BY_TAG.get(normalized_tag)
    if not replace and (existing_type is not None or existing_tag is not None):
        raise ValidationError("a portable codec is already registered for this type or tag")
    if replace:
        if existing_type is not None:
            _CODECS_BY_TAG.pop(existing_type.tag, None)
        if existing_tag is not None:
            _CODECS_BY_TYPE.pop(existing_tag.type_, None)
    codec = PortableCodec(
        normalized_tag,
        normalized_version,
        type_,
        encoder,
        decoder,
    )
    _CODECS_BY_TYPE[type_] = codec
    _CODECS_BY_TAG[normalized_tag] = codec


def register_portable_migration(
    *,
    tag: str,
    from_version: int,
    to_version: int,
    migrator: Migrator,
    replace: bool = False,
) -> None:
    """Register one deterministic adjacent migration for codec data.

    Migrations are intentionally restricted to ``n -> n + 1`` steps.  A load
    applies every registered step in sequence until the active codec version is
    reached.  The migrator receives already decoded safe data and cannot select
    or import a Python class from the persisted payload.
    """

    normalized_tag = _validate_tag(tag)
    source = _validate_version(from_version)
    target = _validate_version(to_version)
    if target != source + 1:
        raise ValidationError("portable migrations must advance exactly one codec version")
    if not callable(migrator):
        raise ValidationError("portable migration must be callable")
    migrations = _MIGRATIONS_BY_TAG.setdefault(normalized_tag, {})
    if source in migrations and not replace:
        raise ValidationError(
            f"a portable migration is already registered for {normalized_tag!r} version {source}"
        )
    migrations[source] = PortableMigration(
        normalized_tag,
        source,
        target,
        migrator,
    )


def portable_migrations() -> tuple[PortableMigration, ...]:
    """Return registered codec migrations in deterministic order."""

    _ensure_builtin_codecs()
    return tuple(
        _MIGRATIONS_BY_TAG[tag][version]
        for tag in sorted(_MIGRATIONS_BY_TAG)
        for version in sorted(_MIGRATIONS_BY_TAG[tag])
    )


def _migrate_codec_data(
    tag: str,
    version: int,
    data: Any,
    *,
    target_version: int,
) -> Any:
    if version > target_version:
        raise ValidationError(
            f"unsupported future codec version {version} for {tag!r}; "
            f"current version is {target_version}"
        )
    current = version
    migrated = data
    while current < target_version:
        try:
            migration = _MIGRATIONS_BY_TAG[tag][current]
        except KeyError as exc:
            raise ValidationError(
                f"unsupported codec version {version} for {tag!r}; "
                f"no migration path to version {target_version}"
            ) from exc
        try:
            migrated = migration.migrator(migrated)
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(
                f"portable migration for {tag!r} from version {current} failed"
            ) from exc
        current = migration.to_version
    return migrated


_CERTIFICATE_RECORD_KIND = "historical_certificate_record"
_CERTIFICATE_V1_FIELDS = {
    "valid",
    "exactness",
    "residuals",
    "tolerance",
    "assumptions_checked",
    "assumptions_unchecked",
    "diagnostics",
    "theorem_id",
    "message",
}
_CERTIFICATE_V2_FIELDS = {
    "record_kind",
    "recorded_valid",
    "exactness",
    "residuals",
    "tolerance",
    "assumptions_checked",
    "assumptions_unchecked",
    "diagnostics",
    "theorem_id",
    "message",
}


def _certificate_to_data(value: Certificate) -> dict[str, Any]:
    return {
        "record_kind": _CERTIFICATE_RECORD_KIND,
        "recorded_valid": value.recorded_valid,
        "exactness": value.exactness.value,
        "residuals": value.residuals,
        "tolerance": value.tolerance,
        "assumptions_checked": value.assumptions_checked,
        "assumptions_unchecked": value.assumptions_unchecked,
        "diagnostics": value.diagnostics,
        "theorem_id": value.theorem_id,
        "message": value.message,
    }


def _certificate_v1_to_v2(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping) or set(data) != _CERTIFICATE_V1_FIELDS:
        raise ValidationError("portable certificate fields do not match version 1 schema")
    return {
        "record_kind": _CERTIFICATE_RECORD_KIND,
        "recorded_valid": data["valid"],
        "exactness": data["exactness"],
        "residuals": data["residuals"],
        "tolerance": data["tolerance"],
        "assumptions_checked": data["assumptions_checked"],
        "assumptions_unchecked": data["assumptions_unchecked"],
        "diagnostics": data["diagnostics"],
        "theorem_id": data["theorem_id"],
        "message": data["message"],
    }


def _finite_certificate_number(value: Any, *, label: str, nonnegative: bool = False) -> float | int | Fraction:
    if isinstance(value, (int, np.integer, Fraction)) and not isinstance(value, (bool, np.bool_)):
        normalized = int(value) if isinstance(value, (int, np.integer)) else value
        if nonnegative and normalized < 0:
            raise ValidationError(f"portable certificate {label} must be finite nonnegative")
        # Exact integer and rational evidence must not round through float64.
        return normalized
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationError(f"portable certificate {label} must be a finite real number")
    normalized = float(value)
    if not isfinite(normalized) or (nonnegative and normalized < 0.0):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise ValidationError(f"portable certificate {label} must be {qualifier}")
    return normalized


def _certificate_from_data(data: Mapping[str, Any]) -> Certificate:
    if not isinstance(data, Mapping) or set(data) != _CERTIFICATE_V2_FIELDS:
        raise ValidationError("portable certificate fields do not match version 2 schema")
    if data["record_kind"] != _CERTIFICATE_RECORD_KIND:
        raise ValidationError("portable certificate record kind is unsupported")
    if type(data["recorded_valid"]) is not bool:
        raise ValidationError("portable certificate recorded validity must be boolean")
    try:
        exactness = Exactness(data["exactness"])
    except (TypeError, ValueError) as exc:
        raise ValidationError("portable certificate exactness is invalid") from exc
    raw_residuals = data["residuals"]
    if not isinstance(raw_residuals, Mapping) or any(type(key) is not str for key in raw_residuals):
        raise ValidationError("portable certificate residuals must be a string-keyed mapping")
    residuals = {
        key: _finite_certificate_number(value, label=f"residual {key!r}")
        for key, value in raw_residuals.items()
    }
    tolerance = data["tolerance"]
    if tolerance is not None:
        tolerance = _finite_certificate_number(
            tolerance, label="tolerance", nonnegative=True
        )
    checked = data["assumptions_checked"]
    unchecked = data["assumptions_unchecked"]
    if not isinstance(checked, tuple) or any(type(value) is not str for value in checked):
        raise ValidationError("portable certificate checked assumptions must be a tuple of strings")
    if not isinstance(unchecked, tuple) or any(type(value) is not str for value in unchecked):
        raise ValidationError("portable certificate unchecked assumptions must be a tuple of strings")
    diagnostics = data["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise ValidationError("portable certificate diagnostics must be a mapping")
    theorem_id = data["theorem_id"]
    if theorem_id is not None and type(theorem_id) is not str:
        raise ValidationError("portable certificate theorem_id must be text or null")
    if type(data["message"]) is not str:
        raise ValidationError("portable certificate message must be text")
    certificate = Certificate(
        valid=False,
        exactness=exactness,
        residuals=residuals,
        tolerance=tolerance,
        assumptions_checked=checked,
        assumptions_unchecked=unchecked,
        diagnostics=dict(diagnostics),
        theorem_id=theorem_id,
        message=data["message"],
    )
    object.__setattr__(certificate, "_recorded_valid", data["recorded_valid"])
    object.__setattr__(certificate, "_revalidation_required", True)
    return certificate


def _packet_jet_snapshot_from_data(data: Mapping[str, Any]) -> Any:
    """Decode the current packet-jet snapshot payload after migration."""

    from ..nonlinear.snapshots import PacketJetSnapshot

    ordering = str(data["coefficient_ordering"])
    if ordering != "lexicographic":
        raise ValidationError("packet-jet snapshot coefficient ordering is unsupported")
    return PacketJetSnapshot(
        address=data["address"],
        source_dimension=int(data["source_dimension"]),
        target_dimension=int(data["target_dimension"]),
        provider_name=str(data["provider"]),
        exactness=Exactness(data["exactness"]),
        point=tuple(data["point"]),
        coefficients=tuple(data["coefficients"]),
        capture_certificate=data["capture_certificate"],
    )


def _ensure_builtin_codecs() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from ..deformation.forests import LiftBranch
    from ..nonlinear.filtered import (
        FilteredSpectralSequenceResult,
        SpectralSequenceEntry,
        SpectralSequencePage,
    )
    from ..nonlinear.jets import PacketJetAddress
    from ..nonlinear.lifting import FiniteLiftTree, LiftTreeNode
    from ..nonlinear.snapshots import PacketJetCoefficient, PacketJetSnapshot
    from ..nonlinear.sparse import SparseSymmetricTensor, SparseTensorEntry
    from ..nonlinear.trees import OrderedRootedTree

    register_portable_codec(
        Certificate,
        tag="core.certificate",
        version=2,
        encoder=_certificate_to_data,
        decoder=_certificate_from_data,
        replace=True,
    )
    register_portable_migration(
        tag="core.certificate",
        from_version=1,
        to_version=2,
        migrator=_certificate_v1_to_v2,
        replace=True,
    )
    register_portable_codec(
        OrderedRootedTree,
        tag="nonlinear.ordered_rooted_tree",
        version=1,
        encoder=lambda value: {"children": value.children},
        decoder=lambda data: OrderedRootedTree(tuple(data["children"])),
        replace=True,
    )
    register_portable_codec(
        PacketJetAddress,
        tag="nonlinear.packet_jet_address",
        version=1,
        encoder=lambda value: {"output": value.output, "inputs": value.inputs},
        decoder=lambda data: PacketJetAddress(str(data["output"]), tuple(data["inputs"])),
        replace=True,
    )
    register_portable_codec(
        PacketJetCoefficient,
        tag="nonlinear.packet_jet_coefficient",
        version=1,
        encoder=lambda value: {"indices": value.indices, "value": value.value},
        decoder=lambda data: PacketJetCoefficient(tuple(data["indices"]), tuple(data["value"])),
        replace=True,
    )
    register_portable_codec(
        PacketJetSnapshot,
        tag="nonlinear.packet_jet_snapshot",
        version=2,
        encoder=lambda value: {
            "address": value.address,
            "source_dimension": value.source_dimension,
            "target_dimension": value.target_dimension,
            "provider": value.provider_name,
            "exactness": value.exactness.value,
            "point": value.point,
            "coefficient_ordering": "lexicographic",
            "coefficients": value.coefficients,
            "capture_certificate": value.capture_certificate,
        },
        decoder=_packet_jet_snapshot_from_data,
        replace=True,
    )
    register_portable_migration(
        tag="nonlinear.packet_jet_snapshot",
        from_version=1,
        to_version=2,
        migrator=lambda data: {
            "address": data["address"],
            "source_dimension": data["source_dimension"],
            "target_dimension": data["target_dimension"],
            "provider": data["provider_name"],
            "exactness": data["exactness"],
            "point": data["point"],
            "coefficient_ordering": "lexicographic",
            "coefficients": data["coefficients"],
            "capture_certificate": data["capture_certificate"],
        },
        replace=True,
    )
    register_portable_codec(
        SparseSymmetricTensor,
        tag="nonlinear.sparse_symmetric_tensor",
        version=1,
        encoder=lambda value: {
            "order": value.order,
            "source_dimension": value.source_dimension,
            "target_dimension": value.target_dimension,
            "entries": tuple(
                {
                    "output": entry.output,
                    "inputs": entry.inputs,
                    "value": entry.value,
                }
                for entry in value.entries
            ),
        },
        decoder=lambda data: SparseSymmetricTensor(
            int(data["order"]),
            int(data["source_dimension"]),
            int(data["target_dimension"]),
            tuple(
                SparseTensorEntry(
                    int(entry["output"]),
                    tuple(int(index) for index in entry["inputs"]),
                    entry["value"],
                )
                for entry in data["entries"]
            ),
        ),
        replace=True,
    )
    register_portable_codec(
        LiftBranch,
        tag="deformation.lift_branch",
        version=1,
        encoder=lambda value: {"nodes": value.nodes},
        decoder=lambda data: LiftBranch(tuple(data["nodes"])),
        replace=True,
    )
    register_portable_codec(
        LiftTreeNode,
        tag="nonlinear.lift_tree_node",
        version=1,
        encoder=lambda value: {
            "value": value.value,
            "depth": value.depth,
            "parent_index": value.parent_index,
        },
        decoder=lambda data: LiftTreeNode(
            tuple(int(value) for value in data["value"]),
            int(data["depth"]),
            None if data["parent_index"] is None else int(data["parent_index"]),
        ),
        replace=True,
    )
    register_portable_codec(
        FiniteLiftTree,
        tag="nonlinear.finite_lift_tree",
        version=1,
        encoder=lambda value: {"prime": value.prime, "levels": value.levels},
        decoder=lambda data: FiniteLiftTree(int(data["prime"]), tuple(tuple(level) for level in data["levels"])),
        replace=True,
    )
    register_portable_codec(
        SpectralSequenceEntry,
        tag="nonlinear.spectral_sequence_entry",
        version=1,
        encoder=lambda value: {
            "page": value.page,
            "filtration_degree": value.filtration_degree,
            "complementary_degree": value.complementary_degree,
            "total_degree": value.total_degree,
            "dimension": value.dimension,
            "outgoing_rank": value.outgoing_rank,
            "incoming_rank": value.incoming_rank,
        },
        decoder=lambda data: SpectralSequenceEntry(
            int(data["page"]),
            int(data["filtration_degree"]),
            int(data["complementary_degree"]),
            int(data["total_degree"]),
            int(data["dimension"]),
            int(data["outgoing_rank"]),
            int(data["incoming_rank"]),
        ),
        replace=True,
    )
    register_portable_codec(
        SpectralSequencePage,
        tag="nonlinear.spectral_sequence_page",
        version=1,
        encoder=lambda value: {
            "page": value.page,
            "entries": value.entries,
            "total_dimension": value.total_dimension,
            "total_differential_rank": value.total_differential_rank,
            "certificate": value.certificate,
        },
        decoder=lambda data: SpectralSequencePage(
            int(data["page"]),
            tuple(data["entries"]),
            int(data["total_dimension"]),
            int(data["total_differential_rank"]),
            data["certificate"],
        ),
        replace=True,
    )
    register_portable_codec(
        FilteredSpectralSequenceResult,
        tag="nonlinear.filtered_spectral_sequence_result",
        version=1,
        encoder=lambda value: {
            "pages": value.pages,
            "total_cohomology_dimensions": value.total_cohomology_dimensions,
            "limiting_dimensions": value.limiting_dimensions,
            "certificate": value.certificate,
        },
        decoder=lambda data: FilteredSpectralSequenceResult(
            tuple(data["pages"]),
            {int(key): int(value) for key, value in data["total_cohomology_dimensions"].items()},
            {int(key): int(value) for key, value in data["limiting_dimensions"].items()},
            data["certificate"],
        ),
        replace=True,
    )
    from ..generating.portable import register_generating_codecs
    from ..coordinates.portable import register_coordinate_codecs
    from ..refinement.portable import register_refinement_codecs

    register_generating_codecs()
    register_coordinate_codecs()
    register_refinement_codecs()
    _BUILTINS_REGISTERED = True


def portable_codecs() -> tuple[PortableCodec, ...]:
    """Return the active codecs ordered by stable tag."""

    _ensure_builtin_codecs()
    return tuple(_CODECS_BY_TAG[tag] for tag in sorted(_CODECS_BY_TAG))


def _codec_for(value: Any) -> PortableCodec | None:
    _ensure_builtin_codecs()
    for type_ in type(value).__mro__:
        codec = _CODECS_BY_TYPE.get(type_)
        if codec is not None:
            return codec
    return None


def _tagged(tag: str, version: int, data: Any) -> dict[str, Any]:
    return {_TYPE_KEY: tag, _VERSION_KEY: version, _DATA_KEY: data}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate keys."""

    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValidationError(f"portable JSON contains duplicate object key {key!r}")
        output[key] = value
    return output


def _reject_nonfinite_json_constant(value: str) -> Any:
    """Reject the nonstandard NaN and infinity constants accepted by ``json``."""

    raise ValidationError(f"portable JSON contains nonfinite constant {value}")


def _encode(value: Any, *, depth: int, max_depth: int) -> Any:
    if depth > max_depth:
        raise ValidationError("maximum portable serialization depth exceeded")
    next_depth = depth + 1

    codec = _codec_for(value)
    if codec is not None:
        return _tagged(
            codec.tag,
            codec.version,
            _encode(codec.encoder(value), depth=next_depth, max_depth=max_depth),
        )

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValidationError("portable serialization rejects nonfinite floats")
        return value
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, np.generic):
            raise ValidationError(
                "portable serialization does not support extended NumPy scalar precision"
            )
        return _encode(item, depth=next_depth, max_depth=max_depth)
    if isinstance(value, Fraction):
        return _tagged(
            "python.fraction",
            1,
            # Fraction can retain fixed-width NumPy integer components. JSON
            # requires native integers, and later exact arithmetic must not
            # inherit fixed-width overflow from those scalar implementations.
            {"numerator": int(value.numerator), "denominator": int(value.denominator)},
        )
    if isinstance(value, complex):
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValidationError("portable serialization rejects nonfinite complex values")
        return _tagged("python.complex", 1, {"real": value.real, "imag": value.imag})
    if isinstance(value, Path):
        return _tagged("python.path", 1, str(value))
    if isinstance(value, bytes):
        return _tagged("python.bytes", 1, base64.b64encode(value).decode("ascii"))
    if isinstance(value, np.ndarray):
        if value.dtype.fields is not None or value.dtype.subdtype is not None:
            raise ValidationError("portable ndarray structured/subarray dtype is unsupported")
        flat = [
            _encode(item, depth=next_depth, max_depth=max_depth)
            for item in value.reshape(-1).tolist()
        ]
        return _tagged(
            "numpy.ndarray",
            1,
            {
                "dtype": str(value.dtype),
                "shape": [int(entry) for entry in value.shape],
                "data": flat,
            },
        )
    if isinstance(value, tuple):
        return _tagged(
            "python.tuple",
            1,
            [_encode(item, depth=next_depth, max_depth=max_depth) for item in value],
        )
    if isinstance(value, list):
        return [
            _encode(item, depth=next_depth, max_depth=max_depth)
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        encoded = [
            _encode(item, depth=next_depth, max_depth=max_depth)
            for item in value
        ]
        encoded.sort(key=_canonical_json)
        return _tagged(
            "python.frozenset" if isinstance(value, frozenset) else "python.set",
            1,
            encoded,
        )
    if isinstance(value, Mapping):
        pairs = [
            [
                _encode(key, depth=next_depth, max_depth=max_depth),
                _encode(item, depth=next_depth, max_depth=max_depth),
            ]
            for key, item in value.items()
        ]
        pairs.sort(key=lambda pair: _canonical_json(pair[0]))
        return _tagged("python.mapping", 1, pairs)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _tagged(
            "python.tuple",
            1,
            [_encode(item, depth=next_depth, max_depth=max_depth) for item in value],
        )
    raise ValidationError(
        f"object of type {type(value).__module__}.{type(value).__qualname__} has no portable codec"
    )


def _decode(value: Any, *, depth: int, max_depth: int) -> Any:
    if depth > max_depth:
        raise ValidationError("maximum portable deserialization depth exceeded")
    next_depth = depth + 1
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValidationError("portable payload contains a nonfinite float")
        return value
    if isinstance(value, list):
        return [_decode(item, depth=next_depth, max_depth=max_depth) for item in value]
    if not isinstance(value, dict):
        raise ValidationError("portable payload contains an invalid JSON node")
    if _TYPE_KEY not in value:
        return {
            key: _decode(item, depth=next_depth, max_depth=max_depth)
            for key, item in value.items()
        }
    if set(value) != {_TYPE_KEY, _VERSION_KEY, _DATA_KEY}:
        raise ValidationError("portable tagged values must contain exactly type, version, and data")
    if not isinstance(value[_TYPE_KEY], str):
        raise ValidationError("portable codec tag must be a string")
    tag = value[_TYPE_KEY]
    version = _validate_version(value[_VERSION_KEY])
    raw_data = value[_DATA_KEY]

    if tag == "python.fraction":
        if version != 1:
            raise ValidationError("unsupported Fraction codec version")
        if not isinstance(raw_data, Mapping) or set(raw_data) != {"numerator", "denominator"}:
            raise ValidationError("portable Fraction data does not match its schema")
        try:
            return Fraction(
                _portable_integer(raw_data["numerator"], label="Fraction numerator"),
                _portable_integer(raw_data["denominator"], label="Fraction denominator"),
            )
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValidationError("portable Fraction data is invalid") from exc
    if tag == "python.complex":
        if version != 1:
            raise ValidationError("unsupported complex codec version")
        if not isinstance(raw_data, Mapping) or set(raw_data) != {"real", "imag"}:
            raise ValidationError("portable complex data does not match its schema")
        try:
            real = float(raw_data["real"])
            imag = float(raw_data["imag"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("portable complex data is invalid") from exc
        if not isfinite(real) or not isfinite(imag):
            raise ValidationError("portable complex data contains a nonfinite component")
        return complex(real, imag)
    if tag == "python.path":
        if version != 1:
            raise ValidationError("unsupported path codec version")
        return Path(str(raw_data))
    if tag == "python.bytes":
        if version != 1:
            raise ValidationError("unsupported bytes codec version")
        try:
            return base64.b64decode(str(raw_data), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValidationError("portable bytes payload is not valid base64") from exc
    if tag == "python.tuple":
        if version != 1:
            raise ValidationError("unsupported tuple codec version")
        if not isinstance(raw_data, list):
            raise ValidationError("portable tuple data must be an array")
        return tuple(_decode(item, depth=next_depth, max_depth=max_depth) for item in raw_data)
    if tag in {"python.set", "python.frozenset"}:
        if version != 1:
            raise ValidationError("unsupported set codec version")
        if not isinstance(raw_data, list):
            raise ValidationError("portable set data must be an array")
        decoded_items = [_decode(item, depth=next_depth, max_depth=max_depth) for item in raw_data]
        try:
            return frozenset(decoded_items) if tag == "python.frozenset" else set(decoded_items)
        except TypeError as exc:
            raise ValidationError("portable set contains an unhashable item") from exc
    if tag == "python.mapping":
        if version != 1:
            raise ValidationError("unsupported mapping codec version")
        if not isinstance(raw_data, list):
            raise ValidationError("portable mapping data must be an array")
        output: dict[Any, Any] = {}
        for raw_pair in raw_data:
            if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                raise ValidationError("portable mapping entries must be key-value pairs")
            raw_key, raw_item = raw_pair
            key = _decode(raw_key, depth=next_depth, max_depth=max_depth)
            item = _decode(raw_item, depth=next_depth, max_depth=max_depth)
            try:
                if key in output:
                    raise ValidationError("portable mapping contains duplicate keys")
                output[key] = item
            except TypeError as exc:
                raise ValidationError("portable mapping key is not hashable") from exc
        return output
    if tag == "numpy.ndarray":
        if version != 1:
            raise ValidationError("unsupported ndarray codec version")
        if not isinstance(raw_data, Mapping) or set(raw_data) != {"dtype", "shape", "data"}:
            raise ValidationError("portable ndarray data does not match its schema")
        if not isinstance(raw_data["shape"], list) or not isinstance(raw_data["data"], list):
            raise ValidationError("portable ndarray shape and data must be arrays")
        if not isinstance(raw_data["dtype"], str):
            raise ValidationError("portable ndarray dtype must be a string")
        try:
            dtype = np.dtype(raw_data["dtype"])
            shape = tuple(_portable_integer(entry, label="ndarray dimension")
                          for entry in raw_data["shape"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("portable ndarray dtype or shape is invalid") from exc
        if dtype.fields is not None or dtype.subdtype is not None:
            raise ValidationError("portable ndarray structured/subarray dtype is unsupported")
        if any(entry < 0 for entry in shape):
            raise ValidationError("portable ndarray shape cannot contain negative dimensions")
        intp_max = int(np.iinfo(np.intp).max)
        if any(entry > intp_max for entry in shape):
            raise ValidationError("portable ndarray shape exceeds platform index range")
        expected = prod(shape) if shape else 1
        if len(raw_data["data"]) != expected:
            raise ValidationError("ndarray payload length does not match its declared shape")
        if expected > intp_max or expected * dtype.itemsize > intp_max:
            raise ValidationError("portable ndarray storage exceeds platform index range")
        # Only decode nested values after checking the declared cardinality.
        decoded = [
            _decode(item, depth=next_depth, max_depth=max_depth)
            for item in raw_data["data"]
        ]
        if dtype.kind in "iu":
            bounds = np.iinfo(dtype)
            if any(type(item) is not int or not bounds.min <= item <= bounds.max
                   for item in decoded):
                raise ValidationError("portable integer ndarray data must contain in-range integers")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                if dtype.kind == "O":
                    # np.asarray recursively expands nested lists and arrays,
                    # even with dtype=object. Each decoded value is one cell in
                    # the declared outer shape and must remain one cell.
                    array = np.empty(expected, dtype=dtype)
                    for index, item in enumerate(decoded):
                        array[index] = item
                    array = array.reshape(shape)
                else:
                    array = np.asarray(decoded, dtype=dtype).reshape(shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("ndarray payload is incompatible with its declared dtype") from exc
        if dtype.kind in "fc" and not bool(np.all(np.isfinite(array))):
            raise ValidationError("portable ndarray conversion produced nonfinite values")
        return array

    _ensure_builtin_codecs()
    try:
        codec = _CODECS_BY_TAG[tag]
    except KeyError as exc:
        raise ValidationError(f"unknown portable codec tag {tag!r}") from exc
    data = _decode(raw_data, depth=next_depth, max_depth=max_depth)
    if version != codec.version:
        data = _migrate_codec_data(
            tag,
            version,
            data,
            target_version=codec.version,
        )
    try:
        return codec.decoder(data)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"failed to decode portable type {tag!r}") from exc


def to_portable_data(value: Any, *, max_depth: int = 128) -> Any:
    """Encode one value into the versioned portable payload tree."""

    limit = _validate_version(max_depth)
    return _encode(value, depth=0, max_depth=limit)


def from_portable_data(value: Any, *, max_depth: int = 128) -> Any:
    """Decode one previously encoded portable payload tree."""

    limit = _validate_version(max_depth)
    try:
        return _decode(value, depth=0, max_depth=limit)
    except RecursionError as exc:
        raise ValidationError("portable deserialization exceeds interpreter depth") from exc


def portable_digest(value: Any, *, max_depth: int = 128) -> str:
    """Return the canonical SHA256 digest of one encoded payload."""

    encoded = to_portable_data(value, max_depth=max_depth)
    return sha256(_canonical_json(encoded).encode("utf-8")).hexdigest()


def dumps_portable(value: Any, *, indent: int | None = 2, max_depth: int = 128) -> str:
    """Serialize one value into a complete digest-checked JSON envelope."""

    encoded = to_portable_data(value, max_depth=max_depth)
    digest = sha256(_canonical_json(encoded).encode("utf-8")).hexdigest()
    envelope = {
        "schema": PORTABLE_SCHEMA,
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "producer_version": __version__,
        "payload_sha256": digest,
        "payload": encoded,
    }
    return json.dumps(
        envelope,
        sort_keys=True,
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def loads_portable(text: str | bytes, *, max_depth: int = 128) -> Any:
    """Validate and deserialize one complete portable JSON envelope."""

    limit = _validate_version(max_depth)
    if not isinstance(text, (str, bytes)):
        raise ValidationError("portable data must be UTF-8 text or bytes")
    try:
        raw_text = text.decode("utf-8") if isinstance(text, bytes) else text
        _check_json_nesting(raw_text, limit)
        envelope = json.loads(
            raw_text,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except RecursionError as exc:
        raise ValidationError("portable JSON exceeds interpreter nesting depth") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("portable data is not valid UTF-8 JSON") from exc
    if not isinstance(envelope, dict):
        raise ValidationError("portable envelope must be a JSON object")
    required = {"schema", "schema_version", "producer_version", "payload_sha256", "payload"}
    if set(envelope) != required:
        raise ValidationError("portable envelope fields do not match the schema")
    if envelope["schema"] != PORTABLE_SCHEMA:
        raise ValidationError("portable envelope uses an unknown schema")
    if (type(envelope["schema_version"]) is not int
            or envelope["schema_version"] != PORTABLE_SCHEMA_VERSION):
        raise ValidationError("portable envelope schema version is unsupported")
    if (not isinstance(envelope["producer_version"], str)
            or not envelope["producer_version"].strip()):
        raise ValidationError("portable producer_version must be a nonempty string")
    payload = envelope["payload"]
    try:
        canonical_payload = _canonical_json(payload)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValidationError("portable payload is not canonical finite JSON") from exc
    observed = sha256(canonical_payload.encode("utf-8")).hexdigest()
    expected = str(envelope["payload_sha256"])
    if observed != expected:
        raise ValidationError("portable payload SHA256 mismatch")
    return from_portable_data(payload, max_depth=max_depth)


def _publish_exclusive(
    temporary: Path,
    destination: Path,
    *,
    windows: bool | None = None,
) -> None:
    """Publish a complete file without replacing an existing destination.

    Windows rename refuses an existing destination and works on FAT/exFAT as
    well as NTFS. POSIX rename replaces existing files, so that branch requires
    same-directory hard-link support instead. The private override permits
    isolated platform-branch tests without changing the process's OS identity.
    """

    use_rename = os.name == "nt" if windows is None else windows
    try:
        if use_rename:
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
    except FileExistsError as exc:
        raise ValidationError(f"refusing to overwrite existing portable file {destination}") from exc
    temporary.unlink(missing_ok=True)


def save_portable(
    value: Any,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    indent: int | None = 2,
    max_depth: int = 128,
) -> Path:
    """Atomically write a portable envelope to disk.

    Exclusive writes use Windows no-replacement rename or POSIX hard links.
    The POSIX destination filesystem must therefore support hard links when
    ``overwrite`` is false.
    """

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise ValidationError(f"refusing to overwrite existing portable file {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = dumps_portable(value, indent=indent, max_depth=max_depth)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Publish the completed file exclusively. An existence check followed
            # by replace would let concurrent writers silently overwrite each other.
            _publish_exclusive(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_portable(path: str | os.PathLike[str], *, max_depth: int = 128) -> Any:
    """Read and validate a portable envelope from disk."""

    source = Path(path)
    try:
        content = source.read_bytes()
    except OSError as exc:
        raise ValidationError(f"could not read portable file {source}") from exc
    return loads_portable(content, max_depth=max_depth)


__all__ = [
    "PORTABLE_SCHEMA",
    "PORTABLE_SCHEMA_VERSION",
    "PortableCodec",
    "PortableMigration",
    "dumps_portable",
    "from_portable_data",
    "load_portable",
    "loads_portable",
    "portable_codecs",
    "portable_digest",
    "portable_migrations",
    "register_portable_codec",
    "register_portable_migration",
    "save_portable",
    "to_portable_data",
]
