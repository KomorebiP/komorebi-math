"""Chunked, resumable persistence for finite packet-jet snapshots.

The ordinary portable envelope is appropriate when a complete finite object fits
comfortably in memory.  This module stores the standard-basis coefficients of a
packet jet as independently digested chunks inside a directory.  Readers verify
and decode one chunk at a time, so evaluation, selection, and integrity checks
have memory bounded by the configured chunk size rather than by the complete
snapshot.

A direct capture writes into ``<destination>.partial`` and records progress after
every completed chunk.  A later call with ``resume=True`` continues from the
recorded basis ordinal.  Only a complete, verified staging directory is promoted
to the requested destination.  The format provides integrity and resumability,
not authentication or protection against an adversary able to replace every
digest and payload.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .._residuals import coefficient_storage_decision, python_integer_entries
from .._version import __version__
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..export.persistence import from_portable_data, to_portable_data
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .jets import PacketJet, PacketJetAddress
from .snapshots import PacketJetCoefficient, PacketJetSnapshot

PACKET_JET_STREAM_SCHEMA = "komorebi-math.packet-jet-stream"
PACKET_JET_STREAM_SCHEMA_VERSION = 1
PACKET_JET_CHUNK_SCHEMA = "komorebi-math.packet-jet-stream.chunk"
PACKET_JET_CHUNK_SCHEMA_VERSION = 1
STREAM_MANIFEST_NAME = "manifest.json"
STREAM_CHUNK_DIRECTORY = "chunks"
STREAM_PARTIAL_SUFFIX = ".partial"

StreamState = Literal["in_progress", "complete"]
CaptureMode = Literal["direct", "snapshot"]


def _positive_integer(value: Any, *, name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or normalized != value or normalized < 1:
        raise ValidationError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_integer(value: Any, *, name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a nonnegative integer") from exc
    if isinstance(value, bool) or normalized != value or normalized < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return normalized


def _finite_nonnegative(value: Any, *, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be finite and nonnegative") from exc
    if not isfinite(normalized) or normalized < 0:
        raise ValidationError(f"{name} must be finite and nonnegative")
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValidationError(f"stream JSON contains duplicate object key {key!r}")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> Any:
    raise ValidationError(f"stream JSON contains nonfinite constant {value}")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _envelope(schema: str, version: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": schema,
        "schema_version": version,
        "payload_sha256": _digest(payload),
        "payload": dict(payload),
    }


def _read_envelope(
    path: Path,
    *,
    schema: str,
    version: int,
) -> dict[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except OSError as exc:
        raise ValidationError(f"could not read packet-jet stream file {path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError(f"packet-jet stream file {path} is not valid UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError(f"packet-jet stream file {path} must contain a JSON object")
    if set(raw) != {"schema", "schema_version", "payload_sha256", "payload"}:
        raise ValidationError(f"packet-jet stream file {path} has invalid envelope fields")
    if raw["schema"] != schema or raw["schema_version"] != version:
        raise ValidationError(f"packet-jet stream file {path} uses an unsupported schema")
    payload = raw["payload"]
    if not isinstance(payload, Mapping):
        raise ValidationError(f"packet-jet stream file {path} has a non-object payload")
    observed = _digest(payload)
    if observed != str(raw["payload_sha256"]):
        raise ValidationError(f"packet-jet stream payload SHA256 mismatch in {path}")
    return dict(payload)


def _staging_path(destination: Path) -> Path:
    return destination.with_name(destination.name + STREAM_PARTIAL_SUFFIX)


def _vector_entries(value: Any, *, dimension: int, name: str) -> tuple[Any, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        normalized_shape = tuple(int(entry) for entry in shape)
        if normalized_shape == (dimension,):
            return tuple(value[index] for index in range(dimension))
        if normalized_shape == (dimension, 1):
            return tuple(value[index, 0] for index in range(dimension))
        if normalized_shape == (1, dimension):
            return tuple(value[0, index] for index in range(dimension))
    array = np.asarray(value, dtype=object)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.shape[0] != dimension:
        raise DimensionError(f"{name} must have dimension {dimension}")
    return tuple(array.tolist())


def _direction_vectors(
    directions: Sequence[Any],
    *,
    order: int,
    source_dimension: int,
) -> tuple[np.ndarray, ...]:
    values = tuple(directions)
    if len(values) != order:
        raise ValidationError(f"order {order} stream requires exactly {order} directions")
    output: list[np.ndarray] = []
    for value in values:
        array = np.asarray(value)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1 or array.shape[0] != source_dimension:
            raise DimensionError(
                f"packet-jet stream direction must have dimension {source_dimension}"
            )
        output.append(array)
    return tuple(output)


def _ordinal_indices(ordinal: int, dimension: int, order: int) -> tuple[int, ...]:
    value = ordinal
    digits = [0] * order
    for position in range(order - 1, -1, -1):
        digits[position] = value % dimension
        value //= dimension
    return tuple(digits)


def _coefficient_to_data(value: PacketJetCoefficient) -> dict[str, Any]:
    return {
        "indices": list(value.indices),
        "value": to_portable_data(tuple(value.value)),
    }


def _coefficient_from_data(
    value: Mapping[str, Any],
    *,
    order: int,
    source_dimension: int,
    target_dimension: int,
) -> PacketJetCoefficient:
    if set(value) != {"indices", "value"}:
        raise ValidationError("stream coefficient fields do not match the schema")
    raw_indices = value["indices"]
    if not isinstance(raw_indices, list):
        raise ValidationError("stream coefficient indices must be an array")
    indices = tuple(int(entry) for entry in raw_indices)
    if len(indices) != order:
        raise DimensionError("stream coefficient order does not match the packet address")
    if any(index < 0 or index >= source_dimension for index in indices):
        raise DimensionError("stream coefficient index is outside the source dimension")
    decoded = from_portable_data(value["value"])
    if not isinstance(decoded, tuple):
        raise ValidationError("stream coefficient value must decode to a tuple")
    if len(decoded) != target_dimension:
        raise DimensionError("stream coefficient value has the wrong target dimension")
    return PacketJetCoefficient(indices, decoded)


def _metadata_payload(
    *,
    address: PacketJetAddress,
    source_dimension: int,
    target_dimension: int,
    provider_name: str,
    exactness: Exactness,
    point: Sequence[Any],
    tolerance: float,
    dense_coefficient_count: int,
    capture_mode: CaptureMode,
    source_digest: str,
) -> dict[str, Any]:
    return {
        "address": to_portable_data(address),
        "source_dimension": source_dimension,
        "target_dimension": target_dimension,
        "provider_name": str(provider_name),
        "exactness": Exactness(exactness).value,
        "point": to_portable_data(tuple(point)),
        "tolerance": tolerance,
        "dense_coefficient_count": dense_coefficient_count,
        "capture_mode": capture_mode,
        "source_digest": source_digest,
    }


def _stream_content_digest(payload: Mapping[str, Any]) -> str:
    chunks = payload.get("chunks", [])
    summary = {
        "metadata_sha256": payload["metadata_sha256"],
        "state": payload["state"],
        "evaluations_completed": payload["evaluations_completed"],
        "coefficients_completed": payload["coefficients_completed"],
        "maximum_omitted_basis_norm": payload["maximum_omitted_basis_norm"],
        "chunk_payload_sha256": [chunk["payload_sha256"] for chunk in chunks],
        "capture_certificate_sha256": None
        if payload.get("capture_certificate") is None
        else _digest(payload["capture_certificate"]),
    }
    return _digest(summary)


@dataclass(frozen=True, slots=True)
class PacketJetStreamChunkRecord:
    """One immutable chunk index entry in a packet-jet stream manifest."""

    index: int
    filename: str
    basis_start: int
    basis_stop: int
    coefficient_start: int
    coefficient_stop: int
    payload_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        index = _nonnegative_integer(self.index, name="chunk index")
        basis_start = _nonnegative_integer(self.basis_start, name="chunk basis start")
        basis_stop = _nonnegative_integer(self.basis_stop, name="chunk basis stop")
        coefficient_start = _nonnegative_integer(
            self.coefficient_start,
            name="chunk coefficient start",
        )
        coefficient_stop = _nonnegative_integer(
            self.coefficient_stop,
            name="chunk coefficient stop",
        )
        byte_count = _positive_integer(self.byte_count, name="chunk byte count")
        if basis_stop < basis_start or coefficient_stop < coefficient_start:
            raise ValidationError("packet-jet stream chunk ranges must be nondecreasing")
        filename = str(self.filename)
        if filename != f"chunk-{index:08d}.json":
            raise ValidationError("packet-jet stream chunk filename is not canonical")
        digest = str(self.payload_sha256)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValidationError("packet-jet stream chunk digest is not lowercase SHA-256")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "basis_start", basis_start)
        object.__setattr__(self, "basis_stop", basis_stop)
        object.__setattr__(self, "coefficient_start", coefficient_start)
        object.__setattr__(self, "coefficient_stop", coefficient_stop)
        object.__setattr__(self, "byte_count", byte_count)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "payload_sha256", digest)

    @property
    def coefficient_count(self) -> int:
        return self.coefficient_stop - self.coefficient_start

    @property
    def basis_count(self) -> int:
        return self.basis_stop - self.basis_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "filename": self.filename,
            "basis_start": self.basis_start,
            "basis_stop": self.basis_stop,
            "coefficient_start": self.coefficient_start,
            "coefficient_stop": self.coefficient_stop,
            "payload_sha256": self.payload_sha256,
            "byte_count": self.byte_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PacketJetStreamChunkRecord":
        required = {
            "index",
            "filename",
            "basis_start",
            "basis_stop",
            "coefficient_start",
            "coefficient_stop",
            "payload_sha256",
            "byte_count",
        }
        if set(value) != required:
            raise ValidationError("packet-jet stream chunk record fields do not match the schema")
        return cls(
            int(value["index"]),
            str(value["filename"]),
            int(value["basis_start"]),
            int(value["basis_stop"]),
            int(value["coefficient_start"]),
            int(value["coefficient_stop"]),
            str(value["payload_sha256"]),
            int(value["byte_count"]),
        )


@dataclass(frozen=True, slots=True)
class PacketJetStreamManifest:
    """Validated metadata and progress for one chunked packet-jet stream."""

    state: StreamState
    producer_version: str
    address: PacketJetAddress
    source_dimension: int
    target_dimension: int
    provider_name: str
    exactness: Exactness
    point: tuple[Any, ...]
    tolerance: float
    dense_coefficient_count: int
    capture_mode: CaptureMode
    source_digest: str
    metadata_sha256: str
    chunk_size: int
    evaluations_completed: int
    coefficients_completed: int
    maximum_omitted_basis_norm: float
    chunks: tuple[PacketJetStreamChunkRecord, ...]
    capture_certificate: Certificate | None
    stream_sha256: str

    def __post_init__(self) -> None:
        if self.state not in {"in_progress", "complete"}:
            raise ValidationError("packet-jet stream state is invalid")
        source_dimension = _positive_integer(self.source_dimension, name="stream source dimension")
        target_dimension = _positive_integer(self.target_dimension, name="stream target dimension")
        dense_count = _positive_integer(
            self.dense_coefficient_count,
            name="stream dense coefficient count",
        )
        chunk_size = _positive_integer(self.chunk_size, name="stream chunk size")
        evaluations_completed = _nonnegative_integer(
            self.evaluations_completed,
            name="stream evaluations completed",
        )
        coefficients_completed = _nonnegative_integer(
            self.coefficients_completed,
            name="stream coefficients completed",
        )
        maximum_omitted = _finite_nonnegative(
            self.maximum_omitted_basis_norm,
            name="maximum omitted basis norm",
        )
        expected_dense_count = source_dimension**self.address.order
        if dense_count != expected_dense_count:
            raise ValidationError(
                "stream dense coefficient count does not match source dimension and order"
            )
        if evaluations_completed > dense_count:
            raise ValidationError("stream evaluations exceed the dense coefficient count")
        if coefficients_completed > dense_count:
            raise ValidationError("stream coefficients exceed the dense coefficient count")
        if self.capture_mode not in {"direct", "snapshot"}:
            raise ValidationError("packet-jet stream capture mode is invalid")
        if self.state == "complete" and evaluations_completed != dense_count:
            raise ValidationError("complete packet-jet stream has unfinished dense-basis coverage")
        if self.state == "complete" and self.capture_certificate is None:
            raise ValidationError("complete packet-jet stream requires a capture certificate")
        if self.state == "in_progress" and self.capture_certificate is not None:
            raise ValidationError("in-progress packet-jet stream cannot have a final certificate")
        if len(self.point) != source_dimension:
            raise DimensionError("stream point does not match the source dimension")
        chunks = tuple(self.chunks)
        expected_basis = 0
        expected_coefficient = 0
        for expected_index, chunk in enumerate(chunks):
            if chunk.index != expected_index:
                raise ValidationError("packet-jet stream chunk indices are not contiguous")
            if chunk.basis_start != expected_basis:
                raise ValidationError("packet-jet stream basis ranges are not contiguous")
            if chunk.coefficient_start != expected_coefficient:
                raise ValidationError("packet-jet stream coefficient ranges are not contiguous")
            if self.capture_mode == "direct":
                if chunk.basis_count < 1 or chunk.basis_count > chunk_size:
                    raise ValidationError("direct stream chunk basis count is outside its chunk size")
                if chunk.coefficient_count > chunk.basis_count:
                    raise ValidationError(
                        "direct stream chunk stores more coefficients than basis evaluations"
                    )
            elif chunk.coefficient_count > chunk_size:
                raise ValidationError(
                    "snapshot stream chunk coefficient count is outside its chunk size"
                )
            expected_basis = chunk.basis_stop
            expected_coefficient = chunk.coefficient_stop
        if expected_basis != evaluations_completed:
            raise ValidationError("stream chunk basis ranges do not match progress")
        if expected_coefficient != coefficients_completed:
            raise ValidationError("stream chunk coefficient ranges do not match progress")
        if self.capture_mode == "snapshot":
            if self.state == "in_progress" and any(chunk.basis_count for chunk in chunks):
                raise ValidationError("in-progress snapshot chunks cannot claim basis coverage")
            if self.state == "complete":
                for chunk in chunks[:-1]:
                    if chunk.basis_count:
                        raise ValidationError(
                            "only the final complete snapshot chunk may close basis coverage"
                        )
                if not chunks or chunks[-1].basis_count != dense_count:
                    raise ValidationError(
                        "complete snapshot stream must close dense-basis coverage in its final chunk"
                    )
        tolerance = _finite_nonnegative(self.tolerance, name="stream tolerance")
        for digest_name, digest_value in (
            ("source", self.source_digest),
            ("metadata", self.metadata_sha256),
            ("stream", self.stream_sha256),
        ):
            digest = str(digest_value)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValidationError(f"packet-jet stream {digest_name} digest is invalid")
        object.__setattr__(self, "source_dimension", source_dimension)
        object.__setattr__(self, "target_dimension", target_dimension)
        object.__setattr__(self, "dense_coefficient_count", dense_count)
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "evaluations_completed", evaluations_completed)
        object.__setattr__(self, "coefficients_completed", coefficients_completed)
        object.__setattr__(self, "maximum_omitted_basis_norm", maximum_omitted)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "exactness", Exactness(self.exactness))
        object.__setattr__(self, "point", tuple(self.point))
        object.__setattr__(self, "chunks", chunks)

    @property
    def complete(self) -> bool:
        return self.state == "complete"

    @property
    def order(self) -> int:
        return self.address.order

    def metadata_dict(self) -> dict[str, Any]:
        return _metadata_payload(
            address=self.address,
            source_dimension=self.source_dimension,
            target_dimension=self.target_dimension,
            provider_name=self.provider_name,
            exactness=self.exactness,
            point=self.point,
            tolerance=self.tolerance,
            dense_coefficient_count=self.dense_coefficient_count,
            capture_mode=self.capture_mode,
            source_digest=self.source_digest,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "state": self.state,
            "producer_version": self.producer_version,
            "metadata": self.metadata_dict(),
            "metadata_sha256": self.metadata_sha256,
            "chunk_size": self.chunk_size,
            "evaluations_completed": self.evaluations_completed,
            "coefficients_completed": self.coefficients_completed,
            "maximum_omitted_basis_norm": self.maximum_omitted_basis_norm,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "capture_certificate": None
            if self.capture_certificate is None
            else to_portable_data(self.capture_certificate),
            "stream_sha256": self.stream_sha256,
        }
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PacketJetStreamManifest":
        required = {
            "state",
            "producer_version",
            "metadata",
            "metadata_sha256",
            "chunk_size",
            "evaluations_completed",
            "coefficients_completed",
            "maximum_omitted_basis_norm",
            "chunks",
            "capture_certificate",
            "stream_sha256",
        }
        if set(payload) != required:
            raise ValidationError("packet-jet stream manifest fields do not match the schema")
        metadata = payload["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValidationError("packet-jet stream metadata must be an object")
        metadata_required = {
            "address",
            "source_dimension",
            "target_dimension",
            "provider_name",
            "exactness",
            "point",
            "tolerance",
            "dense_coefficient_count",
            "capture_mode",
            "source_digest",
        }
        if set(metadata) != metadata_required:
            raise ValidationError("packet-jet stream metadata fields do not match the schema")
        if _digest(metadata) != str(payload["metadata_sha256"]):
            raise ValidationError("packet-jet stream metadata SHA256 mismatch")
        address = from_portable_data(metadata["address"])
        point = from_portable_data(metadata["point"])
        if not isinstance(address, PacketJetAddress):
            raise ValidationError("packet-jet stream address has the wrong portable type")
        if not isinstance(point, tuple):
            raise ValidationError("packet-jet stream point must decode to a tuple")
        raw_chunks = payload["chunks"]
        if not isinstance(raw_chunks, list):
            raise ValidationError("packet-jet stream chunks must be an array")
        certificate_data = payload["capture_certificate"]
        certificate = None if certificate_data is None else from_portable_data(certificate_data)
        if certificate is not None and not isinstance(certificate, Certificate):
            raise ValidationError("packet-jet stream capture certificate has the wrong type")
        manifest = cls(
            state=str(payload["state"]),  # type: ignore[arg-type]
            producer_version=str(payload["producer_version"]),
            address=address,
            source_dimension=int(metadata["source_dimension"]),
            target_dimension=int(metadata["target_dimension"]),
            provider_name=str(metadata["provider_name"]),
            exactness=Exactness(metadata["exactness"]),
            point=point,
            tolerance=float(metadata["tolerance"]),
            dense_coefficient_count=int(metadata["dense_coefficient_count"]),
            capture_mode=str(metadata["capture_mode"]),  # type: ignore[arg-type]
            source_digest=str(metadata["source_digest"]),
            metadata_sha256=str(payload["metadata_sha256"]),
            chunk_size=int(payload["chunk_size"]),
            evaluations_completed=int(payload["evaluations_completed"]),
            coefficients_completed=int(payload["coefficients_completed"]),
            maximum_omitted_basis_norm=float(payload["maximum_omitted_basis_norm"]),
            chunks=tuple(PacketJetStreamChunkRecord.from_dict(value) for value in raw_chunks),
            capture_certificate=certificate,
            stream_sha256=str(payload["stream_sha256"]),
        )
        expected_stream = _stream_content_digest(manifest.to_payload())
        if expected_stream != manifest.stream_sha256:
            raise ValidationError("packet-jet stream content SHA256 mismatch")
        return manifest


@dataclass(frozen=True, slots=True)
class PacketJetStreamWriteResult:
    """Progress returned by a direct or snapshot stream write."""

    destination: Path
    active_path: Path
    completed: bool
    resumed: bool
    chunks_written: int
    evaluations_written: int
    coefficients_written: int
    manifest: PacketJetStreamManifest
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class PacketJetStreamSelection:
    """A bounded coefficient selection loaded from a packet-jet stream."""

    coefficient_start: int
    coefficient_stop: int
    coefficients: tuple[PacketJetCoefficient, ...]
    complete_snapshot: bool
    certificate: Certificate


@dataclass(frozen=True, slots=True)
class StreamedPacketJetEvaluation:
    """One chunk-wise packet-jet evaluation with an explicit coverage record."""

    value: np.ndarray
    coefficient_start: int
    coefficient_stop: int
    total_coefficients: int
    complete_snapshot: bool
    certificate: Certificate


def _manifest_payload(
    *,
    state: StreamState,
    metadata: Mapping[str, Any],
    chunk_size: int,
    evaluations_completed: int,
    coefficients_completed: int,
    maximum_omitted_basis_norm: float,
    chunks: Sequence[PacketJetStreamChunkRecord],
    capture_certificate: Certificate | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": state,
        "producer_version": __version__,
        "metadata": dict(metadata),
        "metadata_sha256": _digest(metadata),
        "chunk_size": chunk_size,
        "evaluations_completed": evaluations_completed,
        "coefficients_completed": coefficients_completed,
        "maximum_omitted_basis_norm": maximum_omitted_basis_norm,
        "chunks": [chunk.to_dict() for chunk in chunks],
        "capture_certificate": None
        if capture_certificate is None
        else to_portable_data(capture_certificate),
        "stream_sha256": "0" * 64,
    }
    payload["stream_sha256"] = _stream_content_digest(payload)
    return payload


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(
        path / STREAM_MANIFEST_NAME,
        _envelope(PACKET_JET_STREAM_SCHEMA, PACKET_JET_STREAM_SCHEMA_VERSION, payload),
    )


def _load_manifest(path: Path) -> PacketJetStreamManifest:
    payload = _read_envelope(
        path / STREAM_MANIFEST_NAME,
        schema=PACKET_JET_STREAM_SCHEMA,
        version=PACKET_JET_STREAM_SCHEMA_VERSION,
    )
    return PacketJetStreamManifest.from_payload(payload)


def _write_chunk(
    root: Path,
    *,
    index: int,
    basis_start: int,
    basis_stop: int,
    coefficient_start: int,
    coefficients: Sequence[PacketJetCoefficient],
) -> PacketJetStreamChunkRecord:
    coefficient_tuple = tuple(coefficients)
    payload = {
        "index": index,
        "basis_start": basis_start,
        "basis_stop": basis_stop,
        "coefficient_start": coefficient_start,
        "coefficient_stop": coefficient_start + len(coefficient_tuple),
        "coefficients": [_coefficient_to_data(value) for value in coefficient_tuple],
    }
    envelope = _envelope(PACKET_JET_CHUNK_SCHEMA, PACKET_JET_CHUNK_SCHEMA_VERSION, payload)
    filename = f"chunk-{index:08d}.json"
    destination = root / STREAM_CHUNK_DIRECTORY / filename
    _atomic_write_json(destination, envelope)
    byte_count = destination.stat().st_size
    return PacketJetStreamChunkRecord(
        index=index,
        filename=filename,
        basis_start=basis_start,
        basis_stop=basis_stop,
        coefficient_start=coefficient_start,
        coefficient_stop=coefficient_start + len(coefficient_tuple),
        payload_sha256=str(envelope["payload_sha256"]),
        byte_count=byte_count,
    )


def _read_chunk(
    root: Path,
    record: PacketJetStreamChunkRecord,
    *,
    order: int,
    source_dimension: int,
    target_dimension: int,
) -> tuple[PacketJetCoefficient, ...]:
    path = root / STREAM_CHUNK_DIRECTORY / record.filename
    payload = _read_envelope(
        path,
        schema=PACKET_JET_CHUNK_SCHEMA,
        version=PACKET_JET_CHUNK_SCHEMA_VERSION,
    )
    required = {
        "index",
        "basis_start",
        "basis_stop",
        "coefficient_start",
        "coefficient_stop",
        "coefficients",
    }
    if set(payload) != required:
        raise ValidationError(f"packet-jet stream chunk {record.index} fields are invalid")
    if _digest(payload) != record.payload_sha256:
        raise ValidationError(f"packet-jet stream chunk {record.index} digest disagrees with manifest")
    expected_values = {
        "index": record.index,
        "basis_start": record.basis_start,
        "basis_stop": record.basis_stop,
        "coefficient_start": record.coefficient_start,
        "coefficient_stop": record.coefficient_stop,
    }
    for name, expected in expected_values.items():
        if payload[name] != expected:
            raise ValidationError(f"packet-jet stream chunk {record.index} {name} mismatch")
    raw_coefficients = payload["coefficients"]
    if not isinstance(raw_coefficients, list):
        raise ValidationError("packet-jet stream chunk coefficients must be an array")
    coefficients = tuple(
        _coefficient_from_data(
            value,
            order=order,
            source_dimension=source_dimension,
            target_dimension=target_dimension,
        )
        for value in raw_coefficients
    )
    if len(coefficients) != record.coefficient_count:
        raise ValidationError("packet-jet stream chunk coefficient count mismatch")
    if path.stat().st_size != record.byte_count:
        raise ValidationError("packet-jet stream chunk byte count mismatch")
    return coefficients


def _verify_committed_prefix(
    root: Path,
    manifest: PacketJetStreamManifest,
) -> None:
    """Re-read every committed chunk before extending an incomplete stream."""

    observed_coefficients = 0
    previous_indices: tuple[int, ...] | None = None
    for record in manifest.chunks:
        coefficients = _read_chunk(
            root,
            record,
            order=manifest.order,
            source_dimension=manifest.source_dimension,
            target_dimension=manifest.target_dimension,
        )
        for coefficient in coefficients:
            if previous_indices is not None and coefficient.indices <= previous_indices:
                raise ValidationError(
                    "packet-jet stream committed coefficients are not strictly lexicographic"
                )
            previous_indices = coefficient.indices
        observed_coefficients += len(coefficients)
    if observed_coefficients != manifest.coefficients_completed:
        raise ValidationError(
            "packet-jet stream committed coefficient count does not match its manifest"
        )


def _clean_orphan_chunks(root: Path, committed_count: int) -> None:
    chunk_directory = root / STREAM_CHUNK_DIRECTORY
    if not chunk_directory.exists():
        return
    for path in chunk_directory.glob("chunk-*.json"):
        try:
            index = int(path.stem.split("-")[-1])
        except ValueError:
            raise ValidationError(f"unexpected packet-jet stream chunk filename {path.name}")
        if index >= committed_count:
            path.unlink()
    for temporary in chunk_directory.glob("*.tmp"):
        temporary.unlink(missing_ok=True)


def _prepare_staging(
    destination: Path,
    *,
    metadata: Mapping[str, Any],
    chunk_size: int,
    resume: bool,
    overwrite: bool,
) -> tuple[Path, PacketJetStreamManifest | None, bool]:
    staging = _staging_path(destination)
    if destination.exists() and not overwrite:
        raise ValidationError(f"refusing to overwrite existing packet-jet stream {destination}")
    if staging.exists():
        if not resume:
            raise ValidationError(
                f"incomplete packet-jet stream exists at {staging}; pass resume=True or remove it"
            )
        manifest = _load_manifest(staging)
        if manifest.complete:
            raise ValidationError("completed packet-jet stream was not promoted from staging")
        if manifest.metadata_sha256 != _digest(metadata):
            raise ValidationError("resume source does not match the incomplete packet-jet stream")
        if manifest.chunk_size != chunk_size:
            raise ValidationError("resume chunk size does not match the incomplete packet-jet stream")
        _verify_committed_prefix(staging, manifest)
        _clean_orphan_chunks(staging, len(manifest.chunks))
        return staging, manifest, True
    staging.mkdir(parents=True, exist_ok=False)
    (staging / STREAM_CHUNK_DIRECTORY).mkdir()
    payload = _manifest_payload(
        state="in_progress",
        metadata=metadata,
        chunk_size=chunk_size,
        evaluations_completed=0,
        coefficients_completed=0,
        maximum_omitted_basis_norm=0.0,
        chunks=(),
        capture_certificate=None,
    )
    _write_manifest(staging, payload)
    return staging, None, False


def _promote(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        os.replace(staging, destination)
        return
    if not overwrite:
        raise ValidationError(f"refusing to overwrite existing packet-jet stream {destination}")
    backup = destination.with_name(
        f".{destination.name}.backup-{uuid.uuid4().hex}"
    )
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        os.replace(backup, destination)
        raise
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink(missing_ok=True)


def _source_digest_for_jet(jet: PacketJet, tolerance: float) -> str:
    identity = {
        "address": to_portable_data(jet.address),
        "source_dimension": jet.source.dimension,
        "target_dimension": jet.target.dimension,
        "provider_name": jet.provider.name,
        "exactness": jet.exactness.value,
        "point": to_portable_data(
            _vector_entries(jet.point, dimension=jet.source.dimension, name="packet-jet point")
        ),
        "tolerance": tolerance,
    }
    return _digest(identity)


def _source_digest_for_snapshot(snapshot: PacketJetSnapshot) -> str:
    return _digest(
        {
            "address": to_portable_data(snapshot.address),
            "source_dimension": snapshot.source_dimension,
            "target_dimension": snapshot.target_dimension,
            "provider_name": snapshot.provider_name,
            "exactness": snapshot.exactness.value,
            "point": to_portable_data(snapshot.point),
            "coefficients": [_coefficient_to_data(value) for value in snapshot.coefficients],
            "capture_certificate": to_portable_data(snapshot.capture_certificate),
        }
    )


def _write_result_certificate(
    *,
    completed: bool,
    resumed: bool,
    manifest: PacketJetStreamManifest,
) -> Certificate:
    remaining_basis = (
        manifest.dense_coefficient_count - manifest.evaluations_completed
        if manifest.capture_mode == "direct"
        else 0
    )
    if completed:
        unchecked: tuple[str, ...] = ()
    elif manifest.capture_mode == "direct":
        unchecked = ("remaining basis evaluations have not yet been captured",)
    else:
        unchecked = ("remaining snapshot coefficients have not yet been persisted",)
    return Certificate(
        valid=True,
        exactness=manifest.exactness if completed else Exactness.CONDITIONAL,
        residuals={
            "remaining_basis_evaluations": float(remaining_basis),
            "incomplete_stream": 0.0 if completed else 1.0,
        },
        tolerance=None if manifest.exactness is Exactness.EXACT else manifest.tolerance,
        assumptions_checked=(
            "every committed chunk has an independent SHA-256 payload digest",
            "manifest progress is updated only after an atomic chunk write",
            "source metadata matches the recorded resume fingerprint",
        ),
        assumptions_unchecked=unchecked,
        diagnostics={
            "completed": completed,
            "resumed": resumed,
            "chunks": len(manifest.chunks),
            "evaluations_completed": manifest.evaluations_completed,
            "stored_coefficients": manifest.coefficients_completed,
        },
        theorem_id="PSC-III-D60",
        message=(
            "packet-jet stream completed and promoted"
            if completed
            else "packet-jet stream progress committed for later resumption"
        ),
    )


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D60",
        title="Chunked resumable packet-jet capture",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Block 11 runtime",
        assumptions=("finite-dimensional multilinear packet jet",),
        limitations=(
            "integrity digests are not signatures",
            "resuming assumes the provider behavior still agrees with its recorded identity",
        ),
        verification_suite="nonlinear.streaming",
    )
)
def capture_packet_jet_stream(
    jet: PacketJet,
    path: str | os.PathLike[str],
    *,
    tolerance: float | None = None,
    max_evaluations: int = 10_000_000,
    chunk_size: int = 1024,
    evaluation_budget: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> PacketJetStreamWriteResult:
    """Capture a packet jet directly into bounded, resumable basis chunks.

    ``evaluation_budget`` limits only the additional work performed by this call.
    When the budget is exhausted, a valid incomplete stream remains in the
    sibling ``.partial`` directory and can be continued with ``resume=True``.
    """

    if not isinstance(jet, PacketJet):
        raise ValidationError("direct stream capture requires a PacketJet")
    destination = Path(path)
    limit = _positive_integer(max_evaluations, name="maximum evaluations")
    size = _positive_integer(chunk_size, name="stream chunk size")
    total = jet.source.dimension**jet.order
    if total > limit:
        raise ValidationError(
            f"packet-jet stream capture requires {total} evaluations, exceeding {limit}"
        )
    if evaluation_budget is None:
        budget = total
    else:
        budget = _nonnegative_integer(evaluation_budget, name="evaluation budget")
    tol = (
        max(jet.source.tolerance, jet.target.tolerance)
        if tolerance is None
        else _finite_nonnegative(tolerance, name="stream capture tolerance")
    )
    point = _vector_entries(jet.point, dimension=jet.source.dimension, name="packet-jet point")
    source_digest = _source_digest_for_jet(jet, tol)
    metadata = _metadata_payload(
        address=jet.address,
        source_dimension=jet.source.dimension,
        target_dimension=jet.target.dimension,
        provider_name=jet.provider.name,
        exactness=jet.exactness,
        point=point,
        tolerance=tol,
        dense_coefficient_count=total,
        capture_mode="direct",
        source_digest=source_digest,
    )
    staging, existing, resumed = _prepare_staging(
        destination,
        metadata=metadata,
        chunk_size=size,
        resume=resume,
        overwrite=overwrite,
    )
    manifest = existing or _load_manifest(staging)
    start_evaluations = manifest.evaluations_completed
    start_coefficients = manifest.coefficients_completed
    chunks = list(manifest.chunks)
    maximum_omitted = manifest.maximum_omitted_basis_norm
    identity = jet.source.backend.eye(jet.source.dimension)
    basis = tuple(identity[:, index] for index in range(jet.source.dimension))
    remaining_budget = min(budget, total - start_evaluations)
    cursor = start_evaluations

    while remaining_budget > 0:
        count = min(size, remaining_budget, total - cursor)
        basis_start = cursor
        basis_stop = cursor + count
        coefficient_values: list[PacketJetCoefficient] = []
        for ordinal in range(basis_start, basis_stop):
            indices = _ordinal_indices(ordinal, jet.source.dimension, jet.order)
            value = jet.evaluate(tuple(basis[index] for index in indices))
            zero, value_norm, _ = coefficient_storage_decision(
                value, exact=jet.exactness is Exactness.EXACT, tolerance=tol)
            if zero:
                maximum_omitted = max(maximum_omitted, value_norm)
                continue
            coefficient_values.append(
                PacketJetCoefficient(
                    indices,
                    _vector_entries(
                        value,
                        dimension=jet.target.dimension,
                        name="packet-jet coefficient",
                    ),
                )
            )
        record = _write_chunk(
            staging,
            index=len(chunks),
            basis_start=basis_start,
            basis_stop=basis_stop,
            coefficient_start=start_coefficients,
            coefficients=coefficient_values,
        )
        chunks.append(record)
        cursor = basis_stop
        start_coefficients = record.coefficient_stop
        remaining_budget -= count
        progress = _manifest_payload(
            state="in_progress",
            metadata=metadata,
            chunk_size=size,
            evaluations_completed=cursor,
            coefficients_completed=start_coefficients,
            maximum_omitted_basis_norm=maximum_omitted,
            chunks=chunks,
            capture_certificate=None,
        )
        _write_manifest(staging, progress)

    completed = cursor == total
    if completed:
        certificate = Certificate(
            valid=True,
            exactness=jet.exactness,
            residuals={"maximum_omitted_basis_norm": maximum_omitted},
            tolerance=None if jet.exactness is Exactness.EXACT else tol,
            assumptions_checked=(
                "complete ambient standard basis",
                "fixed packet address and base point",
                "each committed chunk passed atomic write and digest construction",
            ),
            assumptions_unchecked=(
                "global correctness and multilinearity of the underlying derivative provider",
            ),
            diagnostics={
                "order": jet.order,
                "basis_evaluations": total,
                "stored_coefficients": start_coefficients,
                "chunks": len(chunks),
                "provider": jet.provider.name,
            },
            theorem_id="PSC-III-D60",
            message="packet-jet stream captured on the complete standard basis",
        )
        final_payload = _manifest_payload(
            state="complete",
            metadata=metadata,
            chunk_size=size,
            evaluations_completed=cursor,
            coefficients_completed=start_coefficients,
            maximum_omitted_basis_norm=maximum_omitted,
            chunks=chunks,
            capture_certificate=certificate,
        )
        _write_manifest(staging, final_payload)
        _load_manifest(staging)
        _promote(staging, destination, overwrite=overwrite)
        active = destination
    else:
        active = staging

    final_manifest = _load_manifest(active)
    result_certificate = _write_result_certificate(
        completed=completed,
        resumed=resumed,
        manifest=final_manifest,
    )
    return PacketJetStreamWriteResult(
        destination=destination,
        active_path=active,
        completed=completed,
        resumed=resumed,
        chunks_written=len(final_manifest.chunks) - len(manifest.chunks),
        evaluations_written=final_manifest.evaluations_completed - start_evaluations,
        coefficients_written=final_manifest.coefficients_completed - manifest.coefficients_completed,
        manifest=final_manifest,
        certificate=result_certificate,
    )


def save_packet_jet_snapshot_stream(
    snapshot: PacketJetSnapshot,
    path: str | os.PathLike[str],
    *,
    chunk_size: int = 1024,
    coefficient_budget: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> PacketJetStreamWriteResult:
    """Write an existing snapshot as independently verified coefficient chunks."""

    if not isinstance(snapshot, PacketJetSnapshot):
        raise ValidationError("snapshot stream persistence requires a PacketJetSnapshot")
    destination = Path(path)
    size = _positive_integer(chunk_size, name="stream chunk size")
    source_count = snapshot.nnz
    budget = source_count if coefficient_budget is None else _nonnegative_integer(
        coefficient_budget,
        name="coefficient budget",
    )
    tolerance = 0.0 if snapshot.capture_certificate.tolerance is None else float(
        snapshot.capture_certificate.tolerance
    )
    source_digest = _source_digest_for_snapshot(snapshot)
    metadata = _metadata_payload(
        address=snapshot.address,
        source_dimension=snapshot.source_dimension,
        target_dimension=snapshot.target_dimension,
        provider_name=snapshot.provider_name,
        exactness=snapshot.exactness,
        point=snapshot.point,
        tolerance=tolerance,
        dense_coefficient_count=snapshot.dense_coefficient_count,
        capture_mode="snapshot",
        source_digest=source_digest,
    )
    staging, existing, resumed = _prepare_staging(
        destination,
        metadata=metadata,
        chunk_size=size,
        resume=resume,
        overwrite=overwrite,
    )
    manifest = existing or _load_manifest(staging)
    start_coefficient = manifest.coefficients_completed
    if start_coefficient > source_count:
        raise ValidationError("snapshot stream progress exceeds the source coefficient count")
    chunks = list(manifest.chunks)
    cursor = start_coefficient
    remaining = min(budget, source_count - cursor)
    while remaining > 0:
        count = min(size, remaining)
        selected = snapshot.coefficients[cursor : cursor + count]
        record = _write_chunk(
            staging,
            index=len(chunks),
            basis_start=manifest.evaluations_completed,
            basis_stop=manifest.evaluations_completed,
            coefficient_start=cursor,
            coefficients=selected,
        )
        chunks.append(record)
        cursor += count
        remaining -= count
        progress = _manifest_payload(
            state="in_progress",
            metadata=metadata,
            chunk_size=size,
            evaluations_completed=manifest.evaluations_completed,
            coefficients_completed=cursor,
            maximum_omitted_basis_norm=float(
                snapshot.capture_certificate.residuals.get("maximum_omitted_basis_norm", 0.0)
            ),
            chunks=chunks,
            capture_certificate=None,
        )
        _write_manifest(staging, progress)

    completed = cursor == source_count
    if completed:
        if not chunks:
            chunks.append(
                _write_chunk(
                    staging,
                    index=0,
                    basis_start=0,
                    basis_stop=0,
                    coefficient_start=0,
                    coefficients=(),
                )
            )
        # Snapshot-mode chunks use coefficient ranges rather than basis ranges.
        # Rewrite their payload metadata once so the manifest retains a contiguous
        # complete basis interval without loading any coefficient chunk together.
        rewritten_chunks: list[PacketJetStreamChunkRecord] = []
        basis_cursor = 0
        for index, old_record in enumerate(chunks):
            coefficients = _read_chunk(
                staging,
                old_record,
                order=snapshot.order,
                source_dimension=snapshot.source_dimension,
                target_dimension=snapshot.target_dimension,
            )
            basis_stop = (
                snapshot.dense_coefficient_count
                if index == len(chunks) - 1
                else basis_cursor
            )
            rewritten_chunks.append(
                _write_chunk(
                    staging,
                    index=index,
                    basis_start=basis_cursor,
                    basis_stop=basis_stop,
                    coefficient_start=old_record.coefficient_start,
                    coefficients=coefficients,
                )
            )
            basis_cursor = basis_stop
        final_payload = _manifest_payload(
            state="complete",
            metadata=metadata,
            chunk_size=size,
            evaluations_completed=snapshot.dense_coefficient_count,
            coefficients_completed=cursor,
            maximum_omitted_basis_norm=float(
                snapshot.capture_certificate.residuals.get("maximum_omitted_basis_norm", 0.0)
            ),
            chunks=rewritten_chunks,
            capture_certificate=snapshot.capture_certificate,
        )
        _write_manifest(staging, final_payload)
        _load_manifest(staging)
        _promote(staging, destination, overwrite=overwrite)
        active = destination
    else:
        active = staging

    final_manifest = _load_manifest(active)
    return PacketJetStreamWriteResult(
        destination=destination,
        active_path=active,
        completed=completed,
        resumed=resumed,
        chunks_written=len(final_manifest.chunks) - len(manifest.chunks),
        evaluations_written=0,
        coefficients_written=final_manifest.coefficients_completed - manifest.coefficients_completed,
        manifest=final_manifest,
        certificate=_write_result_certificate(
            completed=completed,
            resumed=resumed,
            manifest=final_manifest,
        ),
    )


class PacketJetSnapshotStreamReader:
    """Bounded-memory reader for a complete or explicitly allowed partial stream."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        allow_incomplete: bool = False,
        verify_chunks: bool = False,
    ) -> None:
        requested = Path(path)
        active = requested
        if not active.exists() and allow_incomplete and _staging_path(requested).exists():
            active = _staging_path(requested)
        if not active.is_dir():
            raise ValidationError(f"packet-jet stream directory does not exist: {active}")
        manifest = _load_manifest(active)
        if not manifest.complete and not allow_incomplete:
            raise ValidationError("packet-jet stream is incomplete")
        self.path = active
        self.manifest = manifest
        if verify_chunks:
            self.verify().require()

    @property
    def complete(self) -> bool:
        return self.manifest.complete

    def _chunk(self, record: PacketJetStreamChunkRecord) -> tuple[PacketJetCoefficient, ...]:
        return _read_chunk(
            self.path,
            record,
            order=self.manifest.order,
            source_dimension=self.manifest.source_dimension,
            target_dimension=self.manifest.target_dimension,
        )

    def iter_chunks(self) -> Iterator[tuple[PacketJetStreamChunkRecord, tuple[PacketJetCoefficient, ...]]]:
        """Yield one verified coefficient chunk at a time."""

        for record in self.manifest.chunks:
            yield record, self._chunk(record)

    def iter_coefficients(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[PacketJetCoefficient]:
        """Yield a coefficient ordinal range while holding at most one chunk."""

        lower = _nonnegative_integer(start, name="coefficient start")
        upper = self.manifest.coefficients_completed if stop is None else _nonnegative_integer(
            stop,
            name="coefficient stop",
        )
        if upper < lower or upper > self.manifest.coefficients_completed:
            raise ValidationError("coefficient selection lies outside the stored stream range")
        for record in self.manifest.chunks:
            if record.coefficient_stop <= lower:
                continue
            if record.coefficient_start >= upper:
                break
            coefficients = self._chunk(record)
            chunk_start = max(lower, record.coefficient_start) - record.coefficient_start
            chunk_stop = min(upper, record.coefficient_stop) - record.coefficient_start
            yield from coefficients[chunk_start:chunk_stop]

    def select(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        max_coefficients: int = 100_000,
    ) -> PacketJetStreamSelection:
        """Load a guarded coefficient ordinal range into one finite tuple."""

        lower = _nonnegative_integer(start, name="coefficient start")
        upper = self.manifest.coefficients_completed if stop is None else _nonnegative_integer(
            stop,
            name="coefficient stop",
        )
        if upper < lower or upper > self.manifest.coefficients_completed:
            raise ValidationError("coefficient selection lies outside the stored stream range")
        limit = _positive_integer(max_coefficients, name="maximum selected coefficients")
        if upper - lower > limit:
            raise ValidationError(
                f"coefficient selection requires {upper - lower} entries, exceeding {limit}"
            )
        coefficients = tuple(self.iter_coefficients(start=lower, stop=upper))
        complete_snapshot = (
            self.complete and lower == 0 and upper == self.manifest.coefficients_completed
        )
        certificate = Certificate(
            valid=len(coefficients) == upper - lower,
            exactness=self.manifest.exactness,
            residuals={
                "missing_selected_coefficients": float(
                    max(0, upper - lower - len(coefficients))
                )
            },
            tolerance=None
            if self.manifest.exactness is Exactness.EXACT
            else self.manifest.tolerance,
            assumptions_checked=(
                "selected chunks passed envelope and manifest digest validation",
                "coefficient ordinal bounds were enforced",
            ),
            assumptions_unchecked=()
            if complete_snapshot
            else ("coefficients outside the selected ordinal interval were not loaded",),
            diagnostics={
                "coefficient_start": lower,
                "coefficient_stop": upper,
                "loaded_coefficients": len(coefficients),
                "complete_snapshot": complete_snapshot,
            },
            theorem_id="PSC-III-D60",
            message="packet-jet stream coefficient selection loaded",
        )
        return PacketJetStreamSelection(
            lower,
            upper,
            coefficients,
            complete_snapshot,
            certificate,
        )

    def evaluate(
        self,
        directions: Sequence[Any],
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> StreamedPacketJetEvaluation:
        """Evaluate a complete stream or a declared coefficient partial sum."""

        vectors = _direction_vectors(
            directions,
            order=self.manifest.order,
            source_dimension=self.manifest.source_dimension,
        )
        lower = _nonnegative_integer(start, name="coefficient start")
        upper = self.manifest.coefficients_completed if stop is None else _nonnegative_integer(
            stop,
            name="coefficient stop",
        )
        if upper < lower or upper > self.manifest.coefficients_completed:
            raise ValidationError("coefficient evaluation range lies outside the stream")
        if self.manifest.exactness is Exactness.EXACT:
            vectors = tuple(python_integer_entries(v) for v in vectors)
        dtype: Any = object if self.manifest.exactness is Exactness.EXACT else complex
        total = np.zeros(self.manifest.target_dimension, dtype=dtype)
        processed = 0
        for coefficient in self.iter_coefficients(start=lower, stop=upper):
            factor: Any = 1
            for slot, index in enumerate(coefficient.indices):
                factor = factor * vectors[slot][index]
            for output, value in enumerate(coefficient.value):
                if self.manifest.exactness is Exactness.EXACT and isinstance(value, np.integer):
                    value = int(value)
                total[output] = total[output] + factor * value
            processed += 1
        complete_snapshot = self.complete and lower == 0 and upper == self.manifest.coefficients_completed
        certificate = Certificate(
            valid=processed == upper - lower,
            exactness=self.manifest.exactness,
            residuals={"unprocessed_selected_coefficients": float(upper - lower - processed)},
            tolerance=None
            if self.manifest.exactness is Exactness.EXACT
            else self.manifest.tolerance,
            assumptions_checked=(
                "each contributing chunk passed digest validation",
                "ordered multilinear coefficient contraction",
            ),
            assumptions_unchecked=()
            if complete_snapshot
            else ("the returned value is only the selected coefficient partial sum",),
            diagnostics={
                "coefficient_start": lower,
                "coefficient_stop": upper,
                "processed_coefficients": processed,
                "complete_snapshot": complete_snapshot,
            },
            theorem_id="PSC-III-D60",
            message=(
                "complete packet-jet stream evaluated"
                if complete_snapshot
                else "packet-jet stream coefficient partial sum evaluated"
            ),
        )
        return StreamedPacketJetEvaluation(
            total,
            lower,
            upper,
            self.manifest.coefficients_completed,
            complete_snapshot,
            certificate,
        )

    def load_snapshot(self, *, max_coefficients: int = 1_000_000) -> PacketJetSnapshot:
        """Materialize a complete stream under an explicit coefficient guard."""

        if not self.complete:
            raise ValidationError("cannot materialize an incomplete packet-jet stream")
        selection = self.select(max_coefficients=max_coefficients)
        selection.certificate.require()
        certificate = self.manifest.capture_certificate
        if certificate is None:
            raise ValidationError("complete stream is missing its capture certificate")
        return PacketJetSnapshot(
            address=self.manifest.address,
            source_dimension=self.manifest.source_dimension,
            target_dimension=self.manifest.target_dimension,
            provider_name=self.manifest.provider_name,
            exactness=self.manifest.exactness,
            point=self.manifest.point,
            coefficients=selection.coefficients,
            capture_certificate=certificate,
        )

    def verify(self) -> Certificate:
        """Verify every committed chunk without retaining the complete snapshot."""

        observed_coefficients = 0
        observed_basis = 0
        ordering_violations = 0
        previous_indices: tuple[int, ...] | None = None
        for record, coefficients in self.iter_chunks():
            observed_coefficients += len(coefficients)
            observed_basis += record.basis_count
            for coefficient in coefficients:
                if previous_indices is not None and coefficient.indices <= previous_indices:
                    ordering_violations += 1
                previous_indices = coefficient.indices
        valid = (
            observed_coefficients == self.manifest.coefficients_completed
            and observed_basis == self.manifest.evaluations_completed
            and ordering_violations == 0
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals={
                "coefficient_count_mismatch": float(
                    abs(observed_coefficients - self.manifest.coefficients_completed)
                ),
                "basis_count_mismatch": float(
                    abs(observed_basis - self.manifest.evaluations_completed)
                ),
                "coefficient_ordering_violations": float(ordering_violations),
            },
            assumptions_checked=(
                "manifest envelope digest",
                "metadata digest",
                "stream content digest",
                "every chunk envelope digest and manifest cross-reference",
                "contiguous basis and coefficient ranges",
                "strict lexicographic coefficient ordering without duplicates",
            ),
            assumptions_unchecked=(
                "SHA-256 integrity does not establish authenticity or authorship",
            ),
            diagnostics={
                "chunks": len(self.manifest.chunks),
                "coefficients": observed_coefficients,
                "basis_evaluations": observed_basis,
                "complete": self.complete,
            },
            theorem_id="PSC-III-D60",
            message=(
                "packet-jet stream integrity verified"
                if valid
                else "packet-jet stream integrity verification failed"
            ),
        )


def open_packet_jet_snapshot_stream(
    path: str | os.PathLike[str],
    *,
    allow_incomplete: bool = False,
    verify_chunks: bool = False,
) -> PacketJetSnapshotStreamReader:
    """Open one packet-jet stream with explicit incomplete-stream policy."""

    return PacketJetSnapshotStreamReader(
        path,
        allow_incomplete=allow_incomplete,
        verify_chunks=verify_chunks,
    )


def load_packet_jet_snapshot_stream(
    path: str | os.PathLike[str],
    *,
    max_coefficients: int = 1_000_000,
    verify_chunks: bool = True,
) -> PacketJetSnapshot:
    """Load a complete stream into a conventional in-memory snapshot."""

    reader = open_packet_jet_snapshot_stream(path, verify_chunks=verify_chunks)
    return reader.load_snapshot(max_coefficients=max_coefficients)


__all__ = [
    "PACKET_JET_CHUNK_SCHEMA",
    "PACKET_JET_CHUNK_SCHEMA_VERSION",
    "PACKET_JET_STREAM_SCHEMA",
    "PACKET_JET_STREAM_SCHEMA_VERSION",
    "PacketJetSnapshotStreamReader",
    "PacketJetStreamChunkRecord",
    "PacketJetStreamManifest",
    "PacketJetStreamSelection",
    "PacketJetStreamWriteResult",
    "StreamedPacketJetEvaluation",
    "capture_packet_jet_stream",
    "load_packet_jet_snapshot_stream",
    "open_packet_jet_snapshot_stream",
    "save_packet_jet_snapshot_stream",
]
