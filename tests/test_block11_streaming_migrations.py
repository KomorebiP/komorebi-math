from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from komorebi_math import Exactness, PacketSystem
from komorebi_math.certificates import Certificate
from komorebi_math.exceptions import ValidationError
from komorebi_math.export import (
    dumps_portable,
    loads_portable,
    register_portable_codec,
    register_portable_migration,
    to_portable_data,
)
from komorebi_math.nonlinear import (
    PacketJetAddress,
    PacketJetFamily,
    PacketJetSnapshot,
    PolynomialDerivativeProvider,
    capture_packet_jet_stream,
    load_packet_jet_snapshot_stream,
    open_packet_jet_snapshot_stream,
    save_packet_jet_snapshot_stream,
    snapshot_packet_jet,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _rehash(envelope: dict[str, object]) -> str:
    envelope["payload_sha256"] = sha256(_canonical(envelope["payload"]).encode()).hexdigest()
    return _canonical(envelope)


def _coordinate_family(*, dense: bool = False) -> tuple[PacketJetFamily, tuple[np.ndarray, np.ndarray]]:
    if dense:
        tensor = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        source = PacketSystem((np.eye(2),), names=("all",))
        target = PacketSystem((np.eye(1),), names=("all",))
        provider = PolynomialDerivativeProvider(
            {2: tensor},
            source_dimension=2,
            target_dimension=1,
            name="dense quadratic",
        )
        point = np.zeros(2)
    else:
        tensor = np.zeros((2, 2, 2), dtype=float)
        tensor[0, 0, 0] = 2.0
        tensor[1, 1, 1] = -3.0
        source = PacketSystem(
            (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
            names=("x", "y"),
        )
        target = source
        provider = PolynomialDerivativeProvider(
            {2: tensor},
            source_dimension=2,
            target_dimension=2,
            name="coordinate quadratic",
        )
        point = np.array([0.2, -0.4])
    family = PacketJetFamily(provider, point, source, target, 2)
    directions = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))
    return family, directions


def test_direct_capture_pauses_resumes_and_matches_live_jet(tmp_path: Path) -> None:
    family, directions = _coordinate_family()
    jet = family.jet("x", ("x", "x"))
    destination = tmp_path / "direct.kmstream"

    partial = capture_packet_jet_stream(
        jet,
        destination,
        chunk_size=1,
        evaluation_budget=2,
    )
    assert not partial.completed
    assert partial.active_path.name.endswith(".partial")
    assert partial.evaluations_written == 2
    assert not destination.exists()

    reader = open_packet_jet_snapshot_stream(destination, allow_incomplete=True)
    assert not reader.complete
    reader.verify().require()
    partial_value = reader.evaluate(directions)
    assert not partial_value.complete_snapshot
    assert partial_value.certificate.assumptions_unchecked

    complete = capture_packet_jet_stream(
        jet,
        destination,
        chunk_size=1,
        resume=True,
    )
    complete.certificate.require()
    assert complete.completed and complete.resumed
    assert destination.is_dir()
    assert not partial.active_path.exists()

    final_reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
    result = final_reader.evaluate(directions)
    result.certificate.require()
    assert result.complete_snapshot
    assert np.allclose(np.real_if_close(result.value), jet.evaluate(directions))
    restored = load_packet_jet_snapshot_stream(destination)
    assert np.allclose(np.real_if_close(restored.evaluate(directions)), jet.evaluate(directions))


def test_snapshot_stream_pauses_resumes_and_reads_bounded_ranges(tmp_path: Path) -> None:
    family, directions = _coordinate_family(dense=True)
    snapshot = snapshot_packet_jet(family.jet("all", ("all", "all")))
    assert snapshot.nnz == 4
    destination = tmp_path / "snapshot.kmstream"

    partial = save_packet_jet_snapshot_stream(
        snapshot,
        destination,
        chunk_size=2,
        coefficient_budget=1,
    )
    assert not partial.completed
    assert partial.coefficients_written == 1

    complete = save_packet_jet_snapshot_stream(
        snapshot,
        destination,
        chunk_size=2,
        resume=True,
    )
    assert complete.completed and complete.resumed
    assert len(complete.manifest.chunks) == 3

    reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
    selection = reader.select(start=1, stop=3, max_coefficients=2)
    selection.certificate.require()
    assert len(selection.coefficients) == 2
    assert not selection.complete_snapshot
    with pytest.raises(ValidationError, match="exceeding"):
        reader.select(max_coefficients=3)

    left = reader.evaluate(directions, start=0, stop=2)
    right = reader.evaluate(directions, start=2)
    assert not left.complete_snapshot and not right.complete_snapshot
    full = reader.evaluate(directions)
    full.certificate.require()
    assert np.allclose(np.real_if_close(left.value) + np.real_if_close(right.value), np.real_if_close(full.value))
    assert np.allclose(np.real_if_close(full.value), np.real_if_close(snapshot.evaluate(directions)))


def test_empty_snapshot_stream_roundtrip(tmp_path: Path) -> None:
    certificate = Certificate(
        valid=True,
        exactness=Exactness.EXACT,
        residuals={"maximum_omitted_basis_norm": 0.0},
        message="zero block",
    )
    snapshot = PacketJetSnapshot(
        address=PacketJetAddress("all", ("all", "all")),
        source_dimension=2,
        target_dimension=1,
        provider_name="zero",
        exactness=Exactness.EXACT,
        point=(0, 0),
        coefficients=(),
        capture_certificate=certificate,
    )
    destination = tmp_path / "zero.kmstream"
    result = save_packet_jet_snapshot_stream(snapshot, destination, chunk_size=1)
    result.certificate.require()
    assert result.completed
    assert len(result.manifest.chunks) == 1
    reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
    assert reader.manifest.coefficients_completed == 0
    restored = reader.load_snapshot(max_coefficients=1)
    assert restored.nnz == 0
    assert restored.evaluate((np.ones(2), np.ones(2))).tolist() == [0]


def test_snapshot_stream_explicit_overwrite_replaces_existing_file(tmp_path: Path) -> None:
    family, _ = _coordinate_family(dense=True)
    snapshot = snapshot_packet_jet(family.jet("all", ("all", "all")))
    destination = tmp_path / "replace-file.kmstream"
    destination.write_text("not a stream", encoding="utf-8")

    result = save_packet_jet_snapshot_stream(
        snapshot,
        destination,
        chunk_size=2,
        overwrite=True,
    )

    result.certificate.require()
    assert destination.is_dir()
    open_packet_jet_snapshot_stream(destination, verify_chunks=True).verify().require()


def test_stream_detects_chunk_and_manifest_tampering(tmp_path: Path) -> None:
    family, _ = _coordinate_family(dense=True)
    snapshot = snapshot_packet_jet(family.jet("all", ("all", "all")))

    chunk_destination = tmp_path / "chunk-tamper.kmstream"
    save_packet_jet_snapshot_stream(snapshot, chunk_destination, chunk_size=2)
    chunk = next((chunk_destination / "chunks").glob("chunk-*.json"))
    raw_chunk = json.loads(chunk.read_text(encoding="utf-8"))
    raw_chunk["payload"]["coefficient_stop"] += 1
    chunk.write_text(json.dumps(raw_chunk), encoding="utf-8")
    with pytest.raises(ValidationError, match="SHA256 mismatch"):
        open_packet_jet_snapshot_stream(chunk_destination, verify_chunks=True)

    manifest_destination = tmp_path / "manifest-tamper.kmstream"
    save_packet_jet_snapshot_stream(snapshot, manifest_destination, chunk_size=2)
    manifest = manifest_destination / "manifest.json"
    raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    raw_manifest["payload"]["coefficients_completed"] += 1
    manifest.write_text(json.dumps(raw_manifest), encoding="utf-8")
    with pytest.raises(ValidationError, match="SHA256 mismatch"):
        open_packet_jet_snapshot_stream(manifest_destination)



def test_resume_rejects_tampered_committed_prefix(tmp_path: Path) -> None:
    family, _ = _coordinate_family(dense=True)
    jet = family.jet("all", ("all", "all"))
    destination = tmp_path / "tampered-resume.kmstream"
    partial = capture_packet_jet_stream(
        jet,
        destination,
        chunk_size=1,
        evaluation_budget=1,
    )
    assert not partial.completed
    chunk = next((partial.active_path / "chunks").glob("chunk-*.json"))
    raw = json.loads(chunk.read_text(encoding="utf-8"))
    raw["payload"]["basis_stop"] += 1
    chunk.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="SHA256 mismatch"):
        capture_packet_jet_stream(
            jet,
            destination,
            chunk_size=1,
            resume=True,
        )


def test_stream_manifest_rejects_inconsistent_dense_basis_count(tmp_path: Path) -> None:
    family, _ = _coordinate_family(dense=True)
    snapshot = snapshot_packet_jet(family.jet("all", ("all", "all")))
    destination = tmp_path / "dense-count.kmstream"
    result = save_packet_jet_snapshot_stream(snapshot, destination, chunk_size=2)

    from dataclasses import replace

    with pytest.raises(ValidationError, match="dense coefficient count"):
        replace(
            result.manifest,
            dense_coefficient_count=result.manifest.dense_coefficient_count + 1,
        )

def test_resume_rejects_changed_source_and_chunk_size(tmp_path: Path) -> None:
    family, _ = _coordinate_family()
    destination = tmp_path / "resume.kmstream"
    jet = family.jet("x", ("x", "x"))
    capture_packet_jet_stream(jet, destination, chunk_size=1, evaluation_budget=1)

    with pytest.raises(ValidationError, match="chunk size"):
        capture_packet_jet_stream(jet, destination, chunk_size=2, resume=True)

    other = family.jet("y", ("y", "y"))
    with pytest.raises(ValidationError, match="source does not match"):
        capture_packet_jet_stream(other, destination, chunk_size=1, resume=True)


@dataclass(frozen=True)
class MigratingValue:
    numerator: int
    denominator: int


def test_portable_codec_runs_explicit_adjacent_migration_chain() -> None:
    tag = "tests.block11.migrating_value"
    register_portable_codec(
        MigratingValue,
        tag=tag,
        version=3,
        encoder=lambda value: {
            "numerator": value.numerator,
            "denominator": value.denominator,
            "normalized": True,
        },
        decoder=lambda data: MigratingValue(
            int(data["numerator"]),
            int(data["denominator"]),
        ),
        replace=True,
    )
    register_portable_migration(
        tag=tag,
        from_version=1,
        to_version=2,
        migrator=lambda data: {"value": data["value"], "denominator": 1},
        replace=True,
    )
    register_portable_migration(
        tag=tag,
        from_version=2,
        to_version=3,
        migrator=lambda data: {
            "numerator": data["value"],
            "denominator": data["denominator"],
            "normalized": True,
        },
        replace=True,
    )

    envelope = json.loads(dumps_portable(MigratingValue(7, 1)))
    envelope["payload"]["__version__"] = 1
    envelope["payload"]["data"] = to_portable_data({"value": 7})
    assert loads_portable(_rehash(envelope)) == MigratingValue(7, 1)

    future = json.loads(dumps_portable(MigratingValue(7, 1)))
    future["payload"]["__version__"] = 4
    with pytest.raises(ValidationError, match="future codec version"):
        loads_portable(_rehash(future))


def test_portable_codec_rejects_missing_migration_path() -> None:
    @dataclass(frozen=True)
    class MissingPathValue:
        value: Fraction

    tag = "tests.block11.missing_path"
    register_portable_codec(
        MissingPathValue,
        tag=tag,
        version=3,
        encoder=lambda value: {"value": value.value},
        decoder=lambda data: MissingPathValue(data["value"]),
        replace=True,
    )
    envelope = json.loads(dumps_portable(MissingPathValue(Fraction(1, 3))))
    envelope["payload"]["__version__"] = 1
    with pytest.raises(ValidationError, match="no migration path"):
        loads_portable(_rehash(envelope))
