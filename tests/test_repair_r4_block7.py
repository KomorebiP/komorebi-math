from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from komorebi_math import Certificate, Exactness, PacketSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.export.persistence import dumps_portable, loads_portable
from komorebi_math.nonlinear.derivatives import PolynomialDerivativeProvider
from komorebi_math.nonlinear.jets import PacketJetFamily
from komorebi_math.nonlinear.snapshots import PacketJetSnapshot, snapshot_packet_jet
from komorebi_math.nonlinear.streaming import (
    load_packet_jet_snapshot_stream,
    open_packet_jet_snapshot_stream,
    save_packet_jet_snapshot_stream,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _rehash(envelope: dict) -> str:
    envelope["payload_sha256"] = hashlib.sha256(_canonical(envelope["payload"]).encode("utf-8")).hexdigest()
    return json.dumps(envelope)


def _certificate_pairs(envelope: dict) -> list[list[object]]:
    payload = envelope["payload"]
    assert payload["__komorebi_type__"] == "core.certificate"
    data = payload["data"]
    assert data["__komorebi_type__"] == "python.mapping"
    return data["data"]


def _pair(pairs: list[list[object]], key: str):
    for entry in pairs:
        if entry[0] == key:
            return entry
    raise KeyError(key)


def _replace_pair(pairs: list[list[object]], key: str, value) -> None:
    _pair(pairs, key)[1] = value


def _snapshot() -> PacketJetSnapshot:
    packets = PacketSystem([np.eye(1)], names=("p0",))
    provider = PolynomialDerivativeProvider(
        {1: np.array([[2.0]])},
        source_dimension=1,
        target_dimension=1,
    )
    jet = PacketJetFamily(provider, np.zeros(1), packets, packets, 1).jet("p0", ("p0",))
    return snapshot_packet_jet(jet)


def test_loaded_positive_certificate_is_historical_not_live() -> None:
    original = Certificate(
        True,
        Exactness.EXACT,
        residuals={"identity": 0.0},
        assumptions_checked=("fresh producer executed",),
        theorem_id="BLOCK7-DEMO",
        message="fresh certificate",
    )
    restored = loads_portable(dumps_portable(original))
    assert isinstance(restored, Certificate)
    assert restored.recorded_valid is True
    assert restored.revalidation_required is True
    assert restored.valid is False
    with pytest.raises(ValidationError, match="revalidation"):
        restored.require()


def test_loaded_negative_certificate_preserves_recorded_invalidity() -> None:
    original = Certificate(False, Exactness.NUMERICAL, residuals={"r": 1.0})
    restored = loads_portable(dumps_portable(original))
    assert restored.recorded_valid is False
    assert restored.revalidation_required is True
    assert restored.valid is False


def test_portable_snapshot_keeps_coefficients_but_not_live_capture_validity() -> None:
    snapshot = _snapshot()
    restored = loads_portable(dumps_portable(snapshot))
    assert isinstance(restored, PacketJetSnapshot)
    assert restored.address == snapshot.address
    assert restored.coefficients == snapshot.coefficients
    assert np.array_equal(restored.evaluate(([3.0],)), snapshot.evaluate(([3.0],)))
    assert restored.capture_certificate.recorded_valid is True
    assert restored.capture_certificate.revalidation_required is True
    assert restored.capture_certificate.valid is False


def test_stream_load_separates_integrity_verification_from_capture_certificate(tmp_path: Path) -> None:
    snapshot = _snapshot()
    destination = tmp_path / "snapshot.kmstream"
    save_packet_jet_snapshot_stream(snapshot, destination, chunk_size=1)
    reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
    integrity = reader.verify()
    assert integrity.valid
    assert integrity.revalidation_required is False
    restored = load_packet_jet_snapshot_stream(destination)
    assert restored.capture_certificate.recorded_valid is True
    assert restored.capture_certificate.revalidation_required is True
    assert restored.capture_certificate.valid is False


def test_rehashed_certificate_with_unknown_decoded_field_is_rejected() -> None:
    envelope = json.loads(dumps_portable(Certificate(True, Exactness.EXACT)))
    _certificate_pairs(envelope).append(["unexpected", 7])
    with pytest.raises(ValidationError, match="certificate fields"):
        loads_portable(_rehash(envelope))


def test_rehashed_certificate_rejects_string_nan_residual() -> None:
    envelope = json.loads(dumps_portable(Certificate(True, Exactness.EXACT, residuals={"r": 0.0})))
    residuals = _pair(_certificate_pairs(envelope), "residuals")[1]
    residuals["data"][0][1] = "nan"
    with pytest.raises(ValidationError, match="residual"):
        loads_portable(_rehash(envelope))


@pytest.mark.parametrize("value", ["inf", "nan", -1.0, True])
def test_rehashed_certificate_rejects_invalid_tolerance(value) -> None:
    envelope = json.loads(dumps_portable(Certificate(True, Exactness.NUMERICAL, tolerance=1e-9)))
    _replace_pair(_certificate_pairs(envelope), "tolerance", value)
    with pytest.raises(ValidationError, match="tolerance"):
        loads_portable(_rehash(envelope))


def test_repeated_persistence_does_not_restore_live_validity() -> None:
    first = loads_portable(dumps_portable(Certificate(True, Exactness.EXACT)))
    second = loads_portable(dumps_portable(first))
    assert first.revalidation_required and second.revalidation_required
    assert first.recorded_valid is True
    assert second.recorded_valid is True
    assert first.valid is False and second.valid is False


def test_version1_certificate_record_migrates_to_historical_v2() -> None:
    envelope = json.loads(dumps_portable(Certificate(True, Exactness.EXACT, theorem_id="OLD")))
    envelope["payload"]["__version__"] = 1
    pairs = _certificate_pairs(envelope)
    recorded = _pair(pairs, "recorded_valid")[1]
    pairs[:] = [entry for entry in pairs if entry[0] != "record_kind"]
    _pair(pairs, "recorded_valid")[0] = "valid"
    _pair(pairs, "valid")[1] = recorded
    restored = loads_portable(_rehash(envelope))
    assert restored.recorded_valid is True
    assert restored.revalidation_required is True
    assert restored.valid is False
    assert restored.theorem_id == "OLD"


def test_rehashed_certificate_rejects_nonstring_assumption() -> None:
    envelope = json.loads(dumps_portable(Certificate(True, Exactness.EXACT, assumptions_checked=("ok",))))
    assumptions = _pair(_certificate_pairs(envelope), "assumptions_checked")[1]
    assumptions["data"].append(7)
    with pytest.raises(ValidationError, match="assumptions"):
        loads_portable(_rehash(envelope))


def test_rehashed_certificate_rejects_nontext_theorem_id() -> None:
    envelope = json.loads(dumps_portable(Certificate(True, Exactness.EXACT, theorem_id="T")))
    _replace_pair(_certificate_pairs(envelope), "theorem_id", 17)
    with pytest.raises(ValidationError, match="theorem_id"):
        loads_portable(_rehash(envelope))
