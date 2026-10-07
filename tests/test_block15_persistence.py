"""Block 15 stabilization contract for refinement persistence."""

from fractions import Fraction
import json

import numpy as np
import pytest

from komorebi_math.arithmetic import polyphase_analyze as arithmetic_polyphase_analyze
from komorebi_math.exceptions import ValidationError
from komorebi_math.export.persistence import (
    dumps_portable,
    load_portable,
    from_portable_data,
    loads_portable,
    portable_digest,
    save_portable,
    to_portable_data,
)
from komorebi_math.refinement import (
    LatticePolyphaseRefinement,
    PolyphaseRefinement,
    PolyphaseRefinementTower,
    residue_analyze,
)
from komorebi_math.signals import lattice_polyphase_analyze, polyphase_analyze


def test_refinement_portable_roundtrip() -> None:
    value = PolyphaseRefinement(4, 20)
    restored = loads_portable(dumps_portable(value))
    assert restored == value


def test_refinement_tower_portable_roundtrip() -> None:
    value = PolyphaseRefinementTower((2, 6, 18, 36))
    restored = loads_portable(dumps_portable(value))
    assert restored == value


def test_lattice_refinement_portable_roundtrip() -> None:
    value = LatticePolyphaseRefinement((2, 3, 1), (4, 9, 5))
    restored = loads_portable(dumps_portable(value))
    assert restored == value


def test_signal_coordinates_portable_roundtrip() -> None:
    values = np.arange(4 * 11).reshape(4, 11)
    value = polyphase_analyze(values, 5, axis=1).refine(15)
    restored = loads_portable(dumps_portable(value))
    assert restored.order == value.order
    assert restored.axis == value.axis
    assert np.array_equal(restored.reconstruct(), values)


def test_lattice_coordinates_portable_roundtrip() -> None:
    values = np.arange(5 * 7).reshape(5, 7)
    value = lattice_polyphase_analyze(values, (2, 3)).refine((4, 6))
    restored = loads_portable(dumps_portable(value))
    assert restored.strides == value.strides
    assert np.array_equal(restored.reconstruct(), values)


def test_residue_coordinates_portable_roundtrip() -> None:
    values = np.arange(23)
    value = residue_analyze(values, 4).refine(12)
    restored = loads_portable(dumps_portable(value))
    assert restored.modulus == value.modulus
    assert np.array_equal(restored.reconstruct(), values)


def test_arithmetic_coordinates_portable_roundtrip() -> None:
    values = tuple(Fraction(index + 1, index + 3) for index in range(17))
    value = arithmetic_polyphase_analyze(values, 3).refine(12)
    restored = loads_portable(dumps_portable(value))
    assert restored.modulus == value.modulus
    assert restored.reconstruct() == values


def test_portable_digest_detects_payload_tampering(tmp_path) -> None:
    value = PolyphaseRefinement(3, 12)
    text = dumps_portable(value)
    envelope = json.loads(text)
    envelope["payload"]["data"]["data"][0][1] = 99
    tampered = json.dumps(envelope)
    with pytest.raises(ValidationError, match="SHA256 mismatch"):
        loads_portable(tampered)
    path = tmp_path / "refinement.json"
    save_portable(value, path)
    assert load_portable(path) == value
    assert portable_digest(value) == portable_digest(load_portable(path))


def test_portable_data_mapping_roundtrip_without_json_normalization() -> None:
    value = {Fraction(1, 3): ("alpha", 7)}
    encoded = to_portable_data(value)
    assert all(isinstance(pair, list) for pair in encoded["data"])
    assert from_portable_data(encoded) == value


def test_portable_data_ndarray_roundtrip_without_json_normalization() -> None:
    value = np.arange(12, dtype=np.int16).reshape(3, 4)
    encoded = to_portable_data(value)
    assert isinstance(encoded["data"]["shape"], list)
    restored = from_portable_data(encoded)
    assert np.array_equal(restored, value)
