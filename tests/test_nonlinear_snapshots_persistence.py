from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest

from komorebi_math import Exactness, PacketSystem
from komorebi_math.deformation.forests import LiftBranch
from komorebi_math.exceptions import ValidationError
from komorebi_math.export.persistence import (
    dumps_portable,
    load_portable,
    loads_portable,
    portable_digest,
    save_portable,
)
from komorebi_math.export.serialization import from_serializable, to_serializable
from komorebi_math.nonlinear.derivatives import PolynomialDerivativeProvider
from komorebi_math.nonlinear.filtered import (
    FilteredVectorSpace,
    FiniteFilteredCochainComplex,
)
from komorebi_math.nonlinear.jets import PacketJetFamily
from komorebi_math.nonlinear.snapshots import PacketJetSnapshot, snapshot_packet_jet
from komorebi_math.nonlinear.sparse import SparseSymmetricTensor
from komorebi_math.nonlinear.trees import OrderedRootedTree


def _snapshot() -> tuple[PacketJetSnapshot, object]:
    packets = PacketSystem(
        [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])],
        names=("x", "y"),
    )
    quadratic = np.zeros((2, 2, 2))
    quadratic[0, 0, 0] = 2.0
    quadratic[1, 1, 1] = 2.0
    provider = PolynomialDerivativeProvider(
        {2: quadratic},
        source_dimension=2,
        target_dimension=2,
    )
    family = PacketJetFamily(provider, np.zeros(2), packets, packets, 2)
    jet = family.jet("x", ("x", "x"))
    return snapshot_packet_jet(jet), jet


def _spectral_result():
    c0 = FilteredVectorSpace.from_weights([0, 1], prime=3)
    c1 = FilteredVectorSpace.from_weights([0, 1], prime=3)
    complex_ = FiniteFilteredCochainComplex(
        {0: 2, 1: 2},
        {0: np.zeros((2, 2), dtype=int)},
        {0: c0, 1: c1},
        prime=3,
    )
    return complex_.spectral_sequence()


def test_packet_jet_snapshot_is_provider_independent_and_round_trips() -> None:
    snapshot, jet = _snapshot()
    directions = (np.array([2.0, 3.0]), np.array([-1.0, 4.0]))

    snapshot.capture_certificate.require()
    snapshot.comparison_certificate(jet, directions).require()
    assert np.allclose(snapshot.evaluate(directions), jet.evaluate(directions))
    assert snapshot.nnz == 1

    restored = loads_portable(dumps_portable(snapshot))
    assert isinstance(restored, PacketJetSnapshot)
    assert restored == snapshot
    assert np.allclose(restored.evaluate(directions), snapshot.evaluate(directions))


def test_sparse_tensor_tree_branch_and_spectral_pages_round_trip() -> None:
    dense = np.zeros((1, 2, 2), dtype=object)
    dense[0, 0, 0] = Fraction(2, 3)
    dense[0, 0, 1] = dense[0, 1, 0] = Fraction(1, 5)
    sparse = SparseSymmetricTensor.from_dense(dense, source_dimension=2)
    tree = OrderedRootedTree(
        (
            OrderedRootedTree.leaf(),
            OrderedRootedTree((OrderedRootedTree.leaf(), OrderedRootedTree.leaf())),
        )
    )
    branch = LiftBranch(((1, 2), (3, 4)))
    spectral = _spectral_result()
    payload = {
        "sparse": sparse,
        "tree": tree,
        "branch": branch,
        "page": spectral.pages[0],
        "spectral": spectral,
    }

    restored = loads_portable(dumps_portable(payload))
    assert restored["sparse"] == sparse
    assert restored["tree"] == tree
    assert restored["branch"] == branch
    assert restored["page"] == spectral.pages[0]
    assert restored["spectral"] == spectral


def test_portable_envelope_is_deterministic_and_digest_checked() -> None:
    payload = {"z": (1, 2), "a": Fraction(3, 7), "array": np.arange(6).reshape(2, 3)}
    first = dumps_portable(payload)
    second = dumps_portable(payload)
    assert first == second
    assert portable_digest(payload) == portable_digest(payload)

    envelope = json.loads(first)
    envelope["payload"]["data"][0][1] = 999
    with pytest.raises(ValidationError, match="SHA256 mismatch"):
        loads_portable(json.dumps(envelope))


def test_portable_file_write_is_atomic_and_requires_explicit_overwrite(tmp_path) -> None:
    destination = tmp_path / "state.kmath.json"
    payload = {"value": Fraction(4, 9)}
    assert save_portable(payload, destination) == destination
    assert load_portable(destination) == payload
    with pytest.raises(ValidationError, match="refusing to overwrite"):
        save_portable(payload, destination)
    save_portable({"value": Fraction(5, 9)}, destination, overwrite=True)
    assert load_portable(destination) == {"value": Fraction(5, 9)}
    assert not tuple(tmp_path.glob("*.tmp"))


def test_safe_presentation_decoder_only_instantiates_builtin_markers() -> None:
    original = {
        "fraction": Fraction(2, 5),
        "complex": 1 + 3j,
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
    }
    encoded = to_serializable(original, preserve_types=True)
    decoded = from_serializable(encoded)
    assert decoded["fraction"] == Fraction(2, 5)
    assert decoded["complex"] == 1 + 3j
    assert np.array_equal(decoded["array"], original["array"])

    unknown = {"__type__": "arbitrary.module.Class", "value": 7}
    assert from_serializable(unknown) == unknown
