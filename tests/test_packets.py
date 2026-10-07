from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from komorebi_math import PacketSystem
from komorebi_math.exceptions import ValidationError


def coordinate_packets() -> PacketSystem:
    return PacketSystem(
        [np.diag([1, 1, 0, 0]), np.diag([0, 0, 1, 1])],
        names=["left", "right"],
    )


def test_analysis_synthesis_and_support() -> None:
    packets = coordinate_packets()
    vector = np.array([1, 2, 0, 0], dtype=float)
    coordinates = packets.analyze(vector)
    assert np.allclose(coordinates.reconstruct(), vector)
    assert coordinates.support() == ("left",)
    assert np.allclose(coordinates["left"], vector)


def test_packet_dimensions() -> None:
    assert coordinate_packets().packet_dimensions() == {"left": 2, "right": 2}


def test_aggregate_packets() -> None:
    fine = PacketSystem(
        [np.diag([1, 0, 0]), np.diag([0, 1, 0]), np.diag([0, 0, 1])],
        names=["a", "b", "c"],
    )
    coarse = fine.aggregate({"ab": ["a", "b"], "c": ["c"]})
    assert coarse.packet_dimensions() == {"ab": 2, "c": 1}


def test_aggregate_must_partition() -> None:
    fine = PacketSystem([np.diag([1, 0]), np.diag([0, 1])])
    with pytest.raises(ValidationError):
        fine.aggregate({"only": [0]})


def test_tensor_packets() -> None:
    left = PacketSystem([np.diag([1, 0]), np.diag([0, 1])], names=["0", "1"])
    right = PacketSystem([np.diag([1, 0]), np.diag([0, 1])], names=["a", "b"])
    tensor = left.tensor(right)
    assert len(tensor) == 4
    assert tensor.dimension == 4
    assert tensor.packet_dimensions() == {"0⊗a": 1, "0⊗b": 1, "1⊗a": 1, "1⊗b": 1}


def test_dual_packets_exact() -> None:
    p0 = sp.Matrix([[1, 1], [0, 0]])
    p1 = sp.eye(2) - p0
    packets = PacketSystem([p0, p1], backend="sympy")
    dual = packets.dual()
    assert dual.certificate().valid


def test_hilbert_certificate_distinguishes_oblique_packets() -> None:
    p0 = np.array([[1, 1], [0, 0]], dtype=float)
    p1 = np.eye(2) - p0
    packets = PacketSystem([p0, p1])
    assert packets.certificate().valid
    assert not packets.hilbert_certificate().valid


def test_block_diagonal_projection() -> None:
    packets = coordinate_packets()
    operator = np.arange(16, dtype=float).reshape(4, 4)
    projected = packets.block_diagonal_projection(operator)
    assert np.allclose(projected[:2, 2:], 0)
    assert np.allclose(projected[2:, :2], 0)
