from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.algorithms import (
    NumpyPacketExecutor,
    PacketIR,
    PacketLayout,
    cupy_available,
    discover_block_diagonal_low_rank,
    discover_packet_support,
    packet_executor,
)
from komorebi_math.exceptions import BackendUnavailableError, ValidationError


def test_support_discovery() -> None:
    layout = PacketLayout((2, 2))
    matrix = np.block(
        [
            [np.eye(2), 1e-14 * np.ones((2, 2))],
            [np.ones((2, 2)), 2 * np.eye(2)],
        ]
    )
    result = discover_packet_support(matrix, layout, absolute_tolerance=1e-12)
    assert result.packet_ir.support == frozenset({(0, 0), (1, 0), (1, 1)})
    assert result.certificate.exactness.value == "heuristic"


def test_bdlr_discovery_is_optimal_for_fixed_diagonal() -> None:
    rng = np.random.default_rng(13)
    layout = PacketLayout((3, 2))
    diagonal = np.block([[np.eye(3), np.zeros((3, 2))], [np.zeros((2, 3)), 2 * np.eye(2)]])
    left = np.zeros((5, 1))
    right = np.zeros((5, 1))
    left[:3, 0] = rng.normal(size=3)
    right[3:, 0] = rng.normal(size=2)
    matrix = diagonal + left @ right.T
    result = discover_block_diagonal_low_rank(matrix, layout, 1)
    result.certificate.require()
    assert np.allclose(result.operator.materialize(), matrix, atol=1e-10)


def test_discovery_validation() -> None:
    layout = PacketLayout((1, 1))
    matrix = np.eye(2)
    with pytest.raises(ValidationError):
        discover_packet_support(matrix, layout, relative_tolerance=float("nan"))
    with pytest.raises(ValidationError):
        discover_block_diagonal_low_rank(matrix, layout, 1.5)


def test_numpy_packet_executor() -> None:
    layout = PacketLayout((1, 2))
    ir = PacketIR.square(layout, {(0, 0): [[2.0]], (1, 1): np.eye(2)})
    executor = packet_executor(ir, "numpy")
    assert isinstance(executor, NumpyPacketExecutor)
    x = np.array([1.0, 2.0, 3.0])
    result = executor.apply(x)
    result.certificate.require()
    assert np.allclose(result.value, ir.materialize() @ x)


def test_cupy_executor_availability_boundary() -> None:
    layout = PacketLayout((1,))
    ir = PacketIR.square(layout, {(0, 0): [[1.0]]})
    if not cupy_available():
        with pytest.raises(BackendUnavailableError):
            packet_executor(ir, "cupy")
