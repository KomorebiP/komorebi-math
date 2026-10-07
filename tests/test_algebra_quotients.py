from __future__ import annotations

import numpy as np
import sympy as sp

from komorebi_math import (
    OperatorAlgebra,
    PacketSystem,
    independent_direct_sum_seed_rank,
    matrix_units,
    shared_direct_sum_seed_bounds,
    tensor_seed_rank_upper_bound,
)


def test_full_matrix_algebra_from_matrix_units() -> None:
    algebra = OperatorAlgebra(matrix_units(2))
    assert algebra.dimension == 4
    assert algebra.certificate().valid
    assert algebra.contains(np.array([[1, 2], [3, 4]], dtype=float))


def test_polynomial_algebra_generated_by_jordan_block() -> None:
    jordan = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=float)
    algebra = OperatorAlgebra([jordan])
    assert algebra.dimension == 3
    assert algebra.certificate().valid


def test_exact_sympy_operator_algebra() -> None:
    nilpotent = sp.Matrix([[0, 1], [0, 0]])
    algebra = OperatorAlgebra([nilpotent], backend="sympy")
    assert algebra.dimension == 2
    assert algebra.certificate().valid


def test_algebra_transfer_graph() -> None:
    shift = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    packets = PacketSystem(
        [np.diag([1, 0, 0]), np.diag([0, 1, 0]), np.diag([0, 0, 1])]
    )
    graph = OperatorAlgebra([shift]).transfer_graph(packets)
    assert graph.reachable([0]) == frozenset({0, 1, 2})


def test_packet_coordinate_transport() -> None:
    packets = PacketSystem([np.diag([1, 0]), np.diag([0, 1])], names=["a", "b"])
    change = np.array([[1, 1], [1, -1]], dtype=float)
    transported = packets.transport(change)
    assert transported.certificate().valid
    assert transported.names == packets.names


def test_packet_restriction() -> None:
    packets = PacketSystem(
        [np.diag([1, 0, 0]), np.diag([0, 1, 1])], names=["first", "rest"]
    )
    inclusion = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    left_inverse = inclusion.T
    restricted = packets.restrict(inclusion, left_inverse)
    assert restricted.packet_dimensions() == {"first": 0, "rest": 2}
    assert restricted.certificate().valid


def test_packet_quotient() -> None:
    packets = PacketSystem(
        [np.diag([1, 0, 0]), np.diag([0, 1, 1])], names=["first", "rest"]
    )
    quotient = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
    section = quotient.T
    descended = packets.quotient(quotient, section)
    assert descended.packet_dimensions() == {"first": 0, "rest": 2}
    assert descended.certificate().valid


def test_seed_rank_construction_helpers() -> None:
    assert shared_direct_sum_seed_bounds([2, 5, 3]) == (5, 10)
    assert independent_direct_sum_seed_rank([2, 5, 3]) == 10
    assert tensor_seed_rank_upper_bound(3, 4) == 12
