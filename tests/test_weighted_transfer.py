from __future__ import annotations

import numpy as np
import pytest

from komorebi_math import PacketSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.refinement import WeightedTransferEdge, WeightedTransferGraph


def singleton_packets(size: int, *, backend: str | None = None) -> PacketSystem:
    if backend == "sympy":
        sp = pytest.importorskip("sympy")
        return PacketSystem([sp.diag(*[1 if i == j else 0 for i in range(size)]) for j in range(size)], backend="sympy")
    return PacketSystem([np.diag([1 if i == j else 0 for i in range(size)]) for j in range(size)])


def test_acyclic_weighted_inverse_is_exact() -> None:
    packets = singleton_packets(3)
    edges = [
        WeightedTransferEdge(0, 1, 1, np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=float)),
        WeightedTransferEdge(1, 2, 1, np.array([[0, 0, 0], [0, 0, 0], [0, 3, 0]], dtype=float)),
    ]
    graph = WeightedTransferGraph(packets, np.eye(3), edges)
    result = graph.exact_inverse_if_acyclic()
    result.certificate.require()
    assert np.allclose(result.value, np.linalg.inv(graph.operator))


def test_longer_lower_weight_path_precedes_direct_edge() -> None:
    packets = singleton_packets(3)
    edges = [
        WeightedTransferEdge(0, 2, 3, np.array([[0, 0, 0], [0, 0, 0], [5, 0, 0]], dtype=float)),
        WeightedTransferEdge(0, 1, 1, np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=float)),
        WeightedTransferEdge(1, 2, 1, np.array([[0, 0, 0], [0, 0, 0], [0, 3, 0]], dtype=float)),
    ]
    graph = WeightedTransferGraph(packets, np.eye(3), edges)
    paths = graph.paths_below_weight(3, source=0, target=2)
    assert len(paths) == 1
    assert paths[0].weight == 2
    assert paths[0].vertices == (0, 1, 2)
    assert graph.minimum_path_weight(0, 2) == 2


def test_conditional_filtered_truncation_without_valuation_backend() -> None:
    packets = singleton_packets(2)
    edge = WeightedTransferEdge(0, 1, 2, np.array([[0, 0], [0.25, 0]], dtype=float))
    graph = WeightedTransferGraph(packets, np.eye(2), [edge])
    result = graph.truncated_inverse(2)
    assert result.certificate.valid
    assert result.certificate.exactness.value == "conditional"
    assert result.certificate.assumptions_unchecked


def test_exact_adic_truncation() -> None:
    sp = pytest.importorskip("sympy")
    packets = singleton_packets(2, backend="sympy")
    edge = WeightedTransferEdge(0, 1, 2, sp.Matrix([[0, 0], [4, 0]]))
    graph = WeightedTransferGraph(packets, sp.eye(2), [edge], prime=2)
    shallow = graph.truncated_inverse(2)
    shallow.certificate.require()
    assert shallow.value == sp.eye(2)
    deep = graph.truncated_inverse(3)
    deep.certificate.require()
    assert deep.value == graph.operator.inv()


def test_incorrect_declared_weight_rejected() -> None:
    sp = pytest.importorskip("sympy")
    packets = singleton_packets(2, backend="sympy")
    edge = WeightedTransferEdge(0, 1, 3, sp.Matrix([[0, 0], [4, 0]]))
    with pytest.raises(ValidationError, match="below declared weight"):
        WeightedTransferGraph(packets, sp.eye(2), [edge], prime=2)


def test_transfer_series_matrix() -> None:
    sp = pytest.importorskip("sympy")
    z = sp.symbols("z")
    packets = singleton_packets(2, backend="sympy")
    edge = WeightedTransferEdge(0, 1, 2, sp.Matrix([[0, 0], [4, 0]]))
    graph = WeightedTransferGraph(packets, sp.eye(2), [edge], prime=2)
    series = graph.transfer_series_matrix(z)
    assert series == sp.Matrix([[1, 0], [z**2, 1]])


def test_conditional_truncation_survives_singular_evaluated_operator() -> None:
    packets = singleton_packets(2)
    edges = [
        WeightedTransferEdge(0, 1, 1, np.array([[0.0, 0.0], [1.0, 0.0]])),
        WeightedTransferEdge(1, 0, 1, np.array([[0.0, 1.0], [0.0, 0.0]])),
    ]
    graph = WeightedTransferGraph(packets, np.eye(2), edges)
    assert np.linalg.det(graph.operator) == 0.0
    result = graph.truncated_inverse(2)
    assert result.certificate.valid
    assert result.certificate.exactness.value == "conditional"
    assert result.certificate.diagnostics["ordinary_inverse_available"] is False


def test_weighted_path_enumeration_limit() -> None:
    packets = singleton_packets(1)
    edge = WeightedTransferEdge(0, 0, 1, np.array([[0.25]]))
    graph = WeightedTransferGraph(packets, np.eye(1), [edge])
    with pytest.raises(ValueError, match="path enumeration"):
        graph.paths_below_weight(10, max_paths=3)


def test_weighted_graph_rejects_composite_prime() -> None:
    packets = singleton_packets(1)
    with pytest.raises(ValueError, match="prime"):
        WeightedTransferGraph(packets, np.eye(1), [], prime=4)
