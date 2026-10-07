from __future__ import annotations

import numpy as np

from komorebi_math.nonlinear import TransportEdge, TransportGraph


def triangle(last: np.ndarray) -> TransportGraph:
    identity = np.eye(2)
    return TransportGraph(
        ["a", "b", "c"],
        [
            TransportEdge("a", "b", identity, "ab"),
            TransportEdge("b", "c", identity, "bc"),
            TransportEdge("a", "c", last, "ac"),
        ],
    )


def test_tree_gauge_normalizes_tree_edges() -> None:
    rotation = np.array([[0, -1], [1, 0]], dtype=float)
    graph = triangle(rotation)
    normalized = graph.gauge_transform(graph.tree_gauge("a"))
    _, tree_edges = normalized.spanning_tree("a")
    for index in tree_edges:
        assert np.allclose(normalized.edges[index].matrix, np.eye(2))


def test_trivial_cycle_glues() -> None:
    graph = triangle(np.eye(2))
    graph.gluing_certificate("a").require()
    assert len(graph.fundamental_holonomies("a")) == 1


def test_nontrivial_cycle_detected() -> None:
    rotation = np.array([[0, -1], [1, 0]], dtype=float)
    graph = triangle(rotation)
    certificate = graph.gluing_certificate("a")
    assert not certificate.valid
    assert certificate.diagnostics["betti_number"] == 1


def test_gauge_invariance_of_cycle_conjugacy() -> None:
    rotation = np.array([[0, -1], [1, 0]], dtype=float)
    graph = triangle(rotation)
    gauges = {
        "a": np.array([[2, 0], [0, 1]], dtype=float),
        "b": np.array([[1, 1], [0, 1]], dtype=float),
        "c": np.array([[1, 0], [1, 1]], dtype=float),
    }
    transformed = graph.gauge_transform(gauges)
    original_h = graph.fundamental_holonomies("a")[0].matrix
    transformed_h = transformed.fundamental_holonomies("a")[0].matrix
    expected = gauges["a"] @ original_h @ np.linalg.inv(gauges["a"])
    assert np.allclose(transformed_h, expected)


def test_intertwining_identical_systems() -> None:
    rotation = np.array([[0, -1], [1, 0]], dtype=float)
    graph = triangle(rotation)
    result = graph.intertwine(graph, np.eye(2), root="a")
    result.certificate.require()
    assert all(residual < 1e-10 for residual in result.edge_residuals.values())


def test_intertwining_cycle_obstruction() -> None:
    identity_graph = triangle(np.eye(2))
    rotation_graph = triangle(np.array([[0, -1], [1, 0]], dtype=float))
    result = identity_graph.intertwine(rotation_graph, np.eye(2), root="a")
    assert not result.certificate.valid
    assert any(residual > 0.1 for residual in result.cycle_residuals.values())


def test_duplicate_edge_names_remain_distinct_in_certificates() -> None:
    graph = TransportGraph(
        ["a", "b"],
        [
            TransportEdge("a", "b", np.eye(1), "same"),
            TransportEdge("a", "b", np.array([[2.0]]), "same"),
        ],
    )
    certificate = graph.gluing_certificate()
    assert set(certificate.residuals) == {"same[1]"}
    assert not certificate.valid


def test_gauge_dimension_is_checked() -> None:
    graph = TransportGraph(["a", "b"], [TransportEdge("a", "b", np.eye(2))])
    import pytest
    from komorebi_math.exceptions import DimensionError

    with pytest.raises(DimensionError, match="gauge"):
        graph.gauge_transform({"a": np.eye(2), "b": np.eye(3)})
