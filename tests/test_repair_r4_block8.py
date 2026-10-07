from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear import TransportEdge, TransportGraph


def loop_graph(holonomy: np.ndarray, *, tolerance: float = 1e-10) -> TransportGraph:
    return TransportGraph(
        ["a"],
        [TransportEdge("a", "a", holonomy, "loop")],
        tolerance=tolerance,
    )


def test_coordinate_gluing_certificate_states_noninvariant_semantics() -> None:
    eps = 1e-12
    graph = loop_graph(np.eye(2) + eps * np.array([[0.0, 1.0], [0.0, 0.0]]))
    certificate = graph.gluing_certificate(gauge_regime="coordinate")
    assert certificate.valid
    assert certificate.diagnostics["gauge_regime"] == "coordinate"
    assert "coordinate-dependent" in certificate.diagnostics["residual_semantics"]
    assert "no gauge-invariance claim" in certificate.diagnostics["gauge_invariance"]


def test_full_gl_numerical_tolerance_certificate_fails_closed() -> None:
    graph = loop_graph(np.eye(2))
    with pytest.raises(ValidationError, match="not gauge invariant"):
        graph.gluing_certificate(gauge_regime="full_gl")


def test_unitary_regime_preserves_frobenius_residual() -> None:
    theta = 0.37
    unitary = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    holonomy = np.array([[1.2, 0.3], [-0.1, 0.9]])
    graph = loop_graph(holonomy, tolerance=1e-12)
    before = graph.gluing_certificate(gauge_regime="unitary")
    transformed = graph.gauge_transform({"a": unitary}, gauge_regime="unitary")
    after = transformed.gluing_certificate(gauge_regime="unitary")
    assert np.isclose(before.residuals["loop[0]"], after.residuals["loop[0]"], rtol=1e-12, atol=1e-12)
    assert before.valid == after.valid
    assert "unitary" in after.diagnostics["gauge_invariance"]


def test_unitary_transform_rejects_nonunitary_gauge() -> None:
    graph = loop_graph(np.eye(2))
    gauge = np.diag([10.0, 1.0])
    with pytest.raises(ValidationError, match="not unitary"):
        graph.gauge_transform({"a": gauge}, gauge_regime="unitary")


def test_full_gl_residual_envelope_contains_transformed_residual() -> None:
    eps = 1e-8
    holonomy = np.eye(2) + eps * np.array([[0.0, 1.0], [0.0, 0.0]])
    graph = loop_graph(holonomy, tolerance=1e-10)
    gauge = np.diag([1e4, 1.0])

    envelope = graph.gluing_residual_envelope(gauge)
    transformed = graph.gauge_transform({"a": gauge})
    actual = transformed.gluing_certificate(gauge_regime="coordinate").residuals["loop[0]"]

    assert np.isclose(envelope.condition_number, 1e4, rtol=1e-12)
    assert envelope.lower_bounds["loop[0]"] <= actual <= envelope.upper_bounds["loop[0]"]
    assert np.isclose(envelope.upper_bounds["loop[0]"], actual, rtol=1e-12, atol=1e-15)


def test_coordinate_decision_can_flip_under_full_gl_while_envelope_records_risk() -> None:
    eps = 1e-12
    tolerance = 1e-10
    graph = loop_graph(np.eye(2) + eps * np.array([[0.0, 1.0], [0.0, 0.0]]), tolerance=tolerance)
    gauge = np.diag([1e4, 1.0])

    before = graph.gluing_certificate(gauge_regime="coordinate")
    envelope = graph.gluing_residual_envelope(gauge)
    after_graph = graph.gauge_transform({"a": gauge})
    after = after_graph.gluing_certificate(gauge_regime="coordinate")

    assert before.valid
    assert not after.valid
    assert envelope.source_residuals["loop[0]"] < tolerance
    assert envelope.upper_bounds["loop[0]"] > tolerance


def test_exact_backend_allows_full_gl_flatness_certificate() -> None:
    sp = pytest.importorskip("sympy")
    graph = TransportGraph(
        ["a"],
        [TransportEdge("a", "a", sp.eye(2), "loop")],
        backend="sympy",
    )
    certificate = graph.gluing_certificate(gauge_regime="full_gl")
    assert certificate.valid
    assert certificate.diagnostics["gauge_regime"] == "full_gl"
    assert "full_gl" in certificate.diagnostics["gauge_invariance"]
