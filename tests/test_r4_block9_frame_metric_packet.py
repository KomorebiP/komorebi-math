"""Episode 005 regression tests for quotient metrics and metric-covariant holonomy."""

from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.coordinates import AnalysisSynthesisGauge, AnalysisSynthesisSystem
from komorebi_math.coordinates import ExactAnalysisSynthesisSystem
from komorebi_math.coordinates.exact import matrix_multiply, transpose
from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear import TransportEdge, TransportGraph


def _system(analysis: np.ndarray, *, name: str) -> AnalysisSynthesisSystem:
    system = AnalysisSynthesisSystem.canonical_dual(
        analysis,
        source="physical",
        coordinates=name,
        tolerance=1e-10,
    )
    system.certificate().require()
    return system


def test_transition_pulls_back_target_quotient_metric_exactly() -> None:
    first = _system(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        name="first",
    )
    second = _system(
        np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]]
        ),
        name="second",
    )
    first_gauge = AnalysisSynthesisGauge(first)
    second_gauge = AnalysisSynthesisGauge(second)
    transition = first_gauge.transition_to(second_gauge)
    first_metric = first_gauge.coefficient_metric()
    second_metric = second_gauge.coefficient_metric()

    np.testing.assert_allclose(
        transition.conj().T @ second_metric @ transition,
        first_metric,
        atol=2e-12,
        rtol=2e-12,
    )

    coefficient = np.array([3.0, -4.0, 5.0])
    assert second_gauge.coefficient_seminorm(transition @ coefficient) == pytest.approx(
        first_gauge.coefficient_seminorm(coefficient),
        abs=2e-12,
        rel=2e-12,
    )


def test_transition_is_an_isomorphism_only_after_quotienting_gauge_kernel() -> None:
    first = _system(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        name="first",
    )
    second = _system(
        np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]]
        ),
        name="second",
    )
    first_gauge = AnalysisSynthesisGauge(first)
    second_gauge = AnalysisSynthesisGauge(second)
    forward = first_gauge.transition_to(second_gauge)
    backward = second_gauge.transition_to(first_gauge)

    np.testing.assert_allclose(
        backward @ forward,
        first_gauge.coefficient_projection(),
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        forward @ backward,
        second_gauge.coefficient_projection(),
        atol=2e-12,
        rtol=2e-12,
    )


def test_exact_transition_metric_pullback_has_zero_residual() -> None:
    first = ExactAnalysisSynthesisSystem.canonical_dual(
        ((1, 0), (0, 1), (1, 1)),
        source="physical",
        coordinates="first exact",
    )
    second = ExactAnalysisSynthesisSystem.canonical_dual(
        ((1, 0), (0, 1), (1, -1), (2, 1)),
        source="physical",
        coordinates="second exact",
    )
    transition = first.transition_to(second)
    pulled_back = matrix_multiply(
        matrix_multiply(transpose(transition), second.coefficient_metric()),
        transition,
    )
    assert pulled_back == first.coefficient_metric()


def test_nonunitary_coefficient_reparameterization_distorts_euclidean_norm_not_quotient_metric() -> None:
    analysis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    synthesis = np.linalg.pinv(analysis)
    base = AnalysisSynthesisSystem(analysis, synthesis, source="physical", tolerance=1e-10)
    base_gauge = AnalysisSynthesisGauge(base)

    scale = 1e6
    coordinate_change = np.diag([scale, 1.0, 1.0 / scale])
    inverse = np.linalg.inv(coordinate_change)
    transformed = AnalysisSynthesisSystem(
        coordinate_change @ analysis,
        synthesis @ inverse,
        source="physical",
        coordinates="strongly oblique",
        tolerance=1e-10,
    )
    transformed_gauge = AnalysisSynthesisGauge(transformed)

    coefficient = np.array([3.0, -4.0, 5.0])
    changed = coordinate_change @ coefficient
    assert np.linalg.norm(changed) / np.linalg.norm(coefficient) > 1e5
    assert transformed_gauge.coefficient_seminorm(changed) == pytest.approx(
        base_gauge.coefficient_seminorm(coefficient), abs=2e-12, rel=2e-12
    )

    base_metric = base_gauge.coefficient_metric()
    transformed_metric = transformed_gauge.coefficient_metric()
    np.testing.assert_allclose(
        transformed_metric,
        inverse.conj().T @ base_metric @ inverse,
        atol=1e-4,
        rtol=2e-12,
    )


def test_metric_holonomy_residual_is_full_gl_coordinate_covariant() -> None:
    epsilon = 1e-6
    holonomy = np.array([[1.0, epsilon], [0.0, 1.0]])
    graph = TransportGraph(
        ("v",),
        (TransportEdge("v", "v", holonomy, "loop"),),
        tolerance=1e-12,
    )
    metric = np.array([[3.0, 0.4], [0.4, 1.5]])
    original = graph.metric_gluing_certificate(metric)

    gauge = np.array([[1e4, 2.0], [0.0, 1.0]])
    transformed_graph = graph.gauge_transform({"v": gauge}, gauge_regime="full_gl")
    transformed_metric = graph.transport_root_metric(metric, gauge)
    transformed = transformed_graph.metric_gluing_certificate(transformed_metric)

    assert transformed.residuals["loop[0]"] == pytest.approx(
        original.residuals["loop[0]"], rel=3e-7, abs=3e-12
    )

    coordinate_original = graph.gluing_certificate(gauge_regime="coordinate")
    coordinate_transformed = transformed_graph.gluing_certificate(gauge_regime="coordinate")
    assert coordinate_transformed.residuals["loop[0]"] > 1e3 * coordinate_original.residuals["loop[0]"]


def test_metric_holonomy_contract_rejects_non_positive_metric_and_bare_full_gl_still_fails_closed() -> None:
    graph = TransportGraph(
        ("v",),
        (TransportEdge("v", "v", np.array([[1.0, 1e-8], [0.0, 1.0]]), "loop"),),
        tolerance=1e-10,
    )
    with pytest.raises(ValidationError, match="positive definite"):
        graph.metric_gluing_certificate(np.diag([1.0, 0.0]))
    with pytest.raises(ValidationError, match="not gauge invariant"):
        graph.gluing_certificate(gauge_regime="full_gl")
