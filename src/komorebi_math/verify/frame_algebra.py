"""Installed verification for redundant-frame algebra."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..coordinates import (
    AnalysisSynthesisGauge,
    AnalysisSynthesisMap,
    AnalysisSynthesisSystem,
    CoefficientPartition,
    ExactAnalysisSynthesisMap,
    ExactAnalysisSynthesisSystem,
)
from ..coordinates.exact import matrix_multiply
from ..exceptions import ValidationError
from .core import VerificationCheck, VerificationReport, _capture


def _oblique(*, source: str = "physical") -> AnalysisSynthesisSystem:
    return AnalysisSynthesisSystem(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([[0.5, -0.5, 0.5], [-0.25, 0.75, 0.25]]),
        source=source,
        coordinates="oblique",
    )


def run_frame_algebra_verification() -> VerificationReport:
    """Verify the numerical and exact coefficient-closure laws."""

    checks: list[VerificationCheck] = []

    def gauge_laws() -> None:
        system = _oblique()
        gauge = AnalysisSynthesisGauge(system)
        gauge.certificate().require()
        projection = gauge.coefficient_projection()
        kernel = np.array([-2.0, -1.0, 1.0])
        assert np.allclose(projection @ projection, projection)
        assert np.linalg.norm(projection - projection.T) > 0.4
        assert np.allclose(system.synthesis @ kernel, 0.0)
        assert gauge.gauge_equivalent(np.ones(3), np.ones(3) + kernel)

    checks.append(_capture("oblique coefficient gauge", gauge_laws))

    def map_laws() -> None:
        system = _oblique()
        operator = np.array([[2.0, -1.0], [3.0, 4.0]])
        mapping = AnalysisSynthesisMap(system, system, operator)
        mapping.certificate().require()
        projection = AnalysisSynthesisGauge(system).coefficient_projection()
        lift = mapping.coefficient_lift()
        assert np.allclose(projection @ lift @ projection, lift)
        assert np.allclose(mapping.reconstruct(), operator)

    checks.append(_capture("coefficient-map closure", map_laws))

    def bridge_law() -> None:
        left = AnalysisSynthesisSystem.canonical_dual(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            source="middle",
        )
        right = AnalysisSynthesisSystem.canonical_dual(
            np.array(
                [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]]
            ),
            source="middle",
        )
        bridge = AnalysisSynthesisGauge(left).transition_to(right)
        assert np.allclose(right.synthesis @ bridge @ left.analysis, np.eye(2))

    checks.append(_capture("cross-presentation bridge", bridge_law))

    def quotient_metric_bridge_law() -> None:
        left = AnalysisSynthesisSystem.canonical_dual(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            source="middle",
        )
        right = AnalysisSynthesisSystem.canonical_dual(
            np.array(
                [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]]
            ),
            source="middle",
        )
        left_gauge = AnalysisSynthesisGauge(left)
        right_gauge = AnalysisSynthesisGauge(right)
        bridge = left_gauge.transition_to(right_gauge)
        assert np.allclose(
            bridge.conj().T @ right_gauge.coefficient_metric() @ bridge,
            left_gauge.coefficient_metric(),
        )

    checks.append(_capture("quotient-metric bridge isometry", quotient_metric_bridge_law))

    def coefficient_blocks() -> None:
        system = _oblique()
        partition = CoefficientPartition(3, ((0, 2), (1,)))
        first = AnalysisSynthesisMap(
            system, system, np.array([[2.0, -1.0], [3.0, 4.0]])
        )
        second = AnalysisSynthesisMap(
            system, system, np.array([[1.0, 2.0], [-2.0, 1.0]])
        )
        left = first.blocks(partition, partition)
        right = second.blocks(partition, partition)
        left.certificate().require()
        right.certificate().require()
        product = sum(
            right.blocks[1][middle] @ left.blocks[middle][0]
            for middle in range(2)
        )
        full = second.coefficient_lift() @ first.coefficient_lift()
        assert np.allclose(product, full[np.ix_(partition.groups[1], partition.groups[0])])

    checks.append(_capture("ordinary coefficient-block multiplication", coefficient_blocks))

    def exact_laws() -> None:
        system = ExactAnalysisSynthesisSystem.canonical_dual(
            ((1, 0), (0, 1), (1, 1))
        )
        assert system.synthesis[0][0] == Fraction(2, 3)
        system.certificate().require()
        first = ExactAnalysisSynthesisMap(system, system, ((2, -1), (3, 4)))
        second = ExactAnalysisSynthesisMap(system, system, ((1, 2), (-2, 1)))
        assert first.then(second).coefficient_lift() == matrix_multiply(
            second.coefficient_lift(), first.coefficient_lift()
        )

    checks.append(_capture("exact rational coefficient closure", exact_laws))

    def threshold_regression() -> None:
        try:
            AnalysisSynthesisSystem.canonical_dual(np.diag([1.0, 9e-16]))
        except ValidationError:
            return
        raise AssertionError("near-singular canonical dual was accepted")

    checks.append(_capture("consistent canonical-dual threshold", threshold_regression))

    return VerificationReport(tuple(checks))


__all__ = ["run_frame_algebra_verification"]
