"""Installed verification suite for the nonlinear foundation."""

from __future__ import annotations

import numpy as np

from ..core import PacketSystem
from ..nonlinear import (
    LinearLiftProblem,
    LinearObstructionProblem,
    NonlinearPacketMap,
    NonlinearRefinementDefect,
    PolynomialLiftProblem,
    TransportEdge,
    TransportGraph,
)
from .core import VerificationCheck, VerificationReport, _capture


def run_nonlinear_foundation_verification() -> VerificationReport:
    """Run the installed nonlinear foundation verification suite."""

    checks: list[VerificationCheck] = []

    def coordinate_check() -> None:
        packets = PacketSystem([np.diag([1, 0]), np.diag([0, 1])])
        function = lambda value: np.array([value[0] ** 2 + value[1], value[0] * value[1]])
        packet_map = NonlinearPacketMap(function, packets, packets)
        packet_map.certificate(np.array([2.0, 3.0])).require()

    checks.append(_capture("nonlinear packet coordinates", coordinate_check))

    def defect_check() -> None:
        defect = NonlinearRefinementDefect(
            fine_map=lambda value: 0.5 * np.asarray(value),
            coarse_map=lambda value: 0.5 * np.asarray(value),
            coarsen_source=np.eye(1),
            coarsen_target=np.eye(1),
        )
        assert defect.iterated_orbit_bound(np.array([2.0]), iterations=5, coarse_lipschitz=0.5) == 0.0

    checks.append(_capture("nonlinear refinement defects", defect_check))

    def obstruction_check() -> None:
        result = LinearObstructionProblem([[1, 0], [0, 0]], [1, 1]).solve()
        result.certificate.require()
        assert not result.vanishes
        exact = LinearObstructionProblem([[1, 0], [0, 1]], [1, 2]).solve()
        exact.require_correction()

    checks.append(_capture("cokernel obstruction criterion", obstruction_check))

    def lifting_check() -> None:
        linear = LinearLiftProblem([[2]], [0], [0], prime=2, depth=1).step()
        linear.certificate.require()
        assert len(linear.children) == 2
        function = lambda point: (point[0] * point[0] - 2,)
        jacobian = lambda point: [[2 * point[0]]]
        polynomial = PolynomialLiftProblem(function, jacobian, [3], prime=7, depth=1).step()
        polynomial.certificate.require()
        assert len(polynomial.children) == 1

    checks.append(_capture("prime power lifting", lifting_check))

    def holonomy_check() -> None:
        identity = np.eye(2)
        graph = TransportGraph(
            ["a", "b", "c"],
            [
                TransportEdge("a", "b", identity, "ab"),
                TransportEdge("b", "c", identity, "bc"),
                TransportEdge("a", "c", identity, "ac"),
            ],
        )
        graph.gluing_certificate("a").require()
        graph.intertwine(graph, identity, root="a").certificate.require()

    checks.append(_capture("finite matrix holonomy", holonomy_check))

    def metric_covariant_holonomy_check() -> None:
        graph = TransportGraph(
            ["a"],
            [TransportEdge("a", "a", np.array([[1.0, 1e-6], [0.0, 1.0]]), "loop")],
            tolerance=1e-12,
        )
        metric = np.array([[3.0, 0.4], [0.4, 1.5]])
        before = graph.metric_gluing_certificate(metric)
        gauge = np.array([[1e3, 2.0], [0.0, 1.0]])
        after_graph = graph.gauge_transform({"a": gauge}, gauge_regime="full_gl")
        after_metric = graph.transport_root_metric(metric, gauge)
        after = after_graph.metric_gluing_certificate(after_metric)
        assert np.isclose(
            before.residuals["loop[0]"],
            after.residuals["loop[0]"],
            rtol=1e-8,
            atol=1e-12,
        )

    checks.append(_capture("metric-covariant full-GL holonomy", metric_covariant_holonomy_check))

    return VerificationReport(tuple(checks))


def run_nonlinear_verification() -> VerificationReport:
    """Run the complete nonlinear verification suite."""

    from .nonlinear_advanced import run_advanced_nonlinear_verification

    reports = (run_nonlinear_foundation_verification(), run_advanced_nonlinear_verification())
    return VerificationReport(tuple(check for report in reports for check in report.checks))


__all__ = ["run_nonlinear_foundation_verification", "run_nonlinear_verification"]
