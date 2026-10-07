"""Installed verification suite for refinement and filtered transfer."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np

from ..core import PacketSystem
from ..refinement import (
    LatticePacketData,
    LinearRefinementDefect,
    PacketRefinement,
    SmithCongruenceProblem,
    WeightedTransferEdge,
    WeightedTransferGraph,
    crt_pair,
    nested_residue_analysis,
    residue_analyze,
)
from .core import VerificationCheck, VerificationReport, _capture


def _singleton_packets(size: int) -> PacketSystem:
    return PacketSystem([np.diag([1 if row == column else 0 for row in range(size)]) for column in range(size)])


def run_refinement_verification() -> VerificationReport:
    """Run the installed refinement and filtered transfer verification suite."""

    checks: list[VerificationCheck] = []

    def tower_check() -> None:
        coarse = PacketSystem([np.diag([1, 1, 0, 0]), np.diag([0, 0, 1, 1])])
        fine = _singleton_packets(4)
        refinement = PacketRefinement(coarse, fine, (0, 0, 1, 1))
        vector = np.array([1, 2, 3, 4], dtype=float)
        fine_coordinates = refinement.refine(coarse.analyze(vector))
        assert np.allclose(refinement.coarsen(fine_coordinates).reconstruct(), vector)

    checks.append(_capture("packet refinement coherence", tower_check))

    def residue_check() -> None:
        values = np.arange(24)
        direct = residue_analyze(values, 12)
        nested = nested_residue_analysis(values, 4, 3)
        assert all(np.array_equal(left, right) for left, right in zip(direct.channels, nested.channels, strict=True))
        combined = crt_pair(1, 4, 2, 3)
        assert combined.residue == 5

    checks.append(_capture("residue and CRT coordinates", residue_check))

    def weighted_check() -> None:
        packets = _singleton_packets(3)
        edges = [
            WeightedTransferEdge(0, 1, 1, np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0]], dtype=float)),
            WeightedTransferEdge(1, 2, 1, np.array([[0, 0, 0], [0, 0, 0], [0, 3, 0]], dtype=float)),
        ]
        graph = WeightedTransferGraph(packets, np.eye(3), edges)
        result = graph.exact_inverse_if_acyclic()
        result.certificate.require()
        assert np.allclose(result.value, np.linalg.inv(graph.operator))

    checks.append(_capture("acyclic weighted inverse", weighted_check))

    def defect_check() -> None:
        defect = LinearRefinementDefect([[1, 0]], [[1, 2], [3, 4]], [[1]])
        defect.power(4).certificate.require()
        defect.resolvent(10).certificate.require()

    checks.append(_capture("linear defect propagation", defect_check))

    if find_spec("sympy") is not None:
        def exact_check() -> None:
            import sympy as sp

            problem = SmithCongruenceProblem([[2, 0, 0], [0, 4, 0]], [2, 4], prime=2, depth=3)
            result = problem.solve()
            result.require()
            assert result.solution_count == 64
            p0 = sp.Matrix([[1, sp.Rational(1, 4)], [0, 0]])
            p1 = sp.eye(2) - p0
            data = LatticePacketData(sp.eye(2), [p0, p1], prime=2, split_basis=sp.Matrix([[1, -1], [0, 4]]))
            data.gluing_weight_certificate().require()

        checks.append(_capture("exact Smith and lattice separation", exact_check))

    return VerificationReport(tuple(checks))
