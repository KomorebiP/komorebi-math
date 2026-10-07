"""Installed verification for finite deformation and support models."""

from __future__ import annotations

import numpy as np
from importlib.util import find_spec

from ..deformation import (
    FiniteCochainComplex,
    FrobeniusSpine,
    PresentedModule,
    VisibilityTree,
    finite_frobenius_separator,
    normal_crossing_count,
    normal_crossing_series,
    residue_forest,
)
from .core import VerificationCheck, VerificationReport, _capture


def run_deformation_verification() -> VerificationReport:
    """Run the installed finite deformation verification suite."""

    checks: list[VerificationCheck] = []

    def forest_check() -> None:
        forest = residue_forest({1: (0, 1), 2: (0, 1, 2, 3), 3: (0, 1, 4, 5)}, prime=2)
        report = forest.report()
        report.certificate.require()
        assert len(report.branches) == 4

    checks.append(_capture("finite lift forest stable cores", forest_check))

    def visibility_count_check() -> None:
        tree = VisibilityTree((0, 4, 2, 1), prime=2)
        assert tree.partition(2) == ((0, 1), (2,), (3,))
        assert tree.ultrametric_certificate()
        series = normal_crossing_series(3)
        assert tuple(int(value) for value in series.first(6)[1:]) == tuple(normal_crossing_count(3, depth) for depth in range(1, 6))

    checks.append(_capture("visibility and counting charts", visibility_count_check))

    def complex_check() -> None:
        complex_ = FiniteCochainComplex({0: 2, 1: 2}, {0: [[1, 0], [0, 0]]}, prime=3)
        complex_.certificate().require()
        assert complex_.cohomology(0).cohomology_dimension == 1
        assert complex_.cohomology(1).cohomology_dimension == 1

    checks.append(_capture("prime field cohomology", complex_check))

    def support_check() -> None:
        import sympy as sp

        x, y = sp.symbols("x y")
        module = PresentedModule([[x, 0], [0, y]])
        module.fiber({x: 0, y: 1}).certificate.require()
        assert module.fiber({x: 0, y: 1}).fiber_dimension == 1
        assert not module.fiber({x: 1, y: 1}).occupied

    if find_spec("sympy") is not None:
        checks.append(_capture("Fitting fiber support", support_check))

    def frobenius_check() -> None:
        spine = FrobeniusSpine({2: (1, -3, 2), 3: (1, -4, 3)})
        assert spine.rank == 2
        separator = finite_frobenius_separator({2: (1, 1), 3: (1, -1)})
        separator.certificate.require()
        assert separator.recover((5, 1)) == (3, 2)
        deficient = finite_frobenius_separator({2: (1, 1), 3: (2, 2)})
        assert not deficient.separating
        assert not deficient.certificate.valid

    checks.append(_capture("finite Frobenius separation", frobenius_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_deformation_verification"]
