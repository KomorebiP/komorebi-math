"""Canonical synthesis defaults and explicit singular value cutoffs."""

from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.signals import FiniteFilterBank, PeriodicFilterBank


@pytest.mark.parametrize("options", [{}, {"rcond": None}])
def test_finite_canonical_dual_preserves_both_modes_by_default(options: dict) -> None:
    analysis = np.diag([2.0, 0.1])
    bank = FiniteFilterBank(analysis, np.zeros((2, 2)))

    dual = bank.canonical_dual(**options)

    np.testing.assert_allclose(dual.synthesis_matrix, np.diag([0.5, 10.0]))
    np.testing.assert_allclose(dual.reconstruction_operator(), np.eye(2))


def test_finite_canonical_dual_honors_explicit_cutoff() -> None:
    bank = FiniteFilterBank(np.diag([2.0, 0.1]), np.zeros((2, 2)))

    dual = bank.canonical_dual(rcond=0.1)

    np.testing.assert_allclose(dual.synthesis_matrix, np.diag([0.5, 0.0]))
    np.testing.assert_allclose(dual.reconstruction_operator(), np.diag([1.0, 0.0]))


@pytest.mark.parametrize("options", [{}, {"rcond": None}])
def test_periodic_canonical_synthesis_preserves_both_modes_by_default(options: dict) -> None:
    bank = PeriodicFilterBank(([1.05, 0.95],), ([1.05, 0.95],), decimation=1)

    synthesis = bank.canonical_synthesis_matrix(2, **options)

    np.testing.assert_allclose(synthesis, [[5.25, -4.75], [-4.75, 5.25]])
    np.testing.assert_allclose(synthesis @ bank.analysis_matrix(2), np.eye(2), atol=1e-14)


def test_periodic_canonical_synthesis_honors_explicit_cutoff() -> None:
    bank = PeriodicFilterBank(([1.05, 0.95],), ([1.05, 0.95],), decimation=1)

    synthesis = bank.canonical_synthesis_matrix(2, rcond=0.1)

    np.testing.assert_allclose(synthesis, np.full((2, 2), 0.25))
    np.testing.assert_allclose(synthesis @ bank.analysis_matrix(2), np.full((2, 2), 0.5))
