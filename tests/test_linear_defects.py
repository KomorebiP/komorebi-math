from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.refinement import LinearRefinementDefect, multilevel_defect


def test_linear_defect_matrix() -> None:
    defect = LinearRefinementDefect([[1, 0]], [[1, 2], [3, 4]], [[1]])
    assert np.allclose(defect.matrix, [[0, 2]])
    assert not defect.is_exact()


def test_power_and_polynomial_defects() -> None:
    defect = LinearRefinementDefect([[1, 0]], [[1, 2], [3, 4]], [[1]])
    power = defect.power(4)
    power.certificate.require()
    polynomial = defect.polynomial([3, 2, 1])
    polynomial.certificate.require()
    fine = np.array([[1, 2], [3, 4]], dtype=complex)
    coarse = np.array([[1]], dtype=complex)
    coarsen = np.array([[1, 0]], dtype=complex)
    expected = coarsen @ (3 * np.eye(2) + 2 * fine + fine @ fine) - (3 * np.eye(1) + 2 * coarse + coarse @ coarse) @ coarsen
    assert np.allclose(polynomial.value, expected)


def test_resolvent_defect() -> None:
    defect = LinearRefinementDefect([[1, 0]], [[1, 2], [3, 4]], [[1]])
    result = defect.resolvent(10)
    result.certificate.require()


def test_exponential_defect() -> None:
    pytest.importorskip("scipy")
    defect = LinearRefinementDefect([[1, 0]], [[1, 2], [3, 4]], [[1]])
    result = defect.exponential(0.2)
    result.certificate.require()


def test_multilevel_defect_telescope() -> None:
    c0 = np.array([[1, 0]], dtype=float)
    c1 = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    t0 = np.array([[2]], dtype=float)
    t1 = np.array([[2, 1], [0, 3]], dtype=float)
    t2 = np.array([[2, 1, 0], [0, 3, 4], [0, 0, 5]], dtype=float)
    result = multilevel_defect([c0, c1], [t0, t1, t2])
    result.certificate.require()
    direct = c0 @ c1 @ t2 - t0 @ c0 @ c1
    assert np.allclose(result.value, direct)
