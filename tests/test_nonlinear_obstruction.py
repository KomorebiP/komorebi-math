from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.nonlinear import LinearObstructionProblem, ObstructionSquare


def test_numerical_obstruction_vanishes() -> None:
    problem = LinearObstructionProblem([[1, 0], [0, 1], [1, 1]], [1, 2, 3])
    result = problem.solve()
    result.certificate.require()
    correction = result.require_correction()
    assert np.allclose(np.asarray(problem.correction_map) @ correction + np.asarray(problem.defect), 0)
    assert result.cokernel_dimension == 1


def test_numerical_obstruction_nonzero() -> None:
    problem = LinearObstructionProblem([[1, 0], [0, 0]], [1, 1])
    result = problem.solve()
    result.certificate.require()
    assert not result.vanishes
    assert result.correction is None
    assert abs(result.obstruction_coordinates[0]) > 0.5


def test_exact_obstruction() -> None:
    sp = pytest.importorskip("sympy")
    problem = LinearObstructionProblem(sp.Matrix([[1, 0], [0, 0]]), sp.Matrix([1, 0]), backend="sympy")
    result = problem.solve()
    result.certificate.require()
    assert result.vanishes
    assert problem.correction_map * result.require_correction() + problem.defect == sp.zeros(2, 1)


def test_exact_nonzero_obstruction() -> None:
    sp = pytest.importorskip("sympy")
    problem = LinearObstructionProblem(sp.Matrix([[1], [0]]), sp.Matrix([0, 1]), backend="sympy")
    result = problem.solve()
    result.certificate.require()
    assert not result.vanishes
    assert result.obstruction_coordinates == (1,)


def test_obstruction_square() -> None:
    square = ObstructionSquare(
        source_correction=np.eye(2),
        target_correction=2 * np.eye(2),
        source_map=np.eye(2),
        target_map=2 * np.eye(2),
    )
    square.certificate().require()
    assert np.allclose(square.transport_defect([1, 2]), [2, 4])


def test_noncommuting_obstruction_square_fails() -> None:
    square = ObstructionSquare(
        source_correction=np.eye(2),
        target_correction=np.eye(2),
        source_map=np.eye(2),
        target_map=2 * np.eye(2),
    )
    assert not square.certificate().valid
