from __future__ import annotations

import numpy as np
import pytest

from komorebi_math import Exactness
from komorebi_math.nonlinear.correction import (
    HilbertObstructionProjector,
    LyapunovSchmidtReduction,
    RightInverseCorrectionProblem,
)
from komorebi_math.nonlinear.derivatives import CallableDerivativeProvider, SympyDerivativeProvider
from komorebi_math.nonlinear.elimination import NonlinearSchurElimination


def test_hilbert_obstruction_projection_and_least_norm_correction() -> None:
    projector = HilbertObstructionProjector([[1.0, 0.0], [0.0, 0.0]])
    obstructed = projector.solve([0.0, 1.0])
    obstructed.certificate.require()
    assert not obstructed.vanishes
    assert np.allclose(obstructed.obstruction, [0.0, 1.0])
    assert abs(obstructed.obstruction_norm - 1.0) < 1e-12

    removable = projector.solve([2.0, 0.0])
    removable.certificate.require()
    assert removable.vanishes
    assert np.allclose(removable.least_norm_correction, [-2.0, 0.0])
    assert removable.correction_residual < 1e-12


def test_right_inverse_nonlinear_correction_and_stability() -> None:
    problem = RightInverseCorrectionProblem(
        np.eye(1),
        np.eye(1),
        lambda x: 0.2 * x * x,
        np.array([0.1]),
        radius=0.2,
        lipschitz_bound=0.08,
        tolerance=1e-13,
    )
    problem.hypothesis_certificate().require()
    result = problem.solve()
    result.certificate.require()
    expected = (-1.0 + np.sqrt(1.08)) / 0.4
    assert np.allclose(result.solution, [expected], atol=1e-12)
    assert result.equation_residual < 1e-12
    assert abs(problem.target_stability_bound(0.01) - 0.01 / 0.92) < 1e-14


def exact_schur_problem() -> tuple[object, NonlinearSchurElimination]:
    sp = pytest.importorskip("sympy")
    u, v = sp.symbols("u v")
    retained = SympyDerivativeProvider((3 * u + 2 * v + u * v + v**2,), (u, v))
    hidden = SympyDerivativeProvider((v - 4 * u + u**2 + u * v,), (u, v))
    problem = NonlinearSchurElimination(retained, hidden, 1, 1, sp.Matrix([0]), sp.Matrix([0]))
    return sp, problem


def test_exact_first_and_second_schur_derivatives() -> None:
    sp, problem = exact_schur_problem()
    first = problem.first_derivatives()
    first.certificate.require()
    assert first.hidden_jacobian == sp.Matrix([[1]])
    assert first.implicit_derivative == sp.Matrix([[4]])
    assert first.reduced_derivative == sp.Matrix([[11]])

    second = problem.second_derivative(sp.Matrix([1]), sp.Matrix([1]))
    second.certificate.require()
    assert second.implicit_second_derivative == sp.Matrix([[-10]])
    assert second.reduced_second_derivative == sp.Matrix([[20]])


def test_numerical_hidden_branch_and_reduced_value() -> None:
    retained = CallableDerivativeProvider(
        lambda point: np.array([3 * point[0] + 2 * point[1] + point[0] * point[1] + point[1] ** 2]),
        {
            1: lambda point, directions: np.array([
                (3 + point[1]) * directions[0][0]
                + (2 + point[0] + 2 * point[1]) * directions[0][1]
            ]),
            2: lambda point, directions: np.array([
                directions[0][0] * directions[1][1]
                + directions[0][1] * directions[1][0]
                + 2 * directions[0][1] * directions[1][1]
            ]),
        },
        source_dimension=2,
        target_dimension=1,
    )
    hidden = CallableDerivativeProvider(
        lambda point: np.array([point[1] - 4 * point[0] + point[0] ** 2 + point[0] * point[1]]),
        {
            1: lambda point, directions: np.array([
                (-4 + 2 * point[0] + point[1]) * directions[0][0]
                + (1 + point[0]) * directions[0][1]
            ]),
            2: lambda point, directions: np.array([
                2 * directions[0][0] * directions[1][0]
                + directions[0][0] * directions[1][1]
                + directions[0][1] * directions[1][0]
            ]),
        },
        source_dimension=2,
        target_dimension=1,
    )
    problem = NonlinearSchurElimination(retained, hidden, 1, 1, [0.0], [0.0], tolerance=1e-12)
    u = 0.05
    solve = problem.solve_hidden([u])
    solve.certificate.require()
    expected_hidden = (4 * u - u * u) / (1 + u)
    assert np.allclose(solve.hidden, [expected_hidden], atol=1e-12)
    reduced = problem.reduced_value([u])
    reduced.certificate.require()
    expected_reduced = 3 * u + 2 * expected_hidden + u * expected_hidden + expected_hidden**2
    assert np.allclose(reduced.value, [expected_reduced], atol=1e-12)


def test_lyapunov_schmidt_quadratic_obstruction() -> None:
    provider = CallableDerivativeProvider(
        lambda point: np.array([point[1], point[0] ** 2]),
        {
            1: lambda point, directions: np.array([
                directions[0][1],
                2 * point[0] * directions[0][0],
            ]),
            2: lambda point, directions: np.array([
                0.0,
                2 * directions[0][0] * directions[1][0],
            ]),
        },
        source_dimension=2,
        target_dimension=2,
        exactness=Exactness.NUMERICAL,
    )
    reduction = LyapunovSchmidtReduction(provider, [0.0, 0.0], tolerance=1e-12)
    reduction.decomposition.certificate.require()
    reduction.base_derivative_certificate().require()
    assert reduction.kernel_dimension == 1
    assert reduction.obstruction_dimension == 1

    point = reduction.solve_range([0.3])
    point.certificate.require()
    assert point.range_residual < 1e-12
    assert np.allclose(point.complement_coordinates, [0.0], atol=1e-12)
    assert np.allclose(np.abs(point.reduced_obstruction), [0.09], atol=1e-12)

    second = reduction.reduced_second_derivative([1.0], [1.0])
    second.certificate.require()
    assert np.allclose(np.abs(second.value), [2.0], atol=1e-12)
