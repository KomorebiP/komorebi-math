from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear import (
    LinearLiftProblem,
    PolynomialLiftProblem,
    build_lift_tree,
    solve_mod_prime,
)


def test_solve_mod_prime_affine_space() -> None:
    result = solve_mod_prime([[1, 1, 0], [0, 1, 1]], [1, 0], 3)
    assert result.solvable
    assert result.solution_count == 3
    solutions = result.enumerate()
    assert len(solutions) == 3
    for vector in solutions:
        assert np.all((np.array([[1, 1, 0], [0, 1, 1]]) @ np.array(vector) - np.array([1, 0])) % 3 == 0)


def test_solve_mod_prime_obstruction() -> None:
    result = solve_mod_prime([[0]], [1], 2)
    assert not result.solvable
    assert result.obstruction_rows == (0,)


def test_linear_lift_children() -> None:
    problem = LinearLiftProblem([[2]], [0], [0], prime=2, depth=1)
    result = problem.step()
    result.certificate.require()
    assert result.children == ((0,), (2,))
    assert result.correction_space.solution_count == 2


def test_linear_lift_rejects_invalid_current_point() -> None:
    problem = LinearLiftProblem([[1]], [0], [1], prime=2, depth=1)
    with pytest.raises(ValidationError, match="not a solution"):
        problem.step()


def test_polynomial_hensel_unique_lift() -> None:
    function = lambda x: (x[0] * x[0] - 2,)
    jacobian = lambda x: [[2 * x[0]]]
    problem = PolynomialLiftProblem(function, jacobian, [3], prime=7, depth=1)
    result = problem.step()
    result.certificate.require()
    assert result.correction_space.solution_count == 1
    assert len(result.children) == 1
    assert function(result.children[0])[0] % 49 == 0


def test_singular_polynomial_has_multiple_children() -> None:
    function = lambda x: (x[0] * x[0],)
    jacobian = lambda x: [[2 * x[0]]]
    problem = PolynomialLiftProblem(function, jacobian, [0], prime=2, depth=1)
    result = problem.step()
    assert result.children == ((0,), (2,))


def test_build_finite_lift_tree() -> None:
    function = lambda x: (x[0] * x[0] - 1,)
    jacobian = lambda x: [[2 * x[0]]]

    def factory(point: tuple[int, ...], depth: int) -> PolynomialLiftProblem:
        return PolynomialLiftProblem(function, jacobian, point, prime=3, depth=depth)

    tree = build_lift_tree(factory, [(1,), (2,)], prime=3, start_depth=1, max_depth=4)
    assert len(tree.levels) == 4
    assert all(len(level) == 2 for level in tree.levels)
    branches = tree.terminal_branches()
    assert len(branches) == 2
    assert all(len(branch) == 4 for branch in branches)


def test_lifting_rejects_composite_prime() -> None:
    with pytest.raises(ValueError, match="prime"):
        solve_mod_prime([[1]], [0], 4)
    with pytest.raises(ValueError, match="prime"):
        LinearLiftProblem([[1]], [0], [0], prime=6, depth=1)


def test_polynomial_lift_rejects_incorrect_jacobian() -> None:
    function = lambda x: (x[0] * x[0] - 1,)
    wrong_jacobian = lambda x: [[0]]
    problem = PolynomialLiftProblem(function, wrong_jacobian, [1], prime=3, depth=1)
    with pytest.raises(ValidationError, match="Jacobian"):
        problem.step()


def test_polynomial_lift_marks_unchecked_large_jacobian_conditional() -> None:
    function = lambda x: tuple(value for value in x)
    jacobian = lambda x: np.eye(len(x), dtype=int)
    point = [0] * 8
    problem = PolynomialLiftProblem(function, jacobian, point, prime=3, depth=1)
    with pytest.raises(ValueError, match="Jacobian verification"):
        problem.step(linearization_limit=100)
    result = problem.step(linearization_limit=100, assume_valid_jacobian=True)
    assert result.certificate.exactness.value == "conditional"
    assert result.certificate.assumptions_unchecked
