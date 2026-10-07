from fractions import Fraction

from komorebi_math.coordinates.exact import (
    determinant,
    identity_matrix,
    inverse_matrix,
    matrix_multiply,
    matrix_power,
    matrix_rank,
    matrix_vector_multiply,
    nullspace,
    solve_linear_system,
)


def test_inverse_and_power_exact() -> None:
    matrix = ((Fraction(1), Fraction(2)), (Fraction(3), Fraction(5)))
    inverse = inverse_matrix(matrix)
    assert matrix_multiply(matrix, inverse) == identity_matrix(2)
    assert determinant(matrix) == -1
    assert matrix_power(matrix, 0) == identity_matrix(2)
    assert matrix_power(matrix, 3) == matrix_multiply(matrix_multiply(matrix, matrix), matrix)


def test_rectangular_solver_and_nullspace() -> None:
    matrix = ((Fraction(1), Fraction(1), Fraction(0)), (Fraction(0), Fraction(1), Fraction(1)))
    result = solve_linear_system(matrix, (3, 5))
    assert result.consistent
    assert result.rank == 2
    assert result.solution is not None
    assert matrix_vector_multiply(matrix, result.solution) == (3, 5)
    basis = nullspace(matrix)
    assert len(basis) == 1
    assert matrix_vector_multiply(matrix, basis[0]) == (0, 0)
    assert matrix_rank(matrix) == 2


def test_inconsistent_system() -> None:
    matrix = ((Fraction(1),), (Fraction(1),))
    result = solve_linear_system(matrix, (1, 2))
    assert not result.consistent
    assert result.solution is None
