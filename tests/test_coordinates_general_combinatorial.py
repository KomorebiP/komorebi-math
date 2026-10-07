from fractions import Fraction

from komorebi_math.coordinates.combinatorial import (
    binomial_transform,
    divisor_zeta_transform,
    falling_factorial_to_monomial_transform,
    forward_difference_transform,
    monomial_to_falling_factorial_transform,
    monomial_to_rising_factorial_transform,
    newton_evaluation_transform,
    rising_factorial_to_monomial_transform,
    rising_to_falling_factorial_transform,
    subset_zeta_transform,
)
from komorebi_math.coordinates.exact import identity_matrix, matrix_multiply


def test_stirling_and_lah_transforms_are_exact_connections() -> None:
    degree = 10
    falling = monomial_to_falling_factorial_transform(degree)
    falling_inverse = falling_factorial_to_monomial_transform(degree)
    rising = monomial_to_rising_factorial_transform(degree)
    rising_inverse = rising_factorial_to_monomial_transform(degree)
    lah = rising_to_falling_factorial_transform(degree)
    identity = identity_matrix(degree + 1)
    assert matrix_multiply(falling_inverse.matrix, falling.matrix) == identity
    assert matrix_multiply(rising_inverse.matrix, rising.matrix) == identity
    assert matrix_multiply(falling.matrix, rising_inverse.matrix) == lah.matrix


def test_binomial_and_forward_difference_transforms() -> None:
    forward = binomial_transform(8)
    inverse = binomial_transform(8, inverse=True)
    values = tuple(range(8))
    assert inverse.apply(forward.apply(values)) == tuple(Fraction(value) for value in values)
    differences = forward_difference_transform(5)
    evaluation = newton_evaluation_transform(5)
    squares = (0, 1, 4, 9, 16)
    assert differences.apply(squares) == (0, 1, 2, 0, 0)
    assert evaluation.apply(differences.apply(squares)) == tuple(Fraction(value) for value in squares)


def test_poset_zeta_and_mobius_coordinates() -> None:
    divisor = divisor_zeta_transform(24)
    mobius = divisor_zeta_transform(24, inverse=True)
    values = tuple(Fraction((index * index + 3) % 11) for index in range(24))
    assert mobius.apply(divisor.apply(values)) == values

    subset = subset_zeta_transform(5)
    subset_inverse = subset_zeta_transform(5, inverse=True)
    Boolean_values = tuple(Fraction(index % 7) for index in range(32))
    assert subset_inverse.apply(subset.apply(Boolean_values)) == Boolean_values
