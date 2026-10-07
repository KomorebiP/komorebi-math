from fractions import Fraction

from komorebi_math.coordinates.calculus import (
    affine_substitution_coordinate_operator,
    derivative_coordinate_operator,
    finite_difference_coordinate_operator,
    integral_coordinate_operator,
    multiplication_coordinate_operator,
    transport_polynomial_operator,
)
from komorebi_math.coordinates.classical import (
    chebyshev_t_polynomial_family,
    falling_factorial_polynomial_family,
    legendre_polynomial_family,
)
from komorebi_math.coordinates.polynomials import monomial_polynomial_family


def test_derivative_and_integral_in_multiple_families() -> None:
    for family in (
        monomial_polynomial_family(),
        falling_factorial_polynomial_family(),
        chebyshev_t_polynomial_family(),
        legendre_polynomial_family(),
    ):
        derivative = derivative_coordinate_operator(family, 8)
        integral = integral_coordinate_operator(family, 8)
        coefficients = tuple(Fraction(index * index - 3) for index in range(9))
        integrated = integral.apply(coefficients)
        recovered = derivative_coordinate_operator(family, 9).apply(integrated)
        assert recovered == coefficients + (Fraction(0),)


def test_falling_factorial_difference_is_a_lowering_operator() -> None:
    family = falling_factorial_polynomial_family()
    difference = finite_difference_coordinate_operator(family, 8)
    for degree in range(1, 9):
        basis_vector = tuple(Fraction(int(index == degree)) for index in range(9))
        output = difference.apply(basis_vector)
        expected = tuple(
            Fraction(degree if index == degree - 1 else 0)
            for index in range(9)
        )
        assert output == expected


def test_operator_transport_and_affine_substitution() -> None:
    monomials = monomial_polynomial_family()
    legendre = legendre_polynomial_family()
    derivative = derivative_coordinate_operator(monomials, 7)
    transported = transport_polynomial_operator(derivative, monomials, legendre)
    direct = derivative_coordinate_operator(legendre, 7)
    assert transported.matrix == direct.matrix

    substitution = affine_substitution_coordinate_operator(legendre, 7, scale=2, shift=-3)
    coefficients = tuple(Fraction(index + 1) for index in range(8))
    assert substitution.square
    assert len(substitution.apply(coefficients)) == 8

    multiplication = multiplication_coordinate_operator(monomials, 5, (1, -2, 1))
    assert multiplication.source_degree == 5
    assert multiplication.target_degree == 7
