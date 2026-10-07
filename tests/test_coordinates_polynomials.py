from fractions import Fraction

from komorebi_math.coordinates import (
    connection_transform,
    derivative_operator_in_family,
    euler_polynomial,
    euler_to_fibonacci_coefficient,
    euler_to_fibonacci_transform,
    fibonacci_companion_power,
    fibonacci_polynomial,
    fibonacci_polynomial_family,
    monomial_polynomial_family,
    monomial_to_fibonacci_coefficient,
    operator_change_of_coordinates,
    polynomial_evaluate,
    secant_euler_from_fibonacci_coordinates,
    signed_secant_euler_numbers,
)
from komorebi_math.coordinates.exact import matrix_multiply
from komorebi_math.coordinates.polynomials import euler_polynomial_family


def _add(left, right):
    size = max(len(left), len(right))
    return tuple(
        (left[index] if index < len(left) else 0)
        + (right[index] if index < len(right) else 0)
        for index in range(size)
    )


def _scale(poly, scalar):
    return tuple(Fraction(scalar) * value for value in poly)


def test_fibonacci_companion_polynomial_identity() -> None:
    for n in range(1, 20):
        power = fibonacci_companion_power(n)
        assert power[0][0] == fibonacci_polynomial(n + 1)
        assert power[0][1] == fibonacci_polynomial(n)
        assert power[1][0] == fibonacci_polynomial(n)
        assert power[1][1] == fibonacci_polynomial(n - 1)


def test_monomial_to_fibonacci_formula() -> None:
    for n in range(30):
        reconstructed = (Fraction(0),)
        for k in range(n + 1):
            reconstructed = _add(
                reconstructed,
                _scale(fibonacci_polynomial(k + 1), monomial_to_fibonacci_coefficient(n, k)),
            )
        expected = tuple(Fraction(int(index == n)) for index in range(n + 1))
        assert reconstructed == expected


def test_euler_to_fibonacci_formula_and_matrix() -> None:
    for degree in range(16):
        transform = euler_to_fibonacci_transform(degree)
        for n in range(degree + 1):
            reconstructed = (Fraction(0),)
            for k in range(n + 1):
                coefficient = transform.matrix[k][n]
                assert coefficient == euler_to_fibonacci_coefficient(n, k)
                reconstructed = _add(reconstructed, _scale(fibonacci_polynomial(k + 1), coefficient))
            assert reconstructed == euler_polynomial(n)


def test_connection_composition_and_operator_conjugacy() -> None:
    degree = 12
    euler = euler_polynomial_family()
    fibonacci = fibonacci_polynomial_family()
    monomial = monomial_polynomial_family()
    euler_to_monomial = connection_transform(euler, monomial, degree)
    monomial_to_fibonacci = connection_transform(monomial, fibonacci, degree)
    direct = connection_transform(euler, fibonacci, degree)
    assert euler_to_monomial.then(monomial_to_fibonacci).matrix == direct.matrix
    derivative_euler = derivative_operator_in_family(euler, degree)
    derivative_fibonacci = derivative_operator_in_family(fibonacci, degree)
    conjugated = operator_change_of_coordinates(derivative_euler, direct)
    assert conjugated == derivative_fibonacci


def test_secant_euler_specialization() -> None:
    expected = signed_secant_euler_numbers(18)
    actual = tuple(secant_euler_from_fibonacci_coordinates(index) for index in range(18))
    assert actual == expected
    for n in range(12):
        assert Fraction(2**n) * polynomial_evaluate(euler_polynomial(n), Fraction(1, 2)) == expected[n]
