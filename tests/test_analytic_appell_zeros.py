from __future__ import annotations

from math import comb, cos, factorial, pi, sqrt

import numpy as np
import pytest

from komorebi_math.analytic import (
    CyclicDenominator,
    appell_coefficients,
    appell_polynomial,
    appell_polynomial_symbolic,
    congruential_inverse,
    congruential_reciprocal_coefficients,
    hadamard_partial_product,
    positive_zero_string,
    reciprocal_coefficient_from_zeros,
    reciprocal_coefficients,
    reduced_zero_zeta,
    theta_polynomial,
    theta_polynomial_symbolic,
    zero_count_asymptotic,
)
from komorebi_math.certificates import Exactness
from komorebi_math.exceptions import ValidationError


def test_congruential_inverse_identity() -> None:
    for order in range(2, 6):
        for residue in range(order):
            values = congruential_inverse(order, residue, 6)
            for n in range(7):
                total = sum(comb(order * n + residue, order * q) * values[q] for q in range(n + 1))
                assert total == (1 if n == 0 else 0)


def test_congruential_parameter_degree_values() -> None:
    left = congruential_reciprocal_coefficients(3, 1, 2, 5)
    right = congruential_reciprocal_coefficients(3, 1, 3, 5)
    combined = congruential_reciprocal_coefficients(3, 1, 5, 5)
    for n in range(6):
        convolution = sum(comb(3 * n, 3 * q) * left[q] * right[n - q] for q in range(n + 1))
        assert convolution == combined[n]


def test_appell_polynomial_and_coefficients() -> None:
    degree = 8
    coefficients = appell_coefficients(3, 1, 2, degree)
    x = 0.31
    direct = sum(value * x**power for power, value in enumerate(coefficients))
    assert abs(complex(direct - appell_polynomial(3, 1, 2, degree, x))) < 1e-12
    h = 1e-6
    derivative = (
        appell_polynomial(3, 1, 2, degree, x + h)
        - appell_polynomial(3, 1, 2, degree, x - h)
    ) / (2 * h)
    assert abs(complex(derivative - degree * appell_polynomial(3, 1, 2, degree - 1, x))) < 2e-6


def test_symbolic_appell_and_theta() -> None:
    sympy = pytest.importorskip("sympy")
    x = sympy.Symbol("x")
    polynomial = appell_polynomial_symbolic(2, 0, 1, 5)
    assert sympy.simplify(sympy.diff(polynomial, x) - 5 * appell_polynomial_symbolic(2, 0, 1, 4)) == 0
    theta = theta_polynomial_symbolic(4, 3)
    u = sympy.Symbol("u")
    assert sympy.simplify(theta.subs(u, 0)) == 0
    assert sympy.simplify(theta.subs(u, -1)) == 0


def test_theta_universal_factors() -> None:
    for degree in range(1, 6):
        assert theta_polynomial(3, degree, 0) == 0
        assert theta_polynomial(4, degree, 0) == 0
        assert theta_polynomial(4, degree, -1) == 0


def test_positive_zero_strings_exact_cases() -> None:
    order_two = positive_zero_string(2, 5)
    order_two.certificate.require()
    expected_two = tuple(pi * (index + 0.5) for index in range(5))
    assert max(abs(left - right) for left, right in zip(order_two.zeros, expected_two, strict=True)) < 1e-11

    order_four = positive_zero_string(4, 5)
    order_four.certificate.require()
    expected_four = tuple(sqrt(2) * pi * (index + 0.5) for index in range(5))
    assert max(abs(left - right) for left, right in zip(order_four.zeros, expected_four, strict=True)) < 1e-10


def test_zero_orbits_and_hadamard_partial_product() -> None:
    result = positive_zero_string(3, 10)
    result.certificate.require()
    orbits = result.rotational_orbits()
    assert all(len(orbit) == 3 for orbit in orbits)
    z = 0.15
    product_value = hadamard_partial_product(3, z, result.zeros)
    denominator_value = CyclicDenominator(3)(z)
    assert abs(product_value - denominator_value) < 1e-5


def test_reciprocal_coefficient_from_zero_orbits_converges() -> None:
    exact = reciprocal_coefficients(2, 1, 3)[3]
    result = positive_zero_string(2, 100)
    approximation = reciprocal_coefficient_from_zeros(2, 3, result.zeros)
    assert abs(complex(approximation) - complex(exact)) < 1e-4


def test_zero_count_and_zeta_diagnostics() -> None:
    assert zero_count_asymptotic(3, 100) > 0
    assert zero_count_asymptotic(3, 100, reduced=True) > 0
    result = reduced_zero_zeta(2, 1.2, terms=30)
    assert result.certificate.exactness is Exactness.NUMERICAL
    assert result.value.real > 0
    model = reduced_zero_zeta(2, 1.2, use_computed_zeros=False)
    assert model.certificate.exactness is Exactness.CONDITIONAL


def test_invalid_zero_requests() -> None:
    with pytest.raises(ValidationError):
        positive_zero_string(1, 3)
    with pytest.raises(ValidationError):
        positive_zero_string(3, 0)
