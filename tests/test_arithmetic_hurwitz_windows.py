from fractions import Fraction
from math import comb

import pytest

from komorebi_math.arithmetic import (
    HurwitzSeries,
    StandardWindow,
    compressed_hurwitz_inverse,
    hurwitz_inverse,
    hurwitz_product,
    subgroup_roots,
    subgroup_selector,
)
from komorebi_math.exceptions import ValidationError


def test_hurwitz_mode_addition() -> None:
    left = HurwitzSeries.exponential_mode(2, 8)
    right = HurwitzSeries.exponential_mode(3, 8)
    product = left.multiply(right, degree=8)
    assert product.coefficients == HurwitzSeries.exponential_mode(5, 8).coefficients


def test_exact_hurwitz_inverse() -> None:
    source = HurwitzSeries((1, 2, 3, 4, 5))
    inverse = source.inverse(8)
    product = source.multiply(inverse, degree=8)
    assert product.coefficients == (Fraction(1),) + (Fraction(0),) * 8
    assert hurwitz_product((1, 1), (1, -1), degree=3) == (
        Fraction(1),
        Fraction(0),
        Fraction(-2),
        Fraction(0),
    )
    assert hurwitz_inverse((1, 1), 4) == (Fraction(1), Fraction(-1), Fraction(2), Fraction(-6), Fraction(24))


def test_finite_field_subgroup_selector() -> None:
    roots = subgroup_roots(13, 4)
    assert len(roots) == 4
    selector = subgroup_selector(13, 4, 20)
    assert selector.coefficients == tuple(1 if n % 4 == 0 else 0 for n in range(21))
    inverse = selector.inverse(20)
    product = selector.multiply(inverse, degree=20)
    assert product.coefficients == (1,) + (0,) * 20


def test_standard_window_period_address_independence_and_recurrence() -> None:
    for prime in (3, 5, 7, 11, 13):
        for compression in range(1, prime):
            if (prime - 1) % compression:
                continue
            window = StandardWindow(prime, compression)
            certificate = window.certificate(address=prime - 1, periods=5)
            certificate.certificate.require()
            h = window.period
            values = window.reduced_inverse(5 * h + 2)
            assert values == compressed_hurwitz_inverse(prime, compression, 5 * h + 2)
            for n in range(1, len(values) - h):
                assert values[n + h] == values[n]
            for n in range(len(values)):
                total = sum(comb(compression * n, compression * q) * values[q] for q in range(n + 1)) % prime
                assert total == (1 if n == 0 else 0)


def test_standard_window_generating_function() -> None:
    window = StandardWindow(7, 2)
    generating = window.rational_generating_function()
    assert generating.first(20) == window.reduced_inverse(19)
    assert window.minimum_period() == 3


def test_invalid_hurwitz_and_window_inputs() -> None:
    with pytest.raises(ValidationError):
        HurwitzSeries(())
    with pytest.raises(ValidationError):
        subgroup_roots(7, 4)
    with pytest.raises(ValidationError):
        StandardWindow(7, 4).period


def test_default_hurwitz_product_returns_full_finite_product() -> None:
    assert hurwitz_product((1, 1), (1, 1)) == (Fraction(1), Fraction(2), Fraction(2))


def test_modular_hurwitz_rejects_float_coercion() -> None:
    with pytest.raises(ValidationError, match="integers"):
        HurwitzSeries((1, 2.0), modulus=5)
