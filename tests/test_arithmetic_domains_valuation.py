from fractions import Fraction

import pytest

from komorebi_math.arithmetic import (
    ModularRing,
    PrimeField,
    TruncatedPadic,
    binomial_valuation,
    binomial_valuation_from_digits,
    carmichael_exponent_prime_power,
    digit_sum,
    factorial_valuation,
    falling_factorial,
    padic_valuation,
    unit_difference_value,
    unit_difference_valuation,
)
from komorebi_math.exceptions import ValidationError


def test_modular_and_truncated_local_domains() -> None:
    ring = ModularRing(12)
    assert ring.normalize(-1) == 11
    assert ring.multiply(7, 5) == 11
    assert ring.is_unit(5)
    assert ring.inverse(5) == 5
    with pytest.raises(ValidationError):
        ring.inverse(6)

    field = PrimeField(7)
    assert field.inverse(3) == 5
    with pytest.raises(ValueError):
        PrimeField(9)

    local = TruncatedPadic(3, 4)
    assert local.modulus == 81
    assert local.valuation(27) == 3
    assert local.valuation(0) == local.precision
    assert local.valuation(local.modulus) == local.precision
    assert local.is_unit(10)
    assert local.refine(2).precision == 6
    assert local.coarsen(2).modulus == 9


def test_exact_valuations_and_digit_formulas() -> None:
    assert padic_valuation(Fraction(40, 27), 2) == 3
    assert padic_valuation(Fraction(40, 27), 3) == -3
    assert padic_valuation(0, 5) == float("inf")
    assert digit_sum(31, 2) == 5
    assert factorial_valuation(20, 3) == 8
    for prime in (2, 3, 5, 7):
        for n in range(30):
            for k in range(n + 1):
                assert binomial_valuation(n, k, prime) == binomial_valuation_from_digits(n, k, prime)


def test_falling_factorials_and_carmichael_exponents() -> None:
    assert falling_factorial(7, 3) == 210
    assert falling_factorial(3, 5) == 0
    assert carmichael_exponent_prime_power(5, 3) == 100
    assert carmichael_exponent_prime_power(2, 1) == 1
    assert carmichael_exponent_prime_power(2, 2) == 2
    assert carmichael_exponent_prime_power(2, 6) == 16


def test_unit_mode_difference_identity() -> None:
    value = unit_difference_value(2, 3, 4, 3)
    assert value == 2**3 * (2**4 - 1) ** 3
    assert unit_difference_valuation(2, 3, 4, 3, 5) == 3
    with pytest.raises(ValidationError):
        unit_difference_valuation(5, 0, 4, 1, 5)
