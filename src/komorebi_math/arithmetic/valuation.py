"""Exact valuations, digit sums, and local precision helpers."""

from __future__ import annotations

from fractions import Fraction
from math import comb
from numbers import Integral, Rational

from .._number_theory import require_prime
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _require_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    return int(value)


def digit_sum(value: int, base: int) -> int:
    """Return the sum of the base ``base`` digits of a nonnegative integer."""

    n = _require_integer(value, name="value")
    b = _require_integer(base, name="base")
    if n < 0:
        raise ValidationError("value must be nonnegative")
    if b < 2:
        raise ValidationError("base must be at least two")
    total = 0
    while n:
        total += n % b
        n //= b
    return total


@documented(
    ProvenanceRecord(
        identifier="PSC-IX-D01",
        title="Exact prime valuation",
        status=ClaimStatus.DEFINITION,
        source="Paper IX",
        assumptions=("rational input", "prime valuation base"),
        verification_suite="arithmetic.valuation",
    )
)
def padic_valuation(value: Rational, prime: int) -> int | float:
    """Return the exact ``prime`` adic valuation of an integer or rational.

    Zero has valuation ``float('inf')``.
    """

    p = require_prime(prime)
    if isinstance(value, bool) or not isinstance(value, Rational):
        raise ValidationError("value must be an integer or rational number")
    q = Fraction(value)
    if q == 0:
        return float("inf")
    numerator = abs(q.numerator)
    denominator = q.denominator
    valuation = 0
    while numerator % p == 0:
        numerator //= p
        valuation += 1
    while denominator % p == 0:
        denominator //= p
        valuation -= 1
    return valuation


def factorial_valuation(index: int, prime: int) -> int:
    """Return the exact valuation of ``index!`` at ``prime``."""

    n = _require_integer(index, name="index")
    p = require_prime(prime)
    if n < 0:
        raise ValidationError("index must be nonnegative")
    total = 0
    power = p
    while power <= n:
        total += n // power
        power *= p
    return total


@documented(
    ProvenanceRecord(
        identifier="PSC-IX-P18",
        title="Binomial carry valuation",
        status=ClaimStatus.CLASSICAL,
        source="Paper IX",
        assumptions=("prime valuation base", "valid binomial indices"),
        verification_suite="arithmetic.valuation",
    )
)
def binomial_valuation(index: int, choose: int, prime: int) -> int:
    """Return the exact valuation of ``binomial(index, choose)``."""

    n = _require_integer(index, name="index")
    k = _require_integer(choose, name="choose")
    p = require_prime(prime)
    if n < 0 or k < 0 or k > n:
        raise ValidationError("binomial indices must satisfy 0 <= choose <= index")
    return factorial_valuation(n, p) - factorial_valuation(k, p) - factorial_valuation(n - k, p)


def binomial_valuation_from_digits(index: int, choose: int, prime: int) -> int:
    """Return the binomial valuation through the exact digit sum formula."""

    n = _require_integer(index, name="index")
    k = _require_integer(choose, name="choose")
    p = require_prime(prime)
    if n < 0 or k < 0 or k > n:
        raise ValidationError("binomial indices must satisfy 0 <= choose <= index")
    numerator = digit_sum(k, p) + digit_sum(n - k, p) - digit_sum(n, p)
    return numerator // (p - 1)


def falling_factorial(value: int, order: int) -> int:
    """Return the exact falling factorial ``value * ... * (value-order+1)``."""

    n = _require_integer(value, name="value")
    r = _require_integer(order, name="order")
    if r < 0:
        raise ValidationError("order must be nonnegative")
    if n < 0:
        raise ValidationError("value must be nonnegative")
    if r > n:
        return 0
    result = 1
    for offset in range(r):
        result *= n - offset
    return result


def carmichael_exponent_prime_power(prime: int, depth: int) -> int:
    """Return the Carmichael exponent of ``(Z / prime**depth Z)^*``."""

    p = require_prime(prime)
    level = _require_integer(depth, name="depth")
    if level < 1:
        raise ValidationError("depth must be positive")
    if p != 2:
        return (p - 1) * p ** (level - 1)
    if level == 1:
        return 1
    if level == 2:
        return 2
    return 2 ** (level - 2)


def unit_difference_value(base: int, index: int, step: int, order: int) -> int:
    """Return ``Delta_step**order base**index`` exactly."""

    u = _require_integer(base, name="base")
    n = _require_integer(index, name="index")
    h = _require_integer(step, name="step")
    r = _require_integer(order, name="order")
    if n < 0 or h < 0 or r < 0:
        raise ValidationError("index, step, and order must be nonnegative")
    return u**n * (u**h - 1) ** r


def unit_difference_valuation(base: int, index: int, step: int, order: int, prime: int) -> int | float:
    """Return the valuation of an exact unit mode finite difference."""

    p = require_prime(prime)
    u = _require_integer(base, name="base")
    if u % p == 0:
        raise ValidationError("base must be a unit at the declared prime")
    return padic_valuation(unit_difference_value(u, index, step, order), p)


def verify_binomial_digit_formula(index: int, choose: int, prime: int) -> bool:
    """Return whether the factorial and digit formulas agree."""

    return binomial_valuation(index, choose, prime) == binomial_valuation_from_digits(index, choose, prime)


def exact_binomial(index: int, choose: int) -> int:
    """Return a validated exact binomial coefficient."""

    n = _require_integer(index, name="index")
    k = _require_integer(choose, name="choose")
    if n < 0 or k < 0 or k > n:
        return 0
    return comb(n, k)


__all__ = [
    "binomial_valuation",
    "binomial_valuation_from_digits",
    "carmichael_exponent_prime_power",
    "digit_sum",
    "exact_binomial",
    "factorial_valuation",
    "falling_factorial",
    "padic_valuation",
    "unit_difference_value",
    "unit_difference_valuation",
    "verify_binomial_digit_formula",
]
