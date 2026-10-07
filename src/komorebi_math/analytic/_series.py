"""Finite formal power series helpers for the analytic package."""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from numbers import Number
from typing import TypeVar

from ..exceptions import ValidationError

Scalar = TypeVar("Scalar", bound=Number)


def validate_order(order: int, *, minimum: int = 1) -> int:
    """Validate and return a positive integer order."""

    if isinstance(order, bool) or int(order) != order or int(order) < minimum:
        raise ValidationError(f"order must be an integer at least {minimum}")
    return int(order)


def validate_degree(degree: int) -> int:
    """Validate and return a nonnegative integer degree."""

    if isinstance(degree, bool) or int(degree) != degree or int(degree) < 0:
        raise ValidationError("degree must be a nonnegative integer")
    return int(degree)


def multiply_series(left: list[Scalar], right: list[Scalar], degree: int) -> list[Scalar]:
    """Multiply truncated ordinary power series."""

    nmax = validate_degree(degree)
    if not left or not right:
        return []
    zero = left[0] * 0 + right[0] * 0
    result = [zero for _ in range(nmax + 1)]
    for i, lhs in enumerate(left[: nmax + 1]):
        for j, rhs in enumerate(right[: nmax + 1 - i]):
            result[i + j] += lhs * rhs
    return result


def inverse_series(coefficients: list[Scalar], degree: int) -> list[Scalar]:
    """Invert an ordinary power series with nonzero constant term."""

    nmax = validate_degree(degree)
    if not coefficients or coefficients[0] == 0:
        raise ValidationError("series inversion requires a nonzero constant term")
    zero = coefficients[0] * 0
    result = [zero for _ in range(nmax + 1)]
    result[0] = 1 / coefficients[0]
    for n in range(1, nmax + 1):
        total = zero
        for k in range(1, min(n, len(coefficients) - 1) + 1):
            total += coefficients[k] * result[n - k]
        result[n] = -total / coefficients[0]
    return result


def logarithmic_coefficients(coefficients: list[Scalar], degree: int) -> list[Scalar]:
    """Return ``kappa`` where ``-log A(w)=sum kappa[n] w**n/n``.

    The input must have constant term one. The returned list has length
    ``degree + 1`` and index zero is unused.
    """

    nmax = validate_degree(degree)
    if not coefficients or coefficients[0] != 1:
        raise ValidationError("logarithmic coefficients require constant term one")
    zero = coefficients[0] * 0
    result = [zero for _ in range(nmax + 1)]
    for n in range(1, nmax + 1):
        a_n = coefficients[n] if n < len(coefficients) else zero
        total = n * a_n
        for k in range(1, n):
            a_k = coefficients[k] if k < len(coefficients) else zero
            total += a_k * result[n - k]
        result[n] = -total
    return result


def power_series_from_log(
    kappa: list[Scalar],
    parameter: Scalar,
    degree: int,
) -> list[Scalar]:
    """Return coefficients of ``exp(parameter * sum kappa[n] w**n/n)``."""

    nmax = validate_degree(degree)
    zero = parameter * 0
    result = [zero for _ in range(nmax + 1)]
    result[0] = parameter * 0 + 1
    for n in range(1, nmax + 1):
        total = zero
        for k in range(1, n + 1):
            if k < len(kappa):
                total += kappa[k] * result[n - k]
        result[n] = parameter * total / n
    return result


def signed_denominator_coefficients(order: int, degree: int, *, exact: bool = True) -> list[Number]:
    """Return coefficients of ``sum (-1)^n w^n/(order*n)!``."""

    n = validate_order(order)
    nmax = validate_degree(degree)
    if exact:
        return [Fraction((-1) ** k, factorial(n * k)) for k in range(nmax + 1)]
    return [((-1.0) ** k) / factorial(n * k) for k in range(nmax + 1)]


def positive_section_coefficients(order: int, residue: int, degree: int, *, exact: bool = True) -> list[Number]:
    """Return normalized coefficients ``residue!/(order*n+residue)!``."""

    n = validate_order(order)
    j = int(residue)
    if isinstance(residue, bool) or j != residue or not 0 <= j < n:
        raise ValidationError("residue must lie between zero and order minus one")
    nmax = validate_degree(degree)
    numerator = factorial(j)
    if exact:
        return [Fraction(numerator, factorial(n * k + j)) for k in range(nmax + 1)]
    return [numerator / factorial(n * k + j) for k in range(nmax + 1)]


__all__ = [
    "inverse_series",
    "logarithmic_coefficients",
    "multiply_series",
    "positive_section_coefficients",
    "power_series_from_log",
    "signed_denominator_coefficients",
    "validate_degree",
    "validate_order",
]
