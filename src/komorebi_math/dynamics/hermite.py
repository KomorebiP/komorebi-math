"""Exact finite polynomial Gaussian transfer from Paper VI, Section E7.

Polynomials use ascending monomial coefficients.  For rational correlation
``rho`` in [0, 1], the Ornstein--Uhlenbeck transfer is evaluated from Gaussian
moments without an approximate square root or exponential.  In the paper's
time convention rho = exp(-t).  Rational rho is an exact parameterization
of this finite polynomial sector, not an implementation of the infinite
Gaussian Lp space or a claim that exp(-t) is rational for a supplied time.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from math import comb, factorial
from numbers import Integral

from ..exceptions import ValidationError

Polynomial = tuple[Fraction, ...]


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return int(value)


def _rational(value: int | Fraction, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
        raise ValidationError(f"{name} must be an integer or Fraction")
    # Fraction accepts Integral subclasses without owning their components.
    # Rebuild those components before arithmetic can dispatch to fixed-width
    # NumPy scalars instead of Python's arbitrary-precision integers.
    return (
        Fraction(int(value))
        if isinstance(value, Integral)
        else Fraction(int(value.numerator), int(value.denominator))
    )


def _trim(values: Iterable[Fraction]) -> Polynomial:
    result = list(values)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result) if result else (Fraction(0),)


def _polynomial(values: Iterable[int | Fraction], max_degree: int) -> Polynomial:
    maximum = _integer(max_degree, "max_degree")
    result: list[Fraction] = []
    for value in values:
        if len(result) == maximum + 1:
            raise ValidationError("polynomial exceeds max_degree")
        result.append(_rational(value, "polynomial coefficient"))
    return _trim(result)


def gaussian_moment(order: int, *, max_order: int = 512) -> int:
    """Return the exact order-th moment of the standard real Gaussian."""
    degree = _integer(order, "order")
    maximum = _integer(max_order, "max_order")
    if degree > maximum:
        raise ValidationError("Gaussian moment order exceeds max_order")
    if degree % 2:
        return 0
    half = degree // 2
    return factorial(degree) // (2**half * factorial(half))


def gaussian_polynomial_inner_product(
    left: Iterable[int | Fraction],
    right: Iterable[int | Fraction],
    *,
    max_degree: int = 256,
) -> Fraction:
    """Integrate the product of two real rational polynomials against N(0,1)."""
    maximum = _integer(max_degree, "max_degree")
    first, second = _polynomial(left, maximum), _polynomial(right, maximum)
    return sum(
        (
            a * b * gaussian_moment(i + j, max_order=2 * maximum)
            for i, a in enumerate(first)
            for j, b in enumerate(second)
        ),
        Fraction(0),
    )


def ornstein_uhlenbeck_generator(
    polynomial: Iterable[int | Fraction], *, max_degree: int = 256
) -> Polynomial:
    """Return the exact polynomial generator Lf = f'' - x f'."""
    coefficients = _polynomial(polynomial, max_degree)
    return _trim(
        (index + 2) * (index + 1) * coefficients[index + 2] - index * coefficients[index]
        if index + 2 < len(coefficients)
        else -index * coefficients[index]
        for index in range(len(coefficients))
    )


def gaussian_transfer_polynomial(
    polynomial: Iterable[int | Fraction],
    correlation: int | Fraction,
    *,
    max_degree: int = 256,
) -> Polynomial:
    """Evaluate E[f(rho*x + sqrt(1-rho**2)*Y)] exactly on polynomials.

    Y is standard Gaussian.  Only even moments survive, so all coefficients
    are rational whenever the supplied coefficients and rho are rational.
    The finite degree limit raises rather than truncating the polynomial.
    """
    coefficients = _polynomial(polynomial, max_degree)
    rho = _rational(correlation, "correlation")
    if not 0 <= rho <= 1:
        raise ValidationError("correlation must lie in [0, 1]")
    variance = 1 - rho**2
    result = [Fraction(0)] * len(coefficients)
    for degree, coefficient in enumerate(coefficients):
        for half in range(degree // 2 + 1):
            even = 2 * half
            result[degree - even] += (
                coefficient
                * comb(degree, even)
                * rho ** (degree - even)
                * variance**half
                * gaussian_moment(even, max_order=max_degree)
            )
    return _trim(result)


def gaussian_polynomial_variance(polynomial: Iterable[int | Fraction], *, max_degree: int = 256) -> Fraction:
    """Return the exact variance of a real rational Gaussian polynomial."""
    coefficients = _polynomial(polynomial, max_degree)
    mean = gaussian_polynomial_inner_product(coefficients, (1,), max_degree=max_degree)
    second = gaussian_polynomial_inner_product(coefficients, coefficients, max_degree=max_degree)
    return second - mean**2


__all__ = [
    "gaussian_moment",
    "gaussian_polynomial_inner_product",
    "gaussian_polynomial_variance",
    "gaussian_transfer_polynomial",
    "ornstein_uhlenbeck_generator",
]
