"""Congruential reciprocal, Appell, and theta polynomial families."""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial
from numbers import Number
from typing import Any

from ..exceptions import BackendUnavailableError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._series import (
    logarithmic_coefficients,
    multiply_series,
    positive_section_coefficients,
    power_series_from_log,
    validate_degree,
    validate_order,
)


def congruential_reciprocal_coefficients(
    order: int,
    residue: int,
    parameter: Number,
    degree: int,
) -> tuple[Number, ...]:
    """Return normalized coefficients of a congruential reciprocal power."""

    n = validate_order(order)
    m = validate_degree(degree)
    if not isinstance(parameter, Number):
        raise ValidationError("parameter must be numeric")
    scalar: Number = Fraction(parameter) if isinstance(parameter, (int, Fraction)) and not isinstance(parameter, bool) else complex(parameter)
    base = positive_section_coefficients(n, residue, m, exact=isinstance(scalar, Fraction))
    kappa = logarithmic_coefficients(base, m)
    normalized = power_series_from_log(kappa, scalar, m)
    return tuple(normalized[index] * factorial(n * index) for index in range(m + 1))


@documented(
    ProvenanceRecord(
        identifier="PSC-IV-T28",
        title="Congruential triangular inverse",
        status=ClaimStatus.PROVED,
        source="Paper IV",
        assumptions=("finite cyclic order", "valid residue address"),
        verification_suite="analytic.appell",
    )
)
def congruential_inverse(order: int, residue: int, degree: int) -> tuple[Number, ...]:
    """Return the parameter one congruential reciprocal coefficients."""

    return congruential_reciprocal_coefficients(order, residue, 1, degree)


def appell_polynomial(
    order: int,
    residue: int,
    parameter: Number,
    degree: int,
    x: Number,
) -> Number:
    """Evaluate one congruential Appell polynomial."""

    n = validate_order(order)
    m = validate_degree(degree)
    coefficients = congruential_reciprocal_coefficients(n, residue, parameter, m // n)
    return sum(comb(m, n * k) * coefficients[k] * x ** (m - n * k) for k in range(m // n + 1))


def appell_coefficients(
    order: int,
    residue: int,
    parameter: Number,
    degree: int,
) -> tuple[Number, ...]:
    """Return monomial coefficients in ascending powers for one Appell polynomial."""

    n = validate_order(order)
    m = validate_degree(degree)
    reciprocal = congruential_reciprocal_coefficients(n, residue, parameter, m // n)
    values: list[Number] = [0 for _ in range(m + 1)]
    for k in range(m // n + 1):
        values[m - n * k] = comb(m, n * k) * reciprocal[k]
    return tuple(values)


def appell_polynomial_symbolic(order: int, residue: int, parameter: Any, degree: int) -> Any:
    """Return a SymPy Appell polynomial in its spatial variable."""

    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError("symbolic Appell polynomials require the exact SymPy extra") from exc
    x = sp.Symbol("x")
    n = validate_order(order)
    m = validate_degree(degree)
    phi = sum(sp.factorial(residue) * sp.Symbol("w") ** k / sp.factorial(n * k + residue) for k in range(m // n + 1))
    w = sp.Symbol("w")
    series = sp.series(phi ** (-parameter), w, 0, m // n + 1).removeO().expand()
    return sp.expand(
        sum(
            sp.binomial(m, n * k)
            * series.coeff(w, k)
            * sp.factorial(n * k)
            * x ** (m - n * k)
            for k in range(m // n + 1)
        )
    )


def theta_polynomial(order: int, degree: int, parameter: Number) -> Number:
    """Evaluate the transfer theta polynomial of one degree."""

    n = validate_order(order)
    m = validate_degree(degree)
    if not isinstance(parameter, Number):
        raise ValidationError("parameter must be numeric")
    exact = isinstance(parameter, (int, Fraction)) and not isinstance(parameter, bool)
    u: Number = Fraction(parameter) if exact else complex(parameter)
    h = [Fraction(1, factorial(n * k)) if exact else 1.0 / factorial(n * k) for k in range(m + 1)]
    scaled = [h[k] * u ** (n * k) for k in range(m + 1)]
    inv_scaled_log = logarithmic_coefficients(scaled, m)
    inv_scaled = power_series_from_log(inv_scaled_log, 1, m)
    h_log = logarithmic_coefficients(h, m)
    h_power = power_series_from_log(h_log, u, m)
    product = multiply_series(inv_scaled, h_power, m)
    return product[m] * factorial(n * m)


def theta_polynomial_symbolic(order: int, degree: int) -> Any:
    """Return the exact SymPy transfer theta polynomial."""

    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError("symbolic theta polynomials require the exact SymPy extra") from exc
    n = validate_order(order)
    m = validate_degree(degree)
    u, w = sp.symbols("u w")
    h = sum(w**k / sp.factorial(n * k) for k in range(m + 1))
    expression = sp.series(
        (sum(u ** (n * k) * w**k / sp.factorial(n * k) for k in range(m + 1))) ** -1
        * h ** (-u),
        w,
        0,
        m + 1,
    ).removeO().expand()
    return sp.expand(expression.coeff(w, m) * sp.factorial(n * m))


__all__ = [
    "appell_coefficients",
    "appell_polynomial",
    "appell_polynomial_symbolic",
    "congruential_inverse",
    "congruential_reciprocal_coefficients",
    "theta_polynomial",
    "theta_polynomial_symbolic",
]
