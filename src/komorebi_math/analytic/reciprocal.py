"""Reciprocal powers, logarithmic coefficients, and transfer tangent data."""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from numbers import Number
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import BackendUnavailableError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._series import (
    logarithmic_coefficients as _logarithmic_coefficients,
    power_series_from_log,
    signed_denominator_coefficients,
    validate_degree,
    validate_order,
)
from .sections import denominator, denominator_derivative


@documented(
    ProvenanceRecord(
        identifier="PSC-IV-T17",
        title="Logarithmic reciprocal coefficients",
        status=ClaimStatus.PROVED,
        source="Paper IV",
        assumptions=("finite cyclic order",),
        verification_suite="analytic.reciprocal",
    )
)
def logarithmic_coefficients(order: int, degree: int) -> tuple[Fraction, ...]:
    """Return exact ``kappa`` coefficients of the reduced denominator."""

    n = validate_order(order)
    nmax = validate_degree(degree)
    coefficients = signed_denominator_coefficients(n, nmax, exact=True)
    return tuple(_logarithmic_coefficients(coefficients, nmax))


@documented(
    ProvenanceRecord(
        identifier="PSC-IV-T15",
        title="Reciprocal power coefficients",
        status=ClaimStatus.PROVED,
        source="Paper IV",
        assumptions=("finite cyclic order", "declared reciprocal parameter"),
        verification_suite="analytic.reciprocal",
    )
)
def reciprocal_coefficients(
    order: int,
    parameter: Number,
    degree: int,
) -> tuple[Number, ...]:
    """Return ``P[order,n](parameter)`` for ``0 <= n <= degree``.

    The coefficients use the exponential normalization
    ``D_order(z)**(-parameter) = sum P_n z**(order*n)/(order*n)!``.
    Integer and rational parameters retain exact ``Fraction`` arithmetic.
    """

    n = validate_order(order)
    nmax = validate_degree(degree)
    if not isinstance(parameter, Number):
        raise ValidationError("parameter must be numeric")
    if isinstance(parameter, (int, Fraction)) and not isinstance(parameter, bool):
        scalar: Number = Fraction(parameter)
    else:
        scalar = complex(parameter)
    kappa = list(logarithmic_coefficients(n, nmax))
    normalized = power_series_from_log(kappa, scalar, nmax)
    return tuple(normalized[index] * factorial(n * index) for index in range(nmax + 1))


def reciprocal_polynomial(order: int, degree: int) -> Any:
    """Return the exact SymPy polynomial in the reciprocal parameter."""

    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError("reciprocal_polynomial requires the exact SymPy extra") from exc
    n = validate_order(order)
    m = validate_degree(degree)
    r = sp.Symbol("r")
    kappa = [sp.Rational(value.numerator, value.denominator) for value in logarithmic_coefficients(n, m)]
    coefficients = [sp.Integer(1)] + [sp.Integer(0)] * m
    for index in range(1, m + 1):
        coefficients[index] = sp.expand(r * sum(kappa[k] * coefficients[index - k] for k in range(1, index + 1)) / index)
    return sp.expand(coefficients[m] * factorial(n * m))


def reciprocal_value(order: int, z: complex, parameter: complex = 1.0) -> complex:
    """Evaluate the principal branch of the reciprocal power."""

    value = denominator(order, z)
    if value == 0:
        raise ZeroDivisionError("the cyclic denominator vanishes at the evaluation point")
    return value ** (-complex(parameter))


def transfer_tangent(order: int, z: complex) -> complex:
    """Evaluate ``-D_order'(z)/D_order(z)``."""

    value = denominator(order, z)
    if value == 0:
        raise ZeroDivisionError("the cyclic denominator vanishes at the evaluation point")
    return -denominator_derivative(order, z) / value


def transfer_tangent_coefficients(order: int, degree: int) -> dict[int, Fraction]:
    """Return ordinary series coefficients of the transfer tangent."""

    n = validate_order(order)
    nmax = validate_degree(degree)
    kappa = logarithmic_coefficients(n, nmax + 1)
    return {n * index + n - 1: Fraction(n) * kappa[index + 1] for index in range(nmax + 1)}


def reciprocal_semigroup_certificate(
    order: int,
    left_parameter: Number,
    right_parameter: Number,
    degree: int,
    *,
    tolerance: float = 1e-10,
) -> ComputationResult[tuple[Number, ...]]:
    """Validate the reciprocal parameter convolution through one finite degree."""

    n = validate_order(order)
    nmax = validate_degree(degree)
    left = reciprocal_coefficients(n, left_parameter, nmax)
    right = reciprocal_coefficients(n, right_parameter, nmax)
    combined = reciprocal_coefficients(n, left_parameter + right_parameter, nmax)
    residuals: list[float] = []
    reconstructed: list[Number] = []
    from math import comb

    for index in range(nmax + 1):
        value = sum(comb(n * index, n * q) * left[q] * right[index - q] for q in range(index + 1))
        reconstructed.append(value)
        residuals.append(float(abs(complex(value) - complex(combined[index]))))
    maximum = max(residuals, default=0.0)
    exact = all(isinstance(value, Fraction) for value in (*left, *right, *combined))
    return ComputationResult(
        tuple(reconstructed),
        Certificate(
            valid=maximum <= tolerance,
            exactness=Exactness.EXACT if exact and maximum == 0 else Exactness.NUMERICAL,
            residuals={"maximum_coefficient": maximum},
            tolerance=None if exact and maximum == 0 else tolerance,
            assumptions_checked=("finite reciprocal coefficient system",),
            theorem_id="PSC-IV-T18",
            message="reciprocal parameter convolution agrees through the requested degree",
        ),
    )


__all__ = [
    "logarithmic_coefficients",
    "reciprocal_coefficients",
    "reciprocal_polynomial",
    "reciprocal_semigroup_certificate",
    "reciprocal_value",
    "transfer_tangent",
    "transfer_tangent_coefficients",
]
