"""Installed verification for cyclic analytic functions."""

from __future__ import annotations

from fractions import Fraction
from math import comb, cos, factorial

import numpy as np

from ..analytic import (
    CyclicDenominator,
    appell_polynomial,
    congruential_inverse,
    cyclic_projection_certificate,
    denominator,
    denominator_root_sum,
    positive_zero_string,
    reciprocal_coefficients,
    reciprocal_semigroup_certificate,
    section_certificate,
    theta_polynomial,
)
from .core import VerificationCheck, VerificationReport, _capture


def run_analytic_verification() -> VerificationReport:
    """Run the installed analytic verification suite."""

    checks: list[VerificationCheck] = []

    def cyclic_check() -> None:
        coefficients = np.arange(15, dtype=float)
        result = cyclic_projection_certificate(coefficients, 4)
        result.certificate.require()
        assert np.array_equal(sum(result.value), coefficients)

    checks.append(_capture("cyclic coefficient projection", cyclic_check))

    def section_check() -> None:
        section_certificate(5, 0.4 + 0.2j, family="signed").certificate.require()
        assert abs(denominator(2, 0.4) - cos(0.4)) < 1e-13
        assert abs(denominator(5, 0.3) - denominator_root_sum(5, 0.3)) < 1e-13

    checks.append(_capture("signed cyclic sections", section_check))

    def reciprocal_check() -> None:
        values = reciprocal_coefficients(2, 1, 4)
        assert values == (Fraction(1), Fraction(1), Fraction(5), Fraction(61), Fraction(1385))
        reciprocal_semigroup_certificate(3, 2, 3, 5).certificate.require()

    checks.append(_capture("reciprocal coefficient algebra", reciprocal_check))

    def congruential_check() -> None:
        values = congruential_inverse(3, 1, 6)
        for n in range(7):
            total = sum(comb(3 * n + 1, 3 * q) * values[q] for q in range(n + 1))
            assert total == (1 if n == 0 else 0)

    checks.append(_capture("congruential triangular inverse", congruential_check))

    def appell_check() -> None:
        x = 0.37
        degree = 7
        derivative = degree * appell_polynomial(3, 1, 2, degree - 1, x)
        h = 1e-6
        numerical = (
            appell_polynomial(3, 1, 2, degree, x + h)
            - appell_polynomial(3, 1, 2, degree, x - h)
        ) / (2 * h)
        assert abs(complex(numerical - derivative)) < 1e-6

    checks.append(_capture("congruential Appell derivative", appell_check))

    def theta_check() -> None:
        for n in range(1, 5):
            assert theta_polynomial(4, n, 0) == 0
            assert theta_polynomial(4, n, -1) == 0

    checks.append(_capture("theta polynomial factors", theta_check))

    def zero_check() -> None:
        result = positive_zero_string(3, 4)
        result.certificate.require()
        assert max(result.residuals) < 1e-10
        exact = CyclicDenominator(4).positive_zeros(3)
        exact.certificate.require()
        expected = tuple(np.sqrt(2) * np.pi * (index + 0.5) for index in range(3))
        assert max(abs(left - right) for left, right in zip(exact.zeros, expected, strict=True)) < 1e-10

    checks.append(_capture("cyclic zero strings", zero_check))

    def coefficient_normalization_check() -> None:
        values = reciprocal_coefficients(2, 1, 5)
        ordinary = [complex(values[n]) / factorial(2 * n) for n in range(6)]
        z = 0.2
        truncated = sum(ordinary[n] * z ** (2 * n) for n in range(6))
        assert abs(truncated - 1 / cos(z)) < 1e-10

    checks.append(_capture("reciprocal analytic normalization", coefficient_normalization_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_analytic_verification"]
