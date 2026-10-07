"""Installed verification for local arithmetic packets."""

from __future__ import annotations

from fractions import Fraction
from math import comb

from ..arithmetic import (
    EulerLocalCoordinates,
    HurwitzSeries,
    MahlerSeries,
    StandardWindow,
    affine_kernel_composition_certificate,
    fibonacci_recurrence,
    prime_power_kernel_certificate,
    simultaneous_unit_difference,
)
from .core import VerificationCheck, VerificationReport, _capture


def run_arithmetic_verification() -> VerificationReport:
    """Run the installed local arithmetic verification suite."""

    checks: list[VerificationCheck] = []

    def hurwitz_check() -> None:
        left = HurwitzSeries.exponential_mode(2, 8)
        right = HurwitzSeries.exponential_mode(5, 8)
        assert left.multiply(right, degree=8).coefficients == HurwitzSeries.exponential_mode(7, 8).coefficients
        source = HurwitzSeries((1, 2, 3, 4))
        product = source.multiply(source.inverse(8), degree=8)
        assert product.coefficients == (Fraction(1),) + (Fraction(0),) * 8

    checks.append(_capture("Hurwitz mode and inverse algebra", hurwitz_check))

    def window_check() -> None:
        window = StandardWindow(7, 2)
        window.certificate(5, periods=5).certificate.require()
        assert window.period == 3
        assert window.rational_generating_function().first(20) == window.reduced_inverse(19)

    checks.append(_capture("uniform standard window period", window_check))

    def euler_check() -> None:
        for prime in (2, 3, 5, 7):
            coordinates = EulerLocalCoordinates(prime)
            coordinates.certificate(18).certificate.require()
            coordinates.truncate(18, 2).certificate.require()

    checks.append(_capture("odd and dyadic Euler coordinates", euler_check))

    def mahler_check() -> None:
        values = [n**4 + 2 * n + 3 for n in range(8)]
        series = MahlerSeries.from_values(values)
        series.reconstruction_certificate().certificate.require()
        assert series.first(12) == tuple(Fraction(n**4 + 2 * n + 3) for n in range(12))
        transformed = series.affine_subsequence(3, 2)
        assert transformed.first(8) == tuple(series.evaluate(3 * n + 2) for n in range(8))
        affine_kernel_composition_certificate(3, 2, 4, 1, 10).require()
        prime_power_kernel_certificate(3, 2, 4, 10).require()

    checks.append(_capture("Mahler analysis and affine polyphase", mahler_check))

    def recurrence_check() -> None:
        recurrence = fibonacci_recurrence()
        recurrence.certificate(24).certificate.require()
        packet = recurrence.residue_packet(5, 3)
        packet.certificate(12).certificate.require()
        period = recurrence.period_mod(16)
        period.certificate.require()
        assert period.preperiod == 0
        assert recurrence.term_mod(10_000, 97) == recurrence.term(10_000) % 97
        trace = 11
        determinant = -1
        values = packet.first(10)
        for index in range(8):
            assert values[index + 2] == trace * values[index + 1] - determinant * values[index]

    checks.append(_capture("recurrence packets and modular periods", recurrence_check))

    def coefficient_recurrence_check() -> None:
        window = StandardWindow(11, 5)
        values = window.reduced_inverse(12)
        for n in range(13):
            total = sum(comb(5 * n, 5 * q) * values[q] for q in range(n + 1)) % 11
            assert total == (1 if n == 0 else 0)

    checks.append(_capture("compressed inverse recurrence", coefficient_recurrence_check))

    def local_gain_check() -> None:
        result = simultaneous_unit_difference(5, {2: 4, 3: 3}, index=7, order=3)
        result.certificate.require()
        assert result.value[2] >= 12
        assert result.value[3] >= 9

    checks.append(_capture("simultaneous local unit gains", local_gain_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_arithmetic_verification"]
