"""Exact guards for the exact rational values of normalized binary floats.

The checks certify scalar inequalities only. Actual norm domination and Banach
completeness remain external assumptions. Work budgets are rejection guards,
not proven time/space complexity bounds.
"""
from __future__ import annotations

from fractions import Fraction
from math import inf, isfinite, nextafter
from typing import Mapping
from .exceptions import ValidationError

MAX_GUARD_WORK = 1_000_000


def upper_float(value: Fraction) -> float:
    """Round a nonnegative rational upward, including under/overflow edges."""
    if value < 0:
        raise ValidationError("upper bound must be nonnegative")
    try:
        rounded = float(value)
    except OverflowError:
        return inf
    if not isfinite(rounded):
        return inf
    return nextafter(rounded, inf) if Fraction.from_float(rounded) < value else rounded


def guard_values(a: float, bounds: Mapping[int, float], radius: float) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    aq, rq = Fraction.from_float(a), Fraction.from_float(radius)
    bq = {k: Fraction.from_float(v) for k, v in bounds.items()}
    # Conservative proxy for the integer-power work, checked before powering.
    rbits = rq.numerator.bit_length() + rq.denominator.bit_length()
    work = sum(k * rbits + q.numerator.bit_length() + q.denominator.bit_length()
               for k, q in bq.items() if q)
    if work > MAX_GUARD_WORK:
        raise ValidationError("exact majorant guard work budget exceeded")
    image = aq + sum((q * rq**k for k, q in bq.items() if q), Fraction(0))
    contraction = sum((k * q * rq**(k-1) for k, q in bq.items() if q), Fraction(0))
    return aq, rq, image, contraction


def tail_upper(a: Fraction, contraction: Fraction, depth: int) -> float:
    if contraction < 0 or contraction >= 1:
        raise ValidationError("tail bound requires an exact contraction strictly below one")
    if a == 0:
        return 0.0
    if contraction == 0:
        return 0.0 if depth >= 1 else upper_float(a)
    if depth * (contraction.numerator.bit_length() + contraction.denominator.bit_length()) > MAX_GUARD_WORK:
        raise ValidationError("exact Picard tail work budget exceeded")
    return upper_float(contraction**depth * a / (1-contraction))


def rational_record(value: Fraction) -> str | dict[str, str]:
    """Keep exact diagnostics without decimal integer-conversion size failures."""
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) <= 12000:
        return str(value)
    return {"format": "hex-rational", "numerator": hex(value.numerator),
            "denominator": hex(value.denominator)}
