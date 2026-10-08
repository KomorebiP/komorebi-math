"""Validation and isolated precision contexts for the special-function layer."""

from __future__ import annotations

from fractions import Fraction
from operator import index
from typing import Any

from ..exceptions import BackendUnavailableError, ValidationError


def _boolean(value: Any) -> bool:
    # NumPy 1.26 accepts bool_ in operator.index while NumPy 2 rejects it.
    # Avoid importing any backend merely to keep the domain version independent.
    kind = type(value)
    return isinstance(value, bool) or (kind.__module__ == "numpy" and kind.__name__ == "bool_")


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if _boolean(value):
        raise ValidationError(f"{name} must be an integer at least {minimum}")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValidationError(f"{name} must be an integer at least {minimum}") from exc
    if result < minimum:
        raise ValidationError(f"{name} must be an integer at least {minimum}")
    return result


def rational(value: Any, name: str) -> Fraction:
    """Accept explicit exact rationals, including decimal or fraction strings."""
    if _boolean(value) or isinstance(value, (float, complex)):
        raise ValidationError(f"{name} must be an exact rational, not a float")
    try:
        if isinstance(value, (Fraction, str)):
            return Fraction(value)
        return Fraction(index(value))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValidationError(f"{name} must be an exact rational") from exc


def context(dps: int):
    """Create an mpmath context without changing the process-wide precision."""
    digits = integer(dps, "dps", minimum=15)
    try:
        import mpmath
    except ImportError as exc:
        raise BackendUnavailableError(
            "special-function numerical evaluation requires mpmath; "
            "install komorebi-math[scientific]"
        ) from exc
    result = mpmath.mp.clone()
    result.dps = digits
    return result


def real(ctx, value: Any, name: str, *, positive: bool = False):
    if _boolean(value):
        raise ValidationError(f"{name} must be a finite real number")
    try:
        result = (
            ctx.mpf(value.numerator) / value.denominator
            if isinstance(value, Fraction)
            else ctx.mpf(value)
        )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValidationError(f"{name} must be a finite real number") from exc
    if not ctx.isfinite(result) or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValidationError(f"{name} must be a {qualifier} real number")
    return result


def complex_number(ctx, value: Any, name: str):
    if _boolean(value):
        raise ValidationError(f"{name} must be finite")
    try:
        result = (
            ctx.mpc(ctx.mpf(value.numerator) / value.denominator)
            if isinstance(value, Fraction)
            else ctx.mpc(value)
        )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValidationError(f"{name} must be finite") from exc
    if not ctx.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result
