"""Exact finite-level counts for basic deformation charts."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from typing import Iterable

from .._number_theory import require_integer
from ..exceptions import ValidationError


def _positive(value: int, *, name: str) -> int:
    integer = require_integer(value, name=name)
    if integer < 1:
        raise ValidationError(f"{name} must be positive")
    return integer


def smooth_chart_count(q: int, dimension: int, depth: int) -> int:
    """Return the number of points above one residual root in a smooth chart."""

    size = _positive(q, name="q")
    d = require_integer(dimension, name="dimension")
    m = _positive(depth, name="depth")
    if d < 0:
        raise ValidationError("dimension must be nonnegative")
    return size ** (d * (m - 1))


def weighted_chart_count(q: int, weights: Iterable[int], depth: int) -> int:
    """Return the exact count for delayed visible coordinates."""

    size = _positive(q, name="q")
    m = _positive(depth, name="depth")
    parsed = tuple(_positive(weight, name="weight") for weight in weights)
    exponent = sum(max(m - weight, 0) for weight in parsed)
    return size**exponent


def nilpotent_chart_count(q: int, exponent: int, depth: int) -> int:
    """Count solutions to ``x**exponent = 0`` modulo one local depth."""

    size = _positive(q, name="q")
    e = _positive(exponent, name="exponent")
    m = _positive(depth, name="depth")
    return size ** (m - ceil(m / e))


def diagonal_nilpotent_count(q: int, exponents: Iterable[int], depth: int) -> int:
    """Count a diagonal family of nilpotent equations."""

    size = _positive(q, name="q")
    m = _positive(depth, name="depth")
    result = 1
    for exponent in exponents:
        result *= nilpotent_chart_count(size, exponent, m)
    return result


def normal_crossing_count(q: int, depth: int) -> int:
    """Count pairs satisfying ``x*y = 0`` modulo a discrete valuation depth."""

    size = _positive(q, name="q")
    m = _positive(depth, name="depth")
    return size**m + m * (size - 1) * size ** (m - 1)


@dataclass(frozen=True, slots=True)
class RationalCountingSeries:
    """A rational ordinary generating series with exact coefficients."""

    numerator: tuple[Fraction, ...]
    denominator: tuple[Fraction, ...]

    def coefficient(self, index: int) -> Fraction:
        """Return one coefficient by the denominator recurrence."""

        n = require_integer(index, name="index")
        if n < 0:
            raise ValidationError("series index must be nonnegative")
        if not self.denominator or self.denominator[0] == 0:
            raise ValidationError("denominator must have nonzero constant term")
        values: list[Fraction] = []
        for degree in range(n + 1):
            rhs = self.numerator[degree] if degree < len(self.numerator) else Fraction(0)
            correction = sum(
                self.denominator[j] * values[degree - j]
                for j in range(1, min(degree, len(self.denominator) - 1) + 1)
            )
            values.append((rhs - correction) / self.denominator[0])
        return values[n]

    def first(self, count: int) -> tuple[Fraction, ...]:
        """Return the first ``count`` coefficients."""

        total = require_integer(count, name="count")
        if total < 0:
            raise ValidationError("count must be nonnegative")
        return tuple(self.coefficient(index) for index in range(total))


def smooth_chart_series(q: int, dimension: int) -> RationalCountingSeries:
    """Return the level series ``sum_{m>=1} q^{d(m-1)} T^m``."""

    size = _positive(q, name="q")
    d = require_integer(dimension, name="dimension")
    if d < 0:
        raise ValidationError("dimension must be nonnegative")
    return RationalCountingSeries((Fraction(0), Fraction(1)), (Fraction(1), Fraction(-(size**d))))


def nilpotent_chart_series(q: int, exponent: int) -> RationalCountingSeries:
    """Return the rational level series for ``x**exponent = 0``."""

    size = _positive(q, name="q")
    e = _positive(exponent, name="exponent")
    numerator = [Fraction(0)] + [Fraction(size ** (r - 1)) for r in range(1, e + 1)]
    denominator = [Fraction(1)] + [Fraction(0)] * (e - 1) + [Fraction(-(size ** (e - 1)))]
    return RationalCountingSeries(tuple(numerator), tuple(denominator))


def normal_crossing_series(q: int) -> RationalCountingSeries:
    """Return the rational level series for the chart ``x*y = 0``."""

    size = _positive(q, name="q")
    numerator = (Fraction(0), Fraction(2 * size - 1), Fraction(-(size**2)))
    denominator = (Fraction(1), Fraction(-2 * size), Fraction(size**2))
    return RationalCountingSeries(numerator, denominator)


__all__ = [
    "RationalCountingSeries",
    "diagonal_nilpotent_count",
    "nilpotent_chart_count",
    "nilpotent_chart_series",
    "normal_crossing_count",
    "normal_crossing_series",
    "smooth_chart_count",
    "smooth_chart_series",
    "weighted_chart_count",
]
