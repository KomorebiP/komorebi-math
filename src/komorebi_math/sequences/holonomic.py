"""Exact polynomial coefficient recurrences for holonomic sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Rational
from typing import Iterable

from .._number_theory import require_integer
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from .base import Sequence

Polynomial = tuple[Fraction, ...]


def _polynomial(values: Iterable[Rational]) -> Polynomial:
    coefficients: list[Fraction] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Rational):
            raise ValidationError("polynomial recurrence coefficients must be exact rational numbers")
        coefficients.append(Fraction(value))
    if not coefficients:
        coefficients.append(Fraction(0))
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return tuple(coefficients)


def evaluate_polynomial(coefficients: Iterable[Rational], value: int) -> Fraction:
    """Evaluate an ascending coefficient rational polynomial at an integer."""

    x = require_integer(value, name="value")
    polynomial = _polynomial(coefficients)
    result = Fraction(0)
    for coefficient in reversed(polynomial):
        result = result * x + coefficient
    return result


@dataclass(frozen=True, slots=True)
class HolonomicRecurrence:
    """An exact recurrence ``sum_j P_j(n) a[n+j] = 0``.

    The final polynomial is the leading coefficient used to solve for the next
    term. Its value must be nonzero at every executed index.
    """

    polynomials: tuple[Polynomial, ...]
    initial: tuple[Fraction, ...]
    name: str = "holonomic recurrence"
    _cache: list[Fraction] = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        polynomials: Iterable[Iterable[Rational]],
        initial: Iterable[Rational],
        *,
        name: str = "holonomic recurrence",
    ) -> None:
        equations = tuple(_polynomial(poly) for poly in polynomials)
        if len(equations) < 2:
            raise ValidationError("a holonomic recurrence requires at least two shift coefficients")
        raw_initial = tuple(initial)
        if any(isinstance(value, bool) or not isinstance(value, Rational) for value in raw_initial):
            raise ValidationError("holonomic initial values must be exact rational numbers")
        seeds = tuple(Fraction(value) for value in raw_initial)
        if len(seeds) != len(equations) - 1:
            raise DimensionError("initial value count must equal the recurrence order")
        if all(coefficient == 0 for coefficient in equations[-1]):
            raise ValidationError("the leading recurrence polynomial cannot vanish identically")
        object.__setattr__(self, "polynomials", equations)
        object.__setattr__(self, "initial", seeds)
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "_cache", list(seeds))

    @property
    def order(self) -> int:
        """Return the number of initial values required."""

        return len(self.polynomials) - 1

    def coefficient(self, shift: int, index: int) -> Fraction:
        """Evaluate one shift coefficient polynomial."""

        j = require_integer(shift, name="shift")
        if j < 0 or j > self.order:
            raise ValidationError("shift must lie between zero and the recurrence order")
        return evaluate_polynomial(self.polynomials[j], index)

    def term(self, index: int) -> Fraction:
        """Return one exact sequence term."""

        n = require_integer(index, name="index")
        if n < 0:
            raise ValidationError("index must be nonnegative")
        cache: list[Fraction] = self._cache  # type: ignore[attr-defined]
        while len(cache) <= n:
            start = len(cache) - self.order
            leading = self.coefficient(self.order, start)
            if leading == 0:
                raise ValidationError(f"leading recurrence coefficient vanishes at index {start}")
            numerator = sum(self.coefficient(shift, start) * cache[start + shift] for shift in range(self.order))
            cache.append(-numerator / leading)
        return cache[n]

    def first(self, count: int) -> tuple[Fraction, ...]:
        """Return a finite exact prefix."""

        total = require_integer(count, name="count")
        if total < 0:
            raise ValidationError("count must be nonnegative")
        return tuple(self.term(index) for index in range(total))

    def to_sequence(self) -> Sequence:
        """Return the common lazy sequence wrapper."""

        return Sequence(
            self.term,
            name=self.name,
            exact=True,
            metadata={"holonomic_recurrence": self, "holonomic_certified": True},
        )

    def certificate(self, count: int = 32) -> ComputationResult[tuple[Fraction, ...]]:
        """Validate the polynomial recurrence on a finite prefix."""

        total = require_integer(count, name="count")
        if total < self.order + 1:
            raise ValidationError("count must exceed the recurrence order")
        values = self.first(total)
        valid = True
        for n in range(total - self.order):
            residual = sum(self.coefficient(shift, n) * values[n + shift] for shift in range(self.order + 1))
            valid = valid and residual == 0
        return ComputationResult(
            values,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT,
                assumptions_checked=("exact rational polynomial recurrence", "nonvanishing executed leading coefficients"),
                diagnostics={"order": self.order, "count": total},
                theorem_id="PSC-SEQ-HOLONOMIC",
                message="the exact prefix satisfies the declared polynomial coefficient recurrence",
            ),
        )


def motzkin_recurrence() -> HolonomicRecurrence:
    """Return a polynomial coefficient recurrence for Motzkin numbers."""

    return HolonomicRecurrence(
        ((-3, -3), (-5, -2), (4, 1)),
        (1, 1),
        name="Motzkin",
    )


def apery_recurrence() -> HolonomicRecurrence:
    """Return the standard polynomial coefficient recurrence for Apéry numbers."""

    # (n+1)^3 A_n - Q(n+1) A_(n+1) + (n+2)^3 A_(n+2) = 0
    # Q(x) = 34 x^3 + 51 x^2 + 27 x + 5
    return HolonomicRecurrence(
        ((1, 3, 3, 1), (-117, -231, -153, -34), (8, 12, 6, 1)),
        (1, 5),
        name="Apery",
    )


__all__ = [
    "HolonomicRecurrence",
    "Polynomial",
    "apery_recurrence",
    "evaluate_polynomial",
    "motzkin_recurrence",
]
