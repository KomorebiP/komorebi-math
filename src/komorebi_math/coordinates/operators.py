"""Polynomial identities for lowering and recurrence defect operators."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError

from .exact import Scalar, Vector, as_vector
from .normalized import apply_lowering_polynomial
from .polynomials import polynomial_add, polynomial_multiply, polynomial_scale


def _trim(values: Sequence[Scalar]) -> Vector:
    data = list(as_vector(values))
    if not data:
        return (Fraction(0),)
    while len(data) > 1 and data[-1] == 0:
        data.pop()
    return tuple(data)


def _divmod_polynomial(dividend: Sequence[Scalar], divisor: Sequence[Scalar]) -> tuple[Vector, Vector]:
    numerator = list(_trim(dividend))
    denominator = _trim(divisor)
    if denominator == (0,):
        raise ValidationError("polynomial division by zero")
    if len(numerator) < len(denominator):
        return (Fraction(0),), tuple(numerator)
    quotient = [Fraction(0) for _ in range(len(numerator) - len(denominator) + 1)]
    while len(numerator) >= len(denominator) and any(numerator):
        shift = len(numerator) - len(denominator)
        factor = numerator[-1] / denominator[-1]
        quotient[shift] = factor
        for index, value in enumerate(denominator):
            numerator[index + shift] -= factor * value
        while len(numerator) > 1 and numerator[-1] == 0:
            numerator.pop()
    return _trim(quotient), _trim(numerator)


@dataclass(frozen=True, slots=True)
class ShiftPolynomial:
    r"""A polynomial ``q(L)`` in the canonical coefficient shift operator."""

    coefficients: Vector
    name: str = "shift polynomial"

    def __init__(self, coefficients: Iterable[Scalar], name: str = "shift polynomial") -> None:
        object.__setattr__(self, "coefficients", _trim(tuple(coefficients)))
        object.__setattr__(self, "name", name)

    @property
    def degree(self) -> int:
        """Return the polynomial degree."""

        return len(self.coefficients) - 1

    def apply(self, values: Sequence[Scalar]) -> Vector:
        """Apply the shift polynomial to a finite sequence prefix."""

        return apply_lowering_polynomial(values, self.coefficients)

    def multiply(self, other: "ShiftPolynomial", *, name: str | None = None) -> "ShiftPolynomial":
        """Compose two commuting shift polynomials."""

        return ShiftPolynomial(
            polynomial_multiply(self.coefficients, other.coefficients),
            name or f"{self.name} times {other.name}",
        )


@dataclass(frozen=True, slots=True)
class BezoutDefectIdentity:
    """A finite prefix reconstruction from two coprime defect streams."""

    gcd: Vector
    left_multiplier: Vector
    right_multiplier: Vector
    reconstructed: Vector
    expected: Vector
    valid: bool

    def certificate(self) -> Certificate:
        """Return the exact finite Bézout reconstruction certificate."""

        return Certificate(
            valid=self.valid and self.reconstructed == self.expected,
            exactness=Exactness.EXACT,
            assumptions_checked=("coprime shift polynomials", "finite sequence prefix"),
            diagnostics={"reconstructed_length": len(self.reconstructed)},
            theorem_id="CC.O2",
            message="coprime recurrence defects reconstruct the common interior",
        )


def polynomial_extended_gcd(
    left: Sequence[Scalar],
    right: Sequence[Scalar],
) -> tuple[Vector, Vector, Vector]:
    r"""Return monic ``g,u,v`` with ``u*left+v*right=g``."""

    old_r, r = _trim(left), _trim(right)
    if old_r == (Fraction(0),) and r == (Fraction(0),):
        raise ValidationError("at least one polynomial must be nonzero for a monic gcd")
    old_s, s = (Fraction(1),), (Fraction(0),)
    old_t, t = (Fraction(0),), (Fraction(1),)
    while r != (Fraction(0),):
        quotient, remainder = _divmod_polynomial(old_r, r)
        old_r, r = r, remainder
        old_s, s = s, _trim(polynomial_add(old_s, polynomial_scale(polynomial_multiply(quotient, s), -1)))
        old_t, t = t, _trim(polynomial_add(old_t, polynomial_scale(polynomial_multiply(quotient, t), -1)))
    leading = old_r[-1]
    return (
        _trim(polynomial_scale(old_r, Fraction(1, 1) / leading)),
        _trim(polynomial_scale(old_s, Fraction(1, 1) / leading)),
        _trim(polynomial_scale(old_t, Fraction(1, 1) / leading)),
    )


def bezout_defect_identity(
    values: Sequence[Scalar],
    left: Sequence[Scalar],
    right: Sequence[Scalar],
) -> BezoutDefectIdentity:
    r"""Reconstruct the common finite interior from two coprime defects."""

    data = as_vector(values)
    gcd, u, v = polynomial_extended_gcd(left, right)
    if gcd != (Fraction(1),):
        raise ValidationError("the two shift polynomials must be coprime")
    left_defect = apply_lowering_polynomial(data, left)
    right_defect = apply_lowering_polynomial(data, right)
    left_part = apply_lowering_polynomial(left_defect, u)
    right_part = apply_lowering_polynomial(right_defect, v)
    horizon = min(len(left_part), len(right_part))
    reconstructed = tuple(left_part[index] + right_part[index] for index in range(horizon))
    expected = data[:horizon]
    return BezoutDefectIdentity(gcd, u, v, reconstructed, expected, reconstructed == expected)


__all__ = [
    "BezoutDefectIdentity",
    "ShiftPolynomial",
    "bezout_defect_identity",
    "polynomial_extended_gcd",
]
