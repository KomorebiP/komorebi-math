"""Hurwitz series, exponential modes, and finite field subgroup inversion."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from numbers import Integral, Number
from typing import Iterable

from .._number_theory import require_integer, require_prime
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .domains import ModularRing

ExactNumber = int | Fraction


def _validated_modulus(modulus: int | None) -> int | None:
    if modulus is None:
        return None
    value = require_integer(modulus, name="modulus")
    if value < 2:
        raise ValidationError("modulus must be at least two")
    return value


def _normalize(value: Number, modulus: int | None) -> Number:
    if modulus is None:
        if isinstance(value, bool) or not isinstance(value, Number):
            raise ValidationError("Hurwitz coefficients must be numeric")
        if isinstance(value, Integral):
            return Fraction(int(value))
        if isinstance(value, Fraction):
            return Fraction(int(value.numerator), int(value.denominator))
        return value
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError("modular Hurwitz coefficients must be integers")
    return int(value) % modulus


def hurwitz_product(
    left: Iterable[Number],
    right: Iterable[Number],
    *,
    degree: int | None = None,
    modulus: int | None = None,
) -> tuple[Number, ...]:
    """Return a finite Hurwitz product through the requested degree."""

    modulus = _validated_modulus(modulus)
    a = tuple(_normalize(value, modulus) for value in left)
    b = tuple(_normalize(value, modulus) for value in right)
    if degree is None:
        nmax = len(a) + len(b) - 2
    else:
        nmax = require_integer(degree, name="degree")
        if nmax < 0:
            raise ValidationError("degree must be nonnegative")
    output: list[Number] = []
    for n in range(nmax + 1):
        total: Number = 0
        for r in range(n + 1):
            if r < len(a) and n - r < len(b):
                total += comb(n, r) * a[r] * b[n - r]
        output.append(_normalize(total, modulus))
    return tuple(output)


def hurwitz_inverse(
    coefficients: Iterable[Number],
    degree: int,
    *,
    modulus: int | None = None,
) -> tuple[Number, ...]:
    """Return the finite Hurwitz inverse through ``degree``."""

    modulus = _validated_modulus(modulus)
    nmax = require_integer(degree, name="degree")
    if nmax < 0:
        raise ValidationError("degree must be nonnegative")
    values = tuple(_normalize(value, modulus) for value in coefficients)
    if not values:
        raise ValidationError("at least one coefficient is required")
    a0 = values[0]
    if modulus is None:
        if a0 == 0:
            raise ValidationError("the constant coefficient must be nonzero")
        inverse_a0: Number = Fraction(1, 1) / Fraction(a0) if isinstance(a0, (int, Fraction)) else 1 / a0
    else:
        ring = ModularRing(modulus)
        inverse_a0 = ring.inverse(int(a0))
    result: list[Number] = [inverse_a0]
    for n in range(1, nmax + 1):
        total: Number = 0
        for r in range(1, n + 1):
            if r < len(values):
                total += comb(n, r) * values[r] * result[n - r]
        value = -inverse_a0 * total
        result.append(_normalize(value, modulus))
    return tuple(result)


@documented(
    ProvenanceRecord(
        identifier="PSC-IX-D02",
        title="Finite Hurwitz series",
        status=ClaimStatus.DEFINITION,
        source="Paper IX",
        assumptions=("finite coefficient truncation",),
        verification_suite="arithmetic.hurwitz",
    )
)
@dataclass(frozen=True, slots=True)
class HurwitzSeries:
    """A finite exact or modular Hurwitz coefficient system."""

    coefficients: tuple[Number, ...]
    modulus: int | None = None

    def __init__(self, coefficients: Iterable[Number], modulus: int | None = None) -> None:
        modulus = _validated_modulus(modulus)
        values = tuple(_normalize(value, modulus) for value in coefficients)
        if not values:
            raise ValidationError("HurwitzSeries requires at least one coefficient")
        object.__setattr__(self, "coefficients", values)
        object.__setattr__(self, "modulus", modulus)

    @property
    def degree(self) -> int:
        """Return the highest stored coefficient index."""

        return len(self.coefficients) - 1

    def coefficient(self, index: int) -> Number:
        """Return one coefficient, using zero beyond the stored truncation."""

        n = require_integer(index, name="index")
        if n < 0:
            raise ValidationError("index must be nonnegative")
        return self.coefficients[n] if n < len(self.coefficients) else 0

    def truncate(self, degree: int) -> "HurwitzSeries":
        """Return the series through the requested degree."""

        nmax = require_integer(degree, name="degree")
        if nmax < 0:
            raise ValidationError("degree must be nonnegative")
        return HurwitzSeries((self.coefficient(n) for n in range(nmax + 1)), self.modulus)

    def multiply(self, other: "HurwitzSeries", *, degree: int | None = None) -> "HurwitzSeries":
        """Multiply two Hurwitz series."""

        if self.modulus != other.modulus:
            raise ValidationError("Hurwitz series must use the same coefficient modulus")
        return HurwitzSeries(
            hurwitz_product(self.coefficients, other.coefficients, degree=degree, modulus=self.modulus),
            self.modulus,
        )

    def inverse(self, degree: int | None = None) -> "HurwitzSeries":
        """Return the finite reciprocal series."""

        nmax = self.degree if degree is None else require_integer(degree, name="degree")
        return HurwitzSeries(hurwitz_inverse(self.coefficients, nmax, modulus=self.modulus), self.modulus)

    def evaluate_egf(self, value: Number) -> Number:
        """Evaluate the stored exponential generating polynomial."""

        total: Number = 0
        for index, coefficient in enumerate(self.coefficients):
            total += coefficient * value**index / factorial(index)
        return total

    def __mul__(self, other: "HurwitzSeries") -> "HurwitzSeries":
        return self.multiply(other)

    @classmethod
    def identity(cls, degree: int, *, modulus: int | None = None) -> "HurwitzSeries":
        """Return the Hurwitz multiplicative identity through ``degree``."""

        nmax = require_integer(degree, name="degree")
        if nmax < 0:
            raise ValidationError("degree must be nonnegative")
        return cls((1, *(0 for _ in range(nmax))), modulus)

    @classmethod
    def exponential_mode(cls, value: int, degree: int, *, modulus: int | None = None) -> "HurwitzSeries":
        """Return the mode ``(value**n)`` through ``degree``."""

        modulus = _validated_modulus(modulus)
        nmax = require_integer(degree, name="degree")
        if nmax < 0:
            raise ValidationError("degree must be nonnegative")
        base = require_integer(value, name="value")
        if modulus is None:
            coefficients = tuple(base**n for n in range(nmax + 1))
        else:
            coefficients = tuple(pow(base % modulus, n, modulus) for n in range(nmax + 1))
        return cls(coefficients, modulus)


def subgroup_roots(prime: int, order: int) -> tuple[int, ...]:
    """Return all ``order`` roots of unity in the prime field."""

    p = require_prime(prime)
    m = require_integer(order, name="order")
    if m < 1 or (p - 1) % m:
        raise ValidationError("order must be a positive divisor of prime - 1")
    roots = tuple(value for value in range(1, p) if pow(value, m, p) == 1)
    if len(roots) != m:
        raise RuntimeError("finite field subgroup enumeration failed")
    return roots


@documented(
    ProvenanceRecord(
        identifier="PSC-IX-T03",
        title="Finite field subgroup selector",
        status=ClaimStatus.PROVED,
        source="Paper IX",
        assumptions=("prime field", "subgroup order divides prime minus one"),
        verification_suite="arithmetic.hurwitz",
    )
)
def subgroup_selector(prime: int, order: int, degree: int) -> HurwitzSeries:
    """Return the finite field selector for coefficient degrees divisible by ``order``."""

    p = require_prime(prime)
    m = require_integer(order, name="order")
    nmax = require_integer(degree, name="degree")
    if nmax < 0:
        raise ValidationError("degree must be nonnegative")
    roots = subgroup_roots(p, m)
    inverse_m = pow(m, -1, p)
    coefficients = []
    for n in range(nmax + 1):
        coefficients.append(inverse_m * sum(pow(root, n, p) for root in roots) % p)
    return HurwitzSeries(coefficients, p)


def compressed_hurwitz_inverse(prime: int, compression: int, count: int) -> tuple[int, ...]:
    """Return the compressed subgroup inverse coefficients modulo ``prime``."""

    p = require_prime(prime)
    m = require_integer(compression, name="compression")
    nmax = require_integer(count, name="count")
    if nmax < 0:
        raise ValidationError("count must be nonnegative")
    if m < 1 or (p - 1) % m:
        raise ValidationError("compression must divide prime - 1")
    values = [1]
    for n in range(1, nmax + 1):
        values.append(-sum(comb(m * n, m * q) * values[q] for q in range(n)) % p)
    return tuple(values)


def compressed_period(prime: int, compression: int) -> int:
    """Return the proved positive tail period ``(prime - 1) / compression``."""

    p = require_prime(prime)
    m = require_integer(compression, name="compression")
    if m < 1 or (p - 1) % m:
        raise ValidationError("compression must divide prime - 1")
    return (p - 1) // m


__all__ = [
    "HurwitzSeries",
    "compressed_hurwitz_inverse",
    "compressed_period",
    "hurwitz_inverse",
    "hurwitz_product",
    "subgroup_roots",
    "subgroup_selector",
]
