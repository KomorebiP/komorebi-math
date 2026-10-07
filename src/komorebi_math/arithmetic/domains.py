"""Finite modular and truncated local arithmetic domains."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from numbers import Integral

from .._number_theory import require_prime
from ..exceptions import ValidationError


def _integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class ModularRing:
    """Exact arithmetic in ``Z / modulus Z``."""

    modulus: int

    def __post_init__(self) -> None:
        modulus = _integer(self.modulus, name="modulus")
        if modulus < 2:
            raise ValidationError("modulus must be at least two")
        object.__setattr__(self, "modulus", modulus)

    def normalize(self, value: int) -> int:
        """Return the canonical residue of ``value``."""

        return _integer(value, name="value") % self.modulus

    def add(self, left: int, right: int) -> int:
        """Add two residues."""

        return (self.normalize(left) + self.normalize(right)) % self.modulus

    def subtract(self, left: int, right: int) -> int:
        """Subtract two residues."""

        return (self.normalize(left) - self.normalize(right)) % self.modulus

    def multiply(self, left: int, right: int) -> int:
        """Multiply two residues."""

        return (self.normalize(left) * self.normalize(right)) % self.modulus

    def power(self, base: int, exponent: int) -> int:
        """Raise a residue to a nonnegative integer power."""

        power = _integer(exponent, name="exponent")
        if power < 0:
            raise ValidationError("exponent must be nonnegative")
        return pow(self.normalize(base), power, self.modulus)

    def is_unit(self, value: int) -> bool:
        """Return whether ``value`` is invertible in the ring."""

        return gcd(self.normalize(value), self.modulus) == 1

    def inverse(self, value: int) -> int:
        """Return the modular inverse of a unit."""

        residue = self.normalize(value)
        if not self.is_unit(residue):
            raise ValidationError("value is not a unit modulo the declared modulus")
        return pow(residue, -1, self.modulus)


@dataclass(frozen=True, slots=True)
class PrimeField(ModularRing):
    """The prime field ``F_p`` represented by canonical integer residues."""

    def __post_init__(self) -> None:
        prime = require_prime(self.modulus)
        object.__setattr__(self, "modulus", prime)

    @property
    def prime(self) -> int:
        """Return the field characteristic."""

        return self.modulus


@dataclass(frozen=True, slots=True)
class TruncatedPadic:
    """Arithmetic modulo ``prime**precision`` with explicit local precision."""

    prime: int
    precision: int

    def __post_init__(self) -> None:
        prime = require_prime(self.prime)
        precision = _integer(self.precision, name="precision")
        if precision < 1:
            raise ValidationError("precision must be positive")
        object.__setattr__(self, "prime", prime)
        object.__setattr__(self, "precision", precision)

    @property
    def modulus(self) -> int:
        """Return ``prime**precision``."""

        return self.prime**self.precision

    @property
    def ring(self) -> ModularRing:
        """Return the corresponding modular ring."""

        return ModularRing(self.modulus)

    def normalize(self, value: int) -> int:
        """Return the canonical truncated local residue."""

        return self.ring.normalize(value)

    def is_unit(self, value: int) -> bool:
        """Return whether the residue is a local unit."""

        return self.normalize(value) % self.prime != 0

    def inverse(self, value: int) -> int:
        """Return the inverse of a local unit at the declared precision."""

        return self.ring.inverse(value)

    def valuation(self, value: int) -> int | float:
        """Return the visible valuation, capped by the declared precision.

        The zero residue modulo ``prime**precision`` represents every lift with
        valuation at least ``precision``.  A truncated domain cannot distinguish
        exact zero from a nonzero lift divisible by the full modulus, so the
        visible valuation is ``precision`` rather than infinity.
        """

        residue = self.normalize(value)
        if residue == 0:
            return self.precision
        valuation = 0
        while residue % self.prime == 0:
            residue //= self.prime
            valuation += 1
        return valuation

    def refine(self, additional_digits: int = 1) -> "TruncatedPadic":
        """Return the same local domain at greater precision."""

        extra = _integer(additional_digits, name="additional_digits")
        if extra < 0:
            raise ValidationError("additional_digits must be nonnegative")
        return TruncatedPadic(self.prime, self.precision + extra)

    def coarsen(self, precision: int) -> "TruncatedPadic":
        """Return a shallower local precision."""

        target = _integer(precision, name="precision")
        if target < 1 or target > self.precision:
            raise ValidationError("coarse precision must lie between one and the current precision")
        return TruncatedPadic(self.prime, target)


__all__ = ["ModularRing", "PrimeField", "TruncatedPadic"]
