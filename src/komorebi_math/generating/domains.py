"""Exact coefficient domains for normalized generating functions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..exceptions import ValidationError

Element = TypeVar("Element")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    return value


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor <= isqrt(value):
        if value % divisor == 0:
            return False
        divisor += 2
    return True


@runtime_checkable
class CoefficientDomain(Protocol[Element]):
    """Protocol for exact coefficient arithmetic and safe serialization."""

    name: str
    zero: Element
    one: Element

    def coerce(self, value: Any) -> Element: ...

    def add(self, left: Element, right: Element) -> Element: ...

    def subtract(self, left: Element, right: Element) -> Element: ...

    def multiply(self, left: Element, right: Element) -> Element: ...

    def inverse(self, value: Element) -> Element: ...

    def is_unit(self, value: Element) -> bool: ...

    def equal(self, left: Element, right: Element) -> bool: ...

    def encode(self, value: Element) -> Any: ...

    def decode(self, payload: Any) -> Element: ...

    def descriptor(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class IntegerDomain:
    """The exact integer coefficient domain."""

    name: str = "integers"
    zero: int = 0
    one: int = 1

    def coerce(self, value: Any) -> int:
        fraction = Fraction(value)
        if fraction.denominator != 1:
            raise ValidationError("value is not integral")
        return fraction.numerator

    def add(self, left: int, right: int) -> int:
        return self.coerce(left) + self.coerce(right)

    def subtract(self, left: int, right: int) -> int:
        return self.coerce(left) - self.coerce(right)

    def multiply(self, left: int, right: int) -> int:
        return self.coerce(left) * self.coerce(right)

    def is_unit(self, value: int) -> bool:
        return abs(self.coerce(value)) == 1

    def inverse(self, value: int) -> int:
        normalized = self.coerce(value)
        if not self.is_unit(normalized):
            raise ValidationError("only 1 and -1 are units in the integer domain")
        return normalized

    def equal(self, left: int, right: int) -> bool:
        return self.coerce(left) == self.coerce(right)

    def encode(self, value: int) -> int:
        return self.coerce(value)

    def decode(self, payload: Any) -> int:
        return self.coerce(payload)

    def descriptor(self) -> dict[str, Any]:
        return {"kind": "integers"}


@dataclass(frozen=True, slots=True)
class RationalDomain:
    """The exact rational coefficient field."""

    name: str = "rationals"
    zero: Fraction = Fraction(0)
    one: Fraction = Fraction(1)

    def coerce(self, value: Any) -> Fraction:
        try:
            return Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValidationError("value is not rational") from exc

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        return self.coerce(left) + self.coerce(right)

    def subtract(self, left: Fraction, right: Fraction) -> Fraction:
        return self.coerce(left) - self.coerce(right)

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        return self.coerce(left) * self.coerce(right)

    def is_unit(self, value: Fraction) -> bool:
        return self.coerce(value) != 0

    def inverse(self, value: Fraction) -> Fraction:
        normalized = self.coerce(value)
        if normalized == 0:
            raise ValidationError("zero is not invertible")
        return 1 / normalized

    def equal(self, left: Fraction, right: Fraction) -> bool:
        return self.coerce(left) == self.coerce(right)

    def encode(self, value: Fraction) -> dict[str, int]:
        fraction = self.coerce(value)
        return {"numerator": fraction.numerator, "denominator": fraction.denominator}

    def decode(self, payload: Any) -> Fraction:
        if not isinstance(payload, dict) or set(payload) != {"numerator", "denominator"}:
            raise ValidationError("invalid rational payload")
        numerator = _integer(payload["numerator"], "numerator")
        denominator = _integer(payload["denominator"], "denominator")
        if denominator == 0:
            raise ValidationError("rational denominator must be nonzero")
        return Fraction(numerator, denominator)

    def descriptor(self) -> dict[str, Any]:
        return {"kind": "rationals"}


@dataclass(frozen=True, slots=True)
class ResidueRingDomain:
    """Exact arithmetic in the finite ring ``Z / modulus Z``."""

    modulus: int
    name: str = "residue_ring"

    def __post_init__(self) -> None:
        modulus = _integer(self.modulus, "modulus")
        if modulus < 2:
            raise ValidationError("modulus must be at least two")
        object.__setattr__(self, "modulus", modulus)
        object.__setattr__(self, "name", f"Z/{modulus}Z")

    @property
    def zero(self) -> int:
        return 0

    @property
    def one(self) -> int:
        return 1

    def coerce(self, value: Any) -> int:
        fraction = Fraction(value)
        numerator = fraction.numerator % self.modulus
        denominator = fraction.denominator % self.modulus
        if gcd(denominator, self.modulus) != 1:
            raise ValidationError(
                "the coefficient denominator is not invertible in the residue ring"
            )
        return numerator * pow(denominator, -1, self.modulus) % self.modulus

    def add(self, left: int, right: int) -> int:
        return (self.coerce(left) + self.coerce(right)) % self.modulus

    def subtract(self, left: int, right: int) -> int:
        return (self.coerce(left) - self.coerce(right)) % self.modulus

    def multiply(self, left: int, right: int) -> int:
        return self.coerce(left) * self.coerce(right) % self.modulus

    def is_unit(self, value: int) -> bool:
        return gcd(self.coerce(value), self.modulus) == 1

    def inverse(self, value: int) -> int:
        normalized = self.coerce(value)
        if not self.is_unit(normalized):
            raise ValidationError("value is not a unit in the residue ring")
        return pow(normalized, -1, self.modulus)

    def equal(self, left: int, right: int) -> bool:
        return self.coerce(left) == self.coerce(right)

    def encode(self, value: int) -> int:
        return self.coerce(value)

    def decode(self, payload: Any) -> int:
        return self.coerce(_integer(payload, "residue"))

    def descriptor(self) -> dict[str, Any]:
        return {"kind": "residue_ring", "modulus": self.modulus}


@dataclass(frozen=True, slots=True)
class PrimeFieldDomain(ResidueRingDomain):
    """The prime field ``F_p`` represented by canonical integer residues."""

    name: str = "prime_field"

    def __post_init__(self) -> None:
        if not _is_prime(_integer(self.modulus, "prime")):
            raise ValidationError("prime must be prime")
        object.__setattr__(self, "modulus", int(self.modulus))
        object.__setattr__(self, "name", f"F_{self.modulus}")

    @property
    def prime(self) -> int:
        return self.modulus

    def descriptor(self) -> dict[str, Any]:
        return {"kind": "prime_field", "prime": self.prime}


@dataclass(frozen=True, slots=True)
class TruncatedPadicDomain(ResidueRingDomain):
    """The finite ring ``Z / p^r Z`` with declared prime and precision."""

    prime: int = 2
    precision: int = 1
    modulus: int = 2
    name: str = "truncated_padic"

    def __init__(self, prime: int, precision: int) -> None:
        p = _integer(prime, "prime")
        r = _integer(precision, "precision")
        if not _is_prime(p):
            raise ValidationError("prime must be prime")
        if r < 1:
            raise ValidationError("precision must be positive")
        object.__setattr__(self, "prime", p)
        object.__setattr__(self, "precision", r)
        object.__setattr__(self, "modulus", p**r)
        object.__setattr__(self, "name", f"Z/{p}^{r}Z")

    def valuation(self, value: int) -> int | float:
        """Return the visible valuation capped by this domain's precision.

        The zero residue class records only divisibility by ``prime**precision``;
        it does not distinguish exact zero from a nonzero lift of larger
        valuation.
        """

        residue = self.coerce(value)
        if residue == 0:
            return self.precision
        count = 0
        while residue % self.prime == 0:
            residue //= self.prime
            count += 1
        return count

    def descriptor(self) -> dict[str, Any]:
        return {
            "kind": "truncated_padic",
            "prime": self.prime,
            "precision": self.precision,
        }


def domain_from_descriptor(payload: Any) -> CoefficientDomain[Any]:
    """Construct one supported exact domain from a data-only descriptor."""

    if not isinstance(payload, dict):
        raise ValidationError("coefficient-domain descriptor must be a mapping")
    kind = payload.get("kind")
    if kind == "integers" and set(payload) == {"kind"}:
        return IntegerDomain()
    if kind == "rationals" and set(payload) == {"kind"}:
        return RationalDomain()
    if kind == "residue_ring" and set(payload) == {"kind", "modulus"}:
        return ResidueRingDomain(_integer(payload["modulus"], "modulus"))
    if kind == "prime_field" and set(payload) == {"kind", "prime"}:
        return PrimeFieldDomain(_integer(payload["prime"], "prime"))
    if kind == "truncated_padic" and set(payload) == {"kind", "prime", "precision"}:
        return TruncatedPadicDomain(
            _integer(payload["prime"], "prime"),
            _integer(payload["precision"], "precision"),
        )
    raise ValidationError("unsupported coefficient-domain descriptor")


INTEGER_DOMAIN = IntegerDomain()
RATIONAL_DOMAIN = RationalDomain()


__all__ = [
    "CoefficientDomain",
    "INTEGER_DOMAIN",
    "IntegerDomain",
    "PrimeFieldDomain",
    "RATIONAL_DOMAIN",
    "RationalDomain",
    "ResidueRingDomain",
    "TruncatedPadicDomain",
    "domain_from_descriptor",
]
