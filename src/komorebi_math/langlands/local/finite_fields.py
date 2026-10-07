"""Small exact finite fields for local packet computations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import prod
from typing import Iterable, Iterator

from ..._number_theory import require_integer, require_prime
from ...exceptions import ValidationError


def _trim(poly: list[int], prime: int) -> list[int]:
    result = [coefficient % prime for coefficient in poly]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _poly_add(left: list[int], right: list[int], prime: int) -> list[int]:
    size = max(len(left), len(right))
    return _trim([
        (left[index] if index < len(left) else 0) + (right[index] if index < len(right) else 0)
        for index in range(size)
    ], prime)


def _poly_sub(left: list[int], right: list[int], prime: int) -> list[int]:
    size = max(len(left), len(right))
    return _trim([
        (left[index] if index < len(left) else 0) - (right[index] if index < len(right) else 0)
        for index in range(size)
    ], prime)


def _poly_mul(left: list[int], right: list[int], prime: int) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] = (result[i + j] + a * b) % prime
    return _trim(result, prime)


def _poly_divmod(dividend: list[int], divisor: list[int], prime: int) -> tuple[list[int], list[int]]:
    numerator = _trim(dividend[:], prime)
    denominator = _trim(divisor[:], prime)
    if denominator == [0]:
        raise ZeroDivisionError("polynomial division by zero")
    if len(numerator) < len(denominator):
        return [0], numerator
    quotient = [0] * (len(numerator) - len(denominator) + 1)
    inverse = pow(denominator[-1], -1, prime)
    while len(numerator) >= len(denominator) and numerator != [0]:
        shift = len(numerator) - len(denominator)
        coefficient = numerator[-1] * inverse % prime
        quotient[shift] = coefficient
        subtractor = [0] * shift + [(coefficient * value) % prime for value in denominator]
        numerator = _poly_sub(numerator, subtractor, prime)
    return _trim(quotient, prime), _trim(numerator, prime)


def _poly_mod(poly: list[int], modulus: list[int], prime: int) -> list[int]:
    return _poly_divmod(poly, modulus, prime)[1]


def _poly_gcd(left: list[int], right: list[int], prime: int) -> list[int]:
    a = _trim(left, prime)
    b = _trim(right, prime)
    while b != [0]:
        a, b = b, _poly_mod(a, b, prime)
    inverse = pow(a[-1], -1, prime)
    return [(coefficient * inverse) % prime for coefficient in a]


def _poly_pow_mod(base: list[int], exponent: int, modulus: list[int], prime: int) -> list[int]:
    result = [1]
    factor = _poly_mod(base, modulus, prime)
    power = exponent
    while power:
        if power & 1:
            result = _poly_mod(_poly_mul(result, factor, prime), modulus, prime)
        factor = _poly_mod(_poly_mul(factor, factor, prime), modulus, prime)
        power >>= 1
    return result


def _prime_divisors(value: int) -> tuple[int, ...]:
    n = value
    divisors: list[int] = []
    candidate = 2
    while candidate * candidate <= n:
        if n % candidate == 0:
            divisors.append(candidate)
            while n % candidate == 0:
                n //= candidate
        candidate += 1
    if n > 1:
        divisors.append(n)
    return tuple(divisors)


def is_irreducible_polynomial(coefficients: Iterable[int], prime: int) -> bool:
    """Return whether a monic polynomial is irreducible over ``F_prime``."""

    p = require_prime(prime)
    modulus = _trim([require_integer(value, name="coefficient") for value in coefficients], p)
    if len(modulus) < 2 or modulus[-1] != 1:
        return False
    degree = len(modulus) - 1
    if degree == 1:
        return True
    x = [0, 1]
    if _poly_sub(_poly_pow_mod(x, p**degree, modulus, p), x, p) != [0]:
        return False
    for divisor in _prime_divisors(degree):
        exponent = p ** (degree // divisor)
        test = _poly_sub(_poly_pow_mod(x, exponent, modulus, p), x, p)
        if len(_poly_gcd(modulus, test, p)) > 1:
            return False
    return True


@dataclass(frozen=True, slots=True)
class FiniteFieldElement:
    """One element of an explicitly represented finite field."""

    field: "FiniteField"
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.field.normalize(self.value))

    @property
    def coefficients(self) -> tuple[int, ...]:
        """Return the polynomial basis coefficients."""

        return self.field.decode(self.value)

    def __add__(self, other: object) -> "FiniteFieldElement":
        return self.field.element(self.field.add(self.value, self.field.coerce(other)))

    def __sub__(self, other: object) -> "FiniteFieldElement":
        return self.field.element(self.field.subtract(self.value, self.field.coerce(other)))

    def __neg__(self) -> "FiniteFieldElement":
        return self.field.element(self.field.negate(self.value))

    def __mul__(self, other: object) -> "FiniteFieldElement":
        return self.field.element(self.field.multiply(self.value, self.field.coerce(other)))

    def __truediv__(self, other: object) -> "FiniteFieldElement":
        denominator = self.field.coerce(other)
        return self.field.element(self.field.multiply(self.value, self.field.inverse(denominator)))

    def __pow__(self, exponent: int) -> "FiniteFieldElement":
        return self.field.element(self.field.power(self.value, exponent))

    def __int__(self) -> int:
        return self.value


@dataclass(frozen=True)
class FiniteField:
    """The field ``F_(p^degree)`` in one explicit polynomial basis.

    Coefficients of ``modulus`` are given from constant term to leading term.
    Prime fields use the default linear modulus.
    """

    prime: int
    degree: int = 1
    modulus: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        p = require_prime(self.prime)
        degree = require_integer(self.degree, name="degree")
        if degree < 1:
            raise ValidationError("finite field degree must be positive")
        if self.modulus is None:
            if degree != 1:
                raise ValidationError("an irreducible modulus is required for extension fields")
            modulus = (0, 1)
        else:
            modulus = tuple(require_integer(value, name="modulus coefficient") % p for value in self.modulus)
        if len(modulus) != degree + 1 or modulus[-1] != 1:
            raise ValidationError("finite field modulus must be monic of the declared degree")
        if not is_irreducible_polynomial(modulus, p):
            raise ValidationError("finite field modulus is reducible over the prime field")
        object.__setattr__(self, "prime", p)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "modulus", modulus)

    @property
    def order(self) -> int:
        """Return the field order."""

        return self.prime**self.degree

    def normalize(self, value: int) -> int:
        """Return the canonical integer code of one field element."""

        return require_integer(value, name="field element") % self.order

    def encode(self, coefficients: Iterable[int]) -> int:
        """Encode polynomial basis coefficients as one canonical integer."""

        values = tuple(require_integer(value, name="coefficient") % self.prime for value in coefficients)
        if len(values) > self.degree:
            raise ValidationError("too many polynomial basis coefficients")
        padded = values + (0,) * (self.degree - len(values))
        return sum(value * (self.prime**index) for index, value in enumerate(padded))

    def decode(self, value: int) -> tuple[int, ...]:
        """Decode one canonical integer into polynomial basis coefficients."""

        residue = self.normalize(value)
        coefficients: list[int] = []
        for _ in range(self.degree):
            coefficients.append(residue % self.prime)
            residue //= self.prime
        return tuple(coefficients)

    def element(self, value: int | Iterable[int] | FiniteFieldElement) -> FiniteFieldElement:
        """Return one field element."""

        if isinstance(value, FiniteFieldElement):
            if value.field != self:
                raise ValidationError("cannot coerce an element from a different finite field")
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return FiniteFieldElement(self, value)
        return FiniteFieldElement(self, self.encode(value))

    def coerce(self, value: object) -> int:
        """Coerce an integer or matching field element to its code."""

        if isinstance(value, FiniteFieldElement):
            if value.field != self:
                raise ValidationError("cannot combine elements from different finite fields")
            return value.value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError("finite field operations require integers or matching field elements")
        return self.normalize(value)

    def add(self, left: int, right: int) -> int:
        """Add two element codes."""

        a = self.decode(left)
        b = self.decode(right)
        return self.encode((x + y) % self.prime for x, y in zip(a, b, strict=True))

    def subtract(self, left: int, right: int) -> int:
        """Subtract two element codes."""

        a = self.decode(left)
        b = self.decode(right)
        return self.encode((x - y) % self.prime for x, y in zip(a, b, strict=True))

    def negate(self, value: int) -> int:
        """Negate one element code."""

        return self.encode((-x) % self.prime for x in self.decode(value))

    def multiply(self, left: int, right: int) -> int:
        """Multiply two element codes."""

        if self.degree == 1:
            return (self.normalize(left) * self.normalize(right)) % self.prime
        product_poly = _poly_mul(list(self.decode(left)), list(self.decode(right)), self.prime)
        remainder = _poly_mod(product_poly, list(self.modulus or ()), self.prime)
        return self.encode(remainder)

    def power(self, base: int, exponent: int) -> int:
        """Raise one element to an integer power."""

        power = require_integer(exponent, name="exponent")
        if power < 0:
            return self.power(self.inverse(base), -power)
        result = 1
        factor = self.normalize(base)
        while power:
            if power & 1:
                result = self.multiply(result, factor)
            factor = self.multiply(factor, factor)
            power >>= 1
        return result

    def inverse(self, value: int) -> int:
        """Return the inverse of a nonzero element."""

        residue = self.normalize(value)
        if residue == 0:
            raise ZeroDivisionError("zero has no finite field inverse")
        return self.power(residue, self.order - 2)

    def elements(self) -> Iterator[FiniteFieldElement]:
        """Iterate over every field element."""

        return (self.element(value) for value in range(self.order))

    def trace(self, value: int | FiniteFieldElement) -> int:
        """Return the field trace in the prime subfield."""

        residue = self.coerce(value)
        total = 0
        term = residue
        for _ in range(self.degree):
            total = self.add(total, term)
            term = self.power(term, self.prime)
        coefficients = self.decode(total)
        if any(coefficients[index] for index in range(1, self.degree)):
            raise ArithmeticError("finite field trace did not land in the prime subfield")
        return coefficients[0]

    def norm(self, value: int | FiniteFieldElement) -> int:
        """Return the field norm in the prime subfield."""

        residue = self.coerce(value)
        if residue == 0:
            return 0
        exponent = (self.order - 1) // (self.prime - 1)
        result = self.power(residue, exponent)
        coefficients = self.decode(result)
        if any(coefficients[index] for index in range(1, self.degree)):
            raise ArithmeticError("finite field norm did not land in the prime subfield")
        return coefficients[0]

    @cached_property
    def primitive(self) -> int:
        """Return the smallest primitive multiplicative generator."""

        factors = _prime_divisors(self.order - 1)
        for candidate in range(1, self.order):
            if all(self.power(candidate, (self.order - 1) // factor) != 1 for factor in factors):
                return candidate
        raise ArithmeticError("failed to find a primitive finite field element")

    @cached_property
    def logarithm_table(self) -> dict[int, int]:
        """Return discrete logarithms with respect to ``primitive``."""

        table: dict[int, int] = {}
        value = 1
        for exponent in range(self.order - 1):
            table[value] = exponent
            value = self.multiply(value, self.primitive)
        if len(table) != self.order - 1:
            raise ArithmeticError("primitive element did not generate the nonzero field elements")
        return table

    def discrete_log(self, value: int | FiniteFieldElement) -> int:
        """Return the discrete logarithm of a nonzero element."""

        residue = self.coerce(value)
        if residue == 0:
            raise ValidationError("zero has no multiplicative discrete logarithm")
        return self.logarithm_table[residue]


__all__ = [
    "FiniteField",
    "FiniteFieldElement",
    "is_irreducible_polynomial",
]
