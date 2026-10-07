"""Finite tame Frobenius packets and monodromy chains."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterable

import numpy as np

from ..._number_theory import require_integer
from ...certificates import Certificate, Exactness
from ...exceptions import ValidationError


def _mobius(value: int) -> int:
    n = value
    count = 0
    prime = 2
    while prime * prime <= n:
        if n % prime == 0:
            n //= prime
            count += 1
            if n % prime == 0:
                return 0
            while n % prime == 0:
                n //= prime
        prime += 1
    if n > 1:
        count += 1
    return -1 if count % 2 else 1


def _divisors(value: int) -> tuple[int, ...]:
    output: list[int] = []
    for candidate in range(1, int(value**0.5) + 1):
        if value % candidate == 0:
            output.append(candidate)
            if candidate * candidate != value:
                output.append(value // candidate)
    return tuple(sorted(output))


@dataclass(frozen=True, slots=True)
class FrobeniusOrbit:
    """A tame character orbit under multiplication by the residue cardinality."""

    q: int
    degree: int
    exponent: int

    def __post_init__(self) -> None:
        q = require_integer(self.q, name="q")
        degree = require_integer(self.degree, name="degree")
        exponent = require_integer(self.exponent, name="exponent")
        if q < 2 or degree < 1:
            raise ValidationError("q must be at least two and degree must be positive")
        modulus = q**degree - 1
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "exponent", exponent % modulus)

    @property
    def modulus(self) -> int:
        """Return the tame character exponent modulus."""

        return self.q**self.degree - 1

    @property
    def elements(self) -> tuple[int, ...]:
        """Return the distinct Frobenius orbit exponents."""

        output: list[int] = []
        value = self.exponent
        while value not in output:
            output.append(value)
            value = (self.q * value) % self.modulus
        return tuple(output)

    @property
    def orbit_length(self) -> int:
        """Return the character orbit length."""

        return len(self.elements)

    @property
    def regular(self) -> bool:
        """Return whether the orbit has full declared degree."""

        return self.orbit_length == self.degree

    @property
    def dual_exponent(self) -> int:
        """Return the exponent of the dual tame character."""

        return (-self.exponent) % self.modulus

    @property
    def self_dual(self) -> bool:
        """Return whether the dual character lies in the same Frobenius orbit."""

        return self.dual_exponent in self.elements

    def certificate(self) -> Certificate:
        """Certify closure and regularity of the finite orbit."""

        orbit = self.elements
        valid = all((self.q * orbit[index]) % self.modulus == orbit[(index + 1) % len(orbit)] for index in range(len(orbit)))
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite exponent orbit",),
            diagnostics={"orbit": orbit, "regular": self.regular, "self_dual": self.self_dual},
            theorem_id="PSC-XI-TAME-ORBIT",
            message="valid tame Frobenius orbit" if valid else "Frobenius orbit failed to close",
        )


def regular_orbit_count(q: int, degree: int) -> int:
    """Return the number of full Frobenius orbits of tame characters."""

    size = require_integer(q, name="q")
    n = require_integer(degree, name="degree")
    if size < 2 or n < 1:
        raise ValidationError("q must be at least two and degree must be positive")
    exact_elements = sum(_mobius(n // divisor) * (size**divisor - 1) for divisor in _divisors(n))
    if exact_elements % n:
        raise ArithmeticError("regular orbit count is not integral")
    return exact_elements // n


@dataclass(frozen=True, slots=True)
class TameWeilPacket:
    """A finite matrix model of one regular tame Weil atom."""

    orbit: FrobeniusOrbit
    frobenius_scalar: complex = 1.0

    def __post_init__(self) -> None:
        if not self.orbit.regular:
            raise ValidationError("a tame Weil packet requires a regular orbit")

    @property
    def dimension(self) -> int:
        """Return the packet dimension."""

        return self.orbit.degree

    def inertia_matrix(self) -> np.ndarray:
        """Return a diagonal matrix for one tame inertia generator."""

        roots = np.exp(2j * np.pi * np.asarray(self.orbit.elements) / self.orbit.modulus)
        return np.diag(roots)

    def frobenius_matrix(self) -> np.ndarray:
        """Return the cyclic Frobenius matrix."""

        matrix = np.zeros((self.dimension, self.dimension), dtype=complex)
        for index in range(1, self.dimension):
            matrix[index - 1, index] = 1.0
        matrix[self.dimension - 1, 0] = complex(self.frobenius_scalar)
        return matrix

    @property
    def determinant_orientation(self) -> complex:
        """Return the determinant of the Frobenius cycle."""

        return ((-1) ** (self.dimension - 1)) * complex(self.frobenius_scalar)

    def relation_certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        """Check the tame conjugacy relation between Frobenius and inertia."""

        inertia = self.inertia_matrix()
        frobenius = self.frobenius_matrix()
        left = frobenius @ inertia @ np.linalg.inv(frobenius)
        right = np.linalg.matrix_power(inertia, self.orbit.q)
        residual = float(np.linalg.norm(left - right))
        return Certificate(
            valid=residual <= tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"tame_conjugacy": residual},
            tolerance=tolerance,
            assumptions_checked=("regular tame orbit", "declared Frobenius cycle"),
            theorem_id="PSC-XI-TAME-ATOM",
            message="tame Weil packet relation holds" if residual <= tolerance else "tame Weil packet relation failed",
        )


@dataclass(frozen=True, slots=True)
class TensorRefinement:
    """Greatest common divisor and least common multiple tensor packet data."""

    packet_count: int
    packet_dimension: int
    total_dimension: int
    certificate: Certificate


def tame_tensor_refinement(left_degree: int, right_degree: int) -> TensorRefinement:
    """Return the orbit count and depth of a tame tensor interaction."""

    n = require_integer(left_degree, name="left_degree")
    m = require_integer(right_degree, name="right_degree")
    if n < 1 or m < 1:
        raise ValidationError("tame packet degrees must be positive")
    count = gcd(n, m)
    depth = n * m // count
    return TensorRefinement(
        packet_count=count,
        packet_dimension=depth,
        total_dimension=n * m,
        certificate=Certificate(
            valid=count * depth == n * m,
            exactness=Exactness.EXACT,
            assumptions_checked=("two cyclic Frobenius orbits",),
            diagnostics={"gcd": count, "lcm": depth},
            theorem_id="PSC-XI-TAME-TENSOR",
            message="tame tensor dimensions are conserved",
        ),
    )


@dataclass(frozen=True, slots=True)
class SpecialMonodromyBlock:
    """A finite monodromy chain with a declared Frobenius scale."""

    length: int
    q: float

    def __post_init__(self) -> None:
        length = require_integer(self.length, name="length")
        if length < 1 or not np.isfinite(self.q) or self.q <= 0:
            raise ValidationError("monodromy length and Frobenius scale must be positive")
        object.__setattr__(self, "length", length)

    def nilpotent(self) -> np.ndarray:
        """Return the downward monodromy shift."""

        matrix = np.zeros((self.length, self.length), dtype=float)
        for index in range(1, self.length):
            matrix[index - 1, index] = 1.0
        return matrix

    def frobenius(self) -> np.ndarray:
        """Return a diagonal Frobenius matrix satisfying ``F N F^-1 = q N``."""

        diagonal = [self.q ** (-index) for index in range(self.length)]
        return np.diag(diagonal)

    def certificate(self, *, tolerance: float = 1e-12) -> Certificate:
        """Check the monodromy relation and nilpotence depth."""

        monodromy = self.nilpotent()
        frobenius = self.frobenius()
        residual = float(np.linalg.norm(frobenius @ monodromy @ np.linalg.inv(frobenius) - self.q * monodromy))
        nilpotent = np.linalg.matrix_power(monodromy, self.length)
        nil_residual = float(np.linalg.norm(nilpotent))
        valid = residual <= tolerance and nil_residual <= tolerance
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"monodromy_relation": residual, "nilpotence": nil_residual},
            tolerance=tolerance,
            assumptions_checked=("finite monodromy chain",),
            theorem_id="PSC-XI-MONODROMY",
            message="valid special monodromy block" if valid else "monodromy block validation failed",
        )


__all__ = [
    "FrobeniusOrbit",
    "SpecialMonodromyBlock",
    "TameWeilPacket",
    "TensorRefinement",
    "regular_orbit_count",
    "tame_tensor_refinement",
]
