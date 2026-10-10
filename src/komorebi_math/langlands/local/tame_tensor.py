"""Exact tame tensor character cycles and local polynomials.

Paper XI section 5: keep all gcd/lcm interaction cycles before combining any
isomorphic or reducible character packets. Scalars here are exact rationals;
this is finite tame data, not a construction of the local correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from numbers import Integral

from ..._number_theory import require_integer
from ...certificates import Certificate, Exactness
from ...exceptions import ValidationError
from .tame import FrobeniusOrbit


def _prime_power(value: int) -> bool:
    factor = next((candidate for candidate in range(2, isqrt(value) + 1) if value % candidate == 0), None)
    if factor is None:
        return value >= 2
    remainder = value
    while remainder % factor == 0:
        remainder //= factor
    return remainder == 1


def _scalar(value: int | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
        raise ValidationError("closing scalars must be exact integers or Fractions")
    result = (
        Fraction(int(value))
        if isinstance(value, Integral)
        else Fraction(int(value.numerator), int(value.denominator))
    )
    if not result:
        raise ValidationError("Frobenius closing scalars must be nonzero")
    return result


@dataclass(frozen=True, slots=True)
class ExactTameTensorPacket:
    """One actual simultaneous-translation cycle and lifted inertia exponent."""

    representative: int
    cycle_pairs: tuple[tuple[int, int], ...]
    exponent: int
    exponent_modulus: int
    closing_scalar: Fraction

    @property
    def dimension(self) -> int:
        return len(self.cycle_pairs)

    @property
    def inertia_trivial(self) -> bool:
        return self.exponent == 0

    @property
    def weil_fixed_dimension(self) -> int:
        return int(self.inertia_trivial and self.closing_scalar == 1)


@dataclass(frozen=True, slots=True)
class ExactTameTensorRefinement:
    """Finite exact tensor cycles, fixed dimensions, and ascending local polynomial."""

    q: int
    packets: tuple[ExactTameTensorPacket, ...]
    total_dimension: int
    inertia_fixed_dimension: int
    weil_fixed_dimension: int
    tame_artin_conductor: int
    local_polynomial_coefficients: tuple[Fraction, ...]
    certificate: Certificate


def tame_tensor_character_packets(
    left: FrobeniusOrbit,
    right: FrobeniusOrbit,
    *,
    left_scalar: int | Fraction = 1,
    right_scalar: int | Fraction = 1,
    max_dimension: int = 10000,
    max_modulus_bits: int = 100000,
    max_residue_cardinality: int = 1000000,
) -> ExactTameTensorRefinement:
    """Construct the norm-lift character packets and finite tame local factor.

    ``q`` must be a common prime-power residue cardinality; the input orbits
    must be regular. Closing scalars use forward orbit cycles with
    ``F(e_i)=e_(i+1)`` and closing edge ``a``. The local polynomial is
    ``det(I-u*F | V^I)`` in ascending powers of ``u``. It refers to the supplied
    cycle convention, so arithmetic comparisons must retain that convention.

    The dimension and modulus-bit caps are checked before building the
    Cartesian cycle data or the lifted character modulus. The latter uses
    the conservative bound ``lcm(n,m) * q.bit_length()``; it may reject a
    modulus whose actual bit length would fit the cap.
    """
    if not isinstance(left, FrobeniusOrbit) or not isinstance(right, FrobeniusOrbit):
        raise ValidationError("tame tensor inputs must be FrobeniusOrbit records")
    limit = require_integer(max_dimension, name="max_dimension")
    bit_limit = require_integer(max_modulus_bits, name="max_modulus_bits")
    field_limit = require_integer(max_residue_cardinality, name="max_residue_cardinality")
    n, m = left.degree, right.degree
    if limit < 1 or bit_limit < 1 or n * m > limit:
        raise ValidationError("tame tensor dimension exceeds max_dimension or a cap is invalid")
    if left.q != right.q:
        raise ValidationError("tame tensor orbits must have the same residue cardinality")
    q = left.q
    if field_limit < 2 or q > field_limit:
        raise ValidationError("residue cardinality exceeds max_residue_cardinality")
    count = gcd(n, m)
    length = n * m // count
    if length * q.bit_length() > bit_limit:
        raise ValidationError("lifted tame character modulus exceeds max_modulus_bits")
    if not _prime_power(q):
        raise ValidationError("residue cardinality must be a prime power")
    if not left.regular or not right.regular:
        raise ValidationError("tame tensor atoms require regular input orbits")
    a, b = _scalar(left_scalar), _scalar(right_scalar)
    modulus = q**length - 1
    left_lift = modulus // left.modulus
    right_lift = modulus // right.modulus
    closing = a ** (length // n) * b ** (length // m)
    packets = tuple(
        ExactTameTensorPacket(
            representative,
            tuple((step % n, (representative + step) % m) for step in range(length)),
            (left.exponent * left_lift + right.exponent * q**representative * right_lift) % modulus,
            modulus,
            closing,
        )
        for representative in range(count)
    )
    seen = {pair for packet in packets for pair in packet.cycle_pairs}
    exponents_valid = all(
        (left.exponent * pow(q, i, modulus) * left_lift + right.exponent * pow(q, j, modulus) * right_lift)
        % modulus
        == (packet.exponent * pow(q, step, modulus)) % modulus
        for packet in packets
        for step, (i, j) in enumerate(packet.cycle_pairs)
    )
    polynomial = [Fraction(1)]
    for packet in packets:
        if packet.inertia_trivial:
            previous = polynomial
            polynomial = previous + [Fraction(0)] * length
            for index, coefficient in enumerate(previous):
                polynomial[index + length] -= packet.closing_scalar * coefficient
    inertia_dimension = sum(packet.dimension for packet in packets if packet.inertia_trivial)
    fixed_dimension = sum(packet.weil_fixed_dimension for packet in packets)
    return ExactTameTensorRefinement(
        q,
        packets,
        n * m,
        inertia_dimension,
        fixed_dimension,
        n * m - inertia_dimension,
        tuple(polynomial),
        Certificate(
            valid=len(seen) == n * m
            and sum(packet.dimension for packet in packets) == n * m
            and exponents_valid,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "common prime-power residue cardinality",
                "regular input exponent orbits",
                "exact nonzero rational closing scalars",
                "forward simultaneous translation cycles",
                "finite dimension and modulus caps",
            ),
            assumptions_unchecked=(
                "arithmetic attachment of the supplied tame characters and Frobenius normalization",
            ),
            diagnostics={
                "packet_count": count,
                "cycle_length": length,
                "exponents": tuple(packet.exponent for packet in packets),
                "inertia_fixed_dimension": inertia_dimension,
                "weil_fixed_dimension": fixed_dimension,
                "interaction_packets_need_not_be_irreducible": True,
            },
            theorem_id="PSC-XI-TAME-CHARACTER-TENSOR",
            message="exact tame character interaction cycles and local polynomial constructed",
        ),
    )


__all__ = ["ExactTameTensorPacket", "ExactTameTensorRefinement", "tame_tensor_character_packets"]
