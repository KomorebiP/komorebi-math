"""Chinese remainder addresses for residue packet systems."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import gcd, prod

from ..provenance import ClaimStatus, ProvenanceRecord, documented


@dataclass(frozen=True, slots=True)
class CRTAddress:
    """One residue address together with its component residues."""

    residue: int
    modulus: int
    residues: tuple[int, ...]
    moduli: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.modulus <= 0:
            raise ValueError("modulus must be positive")
        if len(self.residues) != len(self.moduli):
            raise ValueError("residue and modulus counts do not agree")
        if any(modulus <= 0 for modulus in self.moduli):
            raise ValueError("component moduli must be positive")
        if self.modulus != prod(self.moduli):
            raise ValueError("combined modulus must equal the product of component moduli")
        if not 0 <= self.residue < self.modulus:
            raise ValueError("combined residue must lie in its standard range")
        if any(residue % modulus != self.residue % modulus for residue, modulus in zip(self.residues, self.moduli, strict=True)):
            raise ValueError("component residues do not match the combined residue")


@documented(
    ProvenanceRecord(
        identifier="PSC-II-T08",
        title="Coprime factorization of residue addresses",
        status=ClaimStatus.PROVED,
        source="Paper II",
        assumptions=("pairwise coprime positive moduli",),
        verification_suite="refinement.crt",
    )
)
def crt_pair(a: int, modulus_a: int, b: int, modulus_b: int) -> CRTAddress:
    """Combine two coprime residue addresses.

    The returned residue is the unique integer in ``range(modulus_a * modulus_b)``
    with the requested component residues.
    """

    if modulus_a <= 0 or modulus_b <= 0:
        raise ValueError("moduli must be positive")
    if gcd(modulus_a, modulus_b) != 1:
        raise ValueError("CRT moduli must be coprime")
    a %= modulus_a
    b %= modulus_b
    inverse = pow(modulus_a, -1, modulus_b)
    correction = ((b - a) * inverse) % modulus_b
    modulus = modulus_a * modulus_b
    residue = (a + modulus_a * correction) % modulus
    return CRTAddress(residue, modulus, (a, b), (modulus_a, modulus_b))


@documented(
    ProvenanceRecord(
        identifier="PSC-II-C09",
        title="Finite coprime residue factorization",
        status=ClaimStatus.PROVED,
        source="Paper II",
        assumptions=("pairwise coprime positive moduli",),
        verification_suite="refinement.crt",
    )
)
def crt_many(residues: Iterable[int], moduli: Iterable[int]) -> CRTAddress:
    """Combine a finite collection of pairwise coprime residues."""

    residue_values = tuple(int(value) for value in residues)
    modulus_values = tuple(int(value) for value in moduli)
    if not modulus_values:
        raise ValueError("at least one modulus is required")
    if len(residue_values) != len(modulus_values):
        raise ValueError("residue and modulus counts do not agree")
    if any(value <= 0 for value in modulus_values):
        raise ValueError("moduli must be positive")
    for left in range(len(modulus_values)):
        for right in range(left + 1, len(modulus_values)):
            if gcd(modulus_values[left], modulus_values[right]) != 1:
                raise ValueError("CRT moduli must be pairwise coprime")

    current_residue = residue_values[0] % modulus_values[0]
    current_modulus = modulus_values[0]
    for next_residue, next_modulus in zip(residue_values[1:], modulus_values[1:], strict=True):
        combined = crt_pair(current_residue, current_modulus, next_residue, next_modulus)
        current_residue = combined.residue
        current_modulus = combined.modulus
    normalized = tuple(value % modulus for value, modulus in zip(residue_values, modulus_values, strict=True))
    return CRTAddress(current_residue, current_modulus, normalized, modulus_values)


def factor_residue(residue: int, moduli: Iterable[int]) -> CRTAddress:
    """Factor one combined residue into its component addresses."""

    modulus_values = tuple(int(value) for value in moduli)
    if not modulus_values:
        raise ValueError("at least one modulus is required")
    modulus = prod(modulus_values)
    normalized = residue % modulus
    residues = tuple(normalized % value for value in modulus_values)
    # crt_many validates pairwise coprimality and reconstructs the unique address.
    reconstructed = crt_many(residues, modulus_values)
    if reconstructed.residue != normalized:
        raise ArithmeticError("CRT reconstruction failed")
    return reconstructed


def dyadic_odd_factor(residue: int, dyadic_power: int, odd_modulus: int) -> CRTAddress:
    """Factor a residue modulo ``2**dyadic_power * odd_modulus``."""

    if dyadic_power < 0:
        raise ValueError("dyadic power must be nonnegative")
    if odd_modulus <= 0 or odd_modulus % 2 == 0:
        raise ValueError("odd modulus must be a positive odd integer")
    return factor_residue(residue, (2**dyadic_power, odd_modulus))


__all__ = ["CRTAddress", "crt_many", "crt_pair", "dyadic_odd_factor", "factor_residue"]
