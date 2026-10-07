"""Universal custom exponential sections under the basis lowering operator."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from fractions import Fraction
from math import tau

from .basis import WeightedMonomialBasis
from .series import FormalSeries


def custom_exponential(basis: WeightedMonomialBasis, degree: int) -> FormalSeries:
    r"""Return ``E_gamma(z)=sum z^n/gamma_n`` through one degree."""

    if degree < 0:
        raise ValueError("degree must be nonnegative")
    return FormalSeries(basis, [1] * (degree + 1))


def custom_section(
    basis: WeightedMonomialBasis,
    modulus: int,
    residue: int,
    degree: int,
    *,
    signed: bool = False,
) -> FormalSeries:
    r"""Return one residue section of the custom exponential."""

    if modulus < 1 or not 0 <= residue < modulus or degree < 0:
        raise ValueError("invalid section parameters")
    values = []
    for n in range(degree + 1):
        if n % modulus != residue:
            values.append(Fraction(0))
        else:
            quotient = (n - residue) // modulus
            values.append(Fraction(-1 if signed and quotient % 2 else 1))
    return FormalSeries(basis, values)


def root_filter_value(
    basis: WeightedMonomialBasis,
    modulus: int,
    residue: int,
    degree: int,
    z: complex,
) -> complex:
    r"""Evaluate the finite root filter of ``E_gamma``."""

    omega = cmath.exp(1j * tau / modulus)
    exponential = custom_exponential(basis, degree)
    return sum(omega ** (-residue * j) * exponential.evaluate(omega**j * z) for j in range(modulus)) / modulus


@dataclass(frozen=True, slots=True)
class SectionCycleAudit:
    """Finite lowering-cycle and root-filter audit summary."""

    basis: str
    modulus: int
    degree: int
    signed: bool
    one_step_failures: int
    full_cycle_failures: int
    root_filter_residual: float


def audit_section_cycle(
    basis: WeightedMonomialBasis,
    *,
    modulus: int,
    degree: int,
    signed: bool = False,
    z: complex = 0.37 + 0.11j,
) -> SectionCycleAudit:
    """Audit lowering cycles and root-filter projection on finite truncations."""

    sections = [custom_section(basis, modulus, residue, degree, signed=signed) for residue in range(modulus)]
    one_step_failures = 0
    for residue, section in enumerate(sections):
        lowered = section.custom_lower()
        target_residue = (residue - 1) % modulus
        target = custom_section(basis, modulus, target_residue, degree - 1, signed=signed)
        expected = target
        if signed and residue == 0:
            expected = target.scale(-1)
        if lowered.coefficients != expected.coefficients:
            one_step_failures += 1

    full_cycle_failures = 0
    for section in sections:
        lowered = section
        for _ in range(modulus):
            lowered = lowered.custom_lower()
        target = section.truncate(max(degree - modulus, 0))
        if signed:
            target = target.scale(-1)
        if lowered.coefficients != target.coefficients:
            full_cycle_failures += 1

    selected = custom_section(basis, modulus, 0, degree, signed=False)
    residual = abs(selected.evaluate(z) - root_filter_value(basis, modulus, 0, degree, z))
    return SectionCycleAudit(
        basis.name,
        modulus,
        degree,
        signed,
        one_step_failures,
        full_cycle_failures,
        float(residual),
    )


__all__ = [
    "SectionCycleAudit",
    "audit_section_cycle",
    "custom_exponential",
    "custom_section",
    "root_filter_value",
]
