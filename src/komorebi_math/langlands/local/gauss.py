"""Finite field Gauss sums and phase certificates."""

from __future__ import annotations

from cmath import exp, pi
from dataclasses import dataclass

from ...certificates import Certificate, Exactness
from ...exceptions import ValidationError
from .finite_fields import FiniteField


@dataclass(frozen=True, slots=True)
class GaussSumResult:
    """One finite Gauss sum and its normalized phase."""

    value: complex
    magnitude: float
    phase: complex
    certificate: Certificate


def additive_character(field: FiniteField, value: int, *, frequency: int = 1) -> complex:
    """Evaluate the canonical trace additive character."""

    a = frequency % field.prime
    return exp(2j * pi * a * field.trace(value) / field.prime)


def multiplicative_character(field: FiniteField, value: int, exponent: int) -> complex:
    """Evaluate a multiplicative character indexed by an exponent."""

    residue = field.normalize(value)
    if residue == 0:
        return 0.0 + 0.0j
    return exp(2j * pi * (exponent % (field.order - 1)) * field.discrete_log(residue) / (field.order - 1))


def gauss_sum(
    field: FiniteField,
    exponent: int,
    *,
    additive_frequency: int = 1,
    tolerance: float = 1e-9,
) -> GaussSumResult:
    """Compute one finite multiplicative and additive Gauss sum."""

    if additive_frequency % field.prime == 0:
        raise ValidationError("additive frequency must define a nontrivial additive character")
    character_exponent = exponent % (field.order - 1)
    if character_exponent == 0:
        expected = 1.0
    else:
        expected = field.order**0.5
    value = sum(
        multiplicative_character(field, element, character_exponent)
        * additive_character(field, element, frequency=additive_frequency)
        for element in range(field.order)
    )
    magnitude = abs(value)
    phase = value / magnitude if magnitude else 0.0 + 0.0j
    residual = abs(magnitude - expected)
    return GaussSumResult(
        value=value,
        magnitude=magnitude,
        phase=phase,
        certificate=Certificate(
            valid=residual <= tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"magnitude": residual},
            tolerance=tolerance,
            assumptions_checked=("explicit finite field", "nontrivial additive character"),
            theorem_id="PSC-XI-GAUSS",
            message="finite Gauss magnitude verified" if residual <= tolerance else "finite Gauss magnitude failed",
        ),
    )


__all__ = ["GaussSumResult", "additive_character", "gauss_sum", "multiplicative_character"]
