"""Transfer exponential sections and signed cyclic differential bases."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from math import factorial, tau
from typing import Literal

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._series import validate_order


def _section_sum(
    order: int,
    residue: int,
    z: complex,
    *,
    scale: complex,
    signed: bool,
    tolerance: float,
    max_terms: int,
) -> complex:
    n = validate_order(order)
    j = int(residue)
    if isinstance(residue, bool) or j != residue or not 0 <= j < n:
        raise ValidationError("residue must lie between zero and order minus one")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValidationError("tolerance must be positive and finite")
    if isinstance(max_terms, bool) or int(max_terms) != max_terms or int(max_terms) < 1:
        raise ValidationError("max_terms must be a positive integer")
    value = complex(scale) * complex(z)
    term = value**j / factorial(j)
    total = term
    block = value**n
    for k in range(max_terms - 1):
        denominator = 1
        start = n * k + j + 1
        for offset in range(n):
            denominator *= start + offset
        term *= block / denominator
        if signed:
            term = -term
        total += term
        if abs(term) <= tolerance * max(1.0, abs(total)):
            return total
    raise ArithmeticError("cyclic section series did not converge within max_terms")


@documented(
    ProvenanceRecord(
        identifier="PSC-IV-T05",
        title="Transfer exponential section",
        status=ClaimStatus.PROVED,
        source="Paper IV",
        assumptions=("finite order", "convergent entire series"),
        verification_suite="analytic.sections",
    )
)
def transfer_section(
    order: int,
    residue: int,
    z: complex,
    *,
    scale: complex = 1.0,
    tolerance: float = 1e-15,
    max_terms: int = 512,
) -> complex:
    """Evaluate ``sum (scale*z)**(order*n+residue)/(order*n+residue)!``."""

    return _section_sum(
        order,
        residue,
        z,
        scale=scale,
        signed=False,
        tolerance=tolerance,
        max_terms=max_terms,
    )


@documented(
    ProvenanceRecord(
        identifier="PSC-IV-T09",
        title="Signed cyclic section",
        status=ClaimStatus.PROVED,
        source="Paper IV",
        assumptions=("finite order", "convergent entire series"),
        verification_suite="analytic.sections",
    )
)
def signed_section(
    order: int,
    residue: int,
    z: complex,
    *,
    tolerance: float = 1e-15,
    max_terms: int = 512,
) -> complex:
    """Evaluate ``sum (-1)^n z**(order*n+residue)/(order*n+residue)!``."""

    return _section_sum(
        order,
        residue,
        z,
        scale=1.0,
        signed=True,
        tolerance=tolerance,
        max_terms=max_terms,
    )


def transfer_sections(order: int, z: complex, *, scale: complex = 1.0) -> tuple[complex, ...]:
    """Evaluate the complete transfer exponential basis at one point."""

    n = validate_order(order)
    return tuple(transfer_section(n, residue, z, scale=scale) for residue in range(n))


def signed_sections(order: int, z: complex) -> tuple[complex, ...]:
    """Evaluate the complete signed cyclic basis at one point."""

    n = validate_order(order)
    return tuple(signed_section(n, residue, z) for residue in range(n))


def denominator(order: int, z: complex) -> complex:
    """Evaluate the distinguished signed denominator ``D_order(z)``."""

    return signed_section(order, 0, z)


def denominator_root_sum(order: int, z: complex) -> complex:
    """Evaluate the denominator through its finite exponential root sum."""

    n = validate_order(order)
    return sum(cmath.exp(cmath.exp(1j * (2 * k + 1) * np.pi / n) * z) for k in range(n)) / n


def denominator_derivative(order: int, z: complex, derivative_order: int = 1) -> complex:
    """Evaluate a derivative of the distinguished signed denominator."""

    n = validate_order(order)
    r = int(derivative_order)
    if isinstance(derivative_order, bool) or r != derivative_order or r < 0:
        raise ValidationError("derivative_order must be a nonnegative integer")
    return sum(
        eta**r * cmath.exp(eta * z)
        for eta in (cmath.exp(1j * (2 * k + 1) * np.pi / n) for k in range(n))
    ) / n


def transfer_shift_matrix(order: int, scale: complex = 1.0) -> np.ndarray:
    """Return the cyclic companion matrix for ``y**(order)=scale**order*y``."""

    n = validate_order(order)
    matrix = np.zeros((n, n), dtype=np.result_type(scale, complex))
    for column in range(n - 1):
        matrix[column + 1, column] = scale
    matrix[0, n - 1] = scale
    return matrix


def signed_shift_matrix(order: int) -> np.ndarray:
    """Return the signed shift matrix satisfying ``J**order=-I``."""

    n = validate_order(order)
    matrix = np.zeros((n, n), dtype=float)
    for column in range(n - 1):
        matrix[column + 1, column] = 1.0
    matrix[0, n - 1] = -1.0
    return matrix


def cyclic_addition(
    left: tuple[complex, ...],
    right: tuple[complex, ...],
    *,
    signed: bool,
) -> tuple[complex, ...]:
    """Apply the finite cyclic addition law to two section coordinate vectors."""

    if not left or len(left) != len(right):
        raise ValidationError("section vectors must have one common nonzero order")
    n = len(left)
    output: list[complex] = []
    for target in range(n):
        total = 0j
        for source in range(n):
            residue = (target - source) % n
            sign = -1 if signed and source > target else 1
            total += sign * left[source] * right[residue]
        output.append(total)
    return tuple(output)


def section_certificate(
    order: int,
    z: complex,
    *,
    family: Literal["transfer", "signed"] = "signed",
    tolerance: float = 1e-11,
) -> ComputationResult[tuple[complex, ...]]:
    """Validate the differential basis against its finite root sum."""

    n = validate_order(order)
    if family not in {"transfer", "signed"}:
        raise ValidationError("family must be transfer or signed")
    values = transfer_sections(n, z) if family == "transfer" else signed_sections(n, z)
    if family == "transfer":
        roots = tuple(cmath.exp(1j * tau * k / n) for k in range(n))
    else:
        roots = tuple(cmath.exp(1j * (2 * k + 1) * np.pi / n) for k in range(n))
    reconstructed = []
    for residue in range(n):
        reconstructed.append(sum(root ** (-residue) * cmath.exp(root * z) for root in roots) / n)
    residual = max(abs(lhs - rhs) for lhs, rhs in zip(values, reconstructed, strict=True))
    return ComputationResult(
        values,
        Certificate(
            valid=residual <= tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"root_sum": float(residual)},
            tolerance=tolerance,
            assumptions_checked=("finite root system", "entire section series"),
            theorem_id="PSC-IV-T05" if family == "transfer" else "PSC-IV-T09",
            message=f"{family} sections agree with the finite root representation",
        ),
    )


@dataclass(frozen=True, slots=True)
class CyclicSectionFamily:
    """One transfer or signed cyclic differential family."""

    order: int
    family: Literal["transfer", "signed"] = "signed"
    scale: complex = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", validate_order(self.order))
        if self.family not in {"transfer", "signed"}:
            raise ValidationError("family must be transfer or signed")

    def evaluate(self, residue: int, z: complex) -> complex:
        """Evaluate one residue section."""

        if self.family == "transfer":
            return transfer_section(self.order, residue, z, scale=self.scale)
        return signed_section(self.order, residue, self.scale * z)

    def vector(self, z: complex) -> tuple[complex, ...]:
        """Evaluate every residue section."""

        return tuple(self.evaluate(residue, z) for residue in range(self.order))

    def fundamental_matrix(self, z: complex) -> np.ndarray:
        """Return the matrix exponential of the finite companion model."""

        matrix = transfer_shift_matrix(self.order, self.scale) if self.family == "transfer" else self.scale * signed_shift_matrix(self.order)
        try:
            from scipy.linalg import expm
        except ImportError:
            from ..dynamics._linalg import matrix_exponential

            return matrix_exponential(z * matrix)
        return np.asarray(expm(z * matrix))


__all__ = [
    "CyclicSectionFamily",
    "cyclic_addition",
    "denominator",
    "denominator_derivative",
    "denominator_root_sum",
    "section_certificate",
    "signed_section",
    "signed_sections",
    "signed_shift_matrix",
    "transfer_section",
    "transfer_sections",
    "transfer_shift_matrix",
]
