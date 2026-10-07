"""Finite positive fractional operators and refinement certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _hermitian_psd(matrix: Any, *, tolerance: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise DimensionError("fractional operator must be square")
    hermitian_residual = float(np.linalg.norm(value - value.T.conj(), ord=2))
    if hermitian_residual > tolerance:
        raise ValidationError("fractional operator must be Hermitian")
    eigenvalues, eigenvectors = np.linalg.eigh((value + value.T.conj()) / 2.0)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValidationError("fractional operator must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return value, eigenvalues, eigenvectors


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T24",
        title="Finite spectral fractional operator",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("finite positive semidefinite Hermitian operator", "positive exponent"),
        verification_suite="dynamics.fractional",
    )
)
def fractional_power(matrix: Any, exponent: float, *, tolerance: float = 1e-12) -> np.ndarray:
    """Return the positive spectral power of a finite Hermitian matrix."""

    alpha = float(exponent)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValidationError("fractional exponent must be positive and finite")
    _, eigenvalues, eigenvectors = _hermitian_psd(matrix, tolerance=tolerance)
    return (eigenvectors * eigenvalues**alpha) @ eigenvectors.T.conj()


def fractional_semigroup(
    matrix: Any,
    exponent: float,
    time: float,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return ``exp(-time * matrix**exponent)`` by spectral calculus."""

    t = float(time)
    if not np.isfinite(t) or t < 0:
        raise ValidationError("semigroup time must be finite and nonnegative")
    alpha = float(exponent)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValidationError("fractional exponent must be positive and finite")
    _, eigenvalues, eigenvectors = _hermitian_psd(matrix, tolerance=tolerance)
    multipliers = np.exp(-t * eigenvalues**alpha)
    return (eigenvectors * multipliers) @ eigenvectors.T.conj()


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T25",
        title="Exact finite fractional naturality",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("positive self adjoint fine and coarse operators", "exact intertwining"),
        verification_suite="dynamics.fractional",
    )
)
def fractional_naturality_certificate(
    coarsen: Any,
    fine_operator: Any,
    coarse_operator: Any,
    exponent: float,
    *,
    tolerance: float = 1e-10,
) -> Certificate:
    """Certify transport of a positive spectral power through an intertwiner."""

    c = np.asarray(coarsen)
    fine = np.asarray(fine_operator)
    coarse = np.asarray(coarse_operator)
    if fine.ndim != 2 or coarse.ndim != 2 or c.shape != (coarse.shape[0], fine.shape[0]):
        raise DimensionError("coarsening and operator dimensions do not agree")
    generator_residual = float(np.linalg.norm(c @ fine - coarse @ c, ord=2))
    fractional_residual = float(
        np.linalg.norm(
            c @ fractional_power(fine, exponent) - fractional_power(coarse, exponent) @ c,
            ord=2,
        )
    )
    valid = max(generator_residual, fractional_residual) <= tolerance
    return Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={"generator_intertwining": generator_residual, "fractional_intertwining": fractional_residual},
        tolerance=tolerance,
        assumptions_checked=("finite positive Hermitian operators",),
        theorem_id="PSC-VI-T25",
        message="fractional naturality verified" if valid else "fractional naturality hypotheses or conclusion failed",
    )


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T26",
        title="Finite fractional defect bound",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("exponent strictly between zero and one", "common positive spectral gap"),
        verification_suite="dynamics.fractional",
    )
)
@dataclass(frozen=True, slots=True)
class FractionalDefectResult:
    """Actual fractional defect and the positive gap theorem bound."""

    defect: np.ndarray
    bound: float
    generator_defect_norm: float
    spectral_gap: float
    certificate: Certificate


def fractional_defect(
    coarsen: Any,
    fine_operator: Any,
    coarse_operator: Any,
    exponent: float,
    *,
    tolerance: float = 1e-12,
) -> FractionalDefectResult:
    """Compute and certify the finite positive gap fractional defect estimate."""

    alpha = float(exponent)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValidationError("the fractional defect theorem requires an exponent strictly between zero and one")
    c = np.asarray(coarsen)
    fine, fine_values, _ = _hermitian_psd(fine_operator, tolerance=tolerance)
    coarse, coarse_values, _ = _hermitian_psd(coarse_operator, tolerance=tolerance)
    if c.shape != (coarse.shape[0], fine.shape[0]):
        raise DimensionError("coarsening map shape does not match the operators")
    gap = float(min(np.min(fine_values), np.min(coarse_values)))
    if gap <= tolerance:
        raise ValidationError("the fractional defect norm bound requires a strictly positive spectral gap")
    generator_defect = c @ fine - coarse @ c
    generator_norm = float(np.linalg.norm(generator_defect, ord=2))
    defect = c @ fractional_power(fine, alpha) - fractional_power(coarse, alpha) @ c
    actual = float(np.linalg.norm(defect, ord=2))
    bound = float(alpha * gap ** (alpha - 1.0) * generator_norm)
    valid = actual <= bound + max(tolerance, 1e-10 * max(1.0, bound))
    return FractionalDefectResult(
        defect,
        bound,
        generator_norm,
        gap,
        Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"actual_defect": actual, "theorem_bound": bound, "bound_excess": max(0.0, actual - bound)},
            tolerance=max(tolerance, 1e-10 * max(1.0, bound)),
            assumptions_checked=("positive Hermitian operators", "strict positive common gap", "zero to one exponent"),
            theorem_id="PSC-VI-T26",
            message="fractional defect bound verified" if valid else "fractional defect exceeded the theorem bound",
        ),
    )


__all__ = [
    "FractionalDefectResult",
    "fractional_defect",
    "fractional_naturality_certificate",
    "fractional_power",
    "fractional_semigroup",
]
