"""Finite constraint closure and refinement anomaly diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import as_square_matrix, require_hermitian


@dataclass(frozen=True, slots=True)
class ConstraintClosureReport:
    """Least squares structure coefficients and commutator residuals."""

    coefficients: tuple[tuple[np.ndarray, ...], ...]
    residuals: np.ndarray
    maximum_residual: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VII-T64",
        title="Constraint quotient obstruction",
        status=ClaimStatus.PROVED,
        source="Paper VII",
        assumptions=("finite declared constraint span",),
        verification_suite="quantum.constraints",
    )
)
def constraint_closure(
    constraints: Iterable[Any],
    *,
    tolerance: float = 1e-10,
) -> ConstraintClosureReport:
    """Project every commutator onto the declared complex linear constraint span."""

    values = tuple(require_hermitian(value, tolerance=tolerance, name="constraint") for value in constraints)
    if not values:
        raise ValidationError("at least one constraint is required")
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise DimensionError("all constraints must act on one common space")
    basis = np.column_stack([value.reshape(-1, order="F") for value in values])
    coefficients: list[tuple[np.ndarray, ...]] = []
    residuals = np.zeros((len(values), len(values)), dtype=float)
    for left, first in enumerate(values):
        row: list[np.ndarray] = []
        for right, second in enumerate(values):
            commutator = first @ second - second @ first
            solution, *_ = np.linalg.lstsq(basis, commutator.reshape(-1, order="F"), rcond=None)
            reconstructed = (basis @ solution).reshape(shape, order="F")
            residuals[left, right] = np.linalg.norm(commutator - reconstructed, ord=2)
            row.append(solution)
        coefficients.append(tuple(row))
    maximum = float(np.max(residuals))
    scale = max(1.0, max(float(np.linalg.norm(value, ord=2)) for value in values))
    return ConstraintClosureReport(
        tuple(coefficients),
        residuals,
        maximum,
        Certificate(
            valid=maximum <= tolerance * scale,
            exactness=Exactness.NUMERICAL,
            residuals={"closure_obstruction": maximum},
            tolerance=tolerance * scale,
            assumptions_checked=("finite complex linear constraint span",),
            theorem_id="PSC-VII-T64",
            message="constraint commutators close in the declared span",
        ),
    )


def transported_constraint_defect(
    fine_constraints: Iterable[Any],
    coarse_constraints: Iterable[Any],
    coarsening: Any,
) -> tuple[np.ndarray, ...]:
    """Return ``C*C_fine - C_coarse*C`` for matching constraint families."""

    fine = tuple(as_square_matrix(value, name="fine constraint") for value in fine_constraints)
    coarse = tuple(as_square_matrix(value, name="coarse constraint") for value in coarse_constraints)
    c = np.asarray(coarsening, dtype=complex)
    if not fine or len(fine) != len(coarse):
        raise ValidationError("fine and coarse constraint families must have one common nonzero length")
    fine_dimension = fine[0].shape[0]
    coarse_dimension = coarse[0].shape[0]
    if c.shape != (coarse_dimension, fine_dimension):
        raise DimensionError("coarsening map has incompatible dimensions")
    if any(value.shape != (fine_dimension, fine_dimension) for value in fine):
        raise DimensionError("fine constraints have inconsistent dimensions")
    if any(value.shape != (coarse_dimension, coarse_dimension) for value in coarse):
        raise DimensionError("coarse constraints have inconsistent dimensions")
    return tuple(c @ fine_value - coarse_value @ c for fine_value, coarse_value in zip(fine, coarse, strict=True))


__all__ = ["ConstraintClosureReport", "constraint_closure", "transported_constraint_defect"]
