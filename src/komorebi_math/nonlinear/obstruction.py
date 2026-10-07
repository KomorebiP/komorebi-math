"""Finite dimensional cokernel obstruction computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backends import LinearAlgebraBackend, get_backend
from .._residuals import value_audit, finite_tolerance
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@dataclass(frozen=True, slots=True)
class ObstructionResult:
    """Computed obstruction coordinates and correction space."""

    vanishes: bool
    obstruction_coordinates: tuple[Any, ...]
    cokernel_dimension: int
    correction: Any | None
    kernel_basis: tuple[Any, ...]
    residual: float
    certificate: Certificate

    def require_correction(self) -> Any:
        self.certificate.require()
        if not self.vanishes or self.correction is None:
            raise ValidationError("the obstruction class is nonzero")
        return self.correction


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D13",
        title="Linear correction datum and cokernel obstruction",
        status=ClaimStatus.DEFINITION,
        source="Paper III",
        assumptions=("finite dimensional linear correction map",),
        verification_suite="nonlinear.obstruction",
    )
)
class LinearObstructionProblem:
    """Compute the obstruction to solving ``D u = -defect``."""

    def __init__(
        self,
        correction_map: Any,
        defect: Any,
        *,
        backend: str | LinearAlgebraBackend | None = None,
        tolerance: float = 1e-10,
    ) -> None:
        self.backend = get_backend(backend)
        self.correction_map = self.backend.as_matrix(correction_map)
        self.defect = self.backend.as_vector(defect)
        if self.backend.shape(self.correction_map)[0] != self.backend.shape(self.defect)[0]:
            raise DimensionError("defect dimension must equal the correction map codomain")
        self.tolerance = float(tolerance)

    @documented(
        ProvenanceRecord(
            identifier="PSC-III-T14",
            title="Cokernel obstruction criterion",
            status=ClaimStatus.PROVED,
            source="Paper III",
            assumptions=("finite dimensional correction problem",),
            verification_suite="nonlinear.obstruction",
        )
    )
    def solve(self) -> ObstructionResult:
        if self.backend.name == "sympy":
            return self._solve_exact()
        return self._solve_numerical()

    def _solve_exact(self) -> ObstructionResult:
        import sympy as sp

        matrix = self.correction_map
        target = -self.defect
        augmented = matrix.row_join(target)
        rank = int(matrix.rank())
        augmented_rank = int(augmented.rank())
        vanishes = rank == augmented_rank
        left_nullspace = tuple(matrix.T.nullspace())
        coordinates = tuple((basis.T * self.defect)[0] for basis in left_nullspace)
        kernel = tuple(matrix.nullspace())
        correction = None
        residual = 0.0
        if vanishes:
            solution, parameters = matrix.gauss_jordan_solve(target)
            substitutions = {parameter: 0 for parameter in parameters}
            correction = solution.subs(substitutions)
            residual_matrix = matrix * correction + self.defect
            residual = 0.0 if residual_matrix == sp.zeros(matrix.rows, 1) else float(residual_matrix.norm())
        valid = all(coordinate == 0 for coordinate in coordinates) == vanishes
        certificate = Certificate(
            valid=bool(valid and residual == 0.0),
            exactness=Exactness.EXACT,
            residuals={"correction_residual": residual},
            assumptions_checked=("exact rank comparison", "left nullspace quotient coordinates", "kernel basis"),
            diagnostics={"vanishes": vanishes, "rank": rank, "cokernel_dimension": matrix.rows - rank},
            theorem_id="PSC-III-T14",
            message="cokernel obstruction computed exactly",
        )
        return ObstructionResult(
            vanishes,
            coordinates,
            matrix.rows - rank,
            correction,
            kernel,
            residual,
            certificate,
        )

    def _solve_numerical(self) -> ObstructionResult:
        matrix = np.asarray(self.correction_map)
        defect = np.asarray(self.defect).reshape(-1)
        correction, _, rank, _ = np.linalg.lstsq(matrix, -defect, rcond=None)
        residual_vector = matrix @ correction + defect
        residual = float(np.linalg.norm(residual_vector))
        vanishes = residual <= self.tolerance
        left_nullspace = self.backend.nullspace(self.backend.adjoint(matrix), tolerance=self.tolerance)
        coordinates = tuple(complex(np.vdot(vector, defect)) for vector in left_nullspace)
        kernel = tuple(self.backend.nullspace(matrix, tolerance=self.tolerance))
        coordinate_norm = max((abs(value) for value in coordinates), default=0.0)
        valid = (coordinate_norm <= 10 * self.tolerance) == vanishes
        certificate = Certificate(
            valid=bool(valid),
            exactness=Exactness.NUMERICAL,
            residuals={"correction_residual": residual, "obstruction_coordinate_norm": float(coordinate_norm)},
            tolerance=self.tolerance,
            assumptions_checked=("least squares correction", "orthogonal cokernel coordinates", "numerical rank"),
            diagnostics={"vanishes": vanishes, "rank": int(rank), "cokernel_dimension": matrix.shape[0] - int(rank)},
            theorem_id="PSC-III-T14",
            message="cokernel obstruction computed numerically",
        )
        return ObstructionResult(
            vanishes,
            coordinates,
            matrix.shape[0] - int(rank),
            correction if vanishes else None,
            kernel,
            residual,
            certificate,
        )


@dataclass(frozen=True, slots=True)
class ObstructionSquare:
    """A commuting square inducing a map between cokernels."""

    source_correction: Any
    target_correction: Any
    source_map: Any
    target_map: Any
    backend: str | LinearAlgebraBackend | None = None
    tolerance: float = 1e-10

    def certificate(self) -> Certificate:
        tol = finite_tolerance(self.tolerance)
        algebra = get_backend(self.backend)
        source_correction = algebra.as_matrix(self.source_correction)
        target_correction = algebra.as_matrix(self.target_correction)
        source_map = algebra.as_matrix(self.source_map)
        target_map = algebra.as_matrix(self.target_map)
        source_rows, source_columns = algebra.shape(source_correction)
        target_rows, target_columns = algebra.shape(target_correction)
        if algebra.shape(source_map) != (target_columns, source_columns):
            raise DimensionError("source map has the wrong dimensions for the correction square")
        if algebra.shape(target_map) != (target_rows, source_rows):
            raise DimensionError("target map has the wrong dimensions for the correction square")
        residual_matrix = algebra.subtract(
            algebra.matmul(target_map, source_correction),
            algebra.matmul(target_correction, source_map),
        )
        audit = value_audit(exact=algebra.exact, tolerance=tol)
        residual = audit.check_value("commuting_square", residual_matrix)
        valid = audit.valid
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT if algebra.exact else Exactness.NUMERICAL,
            residuals={"commuting_square": residual},
            tolerance=None if algebra.exact else tol,
            assumptions_checked=("linear square dimensions",),
            assumptions_unchecked=audit.unresolved,
            diagnostics={"residual_decisions": audit.decisions},
            theorem_id="PSC-III-T15",
            message="commuting square induces a cokernel map" if valid else "obstruction square does not commute",
        )

    def transport_defect(self, defect: Any) -> Any:
        self.certificate().require()
        algebra = get_backend(self.backend)
        return algebra.matmul(algebra.as_matrix(self.target_map), algebra.as_vector(defect))


__all__ = ["LinearObstructionProblem", "ObstructionResult", "ObstructionSquare"]
