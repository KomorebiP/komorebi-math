"""Nonlinear refinement defects and stability bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _apply_linear(map_value: Any, vector: Any) -> Any:
    return map_value(vector) if callable(map_value) else np.asarray(map_value) @ np.asarray(vector)


@documented(
    ProvenanceRecord(
        identifier="PSC-III-D10",
        title="Nonlinear refinement defect",
        status=ClaimStatus.DEFINITION,
        source="Paper III",
        assumptions=("fine and coarse maps", "linear coarsening maps"),
        verification_suite="nonlinear.defects",
    )
)
@dataclass(slots=True)
class NonlinearRefinementDefect:
    """Pointwise defect ``C_Y F_f(x) - F_c(C_X x)``."""

    fine_map: Callable[[Any], Any]
    coarse_map: Callable[[Any], Any]
    coarsen_source: Any
    coarsen_target: Any

    def evaluate(self, point: Any) -> np.ndarray:
        fine_value = self.fine_map(point)
        coarse_input = _apply_linear(self.coarsen_source, point)
        return np.asarray(_apply_linear(self.coarsen_target, fine_value)) - np.asarray(self.coarse_map(coarse_input))

    def norm(self, point: Any) -> float:
        return float(np.linalg.norm(self.evaluate(point)))

    @documented(
        ProvenanceRecord(
            identifier="PSC-III-T11",
            title="Exact nonlinear defect under composition",
            status=ClaimStatus.PROVED,
            source="Paper III",
            assumptions=("composable fine and coarse maps",),
            verification_suite="nonlinear.defects",
        )
    )
    def composition_identity(
        self,
        outer_fine: Callable[[Any], Any],
        outer_coarse: Callable[[Any], Any],
        outer_coarsen_target: Any,
        point: Any,
    ) -> ComputationResult[np.ndarray]:
        inner_fine = self.fine_map(point)
        inner_coarse_input = _apply_linear(self.coarsen_source, point)
        inner_coarse = self.coarse_map(inner_coarse_input)
        inner_defect = self.evaluate(point)
        outer_defect = np.asarray(_apply_linear(outer_coarsen_target, outer_fine(inner_fine))) - np.asarray(
            outer_coarse(_apply_linear(self.coarsen_target, inner_fine))
        )
        formula = outer_defect + np.asarray(outer_coarse(inner_coarse + inner_defect)) - np.asarray(outer_coarse(inner_coarse))
        # Compute the true composite defect independently.
        true_direct = np.asarray(_apply_linear(outer_coarsen_target, outer_fine(self.fine_map(point)))) - np.asarray(
            outer_coarse(self.coarse_map(_apply_linear(self.coarsen_source, point)))
        )
        residual = float(np.linalg.norm(formula - true_direct))
        return ComputationResult(
            formula,
            Certificate(
                valid=residual <= 1e-10,
                exactness=Exactness.NUMERICAL,
                residuals={"composition_identity": residual},
                tolerance=1e-10,
                assumptions_checked=("pointwise map evaluation",),
                theorem_id="PSC-III-T11",
                message="nonlinear composition defect identity verified" if residual <= 1e-10 else "composition defect identity failed",
            ),
        )

    def iterated_orbit_bound(
        self,
        point: Any,
        *,
        iterations: int,
        coarse_lipschitz: float,
    ) -> float:
        if iterations < 0:
            raise ValueError("iteration count must be nonnegative")
        if coarse_lipschitz < 0:
            raise ValueError("Lipschitz constant must be nonnegative")
        current = point
        errors: list[float] = []
        for _ in range(iterations):
            errors.append(self.norm(current))
            current = self.fine_map(current)
        return sum(coarse_lipschitz ** (iterations - 1 - index) * error for index, error in enumerate(errors))

    def fixed_point_bound(
        self,
        fine_fixed_point: Any,
        *,
        coarse_contraction: float,
        verify_fixed_point: bool = True,
        tolerance: float = 1e-10,
    ) -> float:
        """Bound the coarse fixed point error from a verified fine fixed point."""

        if not 0 <= coarse_contraction < 1:
            raise ValueError("coarse contraction constant must lie in [0, 1)")
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        if verify_fixed_point:
            point = np.asarray(fine_fixed_point)
            residual = float(np.linalg.norm(np.asarray(self.fine_map(point)) - point))
            if residual > tolerance:
                raise ValidationError(
                    "the supplied point is not a fine fixed point within the declared tolerance"
                )
        return self.norm(fine_fixed_point) / (1 - coarse_contraction)


__all__ = ["NonlinearRefinementDefect"]
