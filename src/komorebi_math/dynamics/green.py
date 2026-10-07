"""Exact one dimensional finite difference Green operators."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T49",
        title="Exact discrete Dirichlet Green reconstruction",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("uniform one dimensional Dirichlet grid",),
        verification_suite="dynamics.green",
    )
)
@dataclass(frozen=True, slots=True)
class DirichletGreen1D:
    """Exact finite difference Green operator on a uniform Dirichlet grid."""

    interior_points: int

    def __post_init__(self) -> None:
        if isinstance(self.interior_points, bool) or int(self.interior_points) != self.interior_points or self.interior_points < 1:
            raise ValidationError("interior_points must be a positive integer")

    @property
    def spacing(self) -> float:
        return 1.0 / (self.interior_points + 1)

    @property
    def grid(self) -> np.ndarray:
        return np.arange(1, self.interior_points + 1, dtype=float) * self.spacing

    @property
    def operator(self) -> np.ndarray:
        n = self.interior_points
        return (2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)) / self.spacing**2

    @property
    def continuum_kernel_samples(self) -> np.ndarray:
        x = self.grid
        return np.minimum.outer(x, x) - np.outer(x, x)

    @property
    def inverse(self) -> np.ndarray:
        return self.spacing * self.continuum_kernel_samples

    def solve(self, source: Sequence[float]) -> np.ndarray:
        value = np.asarray(source, dtype=float)
        if value.shape != (self.interior_points,):
            raise DimensionError("source vector has the wrong grid dimension")
        return self.inverse @ value

    def eigenvalues(self) -> np.ndarray:
        indices = np.arange(1, self.interior_points + 1, dtype=float)
        return 4.0 / self.spacing**2 * np.sin(indices * pi * self.spacing / 2.0) ** 2

    def eigenvalue_error_bound(self, mode: int) -> float:
        if mode < 1 or mode > self.interior_points:
            raise ValidationError("mode must index an interior eigenvalue")
        continuum = (mode * pi) ** 2
        return continuum**2 * self.spacing**2 / 12.0

    def certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        residual = float(np.linalg.norm(self.operator @ self.inverse - np.eye(self.interior_points), 2))
        valid = residual <= tolerance
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"inverse_identity": residual},
            tolerance=tolerance,
            assumptions_checked=("uniform Dirichlet grid",),
            theorem_id="PSC-VI-T49",
            message="discrete Green identity verified" if valid else "discrete Green identity failed",
        )


__all__ = ["DirichletGreen1D"]
