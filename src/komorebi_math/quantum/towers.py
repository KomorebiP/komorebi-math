"""Finite bounded Hamiltonian towers and compatibility diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, Exactness
from ..dynamics._linalg import matrix_exponential
from ..exceptions import DimensionError, ValidationError
from ._validation import require_hermitian


@dataclass(frozen=True, slots=True)
class HamiltonianTowerReport:
    """Finite Cauchy and unitary convergence diagnostics for a Hamiltonian tower."""

    adjacent_differences: tuple[float, ...]
    total_variation: float
    unitary_bounds: tuple[float, ...]
    certificate: Certificate


class BoundedHamiltonianTower:
    """A finite sequence of bounded Hermitian matrices on one Hilbert space."""

    def __init__(self, hamiltonians: Iterable[Any], *, tolerance: float = 1e-10) -> None:
        values = tuple(require_hermitian(value, tolerance=tolerance, name="tower Hamiltonian") for value in hamiltonians)
        if not values:
            raise ValidationError("a Hamiltonian tower requires at least one level")
        shape = values[0].shape
        if any(value.shape != shape for value in values):
            raise DimensionError("all Hamiltonians must act on one common space")
        self.hamiltonians = values
        self.dimension = shape[0]
        self.tolerance = float(tolerance)

    def adjacent_differences(self) -> tuple[float, ...]:
        """Return spectral norms of adjacent Hamiltonian differences."""

        return tuple(float(np.linalg.norm(right - left, ord=2)) for left, right in zip(self.hamiltonians[:-1], self.hamiltonians[1:], strict=True))

    def report(self, time: float) -> HamiltonianTowerReport:
        """Return finite Cauchy diagnostics and Duhamel unitary bounds."""

        t = float(time)
        if not np.isfinite(t):
            raise ValidationError("time must be finite")
        differences = self.adjacent_differences()
        unitary_bounds = tuple(abs(t) * difference for difference in differences)
        observed = []
        for left, right in zip(self.hamiltonians[:-1], self.hamiltonians[1:], strict=True):
            observed.append(float(np.linalg.norm(matrix_exponential(-1j * t * right) - matrix_exponential(-1j * t * left), ord=2)))
        violation = max((max(0.0, value - bound) for value, bound in zip(observed, unitary_bounds, strict=True)), default=0.0)
        return HamiltonianTowerReport(
            differences,
            float(sum(differences)),
            unitary_bounds,
            Certificate(
                valid=violation <= self.tolerance * 20,
                exactness=Exactness.NUMERICAL,
                residuals={"unitary_bound_violation": violation},
                tolerance=self.tolerance * 20,
                assumptions_checked=("finite bounded Hermitian tower",),
                assumptions_unchecked=("infinite tail summability beyond supplied levels",),
                theorem_id="PSC-VII-T70",
                message="finite Hamiltonian tower obeys adjacent unitary perturbation bounds",
            ),
        )

    @property
    def final_hamiltonian(self) -> np.ndarray:
        """Return the last supplied finite tower level."""

        return self.hamiltonians[-1]


__all__ = ["BoundedHamiltonianTower", "HamiltonianTowerReport"]
