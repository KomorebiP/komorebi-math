"""Projective intertwiner envelopes and commutator forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from ..._number_theory import require_integer, require_prime
from ...certificates import Certificate, Exactness
from ...exceptions import DimensionError, ValidationError
from .symplectic import AlternatingForm


@dataclass(frozen=True, slots=True)
class EnvelopeRecovery:
    """Recovered alternating form and scalar commutator diagnostics."""

    form: AlternatingForm
    scalar_residuals: Mapping[str, float]
    certificate: Certificate


class ProjectiveIntertwinerEnvelope:
    """Recover a finite commutator pairing from projective intertwiners."""

    def __init__(
        self,
        *,
        prime: int,
        labels: Iterable[Iterable[int]],
        operators: Iterable[Any],
        tolerance: float = 1e-8,
    ) -> None:
        self.prime = require_prime(prime)
        self.labels = tuple(tuple(require_integer(value, name="label entry") % self.prime for value in label) for label in labels)
        self.operators = tuple(np.asarray(operator, dtype=complex) for operator in operators)
        self.tolerance = float(tolerance)
        if not self.labels or len(self.labels) != len(self.operators):
            raise ValidationError("labels and operators must be nonempty and aligned")
        rank = len(self.labels[0])
        if rank < 1 or any(len(label) != rank for label in self.labels):
            raise DimensionError("all projective labels must have one positive dimension")
        shape = self.operators[0].shape
        if len(shape) != 2 or shape[0] != shape[1] or any(operator.shape != shape for operator in self.operators):
            raise DimensionError("all projective intertwiners must be square matrices of one dimension")
        for operator in self.operators:
            np.linalg.inv(operator)
        self.rank = rank

    def _operator_by_label(self) -> dict[tuple[int, ...], np.ndarray]:
        table = dict(zip(self.labels, self.operators, strict=True))
        zero = (0,) * self.rank
        if zero not in table:
            table[zero] = np.eye(self.operators[0].shape[0], dtype=complex)
        return table

    def recover(self) -> EnvelopeRecovery:
        """Recover the alternating commutator matrix from basis labels."""

        table = self._operator_by_label()
        basis = [tuple(1 if i == j else 0 for i in range(self.rank)) for j in range(self.rank)]
        if any(vector not in table for vector in basis):
            raise ValidationError("operators for every standard basis label are required")
        root = np.exp(2j * np.pi / self.prime)
        form = np.zeros((self.rank, self.rank), dtype=int)
        residuals: dict[str, float] = {}
        valid = True
        dimension = self.operators[0].shape[0]
        identity = np.eye(dimension, dtype=complex)
        for i, left_label in enumerate(basis):
            left = table[left_label]
            for j, right_label in enumerate(basis):
                right = table[right_label]
                commutator = left @ right @ np.linalg.inv(left) @ np.linalg.inv(right)
                scalar = np.trace(commutator) / dimension
                scalar_residual = float(np.linalg.norm(commutator - scalar * identity))
                residuals[f"scalar_{i}_{j}"] = scalar_residual
                valid = valid and scalar_residual <= self.tolerance
                candidates = [abs(scalar - root**power) for power in range(self.prime)]
                exponent = int(np.argmin(candidates))
                phase_residual = float(candidates[exponent])
                residuals[f"phase_{i}_{j}"] = phase_residual
                valid = valid and phase_residual <= self.tolerance
                form[i, j] = exponent
        alternating = AlternatingForm(form, self.prime)
        return EnvelopeRecovery(
            form=alternating,
            scalar_residuals=residuals,
            certificate=Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals=dict(residuals),
                tolerance=self.tolerance,
                assumptions_checked=("projective basis intertwiners", "scalar commutators"),
                diagnostics={"rank": self.rank, "form_rank": alternating.rank},
                theorem_id="PSC-XI-PROJECTIVE-ENVELOPE",
                message="projective commutator envelope recovered" if valid else "projective commutators are not scalar prime phases",
            ),
        )


__all__ = ["EnvelopeRecovery", "ProjectiveIntertwinerEnvelope"]
