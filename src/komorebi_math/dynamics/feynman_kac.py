"""Finite state Feynman Kac transfer operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._linalg import as_square_matrix, matrix_exponential
from .markov import MarkovLumping

FeynmanKacKind = Literal["generator", "transition"]


def _labels(partition: Sequence[int] | Sequence[Sequence[int]], size: int) -> tuple[int, ...]:
    values = tuple(partition)
    if not values:
        raise ValidationError("partition cannot be empty")
    if isinstance(values[0], (list, tuple, np.ndarray)):
        labels = [-1] * size
        for block_index, block in enumerate(values):
            for state in block:  # type: ignore[union-attr]
                index = int(state)
                if index < 0 or index >= size or labels[index] >= 0:
                    raise ValidationError("partition blocks must contain every state exactly once")
                labels[index] = block_index
        if any(value < 0 for value in labels):
            raise ValidationError("partition blocks must contain every state exactly once")
        return tuple(labels)
    labels = tuple(int(value) for value in values)  # type: ignore[arg-type]
    if len(labels) != size:
        raise DimensionError("partition labels must match the state dimension")
    return labels


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T55",
        title="Finite state Feynman Kac transfer",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("finite column Markov model", "nonnegative killing potential"),
        verification_suite="dynamics.feynman_kac",
    )
)
@dataclass(frozen=True, slots=True)
class FiniteFeynmanKac:
    """A continuous or discrete finite Feynman Kac model on observables."""

    markov_operator: np.ndarray
    potential: np.ndarray
    kind: FeynmanKacKind = "generator"
    tolerance: float = 1e-12

    def __init__(
        self,
        markov_operator: Any,
        potential: Any,
        *,
        kind: FeynmanKacKind = "generator",
        tolerance: float = 1e-12,
    ) -> None:
        matrix = np.asarray(as_square_matrix(markov_operator, name="Markov operator"), dtype=float)
        values = np.asarray(potential, dtype=float)
        if values.shape != (matrix.shape[0],):
            raise DimensionError("potential must have one value per state")
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValidationError("potential must be finite and nonnegative")
        tol = float(tolerance)
        if not np.isfinite(tol) or tol < 0:
            raise ValidationError("tolerance must be finite and nonnegative")
        if kind == "generator":
            off = matrix.copy()
            np.fill_diagonal(off, 0.0)
            if np.min(off) < -tol or np.max(np.abs(matrix.sum(axis=0))) > tol:
                raise ValidationError("continuous Feynman Kac requires a column Markov generator")
        elif kind == "transition":
            if np.min(matrix) < -tol or np.max(np.abs(matrix.sum(axis=0) - 1.0)) > tol:
                raise ValidationError("discrete Feynman Kac requires a column stochastic transition")
        else:
            raise ValidationError("kind must be generator or transition")
        object.__setattr__(self, "markov_operator", matrix)
        object.__setattr__(self, "potential", values)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "tolerance", tol)

    @property
    def states(self) -> int:
        return int(self.markov_operator.shape[0])

    @property
    def backward_generator(self) -> np.ndarray:
        if self.kind != "generator":
            raise ValidationError("a discrete model has no continuous backward generator")
        return self.markov_operator.T - np.diag(self.potential)

    @property
    def one_step_operator(self) -> np.ndarray:
        if self.kind != "transition":
            raise ValidationError("a continuous model has no declared one step operator")
        return np.diag(np.exp(-self.potential)) @ self.markov_operator.T

    def operator(self, amount: float | int) -> np.ndarray:
        if self.kind == "generator":
            time = float(amount)
            if not np.isfinite(time) or time < 0:
                raise ValidationError("continuous propagation time must be finite and nonnegative")
            return matrix_exponential(time * self.backward_generator)
        if isinstance(amount, (bool, np.bool_)) or int(amount) != amount or amount < 0:
            raise ValidationError("discrete propagation requires a nonnegative integer step count")
        return np.linalg.matrix_power(self.one_step_operator, int(amount))

    def apply(self, observable: Any, amount: float | int) -> np.ndarray:
        value = np.asarray(observable)
        if value.ndim != 1 or value.size != self.states:
            raise DimensionError("observable has the wrong state dimension")
        return self.operator(amount) @ value

    def potential_perturbation_bound(self, other_potential: Any, time: float) -> float:
        """Return the elementary continuous time potential perturbation bound."""

        if self.kind != "generator":
            raise ValidationError("the continuous perturbation bound requires generator mode")
        other = np.asarray(other_potential, dtype=float)
        if other.shape != self.potential.shape or np.any(other < 0):
            raise DimensionError("other_potential must be a nonnegative vector of the same size")
        t = float(time)
        if not np.isfinite(t) or t < 0:
            raise ValidationError("time must be finite and nonnegative")
        generator_norm = float(np.linalg.norm(self.markov_operator.T, 2))
        potential_norm = max(float(np.max(self.potential)), float(np.max(other)))
        return float(t * np.exp(t * (generator_norm + potential_norm)) * np.max(np.abs(self.potential - other)))

    def coarse_model(
        self,
        partition: Sequence[int] | Sequence[Sequence[int]],
        *,
        tolerance: float = 1e-10,
    ) -> ComputationResult["FiniteFeynmanKac | None"]:
        labels = _labels(partition, self.states)
        lumping = MarkovLumping(self.markov_operator, labels, kind=self.kind)
        lumping_certificate = lumping.certificate(tolerance=tolerance)
        blocks = lumping.blocks
        variations = tuple(float(np.ptp(self.potential[list(block)])) for block in blocks)
        variation = max(variations, default=0.0)
        valid = lumping_certificate.valid and variation <= tolerance
        if not valid:
            return ComputationResult(
                None,
                Certificate(
                    valid=False,
                    exactness=Exactness.CONDITIONAL,
                    residuals={
                        "lumping": max(lumping_certificate.residuals.values(), default=0.0),
                        "potential_variation": variation,
                    },
                    tolerance=tolerance,
                    assumptions_checked=("finite partition",),
                    assumptions_unchecked=("exact Markov lumping", "block constant potential"),
                    theorem_id="PSC-VI-C57",
                    message="coarse Feynman Kac hypotheses are not satisfied",
                ),
            )
        coarse_operator = lumping.exact_coarse_operator(tolerance=tolerance)
        coarse_potential = np.asarray([self.potential[block[0]] for block in blocks], dtype=float)
        result = FiniteFeynmanKac(coarse_operator, coarse_potential, kind=self.kind, tolerance=tolerance)
        return ComputationResult(
            result,
            Certificate(
                valid=True,
                exactness=Exactness.NUMERICAL,
                residuals={"lumping": 0.0, "potential_variation": variation},
                tolerance=tolerance,
                assumptions_checked=("exact Markov lumping", "block constant potential"),
                theorem_id="PSC-VI-C57",
                message="coarse Feynman Kac model is exact",
            ),
        )


__all__ = ["FiniteFeynmanKac", "FeynmanKacKind"]
