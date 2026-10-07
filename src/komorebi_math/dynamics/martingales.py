"""Finite conditional expectations and martingale packet towers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _partition_labels(partition: Iterable[Iterable[int]], count: int) -> tuple[tuple[int, ...], ...]:
    blocks = tuple(tuple(int(index) for index in block) for block in partition)
    if not blocks or any(not block for block in blocks):
        raise ValidationError("partition blocks must be nonempty")
    flattened = [index for block in blocks for index in block]
    if sorted(flattened) != list(range(count)):
        raise ValidationError("partition blocks must contain every state exactly once")
    return blocks


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T17",
        title="Finite weighted conditional expectation",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("positive finite probability weights", "finite partition"),
        verification_suite="dynamics.martingales",
    )
)
@dataclass(frozen=True, slots=True)
class ConditionalExpectation:
    """Weighted conditional expectation onto block constant functions."""

    blocks: tuple[tuple[int, ...], ...]
    weights: np.ndarray

    def __init__(self, partition: Iterable[Iterable[int]], weights: Any) -> None:
        probability = np.asarray(weights, dtype=float)
        if probability.ndim != 1 or probability.size == 0:
            raise DimensionError("weights must be a nonempty one dimensional array")
        if np.any(probability <= 0) or not np.all(np.isfinite(probability)):
            raise ValidationError("conditional expectation weights must be positive and finite")
        probability = probability / np.sum(probability)
        object.__setattr__(self, "blocks", _partition_labels(partition, probability.size))
        object.__setattr__(self, "weights", probability)

    @property
    def matrix(self) -> np.ndarray:
        result = np.zeros((self.weights.size, self.weights.size))
        for block in self.blocks:
            indices = np.asarray(block, dtype=int)
            mass = float(np.sum(self.weights[indices]))
            row = self.weights[indices] / mass
            result[np.ix_(indices, indices)] = np.tile(row, (indices.size, 1))
        return result

    def apply(self, values: Any) -> np.ndarray:
        vector = np.asarray(values)
        if vector.ndim != 1 or vector.size != self.weights.size:
            raise DimensionError("function values do not match the probability space")
        return self.matrix @ vector

    def inner(self, left: Any, right: Any) -> complex:
        x = np.asarray(left)
        y = np.asarray(right)
        if x.shape != self.weights.shape or y.shape != self.weights.shape:
            raise DimensionError("inner product vectors do not match the probability space")
        return complex(np.vdot(x, self.weights * y))

    def certificate(self, *, tolerance: float = 1e-11) -> Certificate:
        matrix = self.matrix
        idempotence = float(np.linalg.norm(matrix @ matrix - matrix, ord=2))
        weighted_adjoint = np.diag(self.weights) @ matrix - matrix.T.conj() @ np.diag(self.weights)
        self_adjoint = float(np.linalg.norm(weighted_adjoint, ord=2))
        valid = max(idempotence, self_adjoint) <= tolerance
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"idempotence": idempotence, "weighted_self_adjointness": self_adjoint},
            tolerance=tolerance,
            assumptions_checked=("positive normalized weights", "finite partition"),
            theorem_id="PSC-VI-T17",
            message="weighted conditional expectation verified" if valid else "conditional expectation validation failed",
        )


def partition_refines(fine: tuple[tuple[int, ...], ...], coarse: tuple[tuple[int, ...], ...]) -> bool:
    """Return whether every fine block is contained in one coarse block."""

    coarse_lookup: dict[int, int] = {}
    for block_index, block in enumerate(coarse):
        for state in block:
            coarse_lookup[state] = block_index
    return all(len({coarse_lookup[state] for state in block}) == 1 for block in fine)


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T18",
        title="Finite martingale packet tower",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("nested finite partitions", "common positive probability weights"),
        verification_suite="dynamics.martingales",
    )
)
@dataclass(frozen=True, slots=True)
class MartingaleTower:
    """A nested family of finite conditional expectations."""

    expectations: tuple[ConditionalExpectation, ...]

    def __init__(self, partitions: Iterable[Iterable[Iterable[int]]], weights: Any) -> None:
        values = tuple(ConditionalExpectation(partition, weights) for partition in partitions)
        if not values:
            raise ValidationError("a martingale tower requires at least one partition")
        for coarse, fine in zip(values, values[1:], strict=False):
            if not partition_refines(fine.blocks, coarse.blocks):
                raise ValidationError("partitions must become finer along the martingale tower")
        object.__setattr__(self, "expectations", values)

    @property
    def weights(self) -> np.ndarray:
        return self.expectations[0].weights

    def project(self, values: Any, level: int) -> np.ndarray:
        return self.expectations[level].apply(values)

    def differences(self, values: Any) -> tuple[np.ndarray, ...]:
        vector = np.asarray(values)
        projected = tuple(expectation.apply(vector) for expectation in self.expectations)
        return (projected[0],) + tuple(
            projected[index] - projected[index - 1]
            for index in range(1, len(projected))
        )

    def energy_decomposition(self, values: Any) -> ComputationResult[tuple[float, ...]]:
        vector = np.asarray(values)
        differences = self.differences(vector)
        energies = tuple(float(np.vdot(value, self.weights * value).real) for value in differences)
        terminal = self.expectations[-1].apply(vector)
        terminal_energy = float(np.vdot(terminal, self.weights * terminal).real)
        residual = abs(sum(energies) - terminal_energy)
        valid = residual <= 1e-10
        return ComputationResult(
            energies,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"energy_decomposition": residual},
                tolerance=1e-10,
                assumptions_checked=("nested finite partitions", "positive weights"),
                theorem_id="PSC-VI-T18",
                message="martingale packet energy verified" if valid else "martingale packet energy failed",
            ),
        )

    def tower_certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        residual = 0.0
        for coarse, fine in zip(self.expectations, self.expectations[1:], strict=False):
            residual = max(
                residual,
                float(np.linalg.norm(coarse.matrix @ fine.matrix - coarse.matrix, ord=2)),
                float(np.linalg.norm(fine.matrix @ coarse.matrix - coarse.matrix, ord=2)),
            )
        valid = residual <= tolerance
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"tower_law": residual},
            tolerance=tolerance,
            assumptions_checked=("nested partition validation",),
            theorem_id="PSC-VI-T18",
            message="conditional expectation tower verified" if valid else "conditional expectation tower law failed",
        )


def blocks_from_labels(labels: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Convert arbitrary finite block labels to an ordered block partition."""

    values = tuple(int(value) for value in labels)
    if not values:
        raise ValidationError("partition labels cannot be empty")
    unique: list[int] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(tuple(index for index, value in enumerate(values) if value == label) for label in unique)


@dataclass(frozen=True, slots=True)
class ConditionalExpectationTower:
    """Convenience wrapper for a martingale tower declared by block labels."""

    tower: MartingaleTower

    def __init__(self, weights: Any, label_partitions: Iterable[Iterable[int]]) -> None:
        partitions = tuple(blocks_from_labels(labels) for labels in label_partitions)
        object.__setattr__(self, "tower", MartingaleTower(partitions, weights))

    @property
    def weights(self) -> np.ndarray:
        return self.tower.weights

    @property
    def expectations(self) -> tuple[ConditionalExpectation, ...]:
        return self.tower.expectations

    def project(self, values: Any, level: int) -> np.ndarray:
        return self.tower.project(values, level)

    def differences(self, values: Any) -> tuple[np.ndarray, ...]:
        return self.tower.differences(values)

    def energy(self, values: Any) -> ComputationResult[tuple[float, ...]]:
        return self.tower.energy_decomposition(values)

    def certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        return self.tower.tower_certificate(tolerance=tolerance)



__all__ = [
    "ConditionalExpectation",
    "ConditionalExpectationTower",
    "MartingaleTower",
    "blocks_from_labels",
    "partition_refines",
]
