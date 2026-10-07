"""Finite Markov operators, aggregation maps, and exact lumping diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import AssumptionError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._linalg import as_square_matrix, matrix_exponential

MarkovKind = Literal["generator", "transition"]


def normalize_partition(
    partition: Sequence[int] | Iterable[Iterable[int]],
    count: int,
) -> tuple[tuple[int, ...], ...]:
    """Normalize labels or explicit blocks into an ordered finite partition."""

    size = int(count)
    if size <= 0:
        raise ValidationError("a Markov partition requires a positive state count")
    raw = tuple(partition)
    if not raw:
        raise ValidationError("partition cannot be empty")
    if len(raw) == size and all(np.isscalar(value) for value in raw):
        labels = tuple(int(value) for value in raw)
        order: list[int] = []
        for label in labels:
            if label not in order:
                order.append(label)
        return tuple(tuple(index for index, label in enumerate(labels) if label == value) for value in order)
    try:
        blocks = tuple(tuple(int(index) for index in block) for block in raw)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError("partition must be state labels or explicit blocks") from exc
    if any(not block for block in blocks):
        raise ValidationError("partition blocks must be nonempty")
    flattened = [index for block in blocks for index in block]
    if sorted(flattened) != list(range(size)):
        raise ValidationError("partition blocks must contain every state exactly once")
    return blocks


@dataclass(frozen=True, slots=True)
class AggregationMap:
    """The block mass map and its uniform right inverse."""

    blocks: tuple[tuple[int, ...], ...]
    state_count: int

    def __init__(
        self,
        partition: Sequence[int] | Iterable[Iterable[int]],
        state_count: int | None = None,
    ) -> None:
        raw = tuple(partition)
        if state_count is None:
            if raw and all(np.isscalar(value) for value in raw):
                state_count = len(raw)
            else:
                flattened = [int(index) for block in raw for index in block]  # type: ignore[union-attr]
                state_count = max(flattened, default=-1) + 1
        blocks = normalize_partition(raw, int(state_count))
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "state_count", int(state_count))

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def labels(self) -> tuple[int, ...]:
        labels = [0] * self.state_count
        for block_index, block in enumerate(self.blocks):
            for state in block:
                labels[state] = block_index
        return tuple(labels)

    @property
    def matrix(self) -> np.ndarray:
        result = np.zeros((self.block_count, self.state_count), dtype=float)
        for block_index, block in enumerate(self.blocks):
            result[block_index, list(block)] = 1.0
        return result

    @property
    def uniform_section(self) -> np.ndarray:
        result = np.zeros((self.state_count, self.block_count), dtype=float)
        for block_index, block in enumerate(self.blocks):
            result[list(block), block_index] = 1.0 / len(block)
        return result

    def kernel_basis(self) -> np.ndarray:
        columns: list[np.ndarray] = []
        for block in self.blocks:
            anchor = block[0]
            for state in block[1:]:
                vector = np.zeros(self.state_count)
                vector[state] = 1.0
                vector[anchor] = -1.0
                columns.append(vector)
        return np.column_stack(columns) if columns else np.zeros((self.state_count, 0))

    def aggregate(self, values: Any) -> np.ndarray:
        vector = np.asarray(values)
        if vector.ndim != 1 or vector.size != self.state_count:
            raise DimensionError("vector dimension does not match the aggregation map")
        return self.matrix @ vector

    def lift_uniform(self, block_values: Any) -> np.ndarray:
        vector = np.asarray(block_values)
        if vector.ndim != 1 or vector.size != self.block_count:
            raise DimensionError("block vector dimension does not match the aggregation map")
        return self.uniform_section @ vector


def aggregation_matrix(partition: Sequence[int] | Iterable[Iterable[int]], count: int | None = None) -> np.ndarray:
    """Return the block mass map for column probability vectors."""

    return AggregationMap(partition, count).matrix


def _validate_markov_operator(operator: Any, kind: MarkovKind, tolerance: float) -> np.ndarray:
    matrix = np.asarray(as_square_matrix(np.asarray(operator), name="Markov operator"), dtype=float)
    if not np.isfinite(matrix).all():
        raise ValidationError("Markov operator entries must be finite")
    if kind == "generator":
        off = matrix.copy()
        np.fill_diagonal(off, 0.0)
        if np.min(off) < -tolerance:
            raise ValidationError("a Markov generator must have nonnegative off diagonal entries")
        if np.max(np.abs(matrix.sum(axis=0))) > tolerance:
            raise ValidationError("columns of a Markov generator must sum to zero")
    elif kind == "transition":
        if np.min(matrix) < -tolerance:
            raise ValidationError("a Markov transition must be nonnegative")
        if np.max(np.abs(matrix.sum(axis=0) - 1.0)) > tolerance:
            raise ValidationError("columns of a Markov transition must sum to one")
    else:
        raise ValidationError("kind must be 'generator' or 'transition'")
    return matrix


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-D18",
        title="Finite Markov operator",
        status=ClaimStatus.DEFINITION,
        source="Paper VI",
        assumptions=("column probability convention",),
        verification_suite="dynamics.markov",
    )
)
@dataclass(frozen=True, slots=True)
class FiniteMarkovChain:
    """A finite continuous or discrete Markov operator on column probabilities."""

    operator: np.ndarray
    kind: MarkovKind = "generator"
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValidationError("tolerance must be finite and nonnegative")
        kind = str(self.kind)
        matrix = _validate_markov_operator(self.operator, kind, tolerance)  # type: ignore[arg-type]
        object.__setattr__(self, "operator", matrix)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def states(self) -> int:
        return int(self.operator.shape[0])

    @property
    def generator(self) -> np.ndarray:
        if self.kind != "generator":
            raise ValidationError("this Markov chain was declared as a transition matrix")
        return self.operator

    @property
    def transition_matrix(self) -> np.ndarray:
        if self.kind != "transition":
            raise ValidationError("this Markov chain was declared as a generator")
        return self.operator

    def evolution(self, amount: float | int) -> np.ndarray:
        if self.kind == "generator":
            value = float(amount)
            if not np.isfinite(value) or value < 0:
                raise ValidationError("continuous time must be finite and nonnegative")
            return np.real_if_close(matrix_exponential(value * self.operator), tol=1000)
        if isinstance(amount, (bool, np.bool_)) or int(amount) != amount or int(amount) < 0:
            raise ValidationError("discrete evolution requires a nonnegative integer step count")
        return np.linalg.matrix_power(self.operator, int(amount))

    def evolve(self, distribution: Any, amount: float | int) -> np.ndarray:
        probability = np.asarray(distribution, dtype=float)
        if probability.ndim != 1 or probability.size != self.states:
            raise DimensionError("distribution dimension does not match the chain")
        if np.min(probability) < -self.tolerance or abs(float(probability.sum()) - 1.0) > self.tolerance:
            raise ValidationError("distribution must be nonnegative and sum to one")
        result = np.asarray(self.evolution(amount) @ probability, dtype=float)
        result[np.abs(result) < 10 * self.tolerance] = 0.0
        return result

    def stationary_distribution(self) -> ComputationResult[np.ndarray]:
        equation = self.operator if self.kind == "generator" else self.operator - np.eye(self.states)
        augmented = np.vstack((equation, np.ones((1, self.states))))
        target = np.concatenate((np.zeros(self.states), [1.0]))
        solution, _, _, _ = np.linalg.lstsq(augmented, target, rcond=None)
        residual = float(np.linalg.norm(equation @ solution, 2))
        mass_residual = abs(float(solution.sum()) - 1.0)
        negativity = max(0.0, float(-np.min(solution)))
        tolerance = max(self.tolerance, 1e-10)
        valid = max(residual, mass_residual, negativity) <= tolerance
        return ComputationResult(
            solution,
            Certificate(
                valid=valid,
                exactness=Exactness.NUMERICAL,
                residuals={"stationarity": residual, "mass": mass_residual, "negativity": negativity},
                tolerance=tolerance,
                assumptions_checked=("finite Markov operator",),
                theorem_id="PSC-VI-C22",
                message="stationary distribution computed" if valid else "stationary distribution check failed",
            ),
        )


@dataclass(frozen=True, slots=True)
class ContinuousTimeMarkovChain:
    """Compatibility wrapper for a finite column Markov generator."""

    generator: np.ndarray
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        chain = FiniteMarkovChain(self.generator, "generator", self.tolerance)
        object.__setattr__(self, "generator", chain.operator)
        object.__setattr__(self, "tolerance", chain.tolerance)

    @property
    def states(self) -> int:
        return int(self.generator.shape[0])

    def transition(self, time: float) -> np.ndarray:
        return FiniteMarkovChain(self.generator, "generator", self.tolerance).evolution(time)

    def evolve(self, distribution: Any, time: float) -> np.ndarray:
        return FiniteMarkovChain(self.generator, "generator", self.tolerance).evolve(distribution, time)

    def stationary_distribution(self) -> ComputationResult[np.ndarray]:
        return FiniteMarkovChain(self.generator, "generator", self.tolerance).stationary_distribution()


def coordinate_lumpability_residual(
    operator: Any,
    partition: Sequence[int] | Iterable[Iterable[int]],
    *,
    kind: MarkovKind = "generator",
) -> float:
    """Return the maximum discrepancy of coarse rates inside source blocks."""

    chain = FiniteMarkovChain(np.asarray(operator), kind)
    aggregation = AggregationMap(partition, chain.states)
    coarse_rows = aggregation.matrix @ chain.operator
    residual = 0.0
    for block in aggregation.blocks:
        reference = coarse_rows[:, block[0]]
        for state in block[1:]:
            residual = max(residual, float(np.max(np.abs(coarse_rows[:, state] - reference))))
    return residual


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-T21",
        title="Exact Markov lumpability criterion",
        status=ClaimStatus.PROVED,
        source="Paper VI",
        assumptions=("finite Markov operator", "fixed state partition"),
        verification_suite="dynamics.markov",
    )
)
@dataclass(frozen=True, slots=True)
class MarkovLumping:
    """A finite state aggregation for a column convention Markov operator."""

    chain: FiniteMarkovChain
    aggregation_map: AggregationMap

    def __init__(
        self,
        operator: Any | FiniteMarkovChain | ContinuousTimeMarkovChain,
        partition: Sequence[int] | Iterable[Iterable[int]],
        *,
        kind: MarkovKind | None = None,
    ) -> None:
        if isinstance(operator, FiniteMarkovChain):
            chain = operator
            if kind is not None and kind != chain.kind:
                raise ValidationError("declared kind disagrees with the supplied FiniteMarkovChain")
        elif isinstance(operator, ContinuousTimeMarkovChain):
            if kind not in (None, "generator"):
                raise ValidationError("ContinuousTimeMarkovChain requires generator kind")
            chain = FiniteMarkovChain(operator.generator, "generator", operator.tolerance)
        else:
            chain = FiniteMarkovChain(np.asarray(operator), "generator" if kind is None else kind)
        object.__setattr__(self, "chain", chain)
        object.__setattr__(self, "aggregation_map", AggregationMap(partition, chain.states))

    @property
    def partition(self) -> tuple[int, ...]:
        return self.aggregation_map.labels

    @property
    def blocks(self) -> tuple[tuple[int, ...], ...]:
        return self.aggregation_map.blocks

    @property
    def aggregation(self) -> np.ndarray:
        return self.aggregation_map.matrix

    @property
    def obstruction(self) -> np.ndarray:
        return self.aggregation @ self.chain.operator @ self.aggregation_map.kernel_basis()

    def is_exact(self, *, tolerance: float = 1e-10) -> bool:
        return max(
            float(np.linalg.norm(self.obstruction, 2)),
            coordinate_lumpability_residual(self.chain.operator, self.blocks, kind=self.chain.kind),
        ) <= tolerance

    def approximate_coarse_operator(self) -> np.ndarray:
        return self.aggregation @ self.chain.operator @ self.aggregation_map.uniform_section

    def exact_coarse_operator(self, *, tolerance: float = 1e-10) -> np.ndarray:
        if not self.is_exact(tolerance=tolerance):
            raise AssumptionError("the Markov operator is not exactly lumpable through this partition")
        result = np.zeros((len(self.blocks), len(self.blocks)), dtype=float)
        coarse_rows = self.aggregation @ self.chain.operator
        for source, block in enumerate(self.blocks):
            result[:, source] = coarse_rows[:, block[0]]
        return result

    @property
    def coarse_operator(self) -> np.ndarray:
        return self.exact_coarse_operator()

    def coarse_generator(self, *, tolerance: float = 1e-10) -> np.ndarray:
        if self.chain.kind != "generator":
            raise ValidationError("coarse_generator is available only for a generator")
        return self.exact_coarse_operator(tolerance=tolerance)

    def coarse_transition(self, *, tolerance: float = 1e-10) -> np.ndarray:
        if self.chain.kind != "transition":
            raise ValidationError("coarse_transition is available only for a transition matrix")
        return self.exact_coarse_operator(tolerance=tolerance)

    def coarse_chain(self, *, tolerance: float = 1e-10) -> FiniteMarkovChain | ContinuousTimeMarkovChain:
        coarse = self.exact_coarse_operator(tolerance=tolerance)
        if self.chain.kind == "generator":
            return ContinuousTimeMarkovChain(coarse, self.chain.tolerance)
        return FiniteMarkovChain(coarse, "transition", self.chain.tolerance)

    def certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        kernel_residual = float(np.linalg.norm(self.obstruction, 2))
        coordinate_residual = coordinate_lumpability_residual(
            self.chain.operator,
            self.blocks,
            kind=self.chain.kind,
        )
        valid = max(kernel_residual, coordinate_residual) <= tolerance
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"kernel_obstruction": kernel_residual, "coordinate_residual": coordinate_residual},
            tolerance=tolerance,
            assumptions_checked=("column Markov convention", "finite partition"),
            theorem_id="PSC-VI-T21",
            message="Markov operator is exactly lumpable" if valid else "Markov operator is not exactly lumpable",
        )

    def aggregate(self, distribution: Any) -> np.ndarray:
        return self.aggregation_map.aggregate(distribution)

    def evolution_defect(self, amount: float | int, *, exact_coarse: bool = False) -> np.ndarray:
        coarse = self.exact_coarse_operator() if exact_coarse else self.approximate_coarse_operator()
        coarse_chain = FiniteMarkovChain(coarse, self.chain.kind)
        return self.aggregation @ self.chain.evolution(amount) - coarse_chain.evolution(amount) @ self.aggregation

    def semigroup_defect(self, time: float, *, use_exact_only: bool = False) -> np.ndarray:
        if self.chain.kind != "generator":
            raise ValidationError("semigroup_defect requires a continuous generator")
        return self.evolution_defect(time, exact_coarse=use_exact_only)

    def error_bound(self, time: float) -> float:
        if self.chain.kind != "generator":
            raise ValidationError("the continuous time error bound requires a generator")
        t = float(time)
        if not np.isfinite(t) or t < 0:
            raise ValidationError("time must be finite and nonnegative")
        coarse = self.approximate_coarse_operator()
        defect = self.aggregation @ self.chain.operator - coarse @ self.aggregation
        exponent = max(float(np.linalg.norm(self.chain.operator, 2)), float(np.linalg.norm(coarse, 2)))
        return float(t * np.exp(t * exponent) * np.linalg.norm(defect, 2))


__all__ = [
    "AggregationMap",
    "ContinuousTimeMarkovChain",
    "FiniteMarkovChain",
    "MarkovKind",
    "MarkovLumping",
    "aggregation_matrix",
    "coordinate_lumpability_residual",
    "normalize_partition",
]
