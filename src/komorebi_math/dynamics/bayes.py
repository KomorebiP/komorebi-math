"""Finite Bayesian updates, sufficient partitions, flows, and coarse filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import AssumptionError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .markov import MarkovLumping


def _probability(values: Any, name: str = "probability") -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size < 1:
        raise DimensionError(f"{name} must be a nonempty one dimensional array")
    if np.any(vector < 0) or not np.all(np.isfinite(vector)):
        raise ValidationError(f"{name} must contain finite nonnegative values")
    mass = float(vector.sum())
    if mass <= 0:
        raise ValidationError(f"{name} must have positive total mass")
    return vector / mass


def _likelihood(values: Any, size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size != size:
        raise DimensionError("likelihood dimension does not match the prior")
    if np.any(vector < 0) or not np.all(np.isfinite(vector)):
        raise ValidationError("likelihood must contain finite nonnegative values")
    if not np.any(vector > 0):
        raise ValidationError("likelihood must be positive on at least one state")
    return vector


def _partition(partition: Sequence[int], size: int) -> tuple[int, ...]:
    if len(partition) != size:
        raise DimensionError("partition length must match the state space")
    labels = tuple(int(value) for value in partition)
    unique = {value: index for index, value in enumerate(sorted(set(labels)))}
    return tuple(unique[value] for value in labels)


@documented(
    ProvenanceRecord(
        identifier="PSC-VI-D33",
        title="Finite Bayesian update",
        status=ClaimStatus.DEFINITION,
        source="Paper VI",
        assumptions=("finite prior", "nonnegative likelihood with positive evidence"),
        verification_suite="dynamics.bayes",
    )
)
@dataclass(frozen=True, slots=True)
class BayesianUpdate:
    """A normalized finite Bayesian update."""

    prior: np.ndarray
    likelihood: np.ndarray

    def __init__(self, prior: Any, likelihood: Any) -> None:
        probability = _probability(prior, "prior")
        weights = _likelihood(likelihood, probability.size)
        object.__setattr__(self, "prior", probability)
        object.__setattr__(self, "likelihood", weights)

    @property
    def evidence(self) -> float:
        return float(self.likelihood @ self.prior)

    @property
    def posterior(self) -> np.ndarray:
        evidence = self.evidence
        if evidence <= 0:
            raise AssumptionError("the likelihood has zero evidence under the prior")
        return self.likelihood * self.prior / evidence

    def perturbation_bound(self, other_likelihood: Any) -> float:
        other = _likelihood(other_likelihood, self.prior.size)
        second_evidence = float(other @ self.prior)
        denominator = max(self.evidence, second_evidence)
        if denominator <= 0:
            raise AssumptionError("at least one compared likelihood must have positive evidence")
        return float(2.0 * np.sum(self.prior * np.abs(self.likelihood - other)) / denominator)


@dataclass(frozen=True, slots=True)
class BayesianPartition:
    """A fixed finite coarse partition for Bayesian updates."""

    labels: tuple[int, ...]

    def __init__(self, partition: Sequence[int]) -> None:
        if not partition:
            raise ValidationError("partition cannot be empty")
        object.__setattr__(self, "labels", _partition(partition, len(partition)))

    @property
    def blocks(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(index for index, value in enumerate(self.labels) if value == block)
            for block in range(max(self.labels) + 1)
        )

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def aggregate(self, probability: Any) -> np.ndarray:
        vector = _probability(probability)
        if vector.size != len(self.labels):
            raise DimensionError("probability dimension does not match the partition")
        return np.asarray([vector[list(block)].sum() for block in self.blocks], dtype=float)

    def effective_likelihood(self, prior: Any, likelihood: Any) -> np.ndarray:
        probability = _probability(prior, "prior")
        weights = _likelihood(likelihood, probability.size)
        if probability.size != len(self.labels):
            raise DimensionError("prior dimension does not match the partition")
        coarse = self.aggregate(probability)
        effective = np.zeros(self.block_count)
        for block_index, block in enumerate(self.blocks):
            if coarse[block_index] > 0:
                effective[block_index] = float(weights[list(block)] @ probability[list(block)] / coarse[block_index])
        return effective

    @documented(
        ProvenanceRecord(
            identifier="PSC-VI-T36",
            title="Exact two stage Bayesian coarsening",
            status=ClaimStatus.PROVED,
            source="Paper VI",
            assumptions=("finite partition", "positive evidence"),
            verification_suite="dynamics.bayes",
        )
    )
    def coarse_posterior(self, prior: Any, likelihood: Any) -> ComputationResult[np.ndarray]:
        probability = _probability(prior, "prior")
        weights = _likelihood(likelihood, probability.size)
        if probability.size != len(self.labels):
            raise DimensionError("prior dimension does not match the partition")
        direct = self.aggregate(BayesianUpdate(probability, weights).posterior)
        coarse_prior = self.aggregate(probability)
        effective = self.effective_likelihood(probability, weights)
        reconstructed = BayesianUpdate(coarse_prior, effective).posterior
        residual = float(np.max(np.abs(direct - reconstructed)))
        return ComputationResult(
            reconstructed,
            Certificate(
                valid=residual <= 1e-12,
                exactness=Exactness.NUMERICAL,
                residuals={"two_stage_update": residual},
                tolerance=1e-12,
                assumptions_checked=("finite partition", "positive evidence"),
                theorem_id="PSC-VI-T36",
                message="two stage Bayesian coarsening verified" if residual <= 1e-12 else "two stage Bayesian coarsening failed",
            ),
        )

    def is_fixed_sufficient(self, likelihoods: Sequence[Any], *, tolerance: float = 1e-12) -> bool:
        for likelihood in likelihoods:
            values = _likelihood(likelihood, len(self.labels))
            for block in self.blocks:
                if np.max(np.abs(values[list(block)] - values[block[0]])) > tolerance:
                    return False
        return True

    def fixed_coarse_likelihood(self, likelihood: Any, *, tolerance: float = 1e-12) -> np.ndarray:
        values = _likelihood(likelihood, len(self.labels))
        if not self.is_fixed_sufficient([values], tolerance=tolerance):
            raise AssumptionError("likelihood is not constant inside every coarse block")
        return np.asarray([values[block[0]] for block in self.blocks], dtype=float)

    def information_loss(self, prior: Any, likelihood: Any, coarse_likelihood: Any) -> float:
        probability = _probability(prior, "prior")
        weights = _likelihood(likelihood, probability.size)
        coarse = np.asarray(coarse_likelihood, dtype=float)
        if coarse.ndim != 1 or coarse.size != self.block_count or np.any(coarse < 0):
            raise DimensionError("coarse_likelihood must have one nonnegative value per block")
        effective = self.effective_likelihood(probability, weights)
        coarse_prior = self.aggregate(probability)
        return float(np.sum(coarse_prior * np.abs(effective - coarse)))


def _likelihood_profiles(likelihoods: Sequence[Any]) -> np.ndarray:
    if not likelihoods:
        raise ValidationError("at least one likelihood is required")
    arrays = tuple(np.asarray(value, dtype=float) for value in likelihoods)
    size = arrays[0].size
    if any(value.ndim != 1 or value.size != size for value in arrays):
        raise DimensionError("every likelihood must have the same one dimensional shape")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValidationError("likelihood profiles must contain finite values")
    return np.column_stack(arrays)


def coarsest_sufficient_partition(likelihoods: Sequence[Any], *, tolerance: float = 0.0) -> BayesianPartition:
    """Return the exact coarsest partition on which all likelihoods are constant.

    The theorem is an equality theorem. A positive numerical tolerance does not
    define an equivalence relation in general, so approximate profile grouping
    is exposed separately through :func:`approximate_likelihood_partition`.
    """

    if float(tolerance) != 0.0:
        raise ValidationError(
            "coarsest_sufficient_partition is exact and requires tolerance=0; "
            "use approximate_likelihood_partition for numerical clustering"
        )
    profiles = _likelihood_profiles(likelihoods)
    labels: list[int] = []
    representatives: dict[tuple[float, ...], int] = {}
    for profile in profiles:
        key = tuple(float(value) for value in profile)
        label = representatives.setdefault(key, len(representatives))
        labels.append(label)
    return BayesianPartition(labels)


def approximate_likelihood_partition(
    likelihoods: Sequence[Any],
    *,
    tolerance: float,
) -> BayesianPartition:
    """Cluster likelihood profiles by connected components of a tolerance graph.

    This is a numerical profile clustering utility. It does not certify exact
    Bayesian sufficiency unless every profile in a returned block is identical.
    """

    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValidationError("tolerance must be finite and nonnegative")
    profiles = _likelihood_profiles(likelihoods)
    count = profiles.shape[0]
    adjacency = [set([index]) for index in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            if float(np.max(np.abs(profiles[left] - profiles[right]))) <= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)
    labels = [-1] * count
    component = 0
    for start in range(count):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = component
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component
                    stack.append(neighbor)
        component += 1
    return BayesianPartition(labels)


def relative_entropy(first: Any, second: Any) -> float:
    """Return the finite Kullback Leibler divergence from ``first`` to ``second``."""

    p = _probability(first, "first")
    q = _probability(second, "second")
    if p.size != q.size:
        raise DimensionError("probabilities must have the same dimension")
    if np.any((p > 0) & (q == 0)):
        return float("inf")
    positive = p > 0
    return float(np.sum(p[positive] * np.log(p[positive] / q[positive])))


def relative_entropy_decomposition(
    first: Any,
    second: Any,
    partition: Sequence[int],
) -> ComputationResult[dict[str, Any]]:
    """Decompose relative entropy into coarse and conditional terms."""

    p = _probability(first, "first")
    q = _probability(second, "second")
    if p.size != q.size:
        raise DimensionError("probabilities must have the same dimension")
    coarse = BayesianPartition(partition)
    if len(coarse.labels) != p.size:
        raise DimensionError("partition dimension does not match the probabilities")
    p_coarse = coarse.aggregate(p)
    q_coarse = coarse.aggregate(q)
    coarse_value = relative_entropy(p_coarse, q_coarse)
    conditional_terms: list[float] = []
    for block_index, block in enumerate(coarse.blocks):
        if p_coarse[block_index] == 0:
            conditional_terms.append(0.0)
            continue
        p_conditional = p[list(block)] / p_coarse[block_index]
        if q_coarse[block_index] == 0:
            conditional_terms.append(float("inf"))
            continue
        q_conditional = q[list(block)] / q_coarse[block_index]
        conditional_terms.append(p_coarse[block_index] * relative_entropy(p_conditional, q_conditional))
    total = coarse_value + sum(conditional_terms)
    direct = relative_entropy(p, q)
    residual = 0.0 if np.isinf(total) and np.isinf(direct) else abs(total - direct)
    valid = residual <= 1e-12
    return ComputationResult(
        {
            "total": total,
            "coarse": coarse_value,
            "conditional_terms": tuple(conditional_terms),
        },
        Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"relative_entropy_decomposition": residual},
            tolerance=1e-12,
            assumptions_checked=("finite probability vectors", "fixed partition"),
            theorem_id="PSC-VI-T41",
            message="relative entropy decomposition verified" if valid else "relative entropy decomposition failed",
        ),
    )


@dataclass(frozen=True, slots=True)
class BayesianFlow:
    """The exponential likelihood or replicator flow of a finite prior."""

    prior: np.ndarray
    scores: np.ndarray

    def __init__(self, prior: Any, scores: Any) -> None:
        probability = _probability(prior, "prior")
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1 or values.size != probability.size or not np.all(np.isfinite(values)):
            raise DimensionError("scores must be a finite vector matching the prior")
        object.__setattr__(self, "prior", probability)
        object.__setattr__(self, "scores", values)

    def at(self, time: float) -> np.ndarray:
        t = float(time)
        if not np.isfinite(t):
            raise ValidationError("time must be finite")
        shifted = t * self.scores
        shifted = shifted - np.max(shifted)
        return BayesianUpdate(self.prior, np.exp(shifted)).posterior

    def derivative(self, time: float) -> np.ndarray:
        probability = self.at(time)
        mean = float(self.scores @ probability)
        return probability * (self.scores - mean)

    def expected_score(self, time: float) -> float:
        return float(self.scores @ self.at(time))

    def score_variance(self, time: float) -> float:
        probability = self.at(time)
        mean = float(self.scores @ probability)
        return float(((self.scores - mean) ** 2) @ probability)


@dataclass(frozen=True, slots=True)
class HiddenMarkovFilter:
    """One step finite hidden Markov prediction and Bayesian correction."""

    transition: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.transition, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise DimensionError("transition must be a square matrix")
        if np.any(matrix < -1e-12) or np.max(np.abs(matrix.sum(axis=0) - 1.0)) > 1e-10:
            raise ValidationError("transition must be column stochastic")
        object.__setattr__(self, "transition", matrix)

    def step(self, prior: Any, likelihood: Any) -> np.ndarray:
        probability = _probability(prior, "prior")
        if probability.size != self.transition.shape[0]:
            raise DimensionError("prior dimension does not match the transition")
        predicted = self.transition @ probability
        return BayesianUpdate(predicted, likelihood).posterior

    def coarse_step(
        self,
        prior: Any,
        likelihood: Any,
        partition: Sequence[int],
        *,
        tolerance: float = 1e-10,
    ) -> ComputationResult[np.ndarray]:
        probability = _probability(prior, "prior")
        values = _likelihood(likelihood, probability.size)
        labels = tuple(partition)
        if probability.size != self.transition.shape[0] or len(labels) != probability.size:
            raise DimensionError("partition and prior must match the transition")
        lumping = MarkovLumping(self.transition, labels, kind="transition")
        bayes_partition = BayesianPartition(labels)
        exact_markov = lumping.is_exact(tolerance=tolerance)
        exact_likelihood = bayes_partition.is_fixed_sufficient([values], tolerance=tolerance)
        direct = bayes_partition.aggregate(self.step(probability, values))
        if not exact_markov or not exact_likelihood:
            return ComputationResult(
                direct,
                Certificate(
                    valid=False,
                    exactness=Exactness.CONDITIONAL,
                    assumptions_checked=("finite filter step",),
                    assumptions_unchecked=tuple(
                        condition
                        for condition, satisfied in (
                            ("transition is exactly lumpable", exact_markov),
                            ("likelihood is block constant", exact_likelihood),
                        )
                        if not satisfied
                    ),
                    theorem_id="PSC-VI-T46",
                    message="exact coarse hidden Markov filtering hypotheses are not satisfied",
                ),
            )
        coarse_transition = lumping.exact_coarse_operator(tolerance=tolerance)
        coarse_likelihood = bayes_partition.fixed_coarse_likelihood(values, tolerance=tolerance)
        coarse_prior = bayes_partition.aggregate(probability)
        reconstructed = BayesianUpdate(coarse_transition @ coarse_prior, coarse_likelihood).posterior
        residual = float(np.max(np.abs(direct - reconstructed)))
        return ComputationResult(
            reconstructed,
            Certificate(
                valid=residual <= tolerance,
                exactness=Exactness.NUMERICAL,
                residuals={"coarse_filter": residual},
                tolerance=tolerance,
                assumptions_checked=("exact Markov lumping", "block constant likelihood"),
                theorem_id="PSC-VI-T46",
                message="coarse hidden Markov filtering verified" if residual <= tolerance else "coarse hidden Markov filtering failed",
            ),
        )


__all__ = [
    "BayesianFlow",
    "BayesianPartition",
    "BayesianUpdate",
    "HiddenMarkovFilter",
    "approximate_likelihood_partition",
    "coarsest_sufficient_partition",
    "relative_entropy",
    "relative_entropy_decomposition",
]
