"""Semigroups, probabilistic refinement, fractional operators, and memory."""

from .._planned import development_status
from ..registry import FeatureSpec
from .bayes import (
    BayesianFlow,
    BayesianPartition,
    approximate_likelihood_partition,
    BayesianUpdate,
    HiddenMarkovFilter,
    coarsest_sufficient_partition,
    relative_entropy,
    relative_entropy_decomposition,
)
from .feshbach import FeshbachReduction
from .feynman_kac import FiniteFeynmanKac
from .fractional import (
    FractionalDefectResult,
    fractional_defect,
    fractional_naturality_certificate,
    fractional_power,
    fractional_semigroup,
)
from .green import DirichletGreen1D
from .markov import (
    AggregationMap,
    ContinuousTimeMarkovChain,
    FiniteMarkovChain,
    MarkovLumping,
    aggregation_matrix,
    coordinate_lumpability_residual,
    normalize_partition,
)
from .martingales import (
    ConditionalExpectation,
    ConditionalExpectationTower,
    MartingaleTower,
    partition_refines,
)
from .memory import (
    MemoryKernel,
    RationalMemoryKernel,
    RationalMemoryRealization,
    kernel_l1_error,
    memory_kernel_error_bound,
    volterra_trajectory_bound,
)
from .semigroups import (
    FiniteSemigroupTower,
    MatrixSemigroup,
    PacketSemigroup,
    ProjectiveSemigroupTower,
    SemigroupRefinement,
    matrix_exponential,
    semigroup_refinement_defect,
)
from .survival import PhaseTypeSurvival

FEATURE_KEY = "dynamics"


def status() -> FeatureSpec:
    """Return the registered implementation status for this domain."""

    return development_status(FEATURE_KEY)


__all__ = [
    "FEATURE_KEY",
    "AggregationMap",
    "BayesianFlow",
    "BayesianPartition",
    "BayesianUpdate",
    "approximate_likelihood_partition",
    "ConditionalExpectation",
    "ConditionalExpectationTower",
    "ContinuousTimeMarkovChain",
    "DirichletGreen1D",
    "FiniteFeynmanKac",
    "FiniteMarkovChain",
    "FiniteSemigroupTower",
    "FeshbachReduction",
    "FractionalDefectResult",
    "HiddenMarkovFilter",
    "MarkovLumping",
    "MartingaleTower",
    "MatrixSemigroup",
    "MemoryKernel",
    "PacketSemigroup",
    "PhaseTypeSurvival",
    "ProjectiveSemigroupTower",
    "RationalMemoryKernel",
    "RationalMemoryRealization",
    "SemigroupRefinement",
    "aggregation_matrix",
    "coarsest_sufficient_partition",
    "coordinate_lumpability_residual",
    "fractional_defect",
    "fractional_naturality_certificate",
    "fractional_power",
    "fractional_semigroup",
    "kernel_l1_error",
    "matrix_exponential",
    "memory_kernel_error_bound",
    "normalize_partition",
    "partition_refines",
    "relative_entropy",
    "relative_entropy_decomposition",
    "semigroup_refinement_defect",
    "status",
    "volterra_trajectory_bound",
]
