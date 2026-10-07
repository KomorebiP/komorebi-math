"""Finite residual deformation, support, and companion models."""

from .companions import (
    CompanionForest,
    CompatibleCarrier,
    FrobeniusSeparationResult,
    FrobeniusSpine,
    finite_frobenius_separator,
)
from .complexes import CohomologyData, FiniteCochainComplex, row_rank_mod
from .counting import (
    RationalCountingSeries,
    diagonal_nilpotent_count,
    nilpotent_chart_count,
    nilpotent_chart_series,
    normal_crossing_count,
    normal_crossing_series,
    smooth_chart_count,
    smooth_chart_series,
    weighted_chart_count,
)
from .forests import FiniteLiftForest, LiftBranch, LiftForestReport, residue_forest
from .support import (
    FiberSupportResult,
    FittingGenerators,
    NestedOccupancyResult,
    PresentedModule,
    nested_subspace_occupancy,
)
from .visibility import VisibilityPair, VisibilityTree, rational_valuation

from ..registry import FeatureSpec
FEATURE_KEY = "deformation"


def status() -> FeatureSpec:
    """Return the registered feature status."""

    from .._planned import development_status

    return development_status(FEATURE_KEY)


def require() -> None:
    """Require the deformation feature to be implemented."""

    from .._planned import require_implemented

    require_implemented(FEATURE_KEY)


__all__ = [
    "CohomologyData",
    "CompanionForest",
    "CompatibleCarrier",
    "FEATURE_KEY",
    "FiberSupportResult",
    "FiniteCochainComplex",
    "FiniteLiftForest",
    "FittingGenerators",
    "FrobeniusSeparationResult",
    "FrobeniusSpine",
    "LiftBranch",
    "LiftForestReport",
    "NestedOccupancyResult",
    "PresentedModule",
    "RationalCountingSeries",
    "VisibilityPair",
    "VisibilityTree",
    "diagonal_nilpotent_count",
    "finite_frobenius_separator",
    "nested_subspace_occupancy",
    "nilpotent_chart_count",
    "nilpotent_chart_series",
    "normal_crossing_count",
    "normal_crossing_series",
    "rational_valuation",
    "require",
    "residue_forest",
    "row_rank_mod",
    "smooth_chart_count",
    "smooth_chart_series",
    "status",
    "weighted_chart_count",
]
