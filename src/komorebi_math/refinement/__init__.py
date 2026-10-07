"""Refinement towers, weighted transfer, integral separation, and linear defects."""

from ..registry import FeatureSpec
from .crt import CRTAddress, crt_many, crt_pair, dyadic_odd_factor, factor_residue
from .defects import LinearRefinementDefect, multilevel_defect
from .lattices import GluingModule, LatticePacketData, PrecisionDelay, refinement_weight_monotonicity
from .polyphase import (
    LatticePolyphaseRefinement,
    PolyphaseCoordinateRecord,
    PolyphaseRefinement,
    PolyphaseRefinementTower,
    coarsen_lattice_polyphase,
    coarsen_polyphase,
    polyphase_refinement_certificate,
    refine_lattice_polyphase,
    refine_polyphase,
)
from .residue import (
    ResidueCoordinates,
    nested_residue_address,
    nested_residue_analysis,
    packet_coordinates_from_residues,
    residue_analyze,
    residue_packet_system,
    residue_synthesize,
)
from .smith import (
    SmithCongruenceProblem,
    SmithCongruenceSolution,
    SmithDecomposition,
    padic_valuation,
    rational_padic_valuation,
    smith_decomposition,
)
from .towers import PacketRefinement, RefinementTower, common_refinement
from .weighted import FilteredInverseExpansion, WeightedPath, WeightedTransferEdge, WeightedTransferGraph

FEATURE_KEY = "refinement"


def status() -> FeatureSpec:
    """Return the feature registry record for this package block."""

    from ..registry import feature

    return feature(FEATURE_KEY)


def require() -> None:
    """Confirm that the refinement block is implemented."""

    return None


__all__ = [
    "CRTAddress",
    "FEATURE_KEY",
    "FilteredInverseExpansion",
    "GluingModule",
    "LatticePacketData",
    "LatticePolyphaseRefinement",
    "LinearRefinementDefect",
    "PacketRefinement",
    "PolyphaseCoordinateRecord",
    "PolyphaseRefinement",
    "PolyphaseRefinementTower",
    "PrecisionDelay",
    "RefinementTower",
    "ResidueCoordinates",
    "SmithCongruenceProblem",
    "SmithCongruenceSolution",
    "SmithDecomposition",
    "WeightedPath",
    "WeightedTransferEdge",
    "WeightedTransferGraph",
    "coarsen_lattice_polyphase",
    "coarsen_polyphase",
    "common_refinement",
    "crt_many",
    "crt_pair",
    "dyadic_odd_factor",
    "factor_residue",
    "multilevel_defect",
    "nested_residue_address",
    "nested_residue_analysis",
    "packet_coordinates_from_residues",
    "padic_valuation",
    "polyphase_refinement_certificate",
    "rational_padic_valuation",
    "refine_lattice_polyphase",
    "refine_polyphase",
    "refinement_weight_monotonicity",
    "require",
    "residue_analyze",
    "residue_packet_system",
    "residue_synthesize",
    "smith_decomposition",
    "status",
]
