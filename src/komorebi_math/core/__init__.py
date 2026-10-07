"""Finite dimensional executable foundation."""

from .algebra import AlgebraBasisResult, OperatorAlgebra, matrix_units
from .grammar import Arrow, OperatorGrammar, OperatorRelation, OperatorWord
from .graphs import SupportPreorder, TransferGraph
from .morphisms import OperatorMorphism, PacketMorphism, SeedMorphism
from .operators import PacketOperator
from .packets import PacketCoordinates, PacketSystem
from .realization import OperatorRealization, RelationResidual
from .seeds import (
    FreePresentation,
    SeedSystem,
    StandardSeedSearchResult,
    independent_direct_sum_seed_rank,
    shared_direct_sum_seed_bounds,
    tensor_seed_rank_upper_bound,
    cyclic_span,
    semisimple_seed_rank,
    single_operator_seed_rank,
)
from .transfer import TransferPath, block_from_paths, path_contributions

__all__ = [
    "AlgebraBasisResult",
    "Arrow",
    "FreePresentation",
    "OperatorAlgebra",
    "OperatorGrammar",
    "OperatorMorphism",
    "OperatorRealization",
    "OperatorRelation",
    "OperatorWord",
    "PacketCoordinates",
    "PacketMorphism",
    "PacketOperator",
    "PacketSystem",
    "RelationResidual",
    "SeedMorphism",
    "SeedSystem",
    "StandardSeedSearchResult",
    "SupportPreorder",
    "TransferGraph",
    "TransferPath",
    "block_from_paths",
    "independent_direct_sum_seed_rank",
    "matrix_units",
    "cyclic_span",
    "path_contributions",
    "semisimple_seed_rank",
    "shared_direct_sum_seed_bounds",
    "single_operator_seed_rank",
    "tensor_seed_rank_upper_bound",
]
