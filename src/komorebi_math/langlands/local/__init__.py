"""Executable finite models for tame and wild local packet geometry."""

from .bridge import LocalPacketBridge, SymplecticBridge
from .envelopes import EnvelopeRecovery, ProjectiveIntertwinerEnvelope
from .finite_fields import FiniteField, FiniteFieldElement, is_irreducible_polynomial
from .gauss import GaussSumResult, additive_character, gauss_sum, multiplicative_character
from .heisenberg import HeisenbergElement, HeisenbergModel
from .symplectic import AlternatingForm, QuadraticRefinement
from .tame import (
    FrobeniusOrbit,
    SpecialMonodromyBlock,
    TameWeilPacket,
    TensorRefinement,
    regular_orbit_count,
    tame_tensor_refinement,
)

from ...registry import FeatureSpec
FEATURE_KEY = "langlands.local"


def status() -> FeatureSpec:
    """Return the registered feature status."""

    from ..._planned import development_status

    return development_status(FEATURE_KEY)


def require() -> None:
    """Require the local finite-model feature to be implemented."""

    from ..._planned import require_implemented

    require_implemented(FEATURE_KEY)


__all__ = [
    "AlternatingForm",
    "EnvelopeRecovery",
    "FEATURE_KEY",
    "FiniteField",
    "FiniteFieldElement",
    "FrobeniusOrbit",
    "GaussSumResult",
    "HeisenbergElement",
    "HeisenbergModel",
    "LocalPacketBridge",
    "ProjectiveIntertwinerEnvelope",
    "QuadraticRefinement",
    "SpecialMonodromyBlock",
    "SymplecticBridge",
    "TameWeilPacket",
    "TensorRefinement",
    "additive_character",
    "gauss_sum",
    "is_irreducible_polynomial",
    "multiplicative_character",
    "regular_orbit_count",
    "require",
    "status",
    "tame_tensor_refinement",
]
