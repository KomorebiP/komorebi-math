"""Finite group, monomial packet, Mackey, and Clifford execution models."""

from .characters import InducedRepresentation, LinearCharacter
from .clifford import (
    CliffordMode,
    CliffordStep,
    CliffordTower,
    CliffordTowerReport,
    VirtualCharacterResolution,
    VirtualCharacterTerm,
)
from .groups import FiniteGroup, FiniteSubgroup, compose_permutations, cyclic_group, symmetric_group
from .mackey import MackeyDecomposition, MackeyPacket, mackey_tensor_packets
from .orbits import MonomialAction, OrbitPacket, weak_compositions
from .plans import AutomorphicInductionPlan

from ...registry import FeatureSpec
FEATURE_KEY = "langlands.finite_image"


def status() -> FeatureSpec:
    """Return the registered feature status."""

    from ..._planned import development_status

    return development_status(FEATURE_KEY)


def require() -> None:
    """Require the finite image feature to be implemented."""

    from ..._planned import require_implemented

    require_implemented(FEATURE_KEY)


__all__ = [
    "AutomorphicInductionPlan",
    "CliffordMode",
    "CliffordStep",
    "CliffordTower",
    "CliffordTowerReport",
    "FEATURE_KEY",
    "FiniteGroup",
    "FiniteSubgroup",
    "InducedRepresentation",
    "LinearCharacter",
    "MackeyDecomposition",
    "MackeyPacket",
    "MonomialAction",
    "OrbitPacket",
    "VirtualCharacterResolution",
    "VirtualCharacterTerm",
    "compose_permutations",
    "cyclic_group",
    "mackey_tensor_packets",
    "require",
    "status",
    "symmetric_group",
    "weak_compositions",
]
