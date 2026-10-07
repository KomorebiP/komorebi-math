"""Finite ramification Clifford tower diagnostics."""

from .towers import (
    ProjectiveMultiplicityPacket,
    RamificationCliffordTower,
    RamificationStage,
    RamificationTowerReport,
)

from ...registry import FeatureSpec
FEATURE_KEY = "langlands.ramification"


def status() -> FeatureSpec:
    """Return the registered feature status."""

    from ..._planned import development_status

    return development_status(FEATURE_KEY)


def require() -> None:
    """Require the ramification feature to be implemented."""

    from ..._planned import require_implemented

    require_implemented(FEATURE_KEY)


__all__ = [
    "FEATURE_KEY",
    "ProjectiveMultiplicityPacket",
    "RamificationCliffordTower",
    "RamificationStage",
    "RamificationTowerReport",
    "require",
    "status",
]
