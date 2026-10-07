"""Conditional global support, Hecke packet, and patching models."""

from .dyadic import DyadicSignDiagnostic, dyadic_sign_diagnostic
from .hecke import FiniteHeckePackets, HeckePacket, SupportComparison
from .patching import PatchingSupportReport

from ...registry import FeatureSpec
FEATURE_KEY = "langlands.global"


def status() -> FeatureSpec:
    """Return the registered feature status."""

    from ..._planned import development_status

    return development_status(FEATURE_KEY)


def require() -> None:
    """Require the global conditional-model feature to be available."""

    from ..._planned import require_implemented

    require_implemented(FEATURE_KEY)


__all__ = [
    "DyadicSignDiagnostic",
    "FEATURE_KEY",
    "FiniteHeckePackets",
    "HeckePacket",
    "PatchingSupportReport",
    "SupportComparison",
    "dyadic_sign_diagnostic",
    "require",
    "status",
]
