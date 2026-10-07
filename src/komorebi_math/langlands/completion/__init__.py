"""Gate-based conditional Langlands completion reports."""

from .gates import (
    AutomorphicCompletionReport,
    CompatiblePropagationReport,
    CompletionGate,
    GateStatus,
)

from ...registry import FeatureSpec
FEATURE_KEY = "langlands.completion"


def status() -> FeatureSpec:
    """Return the registered feature status."""

    from ..._planned import development_status

    return development_status(FEATURE_KEY)


def require() -> None:
    """Require the conditional completion feature to be available."""

    from ..._planned import require_implemented

    require_implemented(FEATURE_KEY)


__all__ = [
    "AutomorphicCompletionReport",
    "CompatiblePropagationReport",
    "CompletionGate",
    "FEATURE_KEY",
    "GateStatus",
    "require",
    "status",
]
