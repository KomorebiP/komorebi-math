"""Offline, bounded mathematical stress testing of the installed package."""

from .paper_suites import paper_checks
from .runner import CheckOutcome, SkipCheck, StressCheck, StressContext, StressReport, run_stress
from .suites import BUILTIN_SUITES, builtin_checks

__all__ = [
    "BUILTIN_SUITES",
    "CheckOutcome",
    "SkipCheck",
    "StressCheck",
    "StressContext",
    "StressReport",
    "builtin_checks",
    "paper_checks",
    "run_stress",
]
