"""Research Gaussian quadratic-form tails with optional directed arithmetic.

Importing this namespace does not import mpmath. Public operations serialize
their precision contexts. Only literal interval endpoints carry enclosures.
See ``docs/quadratic_forms.md`` for the supported domain and trust boundary.
"""

from importlib import import_module

from .api import (
    certified_float64_factor_logsf,
    certified_positive_logsf,
    certify_float64_factor_spectrum,
    design_positive_tail_parameters,
    directed_relative_accuracy,
    even_df_scaled_chi_square_logsf_interval,
    interval_positive_logsf,
)

_RESULT_MODULES = {
    "AccuracyDrivenTailResult": "_accuracy",
    "DirectedAccuracy": "_directed_accuracy",
    "IntervalLogTailResult": "_interval",
    "MatrixTailCertificate": "_spectrum",
    "VerifiedSpectrum": "_spectrum",
}


def __getattr__(name):
    if name in _RESULT_MODULES:
        from .api import _require_mpmath

        _require_mpmath()
        value = getattr(import_module(f"{__name__}.{_RESULT_MODULES[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AccuracyDrivenTailResult",
    "DirectedAccuracy",
    "IntervalLogTailResult",
    "MatrixTailCertificate",
    "VerifiedSpectrum",
    "certified_float64_factor_logsf",
    "certified_positive_logsf",
    "certify_float64_factor_spectrum",
    "design_positive_tail_parameters",
    "directed_relative_accuracy",
    "even_df_scaled_chi_square_logsf_interval",
    "interval_positive_logsf",
]
