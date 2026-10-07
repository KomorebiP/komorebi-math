"""Explicit bridges between formal generating functions and finite signals."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..signals.filtering import FrequencyResponse, fir_frequency_response
from .basis import WeightedMonomialBasis, ordinary_basis
from .series import FormalSeries


@dataclass(frozen=True, slots=True)
class FIRSeriesAdapter:
    r"""A finite transfer polynomial interpreted under ``z -> z**(-1)``."""

    series: FormalSeries
    taps: np.ndarray
    variable_substitution: str = "formal z maps to delay z^(-1)"

    def __post_init__(self) -> None:
        taps = np.asarray(self.taps)
        if taps.ndim != 1 or taps.size != self.series.degree + 1:
            raise DimensionError("adapter taps must match the formal-series degree")
        if not np.all(np.isfinite(taps)):
            raise ValidationError("adapter taps must be finite")
        object.__setattr__(self, "taps", taps.copy())

    def response(
        self,
        *,
        points: int = 512,
        sample_rate: float = 1.0,
        whole: bool = False,
    ) -> FrequencyResponse:
        """Evaluate the associated causal FIR transfer function."""

        return fir_frequency_response(
            self.taps,
            points=points,
            sample_rate=sample_rate,
            whole=whole,
        )

    def certificate(self, *, tolerance: float = 0.0) -> Certificate:
        """Certify the declared coefficient-to-tap conversion."""

        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValidationError("tolerance must be finite and nonnegative")
        expected = np.asarray(
            [float(value) for value in self.series.ordinary_coefficients()],
            dtype=np.result_type(self.taps.dtype, float),
        )
        residual = float(np.max(np.abs(expected - self.taps))) if expected.size else 0.0
        valid = residual <= tolerance
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"coefficient_to_tap": residual},
            tolerance=tolerance,
            assumptions_checked=("formal z is interpreted as one sample of delay",),
            theorem_id="GF-SIGNAL-FIR-BRIDGE",
            message="formal-series FIR bridge verified" if valid else "formal-series FIR bridge failed",
        )


def series_to_fir_taps(
    series: FormalSeries,
    *,
    degree: int | None = None,
    dtype: Any = float,
) -> np.ndarray:
    r"""Map ordinary coefficients to causal taps under ``z -> z**(-1)``."""

    if not isinstance(series, FormalSeries):
        raise ValidationError("series must be a FormalSeries")
    selected = series.degree if degree is None else degree
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
        raise ValidationError("degree must be a nonnegative integer")
    coefficients = series.ordinary_coefficients(selected)
    try:
        taps = np.asarray(coefficients, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("ordinary coefficients cannot be represented in the requested dtype") from exc
    if not np.all(np.isfinite(taps)):
        raise ValidationError("converted FIR taps are not finite")
    return taps


def series_to_fir_adapter(
    series: FormalSeries,
    *,
    degree: int | None = None,
    dtype: Any = float,
) -> FIRSeriesAdapter:
    """Create an explicit formal-series to FIR adapter record."""

    selected = series.degree if degree is None else degree
    truncated = series.truncate(selected)
    return FIRSeriesAdapter(
        truncated,
        series_to_fir_taps(truncated, dtype=dtype),
    )


def fir_taps_to_series(
    taps: Any,
    *,
    basis: WeightedMonomialBasis | None = None,
    max_denominator: int | None = None,
) -> FormalSeries:
    """Convert a finite real FIR tap vector into an exact formal series.

    Floating values are converted through their decimal string representation.
    ``max_denominator`` may be supplied to request a bounded rational
    approximation explicitly.
    """

    array = np.asarray(taps)
    if array.ndim != 1 or array.size == 0:
        raise DimensionError("FIR taps must be a nonempty one-dimensional array")
    if np.iscomplexobj(array) and np.any(np.imag(array) != 0):
        raise ValidationError("FormalSeries currently supports real rational coefficients only")
    real = np.real(array)
    if not np.all(np.isfinite(real)):
        raise ValidationError("FIR taps must be finite")
    if max_denominator is not None and (
        isinstance(max_denominator, bool)
        or not isinstance(max_denominator, int)
        or max_denominator < 1
    ):
        raise ValidationError("max_denominator must be a positive integer")
    ordinary: list[Fraction] = []
    for value in real.tolist():
        if isinstance(value, (int, np.integer)):
            fraction = Fraction(int(value))
        else:
            fraction = Fraction(str(float(value)))
        if max_denominator is not None:
            fraction = fraction.limit_denominator(max_denominator)
        ordinary.append(fraction)
    return FormalSeries.from_ordinary_coefficients(basis or ordinary_basis(), ordinary)


__all__ = [
    "FIRSeriesAdapter",
    "fir_taps_to_series",
    "series_to_fir_adapter",
    "series_to_fir_taps",
]
