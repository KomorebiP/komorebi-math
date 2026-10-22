"""Finite rational resampling with explicit polyphase execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .windows import WindowSpec, get_window


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return int(value)


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or result.size == 0:
        raise DimensionError(f"{name} must be a nonempty one dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} entries must be finite")
    return result


def _reduced_ratio(up: int, down: int) -> tuple[int, int]:
    numerator = _positive_integer(up, "up")
    denominator = _positive_integer(down, "down")
    divisor = gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def resampled_length(input_length: int, up: int, down: int) -> int:
    """Return the finite output length ``ceil(input_length * up / down)``."""

    length = _positive_integer(input_length, "input_length")
    numerator, denominator = _reduced_ratio(up, down)
    return (length * numerator + denominator - 1) // denominator


def upsample(signal: Any, factor: int) -> np.ndarray:
    """Insert ``factor - 1`` zeros between consecutive finite samples."""

    value = _vector(signal, "signal")
    amount = _positive_integer(factor, "factor")
    result = np.zeros(value.size * amount, dtype=value.dtype)
    result[::amount] = value
    return result


def downsample(signal: Any, factor: int, *, phase: int = 0) -> np.ndarray:
    """Retain one residue class from a finite signal."""

    value = _vector(signal, "signal")
    amount = _positive_integer(factor, "factor")
    if isinstance(phase, bool) or int(phase) != phase:
        raise ValidationError("phase must be an integer")
    residue = int(phase)
    if residue < 0 or residue >= amount:
        raise ValidationError("phase must lie between zero and factor minus one")
    return value[residue::amount].copy()


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D11",
        title="Windowed finite rational-resampling anti-alias filter",
        status=ClaimStatus.CLASSICAL,
        source="finite signal processing",
        assumptions=("positive reduced rational rate", "odd finite tap count"),
        limitations=("zero-extension boundary convention", "finite-window approximation"),
        verification_suite="signals.resampling",
    )
)
def design_resample_filter(
    up: int,
    down: int,
    *,
    num_taps: int | None = None,
    window: WindowSpec = ("kaiser", 8.0),
) -> np.ndarray:
    """Design the centered low-pass kernel used by rational resampling.

    The taps live on the upsampled grid, have DC gain ``up``, and use cutoff
    ``1 / (2 * max(up, down))`` cycles per upsampled sample.
    """

    numerator, denominator = _reduced_ratio(up, down)
    rate_bound = max(numerator, denominator)
    count = 20 * rate_bound + 1 if num_taps is None else _positive_integer(num_taps, "num_taps")
    if count % 2 == 0:
        raise ValidationError("num_taps must be odd")
    center = (count - 1) // 2
    index = np.arange(count, dtype=float) - center
    cutoff = 0.5 / rate_bound
    ideal = 2.0 * cutoff * np.sinc(2.0 * cutoff * index)
    tapered = ideal * np.asarray(get_window(window, count, periodic=False), dtype=float)
    total = float(np.sum(tapered))
    if abs(total) <= np.finfo(float).eps:
        raise ValidationError("resampling filter has zero DC gain")
    return tapered * (numerator / total)


@dataclass(frozen=True, slots=True)
class RationalResamplingPlan:
    """A centered zero-extension rational-resampling plan."""

    up: int
    down: int
    taps: np.ndarray

    def __post_init__(self) -> None:
        numerator, denominator = _reduced_ratio(self.up, self.down)
        impulse = _vector(self.taps, "taps")
        if impulse.size % 2 == 0:
            raise ValidationError("resampling taps must have odd length")
        object.__setattr__(self, "up", numerator)
        object.__setattr__(self, "down", denominator)
        object.__setattr__(self, "taps", impulse.copy())

    @property
    def delay(self) -> int:
        """Return the centered filter delay on the upsampled grid."""

        return (self.taps.size - 1) // 2

    @property
    def polyphase_components(self) -> tuple[np.ndarray, ...]:
        """Return the ``up`` residue components of the anti-alias filter."""

        return tuple(self.taps[phase:: self.up].copy() for phase in range(self.up))

    def output_length(self, input_length: int) -> int:
        """Return the finite output length for an input length."""

        return resampled_length(input_length, self.up, self.down)

    def apply(self, signal: Any) -> np.ndarray:
        """Apply the plan directly through its filter residue classes."""

        value = _vector(signal, "signal")
        output_count = self.output_length(value.size)
        dtype = np.result_type(value.dtype, self.taps.dtype)
        output = np.zeros(output_count, dtype=dtype)
        delay = self.delay
        for output_index in range(output_count):
            high_rate_index = delay + output_index * self.down
            phase = high_rate_index % self.up
            accumulator: complex | float = 0.0
            for tap_index in range(phase, self.taps.size, self.up):
                source_index = (high_rate_index - tap_index) // self.up
                if 0 <= source_index < value.size:
                    accumulator = accumulator + self.taps[tap_index] * value[source_index]
            output[output_index] = accumulator
        return np.real_if_close(output, tol=1000)

    def explicit_reference(self, signal: Any) -> np.ndarray:
        """Apply explicit zero insertion, full convolution, delay removal, and decimation."""

        value = _vector(signal, "signal")
        expanded = upsample(value, self.up)
        filtered = np.convolve(expanded, self.taps, mode="full")
        output_count = self.output_length(value.size)
        return np.real_if_close(
            filtered[self.delay : self.delay + output_count * self.down : self.down],
            tol=1000,
        )

    def certificate(self, signal: Any, *, tolerance: float = 1e-12) -> Certificate:
        """Certify polyphase equality with the explicit finite reference construction."""

        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValidationError("tolerance must be finite and nonnegative")
        direct = np.asarray(self.apply(signal))
        reference = np.asarray(self.explicit_reference(signal))
        residual = float(np.max(np.abs(direct - reference))) if direct.size else 0.0
        scale = max(1.0, float(np.max(np.abs(reference))) if reference.size else 0.0)
        dc_residual = abs(complex(np.sum(self.taps)) - self.up)
        valid = residual <= tolerance * scale and dc_residual <= tolerance * max(1.0, self.up)
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"polyphase_reference": residual, "dc_gain": dc_residual},
            tolerance=tolerance,
            assumptions_checked=("centered zero-extension convention", "reduced rational rate"),
            assumptions_unchecked=("application-specific alias rejection", "boundary-transient acceptability"),
            diagnostics={
                "up": self.up,
                "down": self.down,
                "tap_count": int(self.taps.size),
                "delay": self.delay,
            },
            theorem_id="KM-DSP-T06",
            message="polyphase resampling identity verified" if valid else "polyphase resampling identity failed",
        )


@dataclass(frozen=True, slots=True)
class ResamplingResult:
    """A finite rationally resampled signal paired with its execution plan and certificate."""

    signal: np.ndarray
    plan: RationalResamplingPlan
    input_length: int
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-T06",
        title="Finite centered polyphase rational-resampling identity",
        status=ClaimStatus.PROVED,
        source="finite signal processing",
        assumptions=("zero-extension boundary convention", "odd centered FIR kernel"),
        verification_suite="signals.resampling",
    )
)
def resample_poly_result(
    signal: Any,
    up: int,
    down: int,
    *,
    taps: Any | None = None,
    num_taps: int | None = None,
    window: WindowSpec = ("kaiser", 8.0),
    tolerance: float = 1e-12,
) -> ResamplingResult:
    """Return rationally resampled data with an explicit finite identity certificate."""

    value = _vector(signal, "signal")
    numerator, denominator = _reduced_ratio(up, down)
    impulse = (
        design_resample_filter(numerator, denominator, num_taps=num_taps, window=window)
        if taps is None
        else _vector(taps, "taps")
    )
    plan = RationalResamplingPlan(numerator, denominator, impulse)
    output = np.asarray(plan.apply(value))
    certificate = plan.certificate(value, tolerance=tolerance)
    return ResamplingResult(output, plan, int(value.size), certificate)


def resample_poly(
    signal: Any,
    up: int,
    down: int,
    *,
    taps: Any | None = None,
    num_taps: int | None = None,
    window: WindowSpec = ("kaiser", 8.0),
) -> np.ndarray:
    """Return centered zero-extension rational polyphase resampling output."""

    value = _vector(signal, "signal")
    numerator, denominator = _reduced_ratio(up, down)
    impulse = (
        design_resample_filter(numerator, denominator, num_taps=num_taps, window=window)
        if taps is None
        else _vector(taps, "taps")
    )
    return RationalResamplingPlan(numerator, denominator, impulse).apply(value)


def decimate(
    signal: Any,
    factor: int,
    *,
    num_taps: int | None = None,
    window: WindowSpec = ("kaiser", 8.0),
) -> np.ndarray:
    """Low-pass filter and downsample a finite signal by an integer factor."""

    return resample_poly(signal, 1, factor, num_taps=num_taps, window=window)


def interpolate(
    signal: Any,
    factor: int,
    *,
    num_taps: int | None = None,
    window: WindowSpec = ("kaiser", 8.0),
) -> np.ndarray:
    """Upsample and anti-image filter a finite signal by an integer factor."""

    return resample_poly(signal, factor, 1, num_taps=num_taps, window=window)


__all__ = [
    "RationalResamplingPlan",
    "ResamplingResult",
    "decimate",
    "design_resample_filter",
    "downsample",
    "interpolate",
    "resample_poly",
    "resample_poly_result",
    "resampled_length",
    "upsample",
]
