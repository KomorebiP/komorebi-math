"""Deterministic finite windows and overlap-add diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented

WindowName = Literal[
    "boxcar",
    "rectangular",
    "hann",
    "hamming",
    "blackman",
    "bartlett",
    "kaiser",
    "tukey",
]
WindowSpec: TypeAlias = WindowName | str | tuple[str, float] | np.ndarray


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return int(value)


def _canonical_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("_", "-")
    aliases = {
        "rect": "boxcar",
        "rectangle": "boxcar",
        "rectangular": "boxcar",
        "ones": "boxcar",
        "hanning": "hann",
        "triang": "bartlett",
        "triangle": "bartlett",
    }
    return aliases.get(normalized, normalized)


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D01",
        title="Finite deterministic analysis window",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("positive finite window length",),
        verification_suite="signals.windows",
    )
)
def get_window(
    window: WindowSpec,
    length: int,
    *,
    periodic: bool = True,
) -> np.ndarray:
    """Return a finite real window using an explicit periodic or symmetric convention.

    A supplied array is copied and must already have the requested length.  A
    tuple is accepted as ``("kaiser", beta)`` or ``("tukey", alpha)``.  Periodic cosine
    windows are formed by constructing a symmetric window of length ``N + 1``
    and dropping its last sample.
    """

    n_samples = _positive_integer(length, "length")
    if not isinstance(window, (str, tuple)):
        value = np.asarray(window)
        if value.ndim != 1 or value.size != n_samples:
            raise DimensionError("an explicit window must be one dimensional with the requested length")
        if not np.all(np.isfinite(value)):
            raise ValidationError("window entries must be finite")
        return value.copy()

    parameter = 14.0
    if isinstance(window, tuple):
        if len(window) != 2:
            raise ValidationError("a parameterized window must be a two item tuple")
        name = _canonical_name(window[0])
        parameter = float(window[1])
        if name == "kaiser":
            if not np.isfinite(parameter) or parameter < 0:
                raise ValidationError("Kaiser beta must be finite and nonnegative")
        elif name == "tukey":
            if not np.isfinite(parameter) or parameter < 0 or parameter > 1:
                raise ValidationError("Tukey alpha must lie in [0, 1]")
        else:
            raise ValidationError("only Kaiser and Tukey windows accept tuple parameters")
    else:
        name = _canonical_name(window)
        if name == "tukey":
            parameter = 0.5

    if n_samples == 1:
        return np.ones(1, dtype=float)
    evaluation_length = n_samples + 1 if periodic else n_samples
    index = np.arange(evaluation_length, dtype=float)
    denominator = float(evaluation_length - 1)

    if name == "boxcar":
        result = np.ones(evaluation_length, dtype=float)
    elif name == "hann":
        result = 0.5 - 0.5 * np.cos(2.0 * np.pi * index / denominator)
    elif name == "hamming":
        result = 0.54 - 0.46 * np.cos(2.0 * np.pi * index / denominator)
    elif name == "blackman":
        phase = 2.0 * np.pi * index / denominator
        result = 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2.0 * phase)
    elif name == "bartlett":
        result = np.bartlett(evaluation_length)
    elif name == "kaiser":
        result = np.kaiser(evaluation_length, parameter)
    elif name == "tukey":
        alpha = parameter
        if alpha <= 0.0:
            result = np.ones(evaluation_length, dtype=float)
        elif alpha >= 1.0:
            result = 0.5 - 0.5 * np.cos(2.0 * np.pi * index / denominator)
        else:
            result = np.ones(evaluation_length, dtype=float)
            leading = index < alpha * denominator / 2.0
            trailing = index > denominator * (1.0 - alpha / 2.0)
            result[leading] = 0.5 * (
                1.0 + np.cos(np.pi * (2.0 * index[leading] / (alpha * denominator) - 1.0))
            )
            result[trailing] = 0.5 * (
                1.0
                + np.cos(
                    np.pi
                    * (2.0 * index[trailing] / (alpha * denominator) - 2.0 / alpha + 1.0)
                )
            )
    else:
        raise ValidationError(f"unknown window: {name}")
    return np.asarray(result[:-1] if periodic else result, dtype=float)


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    """Finite gain and equivalent-noise-bandwidth diagnostics for a window."""

    length: int
    coherent_gain: float
    power_gain: float
    equivalent_noise_bandwidth_bins: float
    equivalent_noise_bandwidth_hz: float


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-T01",
        title="Finite window gain and equivalent noise bandwidth",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("finite nonzero coherent gain", "positive sample rate"),
        verification_suite="signals.windows",
    )
)
def window_metrics(window: Any, *, sample_rate: float = 1.0) -> WindowMetrics:
    """Return coherent gain, power gain, and equivalent noise bandwidth."""

    value = np.asarray(window)
    if value.ndim != 1 or value.size == 0:
        raise DimensionError("window metrics require a nonempty one dimensional window")
    if not np.all(np.isfinite(value)):
        raise ValidationError("window entries must be finite")
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValidationError("sample_rate must be positive and finite")
    coherent_sum = np.sum(value)
    if abs(complex(coherent_sum)) <= np.finfo(float).eps * value.size:
        raise ValidationError("equivalent noise bandwidth requires nonzero coherent gain")
    coherent_gain = float(abs(coherent_sum) / value.size)
    power_gain = float(np.sum(np.abs(value) ** 2) / value.size)
    bandwidth_bins = float(value.size * np.sum(np.abs(value) ** 2) / abs(coherent_sum) ** 2)
    return WindowMetrics(
        length=int(value.size),
        coherent_gain=coherent_gain,
        power_gain=power_gain,
        equivalent_noise_bandwidth_bins=bandwidth_bins,
        equivalent_noise_bandwidth_hz=bandwidth_bins * rate / value.size,
    )


def overlap_profile(window: Any, hop_length: int, *, squared: bool = True) -> np.ndarray:
    """Return the infinite-lattice overlap profile indexed by hop residue.

    For ``squared=True`` this is the nonzero-overlap-add profile used by an
    analysis/synthesis pair with the same real window.  For ``squared=False``
    it is the ordinary constant-overlap-add profile.
    """

    value = np.asarray(window)
    if value.ndim != 1 or value.size == 0:
        raise DimensionError("overlap profiling requires a nonempty one dimensional window")
    hop = _positive_integer(hop_length, "hop_length")
    if hop > value.size:
        raise ValidationError("hop_length cannot exceed the window length")
    terms = np.abs(value) ** 2 if squared else value
    dtype = np.result_type(terms.dtype, float)
    profile = np.zeros(hop, dtype=dtype)
    for index, coefficient in enumerate(terms):
        profile[index % hop] += coefficient
    return np.real_if_close(profile, tol=1000)


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-C01",
        title="Finite lattice overlap-add certificate",
        status=ClaimStatus.FINITE_CERTIFICATE,
        source="Block 12",
        assumptions=("uniform integer hop", "infinite shift lattice away from finite boundaries"),
        verification_suite="signals.windows",
    )
)
def overlap_add_certificate(
    window: Any,
    hop_length: int,
    *,
    squared: bool = True,
    require_constant: bool = False,
    tolerance: float = 1e-12,
) -> Certificate:
    """Certify nonzero overlap and optionally constant overlap on the shift lattice."""

    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValidationError("tolerance must be finite and nonnegative")
    profile = np.asarray(overlap_profile(window, hop_length, squared=squared))
    magnitude = np.abs(profile)
    minimum = float(np.min(magnitude))
    spread = float(np.max(np.abs(profile - profile[0])))
    scale = max(1.0, float(np.max(magnitude)))
    nonzero = minimum > tolerance * scale
    constant = spread <= tolerance * scale
    valid = nonzero and (constant or not require_constant)
    condition = float(np.max(magnitude) / minimum) if nonzero else float("inf")
    return Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={"overlap_spread": spread, "minimum_overlap": minimum},
        tolerance=tolerance,
        assumptions_checked=("finite window", "uniform integer hop"),
        assumptions_unchecked=("finite-record boundary coverage",),
        diagnostics={
            "squared": bool(squared),
            "constant_overlap": bool(constant),
            "overlap_condition_number": condition,
            "profile": np.real_if_close(profile, tol=1000).tolist(),
        },
        theorem_id="KM-DSP-C01",
        message="overlap-add condition verified" if valid else "overlap-add condition failed",
    )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-T02",
        title="Canonical finite lattice dual window",
        status=ClaimStatus.PROVED,
        source="Block 12",
        assumptions=("nonzero squared overlap profile", "uniform integer hop"),
        verification_suite="signals.windows",
    )
)
def canonical_dual_window(
    window: Any,
    hop_length: int,
    *,
    tolerance: float = 1e-12,
) -> ComputationResult[np.ndarray]:
    """Return the canonical synthesis window for the declared shift lattice."""

    value = np.asarray(window)
    if value.ndim != 1 or value.size == 0:
        raise DimensionError("dual-window construction requires a nonempty one dimensional window")
    profile = np.asarray(overlap_profile(value, hop_length, squared=True), dtype=float)
    scale = max(1.0, float(np.max(np.abs(profile))))
    minimum = float(np.min(np.abs(profile)))
    valid = minimum > tolerance * scale
    if valid:
        denominator = profile[np.arange(value.size) % int(hop_length)]
        dual = np.conjugate(value) / denominator
        cross_profile = overlap_profile(value * dual, hop_length, squared=False)
        residual = float(np.max(np.abs(cross_profile - 1.0)))
        valid = residual <= tolerance * max(1.0, float(np.max(np.abs(cross_profile))))
    else:
        dual = np.zeros_like(value, dtype=np.result_type(value.dtype, float))
        residual = float("inf")
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={"dual_overlap": residual, "minimum_squared_overlap": minimum},
        tolerance=tolerance,
        assumptions_checked=("uniform integer hop",),
        assumptions_unchecked=("finite-record boundary coverage",),
        theorem_id="KM-DSP-T02",
        message="canonical dual window verified" if valid else "canonical dual window does not exist",
    )
    return ComputationResult(np.asarray(dual), certificate)


__all__ = [
    "WindowMetrics",
    "WindowName",
    "WindowSpec",
    "canonical_dual_window",
    "get_window",
    "overlap_add_certificate",
    "overlap_profile",
    "window_metrics",
]
