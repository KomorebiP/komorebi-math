"""Finite recursive digital filters, biquad design, and chunked state transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .filtering import FrequencyResponse

BiquadKind = Literal["lowpass", "highpass", "bandpass", "notch", "allpass"]


def _vector(values: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise DimensionError(f"{name} must be one dimensional")
    if not allow_empty and result.size == 0:
        raise DimensionError(f"{name} cannot be empty")
    if not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must contain only finite values")
    return result


def _positive_rate(value: float) -> float:
    rate = float(value)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValidationError("sample rate must be finite and positive")
    return rate


@dataclass(frozen=True, slots=True)
class FilterOutput:
    """Filtered samples paired with the final direct-form state."""

    samples: np.ndarray
    final_state: np.ndarray


@dataclass(frozen=True, slots=True)
class LinearFilter:
    """Normalized causal FIR/IIR filter in direct-form II transposed form."""

    numerator: np.ndarray
    denominator: np.ndarray

    def __init__(self, numerator: Any, denominator: Any = (1.0,)) -> None:
        b0 = _vector(numerator, "numerator")
        a0 = _vector(denominator, "denominator")
        dtype = np.dtype(np.result_type(b0.dtype, a0.dtype, float))
        b = np.asarray(b0, dtype=dtype)
        a = np.asarray(a0, dtype=dtype)
        if abs(complex(a[0])) <= np.finfo(float).tiny:
            raise ValidationError("the first denominator coefficient cannot be zero")
        b = b / a[0]
        a = a / a[0]
        length = max(b.size, a.size)
        b = np.pad(b, (0, length - b.size))
        a = np.pad(a, (0, length - a.size))
        object.__setattr__(self, "numerator", b)
        object.__setattr__(self, "denominator", a)

    @property
    def order(self) -> int:
        """Return the direct-form state dimension."""

        return int(self.numerator.size - 1)

    @property
    def zeros(self) -> np.ndarray:
        """Return finite transfer-function zeros in the z-plane."""

        if self.numerator.size <= 1:
            return np.zeros(0, dtype=complex)
        first = int(np.argmax(np.abs(self.numerator) > np.finfo(float).tiny))
        coefficients = self.numerator[first:]
        if coefficients.size <= 1:
            return np.zeros(0, dtype=complex)
        return np.roots(coefficients)

    @property
    def poles(self) -> np.ndarray:
        """Return finite transfer-function poles in the z-plane."""

        if self.denominator.size <= 1:
            return np.zeros(0, dtype=complex)
        return np.roots(self.denominator)

    def process(self, signal: Any, *, initial_state: Any | None = None) -> FilterOutput:
        """Filter one finite signal and return the state required for continuation."""

        source = _vector(signal, "signal", allow_empty=True)
        state_dtype = np.asarray(initial_state).dtype if initial_state is not None else float
        dtype = np.dtype(
            np.result_type(source.dtype, self.numerator.dtype, self.denominator.dtype, state_dtype)
        )
        if initial_state is None:
            state = np.zeros(self.order, dtype=dtype)
        else:
            state = np.asarray(initial_state, dtype=dtype).copy()
            if state.ndim != 1 or state.size != self.order:
                raise DimensionError("initial_state must have one entry per filter order")
            if not np.all(np.isfinite(state)):
                raise ValidationError("initial_state entries must be finite")
        output = np.empty(source.size, dtype=dtype)
        if self.order == 0:
            output[:] = self.numerator[0] * source
            return FilterOutput(np.real_if_close(output, tol=1000), state)

        b = self.numerator.astype(dtype, copy=False)
        a = self.denominator.astype(dtype, copy=False)
        for index, sample in enumerate(source):
            current = b[0] * sample + state[0]
            output[index] = current
            if self.order > 1:
                state[:-1] = state[1:] + b[1:-1] * sample - a[1:-1] * current
            state[-1] = b[-1] * sample - a[-1] * current
        return FilterOutput(
            np.real_if_close(output, tol=1000),
            np.real_if_close(state, tol=1000),
        )

    def response(
        self,
        *,
        points: int = 512,
        sample_rate: float = 1.0,
        whole: bool = False,
    ) -> FrequencyResponse:
        """Evaluate the rational transfer function on a deterministic grid."""

        if isinstance(points, bool) or int(points) != points or int(points) <= 0:
            raise ValidationError("frequency-response point count must be positive")
        count = int(points)
        rate = _positive_rate(sample_rate)
        stop = rate if whole else rate / 2.0
        frequencies = np.linspace(0.0, stop, count, endpoint=not whole)
        omega = 2.0 * np.pi * frequencies / rate
        numerator_terms = np.exp(-1j * np.outer(omega, np.arange(self.numerator.size)))
        denominator_terms = np.exp(-1j * np.outer(omega, np.arange(self.denominator.size)))
        top = numerator_terms @ self.numerator
        bottom = denominator_terms @ self.denominator
        if np.any(np.abs(bottom) <= np.finfo(float).tiny):
            raise ValidationError("frequency response is singular on the requested grid")
        return FrequencyResponse(frequencies, top / bottom, rate)

    def stability_certificate(self, *, tolerance: float = 1e-12) -> Certificate:
        """Certify strict BIBO stability by pole containment in the unit disk."""

        allowed = float(tolerance)
        if not np.isfinite(allowed) or allowed < 0.0:
            raise ValidationError("tolerance must be finite and nonnegative")
        poles = self.poles
        radius = float(np.max(np.abs(poles))) if poles.size else 0.0
        margin = 1.0 - radius
        valid = margin > allowed
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"maximum_pole_radius": radius, "stability_margin": margin},
            tolerance=allowed,
            assumptions_checked=("finite rational transfer function",),
            theorem_id="KM-DSP-C04",
            message="all poles lie strictly inside the unit circle" if valid else "strict pole stability failed",
        )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D26",
        title="Direct-form II transposed finite recursion",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("finite normalized transfer coefficients",),
        verification_suite="signals.recursive",
    )
)
def lfilter(
    numerator: Any,
    denominator: Any,
    signal: Any,
    *,
    initial_state: Any | None = None,
) -> FilterOutput:
    """Apply a finite causal rational filter."""

    return LinearFilter(numerator, denominator).process(signal, initial_state=initial_state)


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D27",
        title="Normalized audio-EQ biquad construction",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("frequency strictly inside Nyquist", "positive Q"),
        verification_suite="signals.recursive",
    )
)
def design_biquad(
    kind: BiquadKind,
    frequency: float,
    *,
    q: float = 1.0 / np.sqrt(2.0),
    sample_rate: float = 1.0,
) -> LinearFilter:
    """Design a lowpass, highpass, bandpass, notch, or allpass biquad."""

    rate = _positive_rate(sample_rate)
    center = float(frequency)
    quality = float(q)
    if not np.isfinite(center) or center <= 0.0 or center >= rate / 2.0:
        raise ValidationError("biquad frequency must lie strictly between zero and Nyquist")
    if not np.isfinite(quality) or quality <= 0.0:
        raise ValidationError("biquad Q must be finite and positive")
    omega = 2.0 * np.pi * center / rate
    cosine = np.cos(omega)
    alpha = np.sin(omega) / (2.0 * quality)
    denominator = np.array([1.0 + alpha, -2.0 * cosine, 1.0 - alpha])
    if kind == "lowpass":
        numerator = np.array([(1.0 - cosine) / 2.0, 1.0 - cosine, (1.0 - cosine) / 2.0])
    elif kind == "highpass":
        numerator = np.array([(1.0 + cosine) / 2.0, -(1.0 + cosine), (1.0 + cosine) / 2.0])
    elif kind == "bandpass":
        numerator = np.array([alpha, 0.0, -alpha])
    elif kind == "notch":
        numerator = np.array([1.0, -2.0 * cosine, 1.0])
    elif kind == "allpass":
        numerator = np.array([1.0 - alpha, -2.0 * cosine, 1.0 + alpha])
    else:
        raise ValidationError(f"unsupported biquad kind: {kind}")
    return LinearFilter(numerator, denominator)


@dataclass(slots=True)
class StreamingLinearFilter:
    """Mutable chunked execution of one finite-order linear filter."""

    filter: LinearFilter
    state: np.ndarray = field(init=False)
    processed_samples: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        dtype = np.result_type(self.filter.numerator.dtype, self.filter.denominator.dtype)
        self.state = np.zeros(self.filter.order, dtype=dtype)

    def process(self, chunk: Any) -> np.ndarray:
        """Process one chunk and retain the exact continuation state."""

        result = self.filter.process(chunk, initial_state=self.state)
        self.state = np.asarray(result.final_state).copy()
        self.processed_samples += result.samples.size
        return result.samples

    def reset(self) -> None:
        """Restore zero state and reset the processed-sample count."""

        self.state = np.zeros_like(self.state)
        self.processed_samples = 0


def stream_lfilter(
    filter_: LinearFilter,
    signal: Any,
    chunk_sizes: Any,
    *,
    tolerance: float = 1e-12,
) -> ComputationResult[np.ndarray]:
    """Execute declared chunks and certify equality with one batch recursion."""

    source = _vector(signal, "signal")
    processor = StreamingLinearFilter(filter_)
    pieces: list[np.ndarray] = []
    offset = 0
    for raw_size in chunk_sizes:
        if isinstance(raw_size, bool) or int(raw_size) != raw_size or int(raw_size) <= 0:
            raise ValidationError("chunk sizes must be positive integers")
        if offset >= source.size:
            break
        size = min(int(raw_size), source.size - offset)
        pieces.append(processor.process(source[offset : offset + size]))
        offset += size
    if offset < source.size:
        pieces.append(processor.process(source[offset:]))
    streamed = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.result_type(source, float))
    reference = filter_.process(source).samples
    residual = float(np.max(np.abs(streamed - reference))) if reference.size else 0.0
    scale = max(1.0, float(np.max(np.abs(reference))) if reference.size else 1.0)
    certificate = Certificate(
        valid=bool(streamed.shape == reference.shape and residual <= tolerance * scale),
        exactness=Exactness.NUMERICAL,
        residuals={"streaming_batch_max_abs": residual},
        tolerance=tolerance,
        assumptions_checked=("zero initial state", "ordered contiguous chunks"),
        theorem_id="KM-DSP-T09",
        message="streaming recursive filter equals batch recursion",
    )
    return ComputationResult(streamed, certificate)


__all__ = [
    "BiquadKind",
    "FilterOutput",
    "LinearFilter",
    "StreamingLinearFilter",
    "design_biquad",
    "lfilter",
    "stream_lfilter",
]
