"""Finite FIR design, direct filtering, and stateful streaming execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .windows import WindowSpec, get_window

FIRKind = Literal["lowpass", "highpass", "bandpass", "bandstop"]
FIRMode = Literal["causal", "full", "same", "valid"]


def _vector(values: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise DimensionError(f"{name} must be one dimensional")
    if result.size == 0 and not allow_empty:
        raise DimensionError(f"{name} cannot be empty")
    if not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must contain only finite values")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return int(value)


def _sample_rate(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValidationError("sample_rate must be positive and finite")
    return result


def _extract_taps(taps: Any) -> np.ndarray:
    if isinstance(taps, FIRFilter):
        return taps.taps
    return _vector(taps, "FIR taps")


def _crop(full: np.ndarray, signal_length: int, tap_count: int, mode: FIRMode) -> np.ndarray:
    if mode == "full":
        return full
    if mode == "causal":
        return full[:signal_length]
    if mode == "same":
        start = (tap_count - 1) // 2
        return full[start : start + signal_length]
    if mode == "valid":
        length = max(signal_length - tap_count + 1, 0)
        if length == 0:
            return np.zeros(0, dtype=full.dtype)
        start = tap_count - 1
        return full[start : start + length]
    raise ValidationError(f"unknown FIR mode: {mode}")


@dataclass(frozen=True, slots=True)
class FrequencyResponse:
    """Sampled FIR transfer values on a physical frequency grid."""

    frequencies: np.ndarray
    response: np.ndarray
    sample_rate: float

    @property
    def magnitude(self) -> np.ndarray:
        """Return the response magnitude."""

        return np.abs(self.response)

    @property
    def phase(self) -> np.ndarray:
        """Return the unwrapped phase response."""

        return np.unwrap(np.angle(self.response))

    @property
    def magnitude_db(self) -> np.ndarray:
        """Return finite-clipped response magnitude in decibels."""

        return 20.0 * np.log10(np.maximum(self.magnitude, np.finfo(float).tiny))


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D05",
        title="Finite FIR frequency response",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("finite impulse response", "positive sample rate"),
        verification_suite="signals.filtering",
    )
)
def fir_frequency_response(
    taps: Any,
    *,
    points: int = 512,
    sample_rate: float = 1.0,
    whole: bool = False,
) -> FrequencyResponse:
    """Evaluate a finite impulse response on an equally spaced frequency grid."""

    coefficients = _extract_taps(taps)
    count = _positive_integer(points, "points")
    rate = _sample_rate(sample_rate)
    stop = rate if whole else rate / 2.0
    frequencies = np.linspace(0.0, stop, count, endpoint=not whole)
    omega = 2.0 * np.pi * frequencies / rate
    response = np.exp(-1j * np.outer(omega, np.arange(coefficients.size))) @ coefficients
    return FrequencyResponse(frequencies, response, rate)


@dataclass(frozen=True, slots=True)
class FIRFilter:
    """A finite linear-phase FIR design and its declared design metadata."""

    taps: np.ndarray
    kind: FIRKind
    cutoff: tuple[float, ...]
    sample_rate: float = 1.0
    window_name: str = "custom"
    reference_frequency: float | None = None
    reference_gain: float = 1.0

    def __post_init__(self) -> None:
        coefficients = _vector(self.taps, "FIR taps")
        rate = _sample_rate(self.sample_rate)
        if self.kind not in {"lowpass", "highpass", "bandpass", "bandstop"}:
            raise ValidationError(f"unknown FIR kind: {self.kind}")
        if not self.cutoff:
            raise ValidationError("FIR cutoff metadata cannot be empty")
        object.__setattr__(self, "taps", coefficients.copy())
        object.__setattr__(self, "sample_rate", rate)
        object.__setattr__(self, "cutoff", tuple(float(value) for value in self.cutoff))

    @property
    def order(self) -> int:
        """Return the FIR polynomial order."""

        return int(self.taps.size - 1)

    @property
    def group_delay_samples(self) -> float:
        """Return the constant linear-phase group delay in samples."""

        return 0.5 * self.order

    def apply(self, signal: Any, *, mode: FIRMode = "causal") -> np.ndarray:
        """Apply this design to one finite signal."""

        return fir_filter(signal, self.taps, mode=mode)

    def response(self, *, points: int = 512, whole: bool = False) -> FrequencyResponse:
        """Return this design's sampled frequency response."""

        return fir_frequency_response(self.taps, points=points, sample_rate=self.sample_rate, whole=whole)

    def certificate(self, *, tolerance: float = 1e-10) -> Certificate:
        """Certify symmetry and the declared reference-frequency gain."""

        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValidationError("tolerance must be finite and nonnegative")
        symmetry = float(np.max(np.abs(self.taps - self.taps[::-1])))
        gain_residual = 0.0
        if self.reference_frequency is not None:
            omega = 2.0 * np.pi * self.reference_frequency / self.sample_rate
            value = np.sum(self.taps * np.exp(-1j * omega * np.arange(self.taps.size)))
            gain_residual = abs(float(abs(value)) - float(self.reference_gain))
        scale = max(1.0, float(np.max(np.abs(self.taps))), abs(float(self.reference_gain)))
        valid = symmetry <= tolerance * scale and gain_residual <= tolerance * scale
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"tap_symmetry": symmetry, "reference_gain": gain_residual},
            tolerance=tolerance,
            assumptions_checked=("finite real linear-phase design", "declared gain normalization"),
            theorem_id="KM-DSP-C02",
            message="FIR design certificate verified" if valid else "FIR design certificate failed",
        )


def _ideal_lowpass(num_taps: int, cutoff: float, sample_rate: float) -> np.ndarray:
    centered = np.arange(num_taps, dtype=float) - (num_taps - 1) / 2.0
    normalized_cycles = cutoff / sample_rate
    return 2.0 * normalized_cycles * np.sinc(2.0 * normalized_cycles * centered)


def _window_name(window: WindowSpec) -> str:
    if isinstance(window, tuple):
        return f"{window[0]}({float(window[1]):g})"
    if isinstance(window, str):
        return window
    return "custom"


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D06",
        title="Windowed-sinc finite FIR design",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("cutoffs strictly inside Nyquist", "finite tap count"),
        verification_suite="signals.filtering",
    )
)
def design_fir(
    kind: FIRKind,
    cutoff: float | tuple[float, float],
    *,
    num_taps: int = 101,
    sample_rate: float = 1.0,
    window: WindowSpec = "hann",
    scale: bool = True,
) -> FIRFilter:
    """Design a real linear-phase FIR filter by a windowed ideal impulse response."""

    taps = _positive_integer(num_taps, "num_taps")
    rate = _sample_rate(sample_rate)
    nyquist = rate / 2.0
    cutoffs = np.atleast_1d(np.asarray(cutoff, dtype=float))
    if cutoffs.ndim != 1 or cutoffs.size not in {1, 2}:
        raise DimensionError("cutoff must contain one or two frequencies")
    if np.any(~np.isfinite(cutoffs)) or np.any(cutoffs <= 0.0) or np.any(cutoffs >= nyquist):
        raise ValidationError("cutoffs must lie strictly between zero and Nyquist")
    if cutoffs.size == 2 and cutoffs[0] >= cutoffs[1]:
        raise ValidationError("band cutoffs must be strictly increasing")
    if kind in {"lowpass", "highpass"} and cutoffs.size != 1:
        raise DimensionError(f"{kind} requires one cutoff frequency")
    if kind in {"bandpass", "bandstop"} and cutoffs.size != 2:
        raise DimensionError(f"{kind} requires two cutoff frequencies")
    if taps % 2 == 0:
        raise ValidationError("windowed-sinc linear-phase designs require an odd tap count")

    impulse = np.zeros(taps, dtype=float)
    if taps % 2:
        impulse[taps // 2] = 1.0
    else:
        centered = np.arange(taps, dtype=float) - (taps - 1) / 2.0
        impulse = np.sinc(centered)

    if kind == "lowpass":
        coefficients = _ideal_lowpass(taps, float(cutoffs[0]), rate)
        reference_frequency = 0.0
    elif kind == "highpass":
        coefficients = impulse - _ideal_lowpass(taps, float(cutoffs[0]), rate)
        reference_frequency = nyquist
    elif kind == "bandpass":
        coefficients = _ideal_lowpass(taps, float(cutoffs[1]), rate) - _ideal_lowpass(
            taps, float(cutoffs[0]), rate
        )
        reference_frequency = float(np.mean(cutoffs))
    elif kind == "bandstop":
        coefficients = impulse - (
            _ideal_lowpass(taps, float(cutoffs[1]), rate)
            - _ideal_lowpass(taps, float(cutoffs[0]), rate)
        )
        reference_frequency = 0.0
    else:
        raise ValidationError(f"unknown FIR kind: {kind}")

    coefficients = coefficients * get_window(window, taps, periodic=False)
    reference_gain = 1.0
    if scale:
        omega = 2.0 * np.pi * reference_frequency / rate
        value = np.sum(coefficients * np.exp(-1j * omega * np.arange(taps)))
        magnitude = float(abs(value))
        if magnitude <= np.finfo(float).tiny:
            raise ValidationError("the selected reference frequency has zero FIR gain")
        coefficients = coefficients / magnitude
    else:
        omega = 2.0 * np.pi * reference_frequency / rate
        reference_gain = float(
            abs(np.sum(coefficients * np.exp(-1j * omega * np.arange(taps))))
        )
    return FIRFilter(
        taps=np.asarray(coefficients),
        kind=kind,
        cutoff=tuple(float(value) for value in cutoffs),
        sample_rate=rate,
        window_name=_window_name(window),
        reference_frequency=reference_frequency,
        reference_gain=reference_gain,
    )


def design_lowpass(cutoff: float, **options: Any) -> FIRFilter:
    """Design a lowpass FIR filter."""

    return design_fir("lowpass", cutoff, **options)


def design_highpass(cutoff: float, **options: Any) -> FIRFilter:
    """Design a highpass FIR filter."""

    return design_fir("highpass", cutoff, **options)


def design_bandpass(cutoff: tuple[float, float], **options: Any) -> FIRFilter:
    """Design a bandpass FIR filter."""

    return design_fir("bandpass", cutoff, **options)


def design_bandstop(cutoff: tuple[float, float], **options: Any) -> FIRFilter:
    """Design a bandstop FIR filter."""

    return design_fir("bandstop", cutoff, **options)


def fir_filter(signal: Any, taps: Any, *, mode: FIRMode = "causal") -> np.ndarray:
    """Apply finite convolution with a signal-length-preserving ``same`` convention."""

    source = _vector(signal, "signal")
    coefficients = _extract_taps(taps)
    full = np.convolve(source, coefficients, mode="full")
    return np.real_if_close(_crop(full, source.size, coefficients.size, mode), tol=1000)


@dataclass(slots=True)
class StreamingFIR:
    """Stateful causal FIR execution with bounded input history."""

    taps: np.ndarray
    _history: np.ndarray = field(init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    processed_samples: int = field(default=0, init=False)

    def __init__(self, taps: Any) -> None:
        coefficients = np.asarray(_extract_taps(taps)).copy()
        self.taps = coefficients
        self._history = np.zeros(max(0, coefficients.size - 1), dtype=coefficients.dtype)
        self._closed = False
        self.processed_samples = 0

    def process(self, chunk: Any) -> np.ndarray:
        """Filter a chunk and emit exactly one causal output per input sample."""

        if self._closed:
            raise ValidationError("cannot process samples after StreamingFIR.flush")
        values = _vector(chunk, "chunk", allow_empty=True)
        if values.size == 0:
            return np.zeros(0, dtype=np.result_type(values.dtype, self.taps.dtype))
        dtype = np.result_type(values.dtype, self.taps.dtype)
        history = self._history.astype(dtype, copy=False)
        combined = np.concatenate((history, values.astype(dtype, copy=False)))
        full = np.convolve(combined, self.taps.astype(dtype, copy=False), mode="full")
        start = self.taps.size - 1
        output = full[start : start + values.size]
        if self.taps.size > 1:
            self._history = combined[-(self.taps.size - 1) :].copy()
        self.processed_samples += values.size
        return np.real_if_close(output, tol=1000)

    def flush(self) -> np.ndarray:
        """Emit the remaining full-convolution tail exactly once."""

        if self._closed:
            return np.zeros(0, dtype=np.result_type(self.taps.dtype, float))
        self._closed = True
        if self.taps.size <= 1:
            return np.zeros(0, dtype=np.result_type(self.taps.dtype, float))
        zeros = np.zeros(self.taps.size - 1, dtype=self._history.dtype)
        dtype = np.result_type(zeros.dtype, self.taps.dtype)
        combined = np.concatenate((self._history.astype(dtype, copy=False), zeros))
        full = np.convolve(combined, self.taps.astype(dtype, copy=False), mode="full")
        start = self.taps.size - 1
        tail = full[start : start + zeros.size]
        self._history = np.zeros_like(self._history)
        return np.real_if_close(tail, tol=1000)

    def reset(self) -> None:
        """Reset the finite input history and reopen the processor."""

        self._history = np.zeros_like(self._history)
        self._closed = False
        self.processed_samples = 0


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-T05",
        title="Chunk-invariant finite FIR execution",
        status=ClaimStatus.PROVED,
        source="Block 12",
        assumptions=("ordered nonoverlapping chunks", "common FIR taps"),
        verification_suite="signals.filtering",
    )
)
def stream_fir_filter(
    signal: Any,
    taps: Any,
    chunk_sizes: Iterable[int],
    *,
    tolerance: float = 1e-12,
) -> ComputationResult[np.ndarray]:
    """Filter a signal by declared chunks and certify causal batch equivalence."""

    source = _vector(signal, "signal")
    sizes = tuple(_positive_integer(size, "chunk size") for size in chunk_sizes)
    if sum(sizes) != source.size:
        raise DimensionError("chunk sizes must sum to the signal length")
    processor = StreamingFIR(taps)
    pieces: list[np.ndarray] = []
    start = 0
    for size in sizes:
        pieces.append(processor.process(source[start : start + size]))
        start += size
    streamed = np.concatenate(pieces) if pieces else np.zeros(0, dtype=source.dtype)
    direct = fir_filter(source, taps, mode="causal")
    residual = float(np.linalg.norm(streamed - direct))
    scale = max(1.0, float(np.linalg.norm(direct)))
    valid = residual <= tolerance * scale
    return ComputationResult(
        streamed,
        Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"streaming_batch_equivalence": residual},
            tolerance=tolerance,
            assumptions_checked=("ordered complete chunk partition", "finite causal FIR"),
            theorem_id="KM-DSP-T05",
            message="streaming FIR equals batch FIR" if valid else "streaming FIR differs from batch FIR",
        ),
    )


__all__ = [
    "FIRFilter",
    "FIRKind",
    "FIRMode",
    "FrequencyResponse",
    "StreamingFIR",
    "design_bandpass",
    "design_bandstop",
    "design_fir",
    "design_highpass",
    "design_lowpass",
    "fir_filter",
    "fir_frequency_response",
    "stream_fir_filter",
]
