"""Finite framing, short-time Fourier transforms, and overlap-add synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import AssumptionError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .windows import WindowSpec, get_window

BoundaryMode = Literal["zeros", "reflect", "edge", "none"]
DetrendMode = Literal["none", "constant"]
SpectrogramScaling = Literal["density", "spectrum", "magnitude", "power"]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return int(value)


def _finite_norm(value: np.ndarray) -> float:
    """Evaluate a Euclidean norm without squaring the original scale."""

    magnitude = np.abs(value)
    largest = float(np.max(magnitude))
    if not np.isfinite(largest):
        return float("inf")
    if largest == 0:
        return 0.0
    return largest * float(np.linalg.norm(value / largest))


def _pad(array: np.ndarray, left: int, right: int, boundary: BoundaryMode) -> np.ndarray:
    if left == 0 and right == 0:
        return array.copy()
    if boundary == "none":
        raise ValidationError("boundary='none' cannot be used when padding is required")
    if boundary == "zeros":
        return np.pad(array, (left, right), mode="constant")
    if boundary == "edge":
        return np.pad(array, (left, right), mode="edge")
    if boundary == "reflect":
        if array.size < 2:
            raise ValidationError("reflect padding requires at least two signal samples")
        return np.pad(array, (left, right), mode="reflect")
    raise ValidationError(f"unknown boundary mode: {boundary}")


@dataclass(frozen=True, slots=True)
class FrameBatch:
    """A deterministic finite framing of one one-dimensional signal."""

    frames: np.ndarray
    starts: np.ndarray
    original_length: int
    frame_length: int
    hop_length: int
    pad_left: int
    pad_right: int
    boundary: BoundaryMode

    @property
    def frame_count(self) -> int:
        """Return the number of complete frames."""

        return int(self.frames.shape[0])

    @property
    def padded_length(self) -> int:
        """Return the represented padded signal length."""

        return int(self.original_length + self.pad_left + self.pad_right)

    @property
    def covered_length(self) -> int:
        """Return the sample range covered by the complete frames."""

        return int(self.starts[-1] + self.frame_length)

    def certificate(self) -> Certificate:
        """Validate the finite framing layout."""

        expected_starts = np.arange(self.frame_count, dtype=int) * self.hop_length
        valid = (
            self.frames.ndim == 2
            and self.frames.shape[1] == self.frame_length
            and np.array_equal(self.starts, expected_starts)
            and self.covered_length <= self.padded_length
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals={"uncovered_tail": float(self.padded_length - self.covered_length)},
            assumptions_checked=("complete equal-length frames", "uniform integer hop"),
            theorem_id="KM-DSP-D02",
            message="finite framing layout verified" if valid else "finite framing layout failed",
        )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D02",
        title="Finite uniform signal framing",
        status=ClaimStatus.DEFINITION,
        source="Block 12",
        assumptions=("finite one dimensional signal", "positive frame and hop lengths"),
        verification_suite="signals.time_frequency",
    )
)
def frame_signal(
    signal: Any,
    frame_length: int,
    hop_length: int | None = None,
    *,
    center: bool = False,
    pad_end: bool = False,
    boundary: BoundaryMode = "zeros",
) -> FrameBatch:
    """Split a finite signal into complete uniformly spaced frames."""

    value = np.asarray(signal)
    if value.ndim != 1 or value.size == 0:
        raise DimensionError("framing requires a nonempty one dimensional signal")
    length = _positive_integer(frame_length, "frame_length")
    hop = length if hop_length is None else _positive_integer(hop_length, "hop_length")
    if hop > length:
        raise ValidationError("hop_length cannot exceed frame_length")
    left = length // 2 if center else 0
    right = length // 2 if center else 0
    base_length = value.size + left + right
    if base_length < length:
        if not pad_end:
            raise DimensionError("signal is shorter than frame_length and pad_end is false")
        right += length - base_length
        base_length = length
    if pad_end:
        frame_count = max(1, int(np.ceil((base_length - length) / hop)) + 1)
        covered = (frame_count - 1) * hop + length
        right += covered - base_length
    else:
        frame_count = 1 + (base_length - length) // hop
        covered = (frame_count - 1) * hop + length
    padded = _pad(value, left, right, boundary)
    starts = np.arange(frame_count, dtype=int) * hop
    frames = np.stack([padded[start : start + length] for start in starts], axis=0)
    batch = FrameBatch(
        frames=frames,
        starts=starts,
        original_length=int(value.size),
        frame_length=length,
        hop_length=hop,
        pad_left=left,
        pad_right=right,
        boundary=boundary,
    )
    batch.certificate().require()
    return batch


def overlap_add(
    frames: Any,
    hop_length: int,
    *,
    synthesis_window: Any | None = None,
    length: int | None = None,
) -> np.ndarray:
    """Overlap and add complete frames with an optional synthesis window."""

    value = np.asarray(frames)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise DimensionError("overlap-add requires a nonempty two dimensional frame array")
    hop = _positive_integer(hop_length, "hop_length")
    if hop > value.shape[1]:
        raise ValidationError("hop_length cannot exceed the frame length")
    if synthesis_window is None:
        window = np.ones(value.shape[1], dtype=float)
    else:
        window = np.asarray(synthesis_window)
        if window.ndim != 1 or window.size != value.shape[1]:
            raise DimensionError("synthesis window length must equal the frame length")
    output_length = (value.shape[0] - 1) * hop + value.shape[1]
    result = np.zeros(output_length, dtype=np.result_type(value.dtype, window.dtype))
    for index, frame in enumerate(value):
        start = index * hop
        result[start : start + value.shape[1]] += frame * window
    if length is not None:
        requested = _positive_integer(length, "length")
        if requested > result.size:
            raise DimensionError("requested overlap-add length exceeds the represented range")
        result = result[:requested]
    return np.real_if_close(result, tol=1000)


@dataclass(frozen=True, slots=True)
class STFTResult:
    """Finite short-time Fourier coefficients and complete synthesis metadata."""

    coefficients: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    sample_rate: float
    frame_length: int
    hop_length: int
    fft_length: int
    analysis_window: np.ndarray
    original_length: int
    pad_left: int
    pad_right: int
    onesided: bool
    boundary: BoundaryMode
    detrend: DetrendMode

    @property
    def frame_count(self) -> int:
        """Return the number of time frames."""

        return int(self.coefficients.shape[0])

    def inverse(
        self,
        *,
        synthesis_window: Any | None = None,
        length: int | None = None,
        tolerance: float = 1e-11,
    ) -> ComputationResult[np.ndarray]:
        """Invert this finite STFT by normalized overlap-add."""

        return istft(self, synthesis_window=synthesis_window, length=length, tolerance=tolerance)

    def roundtrip_certificate(self, signal: Any, *, tolerance: float = 1e-10) -> Certificate:
        """Compare normalized synthesis with a supplied source signal."""

        source = np.asarray(signal)
        if source.ndim != 1 or source.size != self.original_length:
            raise DimensionError("roundtrip source must match the STFT original length")
        if not np.all(np.isfinite(source)):
            raise ValidationError("roundtrip source entries must be finite")
        source = np.asarray(source, dtype=np.result_type(source.dtype, float))
        reconstructed = self.inverse(tolerance=tolerance)
        if not reconstructed.valid:
            return reconstructed.certificate
        with np.errstate(over="ignore", invalid="ignore"):
            residual = _finite_norm(reconstructed.value - source)
            scale = max(1.0, _finite_norm(source))
        valid = bool(np.isfinite(residual) and np.isfinite(scale) and residual <= tolerance * scale)
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"stft_roundtrip": residual},
            tolerance=tolerance,
            assumptions_checked=("matching finite source", "normalized overlap-add"),
            theorem_id="KM-DSP-T03",
            message="STFT roundtrip verified" if valid else "STFT roundtrip failed",
        )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D03",
        title="Finite short-time Fourier transform",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("uniform finite framing", "declared window and Fourier length"),
        verification_suite="signals.time_frequency",
    )
)
def stft(
    signal: Any,
    *,
    sample_rate: float = 1.0,
    frame_length: int = 256,
    hop_length: int | None = None,
    fft_length: int | None = None,
    window: WindowSpec = "hann",
    center: bool = True,
    pad_end: bool = True,
    boundary: BoundaryMode = "zeros",
    onesided: bool | None = None,
    detrend: DetrendMode = "none",
) -> STFTResult:
    """Compute a finite STFT with metadata sufficient for normalized synthesis."""

    value = np.asarray(signal)
    if value.ndim != 1 or value.size == 0:
        raise DimensionError("STFT requires a nonempty one dimensional signal")
    if not np.all(np.isfinite(value)):
        raise ValidationError("STFT signal entries must be finite")
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValidationError("sample_rate must be positive and finite")
    length = _positive_integer(frame_length, "frame_length")
    hop = max(1, length // 4) if hop_length is None else _positive_integer(hop_length, "hop_length")
    transform_length = length if fft_length is None else _positive_integer(fft_length, "fft_length")
    if transform_length < length:
        raise ValidationError("fft_length cannot be smaller than frame_length")
    if detrend not in ("none", "constant"):
        raise ValidationError("detrend must be 'none' or 'constant'")
    analysis_window = np.asarray(get_window(window, length, periodic=True))
    one_sided = bool(np.isrealobj(value) and np.isrealobj(analysis_window)) if onesided is None else bool(onesided)
    if one_sided and (not np.isrealobj(value) or not np.isrealobj(analysis_window)):
        raise ValidationError("one-sided STFT requires a real signal and analysis window")
    batch = frame_signal(
        value,
        length,
        hop,
        center=center,
        pad_end=pad_end,
        boundary=boundary,
    )
    frames = np.asarray(batch.frames, dtype=np.result_type(value.dtype, analysis_window.dtype))
    if detrend == "constant":
        frames = frames - np.mean(frames, axis=1, keepdims=True)
    windowed = frames * analysis_window[np.newaxis, :]
    if one_sided:
        coefficients = np.fft.rfft(windowed, n=transform_length, axis=1)
        frequencies = np.fft.rfftfreq(transform_length, d=1.0 / rate)
    else:
        coefficients = np.fft.fft(windowed, n=transform_length, axis=1)
        frequencies = np.fft.fftfreq(transform_length, d=1.0 / rate)
    times = (batch.starts + 0.5 * length - batch.pad_left) / rate
    return STFTResult(
        coefficients=coefficients,
        frequencies=frequencies,
        times=np.asarray(times, dtype=float),
        sample_rate=rate,
        frame_length=length,
        hop_length=hop,
        fft_length=transform_length,
        analysis_window=analysis_window,
        original_length=batch.original_length,
        pad_left=batch.pad_left,
        pad_right=batch.pad_right,
        onesided=one_sided,
        boundary=boundary,
        detrend=detrend,
    )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-T03",
        title="Normalized finite STFT overlap-add reconstruction",
        status=ClaimStatus.PROVED,
        source="Block 12",
        assumptions=("no destructive detrending", "nonzero finite overlap denominator"),
        verification_suite="signals.time_frequency",
    )
)
def istft(
    transform: STFTResult,
    *,
    synthesis_window: Any | None = None,
    length: int | None = None,
    tolerance: float = 1e-11,
) -> ComputationResult[np.ndarray]:
    """Invert a :class:`STFTResult` by normalized finite overlap-add."""

    if not isinstance(transform, STFTResult):
        raise ValidationError("istft requires an STFTResult")
    if transform.detrend != "none":
        raise AssumptionError("constant detrending discards frame means and is not invertible")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValidationError("tolerance must be finite and nonnegative")
    if not np.all(np.isfinite(transform.coefficients)):
        raise ValidationError("STFT coefficients must be finite")
    if not np.all(np.isfinite(transform.analysis_window)):
        raise ValidationError("analysis window entries must be finite")
    if synthesis_window is None:
        synthesis = np.conjugate(np.asarray(transform.analysis_window))
    else:
        synthesis = np.asarray(synthesis_window)
        if synthesis.ndim != 1 or synthesis.size != transform.frame_length:
            raise DimensionError("synthesis window length must equal frame_length")
    if not np.all(np.isfinite(synthesis)):
        raise ValidationError("synthesis window entries must be finite")
    if transform.onesided:
        frames = np.fft.irfft(transform.coefficients, n=transform.fft_length, axis=1)
    else:
        frames = np.fft.ifft(transform.coefficients, n=transform.fft_length, axis=1)
    frames = frames[:, : transform.frame_length]
    represented_length = (transform.frame_count - 1) * transform.hop_length + transform.frame_length
    numerator = np.zeros(represented_length, dtype=np.result_type(frames.dtype, synthesis.dtype))
    denominator = np.zeros(represented_length, dtype=np.result_type(transform.analysis_window.dtype, synthesis.dtype, float))
    cross_window = transform.analysis_window * synthesis
    for index, frame in enumerate(frames):
        start = index * transform.hop_length
        stop = start + transform.frame_length
        numerator[start:stop] += frame * synthesis
        denominator[start:stop] += cross_window
    requested = transform.original_length if length is None else _positive_integer(length, "length")
    if requested > transform.original_length:
        raise DimensionError("requested inverse length exceeds the original signal length")
    start = transform.pad_left
    stop = start + requested
    if stop > represented_length:
        raise DimensionError("STFT frames do not cover the requested inverse range")
    local_denominator = denominator[start:stop]
    scale = float(np.max(np.abs(local_denominator)))
    minimum = float(np.min(np.abs(local_denominator))) if local_denominator.size else 0.0
    valid = bool(
        np.all(np.isfinite(numerator[start:stop]))
        and np.all(np.isfinite(local_denominator))
        and minimum > tolerance * scale
    )
    if valid:
        reconstructed = numerator[start:stop] / local_denominator
        reconstruction_identity = float(
            np.max(np.abs(reconstructed * local_denominator - numerator[start:stop]))
        )
        valid = bool(
            np.all(np.isfinite(reconstructed))
            and np.isfinite(reconstruction_identity)
            and reconstruction_identity <= tolerance * max(1.0, float(np.max(np.abs(numerator[start:stop]))))
        )
    else:
        reconstructed = np.zeros(requested, dtype=numerator.dtype)
        reconstruction_identity = float("inf")
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={
            "normalized_overlap_identity": reconstruction_identity,
            "minimum_overlap": minimum,
        },
        tolerance=tolerance,
        assumptions_checked=("finite Fourier inversion", "finite overlap normalization"),
        assumptions_unchecked=("source agreement requires a separate roundtrip comparison",),
        diagnostics={"represented_length": represented_length, "requested_length": requested},
        theorem_id="KM-DSP-T03",
        message="normalized STFT synthesis verified" if valid else "STFT overlap denominator vanished",
    )
    return ComputationResult(np.real_if_close(reconstructed, tol=1000), certificate)


@dataclass(frozen=True, slots=True)
class SpectrogramResult:
    """A finite time-frequency magnitude or power representation."""

    values: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    scaling: SpectrogramScaling
    sample_rate: float
    transform: STFTResult



def _double_onesided(values: np.ndarray, fft_length: int) -> np.ndarray:
    result = values.copy()
    if result.shape[1] <= 1:
        return result
    if fft_length % 2 == 0:
        result[:, 1:-1] *= 2.0
    else:
        result[:, 1:] *= 2.0
    return result


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D04",
        title="Finite spectrogram scaling",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("finite STFT", "declared spectral scaling"),
        verification_suite="signals.time_frequency",
    )
)
def spectrogram(
    signal: Any,
    *,
    scaling: SpectrogramScaling = "density",
    **stft_options: Any,
) -> SpectrogramResult:
    """Compute a magnitude, power, spectrum, or density spectrogram."""

    transform = stft(signal, **stft_options)
    magnitude = np.abs(transform.coefficients)
    if scaling == "magnitude":
        values = magnitude
    elif scaling == "power":
        values = magnitude**2
        if transform.onesided:
            values = _double_onesided(values, transform.fft_length)
    elif scaling == "density":
        denominator = transform.sample_rate * float(np.sum(np.abs(transform.analysis_window) ** 2))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValidationError("density scaling requires positive finite window energy")
        values = magnitude**2 / denominator
        if transform.onesided:
            values = _double_onesided(values, transform.fft_length)
    elif scaling == "spectrum":
        coherent = abs(complex(np.sum(transform.analysis_window))) ** 2
        if not np.isfinite(coherent) or coherent <= 0:
            raise ValidationError("spectrum scaling requires positive finite squared coherent gain")
        values = magnitude**2 / coherent
        if transform.onesided:
            values = _double_onesided(values, transform.fft_length)
    else:
        raise ValidationError(f"unknown spectrogram scaling: {scaling}")
    return SpectrogramResult(
        values=np.asarray(values, dtype=float),
        frequencies=transform.frequencies,
        times=transform.times,
        scaling=scaling,
        sample_rate=transform.sample_rate,
        transform=transform,
    )


__all__ = [
    "BoundaryMode",
    "DetrendMode",
    "FrameBatch",
    "STFTResult",
    "SpectrogramResult",
    "SpectrogramScaling",
    "frame_signal",
    "istft",
    "overlap_add",
    "spectrogram",
    "stft",
]
