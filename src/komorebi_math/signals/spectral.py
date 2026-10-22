"""Finite Fourier spectra, Welch estimates, cross spectra, and coherence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .time_frequency import DetrendMode, frame_signal
from .windows import WindowSpec, get_window

SpectralScaling = Literal["density", "spectrum"]
SpectralAverage = Literal["mean", "median"]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return int(value)


def _rate(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValidationError("sample_rate must be positive and finite")
    return result


def _validate_signal(signal: Any, name: str = "signal") -> np.ndarray:
    value = np.asarray(signal)
    if value.ndim != 1 or value.size == 0:
        raise DimensionError(f"{name} must be a nonempty one dimensional signal")
    if not np.all(np.isfinite(value)):
        raise ValidationError(f"{name} entries must be finite")
    return np.asarray(value, dtype=np.result_type(value.dtype, float))


def _detrend(frames: np.ndarray, mode: DetrendMode) -> np.ndarray:
    if mode == "none":
        return frames
    if mode == "constant":
        return frames - np.mean(frames, axis=-1, keepdims=True)
    raise ValidationError("detrend must be 'none' or 'constant'")


def _onesided_factor(bin_count: int, fft_length: int) -> np.ndarray:
    factor = np.ones(bin_count, dtype=float)
    if bin_count <= 1:
        return factor
    if fft_length % 2 == 0:
        factor[1:-1] = 2.0
    else:
        factor[1:] = 2.0
    return factor


def _spectral_denominator(window: np.ndarray, sample_rate: float, scaling: SpectralScaling) -> float:
    if scaling == "density":
        denominator = sample_rate * float(np.sum(np.abs(window) ** 2))
    elif scaling == "spectrum":
        coherent = np.abs(np.sum(window))
        denominator = float(coherent * coherent)
    else:
        raise ValidationError(f"unknown spectral scaling: {scaling}")
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValidationError(f"{scaling} scaling requires positive finite window normalization")
    return denominator


def _fft_frames(frames: np.ndarray, fft_length: int, onesided: bool) -> tuple[np.ndarray, np.ndarray]:
    if onesided:
        if not np.isrealobj(frames):
            raise ValidationError("one-sided spectra require real-valued signals")
        return np.fft.rfft(frames, n=fft_length, axis=-1), np.fft.rfftfreq(fft_length)
    return np.fft.fft(frames, n=fft_length, axis=-1), np.fft.fftfreq(fft_length)


@dataclass(frozen=True, slots=True)
class PowerSpectrum:
    """A finite periodogram or averaged power spectral estimate."""

    frequencies: np.ndarray
    power: np.ndarray
    sample_rate: float
    scaling: SpectralScaling
    onesided: bool
    fft_length: int
    segment_count: int
    frame_length: int
    hop_length: int
    window: np.ndarray
    method: str
    certificate: Certificate

    @property
    def frequency_resolution(self) -> float:
        """Return the uniform DFT-bin spacing."""

        return self.sample_rate / self.fft_length

    def dominant_frequency(self, *, ignore_dc: bool = False) -> float:
        """Return the frequency of the largest finite spectral ordinate."""

        start = 1 if ignore_dc and self.power.size > 1 else 0
        index = start + int(np.argmax(self.power[start:]))
        return float(self.frequencies[index])

    def integrated_power(self) -> float:
        """Integrate the represented density or sum the represented spectrum."""

        if self.scaling == "density":
            if self.onesided:
                return float(np.sum(self.power) * self.frequency_resolution)
            return float(np.sum(self.power) * self.frequency_resolution)
        return float(np.sum(self.power))


@dataclass(frozen=True, slots=True)
class CrossSpectrum:
    """Averaged auto spectra, cross spectrum, coherence, and H1 transfer estimate."""

    frequencies: np.ndarray
    cross_power: np.ndarray
    input_power: np.ndarray
    output_power: np.ndarray
    coherence: np.ndarray
    transfer_h1: np.ndarray
    sample_rate: float
    scaling: SpectralScaling
    onesided: bool
    fft_length: int
    segment_count: int
    certificate: Certificate


def _power_certificate(power: np.ndarray, *, theorem_id: str, message: str) -> Certificate:
    minimum = float(np.min(np.real(power)))
    imaginary = float(np.max(np.abs(np.imag(power)))) if np.iscomplexobj(power) else 0.0
    scale = max(1.0, float(np.max(np.abs(power))))
    tolerance = 1e-12
    valid = np.all(np.isfinite(power)) and minimum >= -tolerance * scale and imaginary <= tolerance * scale
    return Certificate(
        valid=bool(valid),
        exactness=Exactness.NUMERICAL,
        residuals={"negative_power": max(0.0, -minimum), "imaginary_power": imaginary},
        tolerance=tolerance,
        assumptions_checked=("finite DFT", "declared deterministic scaling"),
        assumptions_unchecked=("statistical consistency requires a stochastic model",),
        theorem_id=theorem_id,
        message=message if valid else f"{message} failed",
    )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D05",
        title="Finite modified periodogram",
        status=ClaimStatus.CLASSICAL,
        source="finite signal processing",
        assumptions=("finite signal", "declared window and scaling"),
        verification_suite="signals.spectral",
    )
)
def periodogram(
    signal: Any,
    *,
    sample_rate: float = 1.0,
    window: WindowSpec = "boxcar",
    fft_length: int | None = None,
    detrend: DetrendMode = "constant",
    scaling: SpectralScaling = "density",
    onesided: bool | None = None,
) -> PowerSpectrum:
    """Compute a deterministic finite modified periodogram."""

    value = _validate_signal(signal)
    rate = _rate(sample_rate)
    transform_length = value.size if fft_length is None else _positive_integer(fft_length, "fft_length")
    if transform_length < value.size:
        raise ValidationError("fft_length cannot be smaller than the signal length")
    analysis_window = np.asarray(get_window(window, value.size, periodic=False))
    one_sided = bool(np.isrealobj(value) and np.isrealobj(analysis_window)) if onesided is None else bool(onesided)
    frame = _detrend(value[np.newaxis, :], detrend) * analysis_window[np.newaxis, :]
    transform, normalized_frequencies = _fft_frames(frame, transform_length, one_sided)
    denominator = _spectral_denominator(analysis_window, rate, scaling)
    power = np.abs(transform[0]) ** 2 / denominator
    if one_sided:
        power *= _onesided_factor(power.size, transform_length)
    power = np.asarray(np.real_if_close(power, tol=1000), dtype=float)
    certificate = _power_certificate(power, theorem_id="KM-DSP-D05", message="periodogram computed")
    return PowerSpectrum(
        frequencies=normalized_frequencies * rate,
        power=power,
        sample_rate=rate,
        scaling=scaling,
        onesided=one_sided,
        fft_length=transform_length,
        segment_count=1,
        frame_length=int(value.size),
        hop_length=int(value.size),
        window=analysis_window,
        method="periodogram",
        certificate=certificate,
    )


def _average(values: np.ndarray, average: SpectralAverage) -> np.ndarray:
    if average == "mean":
        return np.mean(values, axis=0)
    if average == "median":
        if np.iscomplexobj(values):
            return np.median(values.real, axis=0) + 1.0j * np.median(values.imag, axis=0)
        return np.median(values, axis=0)
    raise ValidationError("average must be 'mean' or 'median'")


def _welch_frames(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    if signal.size < frame_length:
        raise DimensionError("signal is shorter than the Welch frame length")
    return frame_signal(
        signal,
        frame_length,
        hop_length,
        center=False,
        pad_end=False,
        boundary="none",
    ).frames


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D06",
        title="Finite Welch power spectral estimate",
        status=ClaimStatus.CLASSICAL,
        source="finite signal processing",
        assumptions=("complete uniform finite segments", "declared averaging rule"),
        verification_suite="signals.spectral",
    )
)
def welch_psd(
    signal: Any,
    *,
    sample_rate: float = 1.0,
    frame_length: int = 256,
    hop_length: int | None = None,
    fft_length: int | None = None,
    window: WindowSpec = "hann",
    detrend: DetrendMode = "constant",
    scaling: SpectralScaling = "density",
    average: SpectralAverage = "mean",
    onesided: bool | None = None,
) -> PowerSpectrum:
    """Average finite modified periodograms over complete overlapping frames."""

    value = _validate_signal(signal)
    rate = _rate(sample_rate)
    length = _positive_integer(frame_length, "frame_length")
    hop = max(1, length // 2) if hop_length is None else _positive_integer(hop_length, "hop_length")
    if hop > length:
        raise ValidationError("hop_length cannot exceed frame_length")
    transform_length = length if fft_length is None else _positive_integer(fft_length, "fft_length")
    if transform_length < length:
        raise ValidationError("fft_length cannot be smaller than frame_length")
    analysis_window = np.asarray(get_window(window, length, periodic=True))
    one_sided = bool(np.isrealobj(value) and np.isrealobj(analysis_window)) if onesided is None else bool(onesided)
    frames = _detrend(_welch_frames(value, length, hop), detrend)
    transform, normalized_frequencies = _fft_frames(frames * analysis_window[np.newaxis, :], transform_length, one_sided)
    denominator = _spectral_denominator(analysis_window, rate, scaling)
    segment_power = np.abs(transform) ** 2 / denominator
    if one_sided:
        segment_power *= _onesided_factor(segment_power.shape[1], transform_length)[np.newaxis, :]
    power = np.asarray(np.real_if_close(_average(segment_power, average), tol=1000), dtype=float)
    certificate = _power_certificate(power, theorem_id="KM-DSP-D06", message="Welch spectrum computed")
    return PowerSpectrum(
        frequencies=normalized_frequencies * rate,
        power=power,
        sample_rate=rate,
        scaling=scaling,
        onesided=one_sided,
        fft_length=transform_length,
        segment_count=int(frames.shape[0]),
        frame_length=length,
        hop_length=hop,
        window=analysis_window,
        method=f"welch-{average}",
        certificate=certificate,
    )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D07",
        title="Finite Welch cross spectrum and magnitude-squared coherence",
        status=ClaimStatus.CLASSICAL,
        source="finite signal processing",
        assumptions=("paired finite segments", "nonzero auto spectra for interpreted transfer bins"),
        verification_suite="signals.spectral",
    )
)
def cross_spectrum(
    input_signal: Any,
    output_signal: Any,
    *,
    sample_rate: float = 1.0,
    frame_length: int = 256,
    hop_length: int | None = None,
    fft_length: int | None = None,
    window: WindowSpec = "hann",
    detrend: DetrendMode = "constant",
    scaling: SpectralScaling = "density",
    average: SpectralAverage = "mean",
    onesided: bool | None = None,
    regularization: float = 0.0,
) -> CrossSpectrum:
    """Estimate paired auto spectra, cross spectrum, coherence, and H1 transfer."""

    x = _validate_signal(input_signal, "input_signal")
    y = _validate_signal(output_signal, "output_signal")
    if x.shape != y.shape:
        raise DimensionError("input and output signals must have equal length")
    rate = _rate(sample_rate)
    length = _positive_integer(frame_length, "frame_length")
    hop = max(1, length // 2) if hop_length is None else _positive_integer(hop_length, "hop_length")
    if hop > length:
        raise ValidationError("hop_length cannot exceed frame_length")
    transform_length = length if fft_length is None else _positive_integer(fft_length, "fft_length")
    if transform_length < length:
        raise ValidationError("fft_length cannot be smaller than frame_length")
    ridge = float(regularization)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValidationError("regularization must be finite and nonnegative")
    analysis_window = np.asarray(get_window(window, length, periodic=True))
    one_sided = bool(np.isrealobj(x) and np.isrealobj(y) and np.isrealobj(analysis_window)) if onesided is None else bool(onesided)
    x_frames = _detrend(_welch_frames(x, length, hop), detrend) * analysis_window[np.newaxis, :]
    y_frames = _detrend(_welch_frames(y, length, hop), detrend) * analysis_window[np.newaxis, :]
    x_transform, normalized_frequencies = _fft_frames(x_frames, transform_length, one_sided)
    y_transform, _ = _fft_frames(y_frames, transform_length, one_sided)
    denominator = _spectral_denominator(analysis_window, rate, scaling)
    factor = _onesided_factor(x_transform.shape[1], transform_length) if one_sided else np.ones(x_transform.shape[1])
    pxx_segments = np.abs(x_transform) ** 2 / denominator * factor[np.newaxis, :]
    pyy_segments = np.abs(y_transform) ** 2 / denominator * factor[np.newaxis, :]
    pxy_segments = np.conjugate(x_transform) * y_transform / denominator * factor[np.newaxis, :]
    pxx = np.asarray(np.real_if_close(_average(pxx_segments, average), tol=1000), dtype=float)
    pyy = np.asarray(np.real_if_close(_average(pyy_segments, average), tol=1000), dtype=float)
    pxy = np.asarray(_average(pxy_segments, average))
    input_regularized = pxx + ridge
    output_regularized = pyy + ridge
    raw_coherence = np.zeros_like(pxx, dtype=float)
    valid_bins = (input_regularized > 0) & (output_regularized > 0)
    normalized_cross = pxy[valid_bins] / np.sqrt(input_regularized[valid_bins]) / np.sqrt(output_regularized[valid_bins])
    raw_coherence[valid_bins] = np.abs(normalized_cross) ** 2
    coherence = np.clip(raw_coherence, 0.0, 1.0)
    transfer = np.zeros_like(pxy, dtype=np.result_type(pxy.dtype, complex))
    transfer_denominator = input_regularized
    nonzero = transfer_denominator > 0
    transfer[nonzero] = pxy[nonzero] / transfer_denominator[nonzero]
    bound_residual = float(max(0.0, np.max(raw_coherence) - 1.0, -np.min(raw_coherence)))
    finite = bool(
        np.all(np.isfinite(pxx))
        and np.all(np.isfinite(pyy))
        and np.all(np.isfinite(pxy))
        and np.all(np.isfinite(input_regularized))
        and np.all(np.isfinite(output_regularized))
        and np.all(np.isfinite(coherence))
        and np.all(np.isfinite(transfer))
    )
    certificate = Certificate(
        valid=finite and bound_residual <= 1e-12,
        exactness=Exactness.NUMERICAL,
        residuals={"coherence_bound": bound_residual},
        tolerance=1e-12,
        assumptions_checked=("paired complete frames", "finite DFT scaling"),
        assumptions_unchecked=("linear time-invariant data-generating model", "statistical consistency"),
        diagnostics={"regularization": ridge, "segment_count": int(x_frames.shape[0])},
        theorem_id="KM-DSP-D07",
        message="cross spectrum and coherence computed" if finite else "cross spectrum contains nonfinite values",
    )
    return CrossSpectrum(
        frequencies=normalized_frequencies * rate,
        cross_power=pxy,
        input_power=pxx,
        output_power=pyy,
        coherence=coherence,
        transfer_h1=transfer,
        sample_rate=rate,
        scaling=scaling,
        onesided=one_sided,
        fft_length=transform_length,
        segment_count=int(x_frames.shape[0]),
        certificate=certificate,
    )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-T04",
        title="Finite discrete Parseval identity",
        status=ClaimStatus.PROVED,
        source="finite signal processing",
        assumptions=("NumPy DFT normalization convention", "finite one dimensional signal"),
        verification_suite="signals.spectral",
    )
)
def parseval_certificate(signal: Any, *, tolerance: float = 1e-12) -> Certificate:
    """Certify the finite NumPy-normalized DFT Parseval identity."""

    value = _validate_signal(signal)
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValidationError("tolerance must be finite and nonnegative")
    transform = np.fft.fft(value)
    with np.errstate(over="ignore", invalid="ignore"):
        time_energy = float(np.sum(np.abs(value) ** 2))
        frequency_energy = float(np.sum(np.abs(transform / np.sqrt(value.size)) ** 2))
    residual = abs(time_energy - frequency_energy)
    scale = max(1.0, time_energy, frequency_energy)
    valid = bool(
        np.isfinite(time_energy)
        and np.isfinite(frequency_energy)
        and np.isfinite(residual)
        and residual <= tolerance * scale
    )
    return Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={"parseval": residual},
        tolerance=tolerance,
        assumptions_checked=("finite DFT pair",),
        theorem_id="KM-DSP-T04",
        message="finite Parseval identity verified" if valid else "finite Parseval identity failed",
    )


__all__ = [
    "CrossSpectrum",
    "PowerSpectrum",
    "SpectralAverage",
    "SpectralScaling",
    "cross_spectrum",
    "parseval_certificate",
    "periodogram",
    "welch_psd",
]
