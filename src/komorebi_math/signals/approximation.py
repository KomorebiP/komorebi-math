"""Fourier approximation and deliberately narrow packet estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def fft_lowpass_1d(signal: np.ndarray | Sequence[float], modes: int) -> np.ndarray:
    """Keep discrete Fourier modes whose integer frequency has magnitude at most ``modes``."""

    value = np.asarray(signal)
    if value.ndim != 1 or value.size == 0:
        raise DimensionError("one dimensional Fourier approximation requires a nonempty vector")
    if isinstance(modes, bool) or int(modes) != modes or modes < 0:
        raise ValidationError("modes must be a nonnegative integer")
    spectrum = np.fft.fft(value)
    frequencies = np.fft.fftfreq(value.size) * value.size
    return np.real_if_close(np.fft.ifft(spectrum * (np.abs(frequencies) <= int(modes))), tol=1000)


def fft_lowpass_2d(image: np.ndarray, cutoff: int) -> np.ndarray:
    """Apply a rectangular centered Fourier cutoff to a finite two dimensional array."""

    value = np.asarray(image)
    if value.ndim != 2 or not value.size:
        raise DimensionError("two dimensional Fourier approximation requires a nonempty matrix")
    if isinstance(cutoff, bool) or int(cutoff) != cutoff or cutoff < 0:
        raise ValidationError("cutoff must be a nonnegative integer")
    spectrum = np.fft.fftshift(np.fft.fft2(value))
    height, width = value.shape
    yy, xx = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    mask = (np.abs(yy - center_y) <= int(cutoff)) & (np.abs(xx - center_x) <= int(cutoff))
    return np.real_if_close(np.fft.ifft2(np.fft.ifftshift(spectrum * mask)), tol=1000)


def otsu_threshold(values: np.ndarray | Sequence[float], bins: int = 256) -> float:
    """Return the finite histogram threshold that maximizes between class variance."""

    array = np.asarray(values, dtype=float).ravel()
    if not array.size:
        raise ValidationError("Otsu thresholding requires at least one value")
    if bins < 2:
        raise ValidationError("bins must be at least two")
    low, high = float(array.min()), float(array.max())
    if np.isclose(low, high):
        return low
    histogram, edges = np.histogram(array, bins=int(bins), range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    probability = histogram.astype(float) / histogram.sum()
    omega = np.cumsum(probability)
    mean = np.cumsum(probability * centers)
    denominator = omega * (1.0 - omega)
    score = np.zeros_like(denominator)
    valid = denominator > 0
    score[valid] = (mean[-1] * omega[valid] - mean[valid]) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(score))])


@dataclass(frozen=True, slots=True)
class PacketEstimate1D:
    """One heuristic positive contrast packet estimate."""

    mask: np.ndarray
    amplitude: float
    threshold: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-V-H01",
        title="Positive contrast two class packet estimator",
        status=ClaimStatus.HEURISTIC,
        source="Paper V",
        assumptions=("one dominant positive contrast region",),
        limitations=("not a universal edge detector",),
        verification_suite="signals.approximation",
    )
)
def estimate_positive_packet_1d(signal: np.ndarray | Sequence[float], *, bins: int = 256) -> PacketEstimate1D:
    """Estimate one positive contrast region by Otsu thresholding."""

    value = np.asarray(signal, dtype=float)
    if value.ndim != 1 or not value.size:
        raise DimensionError("the estimator requires a nonempty one dimensional signal")
    threshold = otsu_threshold(value, bins=bins)
    mask = value > threshold
    if mask.any():
        amplitude = float(np.median(value[mask]) - np.median(value[~mask])) if (~mask).any() else float(np.median(value[mask]))
    else:
        amplitude = 0.0
    return PacketEstimate1D(
        mask,
        amplitude,
        threshold,
        Certificate(
            valid=True,
            exactness=Exactness.HEURISTIC,
            assumptions_unchecked=("two class positive contrast model", "topological cleanup is unnecessary"),
            diagnostics={"selected_fraction": float(mask.mean())},
            theorem_id="PSC-V-H01",
            message="positive contrast packet estimated heuristically",
        ),
    )


__all__ = [
    "PacketEstimate1D",
    "estimate_positive_packet_1d",
    "fft_lowpass_1d",
    "fft_lowpass_2d",
    "otsu_threshold",
]
