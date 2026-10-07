"""Finite Haar shrinkage and threshold selection for deterministic denoising."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import AssumptionError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .filterbanks import HaarFilterBank

ThresholdMode = Literal["soft", "hard", "garrote"]
ThresholdRule = Literal["universal", "sure", "fixed"]


def _signal(values: Any) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0:
        raise DimensionError("wavelet denoising requires a nonempty one dimensional signal")
    if not np.all(np.isfinite(result)):
        raise ValidationError("signal entries must be finite")
    return result


def _levels(value: int, length: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValidationError("levels must be a positive integer")
    result = int(value)
    if length % (2**result):
        raise DimensionError("signal length must be divisible by 2**levels")
    return result


def median_absolute_deviation(values: Any, *, center: float | None = None) -> float:
    """Return the finite median absolute deviation from a supplied or median center."""

    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise DimensionError("median absolute deviation requires a nonempty one dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValidationError("median absolute deviation requires finite values")
    reference = np.median(array) if center is None else float(center)
    if not np.isfinite(reference):
        raise ValidationError("center must be finite")
    return float(np.median(np.abs(array - reference)))


def estimate_white_noise_sigma(detail_coefficients: Any) -> float:
    """Estimate Gaussian white-noise scale from finest Haar details by MAD."""

    details = np.asarray(detail_coefficients)
    if details.ndim != 1 or details.size == 0:
        raise DimensionError("noise estimation requires nonempty one dimensional detail coefficients")
    if not np.all(np.isfinite(details)):
        raise ValidationError("detail coefficients must be finite")
    normal_median = 0.6744897501960817
    return median_absolute_deviation(details, center=0.0) / normal_median


def universal_threshold(sigma: float, coefficient_count: int) -> float:
    """Return the classical finite universal threshold ``sigma*sqrt(2 log n)``."""

    scale = float(sigma)
    count = int(coefficient_count)
    if not np.isfinite(scale) or scale < 0:
        raise ValidationError("sigma must be finite and nonnegative")
    if count <= 0:
        raise ValidationError("coefficient_count must be positive")
    return scale * float(np.sqrt(2.0 * np.log(count)))


def sure_threshold(coefficients: Any, sigma: float) -> float:
    """Return the finite Stein-risk minimizing soft threshold over observed magnitudes."""

    values = np.asarray(coefficients)
    if values.ndim != 1 or values.size == 0:
        raise DimensionError("SURE thresholding requires nonempty one dimensional coefficients")
    scale = float(sigma)
    if not np.isfinite(scale) or scale < 0:
        raise ValidationError("sigma must be finite and nonnegative")
    if scale == 0.0:
        return 0.0
    normalized = np.sort((np.abs(values) / scale) ** 2)
    count = normalized.size
    cumulative = np.cumsum(normalized)
    ranks = np.arange(1, count + 1, dtype=float)
    risks = count - 2.0 * ranks + cumulative + (count - ranks) * normalized
    index = int(np.argmin(risks))
    return scale * float(np.sqrt(normalized[index]))


def threshold_coefficients(
    coefficients: Any,
    threshold: float,
    *,
    mode: ThresholdMode = "soft",
) -> np.ndarray:
    """Apply hard, soft, or nonnegative-garrote shrinkage coefficientwise."""

    values = np.asarray(coefficients)
    if values.ndim != 1:
        raise DimensionError("thresholding requires a one dimensional coefficient array")
    level = float(threshold)
    if not np.isfinite(level) or level < 0:
        raise ValidationError("threshold must be finite and nonnegative")
    magnitude = np.abs(values)
    normalized = str(mode).lower().strip()
    if normalized == "hard":
        result = np.where(magnitude >= level, values, 0)
    elif normalized == "soft":
        scale = np.zeros_like(magnitude, dtype=float)
        np.divide(np.maximum(magnitude - level, 0.0), magnitude, out=scale, where=magnitude > 0)
        result = values * scale
    elif normalized == "garrote":
        scale = np.zeros_like(magnitude, dtype=float)
        np.divide(
            np.maximum(magnitude**2 - level**2, 0.0),
            magnitude**2,
            out=scale,
            where=magnitude > 0,
        )
        result = values * scale
    else:
        raise ValidationError("threshold mode must be 'soft', 'hard', or 'garrote'")
    return np.real_if_close(result, tol=1000)


@dataclass(frozen=True, slots=True)
class HaarDecomposition:
    """A finite multilevel Haar approximation and ordered detail channels."""

    approximation: np.ndarray
    details: tuple[np.ndarray, ...]
    normalized: bool
    original_length: int

    @property
    def levels(self) -> int:
        """Return the number of retained Haar refinement levels."""

        return len(self.details)

    def reconstruct(self, details: tuple[np.ndarray, ...] | None = None) -> np.ndarray:
        """Reconstruct from the stored or supplied level-ordered detail channels."""

        selected = self.details if details is None else tuple(np.asarray(value) for value in details)
        if len(selected) != self.levels:
            raise DimensionError("one detail channel is required for each decomposition level")
        bank = HaarFilterBank(normalized=self.normalized)
        current = np.asarray(self.approximation)
        for detail in reversed(selected):
            if detail.shape != current.shape:
                raise DimensionError("detail channel shape does not match its Haar approximation")
            current = bank.synthesize((current, detail))
        if current.size != self.original_length:
            raise DimensionError("Haar reconstruction length does not match the original signal")
        return np.real_if_close(current, tol=1000)

    def certificate(self, original: Any, *, tolerance: float = 1e-11) -> Certificate:
        """Certify reconstruction and, when normalized, energy conservation."""

        source = _signal(original)
        if source.size != self.original_length:
            raise DimensionError("original signal length does not match this decomposition")
        reconstructed = self.reconstruct()
        residual = float(np.linalg.norm(reconstructed - source))
        coefficient_energy = float(np.vdot(self.approximation, self.approximation).real)
        coefficient_energy += sum(float(np.vdot(detail, detail).real) for detail in self.details)
        source_energy = float(np.vdot(source, source).real)
        energy_residual = abs(coefficient_energy - source_energy)
        scale = max(1.0, float(np.linalg.norm(source)), source_energy)
        valid = residual <= tolerance * scale and (
            not self.normalized or energy_residual <= tolerance * scale
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"haar_reconstruction": residual, "haar_energy": energy_residual},
            tolerance=tolerance,
            assumptions_checked=("complete finite Haar decomposition",),
            theorem_id="KM-DSP-T06",
            message="multilevel Haar decomposition verified" if valid else "multilevel Haar decomposition failed",
        )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-T06",
        title="Finite multilevel Haar reconstruction",
        status=ClaimStatus.PROVED,
        source="Block 12",
        assumptions=("signal length divisible by the requested dyadic depth",),
        verification_suite="signals.denoise",
    )
)
def haar_decompose(signal: Any, levels: int, *, normalized: bool = True) -> HaarDecomposition:
    """Compute a complete finite multilevel Haar decomposition."""

    source = _signal(signal)
    depth = _levels(levels, source.size)
    bank = HaarFilterBank(normalized=normalized)
    current = source
    details: list[np.ndarray] = []
    for _ in range(depth):
        current, detail = bank.analyze(current)
        details.append(np.asarray(detail))
    result = HaarDecomposition(np.asarray(current), tuple(details), bool(normalized), int(source.size))
    result.certificate(source).require()
    return result


@dataclass(frozen=True, slots=True)
class WaveletDenoiseResult:
    """A thresholded Haar reconstruction with explicit finite diagnostics."""

    signal: np.ndarray
    decomposition: HaarDecomposition
    thresholded_details: tuple[np.ndarray, ...]
    thresholds: tuple[float, ...]
    noise_sigma: float
    rule: ThresholdRule
    mode: ThresholdMode
    certificate: Certificate

    @property
    def removed_energy(self) -> float:
        """Return squared coefficient energy removed by thresholding."""

        return float(
            sum(
                np.vdot(original - reduced, original - reduced).real
                for original, reduced in zip(self.decomposition.details, self.thresholded_details, strict=True)
            )
        )


@documented(
    ProvenanceRecord(
        identifier="KM-DSP-D10",
        title="Finite Haar coefficient shrinkage",
        status=ClaimStatus.CLASSICAL,
        source="Block 12",
        assumptions=("finite dyadic Haar decomposition", "declared threshold rule"),
        limitations=("noise-model optimality is not inferred from one finite signal",),
        verification_suite="signals.denoise",
    )
)
def wavelet_denoise(
    signal: Any,
    *,
    levels: int = 1,
    mode: ThresholdMode = "soft",
    rule: ThresholdRule = "universal",
    threshold: float | None = None,
    sigma: float | None = None,
    level_dependent: bool = False,
    normalized: bool = True,
) -> WaveletDenoiseResult:
    """Denoise a finite signal by thresholding multilevel Haar detail channels."""

    source = _signal(signal)
    decomposition = haar_decompose(source, levels, normalized=normalized)
    estimated_sigma = estimate_white_noise_sigma(decomposition.details[0]) if sigma is None else float(sigma)
    if not np.isfinite(estimated_sigma) or estimated_sigma < 0:
        raise ValidationError("sigma must be finite and nonnegative")
    normalized_rule = str(rule).lower().strip()
    thresholds: list[float] = []
    reduced: list[np.ndarray] = []
    for detail in decomposition.details:
        local_sigma = estimate_white_noise_sigma(detail) if level_dependent and sigma is None else estimated_sigma
        if normalized_rule == "fixed":
            if threshold is None:
                raise ValidationError("rule='fixed' requires an explicit threshold")
            local_threshold = float(threshold)
            if not np.isfinite(local_threshold) or local_threshold < 0:
                raise ValidationError("threshold must be finite and nonnegative")
        elif normalized_rule == "universal":
            if threshold is not None:
                raise ValidationError("an explicit threshold is only valid with rule='fixed'")
            local_threshold = universal_threshold(local_sigma, detail.size)
        elif normalized_rule == "sure":
            if threshold is not None:
                raise ValidationError("an explicit threshold is only valid with rule='fixed'")
            local_threshold = sure_threshold(detail, local_sigma)
        else:
            raise ValidationError("threshold rule must be 'universal', 'sure', or 'fixed'")
        thresholds.append(local_threshold)
        reduced.append(threshold_coefficients(detail, local_threshold, mode=mode))
    selected = tuple(reduced)
    reconstructed = decomposition.reconstruct(selected)
    finite = bool(np.all(np.isfinite(reconstructed)))
    shrinkage_residual = max(
        (float(np.max(np.maximum(np.abs(reduced_detail) - np.abs(original), 0.0))) for original, reduced_detail in zip(decomposition.details, selected, strict=True)),
        default=0.0,
    )
    certificate = Certificate(
        valid=finite and shrinkage_residual <= 1e-12,
        exactness=Exactness.NUMERICAL,
        residuals={"shrinkage_monotonicity": shrinkage_residual},
        tolerance=1e-12,
        assumptions_checked=("finite threshold recurrence", "complete Haar synthesis"),
        assumptions_unchecked=(
            "Gaussian white-noise interpretation" if sigma is None else "declared noise scale matches the application",
            "threshold optimality outside the selected finite risk rule",
        ),
        diagnostics={
            "levels": decomposition.levels,
            "noise_sigma": estimated_sigma,
            "thresholds": tuple(thresholds),
        },
        theorem_id="KM-DSP-D10",
        message="finite wavelet shrinkage completed" if finite else "wavelet shrinkage produced nonfinite output",
    )
    return WaveletDenoiseResult(
        reconstructed,
        decomposition,
        selected,
        tuple(thresholds),
        estimated_sigma,
        normalized_rule,  # type: ignore[arg-type]
        str(mode).lower().strip(),  # type: ignore[arg-type]
        certificate,
    )


def denoising_risk_certificate(
    clean_signal: Any,
    noisy_signal: Any,
    denoised_signal: Any,
    *,
    tolerance: float = 0.0,
) -> Certificate:
    """Compare observed finite squared error before and after denoising."""

    clean = _signal(clean_signal)
    noisy = _signal(noisy_signal)
    denoised = _signal(denoised_signal)
    if clean.shape != noisy.shape or clean.shape != denoised.shape:
        raise DimensionError("clean, noisy, and denoised signals must have equal lengths")
    allowed = float(tolerance)
    if not np.isfinite(allowed) or allowed < 0:
        raise ValidationError("tolerance must be finite and nonnegative")
    noisy_error = float(np.mean(np.abs(noisy - clean) ** 2))
    denoised_error = float(np.mean(np.abs(denoised - clean) ** 2))
    valid = denoised_error <= noisy_error + allowed
    return Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={"noisy_mse": noisy_error, "denoised_mse": denoised_error},
        tolerance=allowed,
        assumptions_checked=("supplied clean finite reference",),
        theorem_id="KM-DSP-C03",
        message="observed denoising error did not increase" if valid else "observed denoising error increased",
    )


__all__ = [
    "HaarDecomposition",
    "ThresholdMode",
    "ThresholdRule",
    "WaveletDenoiseResult",
    "denoising_risk_certificate",
    "estimate_white_noise_sigma",
    "haar_decompose",
    "median_absolute_deviation",
    "sure_threshold",
    "threshold_coefficients",
    "universal_threshold",
    "wavelet_denoise",
]
