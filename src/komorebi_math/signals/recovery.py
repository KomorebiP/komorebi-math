"""Stable recovery of separated jumps and piecewise polynomial breakpoints."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any, Sequence

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import AssumptionError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


@dataclass(frozen=True, slots=True)
class JumpDetection:
    """One detected jump cell with a midpoint location and contrast estimate."""

    cell: int
    left_position: float
    right_position: float
    location: float
    amplitude: float


@dataclass(frozen=True, slots=True)
class JumpRecoveryResult:
    """Detected jumps together with theorem scale error bounds."""

    detections: tuple[JumpDetection, ...]
    threshold: float
    background_bound: float
    location_error_bound: float
    amplitude_error_bound: float
    certificate: Certificate

    @property
    def cells(self) -> tuple[int, ...]:
        return tuple(item.cell for item in self.detections)

    @property
    def locations(self) -> np.ndarray:
        return np.asarray([item.location for item in self.detections], dtype=float)

    @property
    def amplitudes(self) -> np.ndarray:
        return np.asarray([item.amplitude for item in self.detections], dtype=float)


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T36",
        title="Stable recovery of separated one dimensional jumps",
        status=ClaimStatus.CONDITIONAL,
        source="Paper V",
        assumptions=(
            "smooth part has the declared Lipschitz bound",
            "sample noise has the declared uniform bound",
            "every jump magnitude is at least the declared minimum",
            "at most one jump lies in each sample cell",
        ),
        limitations=("one dimensional sampled signals", "separated value jumps"),
        verification_suite="signals.recovery",
    )
)
def recover_separated_jumps(
    samples: Sequence[float],
    *,
    spacing: float,
    lipschitz_bound: float,
    noise_bound: float,
    minimum_jump: float,
    threshold: float | None = None,
    start: float = 0.0,
    minimum_separation: float | None = None,
) -> JumpRecoveryResult:
    """Recover jump cells by thresholding adjacent sample contrasts.

    The theorem guarantee is conditional on the supplied regularity and noise
    bounds describing the unknown signal. The computation itself checks the
    numerical contrast gap and all finite dimensional conditions.
    """

    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise DimensionError("samples must be a one dimensional array with at least two entries")
    if not np.all(np.isfinite(values)):
        raise ValidationError("samples must contain only finite values")
    h = float(spacing)
    m = float(lipschitz_bound)
    epsilon = float(noise_bound)
    a_min = float(minimum_jump)
    origin = float(start)
    if not all(np.isfinite(item) for item in (h, m, epsilon, a_min, origin)):
        raise ValidationError("all recovery parameters must be finite")
    if h <= 0 or m < 0 or epsilon < 0 or a_min <= 0:
        raise ValidationError("spacing and minimum_jump must be positive and the bounds must be nonnegative")
    if minimum_separation is not None:
        separation = float(minimum_separation)
        if not np.isfinite(separation):
            raise ValidationError("minimum_separation must be finite")
        if separation < h:
            raise AssumptionError("minimum_separation must be at least one sample spacing")
    background = m * h + 2.0 * epsilon
    if a_min <= 2.0 * background:
        raise AssumptionError("the declared minimum jump does not exceed twice the smooth and noise contrast bound")
    theta = a_min / 2.0 if threshold is None else float(threshold)
    if not background < theta < a_min - background:
        raise AssumptionError("threshold must lie strictly between the background bound and minimum_jump minus that bound")
    differences = np.diff(values)
    if not np.all(np.isfinite(differences)):
        raise ValidationError("sample contrasts exceed the finite numerical range")
    cells = np.flatnonzero(np.abs(differences) > theta)
    detections = tuple(
        JumpDetection(
            int(cell),
            origin + int(cell) * h,
            origin + (int(cell) + 1) * h,
            origin + (int(cell) + 0.5) * h,
            float(differences[cell]),
        )
        for cell in cells
    )
    if any(not np.isfinite(item.location) or not np.isfinite(item.right_position) for item in detections):
        raise ValidationError("jump locations exceed the finite numerical range")
    certificate = Certificate(
        valid=True,
        exactness=Exactness.CONDITIONAL,
        assumptions_checked=(
            "positive sample spacing",
            "valid threshold gap",
            "declared minimum separation is at least one cell" if minimum_separation is not None else "no finite separation declaration checked",
        ),
        assumptions_unchecked=(
            "unknown smooth part obeys the declared Lipschitz bound",
            "unknown sample noise obeys the declared uniform bound",
            "unknown jumps obey the declared minimum magnitude",
            "no sample cell contains more than one unknown jump",
        ),
        diagnostics={
            "detected_count": len(detections),
            "threshold": theta,
            "background_bound": background,
        },
        theorem_id="PSC-V-T36",
        message="jump cells recovered under the declared theorem hypotheses",
    )
    return JumpRecoveryResult(detections, theta, background, h / 2.0, background, certificate)


def subgaussian_noise_bound(sample_count: int, sigma: float, delta: float) -> float:
    """Return the simultaneous adjacent contrast bound used by Paper V."""

    if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 2:
        raise ValidationError("sample_count must be an integer at least two")
    scale = float(sigma)
    failure = float(delta)
    if not np.isfinite(scale) or scale < 0:
        raise ValidationError("sigma must be nonnegative and finite")
    if not 0 < failure < 1:
        raise ValidationError("delta must lie strictly between zero and one")
    return 2.0 * scale * sqrt(log(2.0 * int(sample_count) / failure))


def recover_jumps_subgaussian(
    samples: Sequence[float],
    *,
    spacing: float,
    lipschitz_bound: float,
    sigma: float,
    delta: float,
    minimum_jump: float,
    threshold: float | None = None,
    start: float = 0.0,
    minimum_separation: float | None = None,
) -> JumpRecoveryResult:
    """Apply the separated jump theorem with a simultaneous sub Gaussian bound."""

    values = np.asarray(samples)
    noise = subgaussian_noise_bound(values.size, sigma, delta)
    result = recover_separated_jumps(
        values,
        spacing=spacing,
        lipschitz_bound=lipschitz_bound,
        noise_bound=noise / 2.0,
        minimum_jump=minimum_jump,
        threshold=threshold,
        start=start,
        minimum_separation=minimum_separation,
    )
    certificate = Certificate(
        valid=result.certificate.valid,
        exactness=Exactness.CONDITIONAL,
        assumptions_checked=result.certificate.assumptions_checked + ("declared sub Gaussian confidence calculation",),
        assumptions_unchecked=(
            "sample noises are independent centered sigma sub Gaussian variables",
        ) + result.certificate.assumptions_unchecked[1:],
        diagnostics={**result.certificate.diagnostics, "delta": float(delta), "simultaneous_noise_bound": noise},
        theorem_id="PSC-V-C37",
        message=f"jump cells recovered with probability at least {1.0 - float(delta):.12g} under the declared model",
    )
    return JumpRecoveryResult(
        result.detections,
        result.threshold,
        result.background_bound,
        result.location_error_bound,
        result.amplitude_error_bound,
        certificate,
    )


def finite_difference(values: Sequence[Any], *, order: int = 1, step: int = 1) -> np.ndarray:
    """Return a forward finite difference of arbitrary positive order and step."""

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise DimensionError("finite differences require a one dimensional sequence")
    if isinstance(order, bool) or int(order) != order or order < 0:
        raise ValidationError("order must be a nonnegative integer")
    if isinstance(step, bool) or int(step) != step or step < 1:
        raise ValidationError("step must be a positive integer")
    result = arr.copy()
    for _ in range(int(order)):
        if result.size <= int(step):
            return np.empty(0, dtype=result.dtype)
        result = result[int(step) :] - result[: -int(step)]
    return result


@dataclass(frozen=True, slots=True)
class BreakpointBandResult:
    """Finite difference indices whose magnitude exceeds a declared noise band."""

    indices: tuple[int, ...]
    response: np.ndarray
    threshold: float
    noise_amplification: float
    certificate: Certificate


def detect_piecewise_polynomial_breaks(
    samples: Sequence[float],
    *,
    degree: int,
    noise_bound: float,
    threshold: float | None = None,
) -> BreakpointBandResult:
    """Detect the finite bands where a piecewise polynomial may change."""

    if isinstance(degree, bool) or int(degree) != degree or degree < 0:
        raise ValidationError("degree must be a nonnegative integer")
    epsilon = float(noise_bound)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValidationError("noise_bound must be nonnegative and finite")
    order = int(degree) + 1
    values = np.asarray(samples)
    if not np.all(np.isfinite(values)):
        raise ValidationError("samples must contain only finite values")
    values = values.astype(np.result_type(values.dtype, float), copy=False)
    response = finite_difference(values, order=order)
    if not np.all(np.isfinite(response)):
        raise ValidationError("finite differences exceed the finite numerical range")
    amplification = (2.0**order) * epsilon
    theta = amplification if threshold is None else float(threshold)
    if not np.isfinite(theta) or theta < amplification:
        raise AssumptionError("threshold must be at least the finite difference noise amplification bound")
    indices = tuple(int(index) for index in np.flatnonzero(np.abs(response) > theta))
    certificate = Certificate(
        valid=True,
        exactness=Exactness.CONDITIONAL,
        assumptions_checked=("finite difference order", "threshold exceeds declared noise amplification"),
        assumptions_unchecked=(
            "the noiseless signal is polynomial of the declared degree on every segment",
            "sample noise obeys the declared uniform bound",
        ),
        diagnostics={"difference_order": order, "detected_band_count": len(indices)},
        theorem_id="PSC-V-T43",
        message="breakpoint response bands computed under the declared piecewise polynomial model",
    )
    return BreakpointBandResult(indices, response, theta, amplification, certificate)


def haar_step_coefficient(amplitude: float, support_length: float, relative_location: float) -> float:
    """Return the magnitude of one normalized Haar coefficient of a step."""

    value = float(relative_location)
    length = float(support_length)
    if length <= 0 or not np.isfinite(length):
        raise ValidationError("support_length must be positive and finite")
    if not 0 <= value <= 1:
        raise ValidationError("relative_location must lie between zero and one")
    return float(abs(amplitude) * sqrt(length) * min(value, 1.0 - value))


def haar_lipschitz_bound(lipschitz_constant: float, support_length: float) -> float:
    """Return the Paper V bound for a smooth Haar coefficient."""

    constant = float(lipschitz_constant)
    length = float(support_length)
    if constant < 0 or not np.isfinite(constant) or length <= 0 or not np.isfinite(length):
        raise ValidationError("the Lipschitz constant must be nonnegative and support_length positive")
    return constant * length ** 1.5 / 4.0


__all__ = [
    "BreakpointBandResult",
    "JumpDetection",
    "JumpRecoveryResult",
    "detect_piecewise_polynomial_breaks",
    "finite_difference",
    "haar_lipschitz_bound",
    "haar_step_coefficient",
    "recover_jumps_subgaussian",
    "recover_separated_jumps",
    "subgaussian_noise_bound",
]
