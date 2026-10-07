"""Certified recovery of separated jumps from finite samples."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .singular_packets import SingularPacket1D


def _sample_matrix(samples: Any) -> tuple[np.ndarray, bool]:
    values = np.asarray(samples)
    if values.ndim == 1:
        return values[:, None], True
    if values.ndim == 2:
        return values, False
    raise DimensionError("samples must have shape (n,) or (n, d)")


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValidationError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class JumpRecoveryResult:
    """Detected jump cells, locations, amplitudes, and theorem certificate."""

    cells: tuple[int, ...]
    locations: np.ndarray
    amplitudes: np.ndarray
    contrasts: np.ndarray
    threshold: float
    background_bound: float
    location_error_bound: float
    amplitude_error_bound: float
    certificate: Certificate

    def packet(self) -> SingularPacket1D:
        if self.amplitudes.ndim != 1:
            raise ValidationError("vector jump amplitudes do not define a scalar SingularPacket1D")
        return SingularPacket1D.from_jumps(self.locations, self.amplitudes)


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T19",
        title="Deterministic separated jump cell recovery",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=(
            "one jump in every active sample cell",
            "Lipschitz smooth component",
            "uniform sample noise bound",
            "minimum jump magnitude exceeds twice the background bound",
        ),
        limitations=("does not certify unknown smoothness or unknown minimum jump size from the samples alone",),
        verification_suite="signals.edge_recovery",
    )
)
def recover_jumps(
    samples: Any,
    *,
    spacing: float,
    lipschitz_bound: float,
    noise_bound: float,
    minimum_jump: float,
    origin: float = 0.0,
    threshold: float | None = None,
) -> JumpRecoveryResult:
    """Recover jump cells by adjacent contrast thresholding."""

    values, scalar = _sample_matrix(samples)
    if values.shape[0] < 2:
        raise DimensionError("at least two samples are required")
    h = float(spacing)
    if not np.isfinite(h) or h <= 0:
        raise ValidationError("spacing must be positive and finite")
    smooth_bound = _finite_nonnegative(lipschitz_bound, "lipschitz bound")
    epsilon = _finite_nonnegative(noise_bound, "noise bound")
    minimum = float(minimum_jump)
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValidationError("minimum jump must be positive and finite")
    background = smooth_bound * h + 2.0 * epsilon
    theorem_separation = minimum > 2.0 * background
    chosen = minimum / 2.0 if threshold is None else float(threshold)
    if not np.isfinite(chosen) or chosen <= 0:
        raise ValidationError("threshold must be positive and finite")
    threshold_valid = background < chosen < minimum - background

    contrasts = np.diff(values, axis=0)
    magnitudes = np.linalg.norm(contrasts, axis=1) if not scalar else np.abs(contrasts[:, 0])
    cells = tuple(int(index) for index in np.flatnonzero(magnitudes > chosen))
    locations = np.asarray([origin + (index + 0.5) * h for index in cells], dtype=float)
    amplitudes_matrix = contrasts[np.asarray(cells, dtype=int)] if cells else np.zeros((0, values.shape[1]))
    amplitudes = amplitudes_matrix[:, 0] if scalar else amplitudes_matrix
    detected_separated = all(right - left > 1 for left, right in zip(cells, cells[1:]))

    unchecked = (
        "each true jump lies in one sampled cell",
        "true jump cells are separated",
        "the supplied Lipschitz and noise bounds hold",
        "every true jump has magnitude at least minimum_jump",
    )
    valid = theorem_separation and threshold_valid
    exactness = Exactness.CONDITIONAL
    message = (
        "jump recovery satisfies the theorem threshold under the listed signal assumptions"
        if valid
        else "jump recovery theorem threshold is not satisfied"
    )
    return JumpRecoveryResult(
        cells,
        locations,
        np.asarray(amplitudes),
        contrasts[:, 0] if scalar else contrasts,
        chosen,
        background,
        h / 2.0,
        background,
        Certificate(
            valid=valid,
            exactness=exactness,
            residuals={
                "background_bound": background,
                "threshold_lower_margin": chosen - background,
                "threshold_upper_margin": minimum - background - chosen,
            },
            assumptions_checked=(
                "positive finite spacing",
                "minimum jump exceeds twice the declared background bound" if theorem_separation else "minimum jump separation failed",
                "threshold lies in the theorem interval" if threshold_valid else "threshold lies outside the theorem interval",
                "detected cells are pairwise nonadjacent" if detected_separated else "detected cells include adjacent indices",
            ),
            assumptions_unchecked=unchecked,
            theorem_id="PSC-V-T19",
            message=message,
        ),
    )


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T20",
        title="Sub Gaussian jump recovery bound",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=(
            "independent centered sub Gaussian sample noise",
            "one separated jump per active cell",
            "known Lipschitz and minimum jump bounds",
        ),
        verification_suite="signals.edge_recovery",
    )
)
def recover_jumps_subgaussian(
    samples: Any,
    *,
    spacing: float,
    lipschitz_bound: float,
    sigma: float,
    minimum_jump: float,
    failure_probability: float,
    origin: float = 0.0,
) -> JumpRecoveryResult:
    """Apply the deterministic theorem with a simultaneous sub Gaussian noise bound."""

    values, _ = _sample_matrix(samples)
    standard = float(sigma)
    delta = float(failure_probability)
    if not np.isfinite(standard) or standard < 0:
        raise ValidationError("sigma must be finite and nonnegative")
    if not np.isfinite(delta) or delta <= 0 or delta >= 1:
        raise ValidationError("failure probability must lie strictly between zero and one")
    cells = values.shape[0] - 1
    dimension = values.shape[1]
    coordinate_bound = 2.0 * standard * sqrt(log(2.0 * max(cells, 1) * dimension / delta))
    noise_difference_bound = sqrt(dimension) * coordinate_bound
    result = recover_jumps(
        samples,
        spacing=spacing,
        lipschitz_bound=lipschitz_bound,
        noise_bound=noise_difference_bound / 2.0,
        minimum_jump=minimum_jump,
        origin=origin,
    )
    certificate = Certificate(
        valid=result.certificate.valid,
        exactness=Exactness.CONDITIONAL,
        residuals={
            **result.certificate.residuals,
            "coordinate_noise_bound": coordinate_bound,
            "simultaneous_noise_difference_bound": noise_difference_bound,
        },
        assumptions_checked=result.certificate.assumptions_checked + (f"declared probability at least {1.0 - delta:.12g}",),
        assumptions_unchecked=(
            "sample noise coordinates are independent and centered",
            f"every coordinate obeys the declared sub Gaussian scale {standard}",
        ) + result.certificate.assumptions_unchecked,
        theorem_id="PSC-V-T20",
        message=(
            f"jump recovery holds with probability at least {1.0 - delta:.12g} under the listed assumptions"
            if result.certificate.valid
            else "probabilistic theorem threshold is not satisfied"
        ),
    )
    return JumpRecoveryResult(
        result.cells,
        result.locations,
        result.amplitudes,
        result.contrasts,
        result.threshold,
        result.background_bound,
        result.location_error_bound,
        result.amplitude_error_bound,
        certificate,
    )


def smooth_haar_coefficient_bound(lipschitz_bound: float, interval_length: float) -> float:
    """Return the normalized Haar coefficient bound ``M L^(3/2) / 4``."""

    bound = _finite_nonnegative(lipschitz_bound, "lipschitz bound")
    length = float(interval_length)
    if not np.isfinite(length) or length <= 0:
        raise ValidationError("interval length must be positive and finite")
    return float(bound * length**1.5 / 4.0)


def step_haar_coefficient(amplitude: float, relative_location: float, interval_length: float) -> float:
    """Return the magnitude of a normalized Haar coefficient of one step."""

    u = float(relative_location)
    length = float(interval_length)
    if not 0.0 <= u <= 1.0:
        raise ValidationError("relative location must lie in the unit interval")
    if not np.isfinite(length) or length <= 0:
        raise ValidationError("interval length must be positive and finite")
    return float(abs(float(amplitude)) * sqrt(length) * min(u, 1.0 - u))


def half_shift_distance(point: float, cell_length: float) -> float:
    """Return the better distance to a boundary among a grid and its half cell shift."""

    x = float(point)
    length = float(cell_length)
    if not np.isfinite(x):
        raise ValidationError("point must be finite")
    if not np.isfinite(length) or length <= 0:
        raise ValidationError("cell length must be positive and finite")
    residue = x % length
    first = min(residue, length - residue)
    shifted_residue = (x - length / 2.0) % length
    second = min(shifted_residue, length - shifted_residue)
    return float(max(first, second))


__all__ = [
    "JumpRecoveryResult",
    "half_shift_distance",
    "recover_jumps",
    "recover_jumps_subgaussian",
    "smooth_haar_coefficient_bound",
    "step_haar_coefficient",
]
