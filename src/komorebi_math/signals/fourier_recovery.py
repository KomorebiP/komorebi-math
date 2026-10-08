"""Finite Fourier/Hankel recovery of mean-zero periodic jump packets.

This NumPy implementation extracts the reusable unknown-edge machinery from
the Signal Reconstruction Final Edition (2026-10-06). Its certificate checks
finite numerical consistency, not an unrestricted noisy recovery theorem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import pi
from numbers import Integral
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import AssumptionError, DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer at least {minimum}")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be an integer at least {minimum}") from exc
    if number != value or number < minimum:
        raise ValidationError(f"{name} must be an integer at least {minimum}")
    return number


def _finite_vector(value: Any, name: str, *, real: bool = False) -> np.ndarray:
    original = np.asarray(value)
    if original.ndim != 1:
        raise DimensionError(f"{name} must be one dimensional")
    if any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValidationError(f"{name} must contain numeric values rather than booleans")
    if real and np.iscomplexobj(original) and np.any(original.imag != 0):
        raise ValidationError(f"{name} must be real-valued")
    array = np.asarray(original.real if real else original, dtype=float if real else complex)
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain only finite values")
    return array


def _tolerance(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValidationError(f"{name} must be positive and finite")
    return result


def _period(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValidationError("period must be positive and finite")
    return result


def _frozen(value: Any) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PeriodicJumpPacket:
    """A real, mean-zero periodic step packet with right-continuous jumps.

    Amplitudes must sum to zero within ``tolerance``. A correction within that
    tolerance is removed before canonical periodic integration. Empty inputs
    represent the zero packet. Locations are reduced modulo ``period``.
    """

    locations: np.ndarray
    amplitudes: np.ndarray
    period: float = 2 * pi

    def __init__(self, locations: Any, amplitudes: Any, *, period: float = 2 * pi,
                 tolerance: float = 1e-10) -> None:
        points = _finite_vector(locations, "locations", real=True)
        jumps = _finite_vector(amplitudes, "amplitudes", real=True)
        if points.shape != jumps.shape:
            raise DimensionError("locations and amplitudes must have equal shape")
        length = _period(period)
        allowed = _tolerance(tolerance, "tolerance")
        points = np.mod(points, length)
        permutation = np.argsort(points)
        points = points[permutation]
        jumps = jumps[permutation].copy()
        if points.size:
            gaps = np.r_[np.diff(points), length - points[-1] + points[0]]
            if np.any(gaps <= 0):
                raise AssumptionError("active jump locations must be distinct modulo the period")
            if np.any(jumps == 0):
                raise AssumptionError("zero jump amplitudes must be removed from the active packet")
            total = float(np.sum(jumps))
            scale = max(1.0, float(np.sum(np.abs(jumps))))
            if not np.isfinite(total) or not np.isfinite(scale) or abs(total) > allowed * scale:
                raise AssumptionError("periodic jump amplitudes must sum to zero")
            jumps -= total / jumps.size
            if jumps.size:
                jumps[-1] = -float(np.sum(jumps[:-1]))
            if np.any(jumps == 0):
                raise AssumptionError("mean-zero correction removed an active jump")
        object.__setattr__(self, "locations", _frozen(points))
        object.__setattr__(self, "amplitudes", _frozen(jumps))
        object.__setattr__(self, "period", length)

    @property
    def order(self) -> int:
        return int(self.locations.size)

    def moments(self, indices: Sequence[int]) -> np.ndarray:
        """Evaluate ``sum a_l exp(-2π i n τ_l / period)``."""

        numbers = np.asarray([_signed_index(index) for index in indices], dtype=np.int64)
        phases = np.exp((-2j * pi) * np.outer(numbers, self.locations / self.period))
        return phases @ self.amplitudes

    def fourier(self, index: int) -> complex:
        """Return the normalized Fourier coefficient on this packet's period."""

        number = _signed_index(index)
        return 0j if number == 0 else complex(self.moments([number])[0] / (2j * pi * number))

    def evaluate(self, points: Any) -> np.ndarray:
        """Evaluate the canonical mean-zero, right-continuous representative."""

        coordinates = np.asarray(points)
        if np.iscomplexobj(coordinates) and np.any(coordinates.imag != 0):
            raise ValidationError("evaluation points must be real-valued")
        coordinates = np.asarray(coordinates.real, dtype=float)
        if not np.all(np.isfinite(coordinates)):
            raise ValidationError("evaluation points must be finite")
        wrapped = np.mod(coordinates, self.period)
        output = np.zeros_like(wrapped)
        for location, jump in zip(self.locations, self.amplitudes, strict=True):
            output += jump * (wrapped >= location)
        mean = float(np.sum(self.amplitudes * (1 - self.locations / self.period)))
        return output - mean


def _signed_index(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValidationError("Fourier indices must be integers")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Fourier indices must be integers") from exc
    if number != value or not np.iinfo(np.int64).min <= number <= np.iinfo(np.int64).max:
        raise ValidationError("Fourier indices must be representable 64-bit integers")
    return number


def edge_moments_from_fourier(measurements: Mapping[int, complex], indices: Sequence[int]) -> np.ndarray:
    """Convert normalized nonzero Fourier modes to periodic derivative moments.

    With Fourier normalization ``T^-1 integral f(x) exp(-2π i n x/T) dx``,
    this conversion is ``2π i n fhat(n)`` for every positive period ``T``.
    """

    values: list[complex] = []
    for index in indices:
        number = _signed_index(index)
        if number == 0:
            raise ValidationError("derivative moments require nonzero Fourier indices")
        if number not in measurements:
            raise ValidationError(f"missing Fourier measurement at index {number}")
        values.append(2j * pi * number * complex(measurements[number]))
    return _finite_vector(values, "derivative moments")


def hankel_moment_matrix(moments: Any, rows: int, columns: int) -> np.ndarray:
    """Build the finite Hankel matrix ``H[i,j] = moments[i+j]``."""

    values = _finite_vector(moments, "moments")
    height = _integer(rows, "rows", minimum=1)
    width = _integer(columns, "columns", minimum=1)
    if values.size < height + width - 1:
        raise DimensionError("not enough consecutive moments for the requested Hankel matrix")
    return values[np.arange(height)[:, None] + np.arange(width)[None, :]]


@dataclass(frozen=True, slots=True)
class FourierJumpRecovery:
    """Recovered periodic packet and finite numerical consistency evidence."""

    packet: PeriodicJumpPacket
    annihilator_ascending: np.ndarray
    hankel_singular_values: np.ndarray
    moment_indices: np.ndarray
    measured_moments: np.ndarray
    predicted_moments: np.ndarray
    certificate: Certificate

    @property
    def order(self) -> int:
        return self.packet.order


@documented(ProvenanceRecord(
    identifier="KM-SIGNAL-PRONY-01",
    title="Finite high-mode Hankel and annihilator jump recovery",
    status=ClaimStatus.CLASSICAL,
    source="Signal Reconstruction Final Edition (2026-10-06), unknown-edge source",
    assumptions=("real periodic mean-zero jump packet", "distinct active nodes",
                 "background Fourier degree at most the declared bound", "enough consecutive high modes"),
    limitations=("floating-point root extraction", "supplied numerical rank threshold",
                 "no model-free or unrestricted noisy guarantee", "no dedicated Lean certificate"),
    verification_suite="signals.fourier_recovery",
))
def recover_fourier_jumps(measurements: Mapping[int, complex], *, background_degree: int,
                          maximum_jumps: int, known_order: int | None = None,
                          period: float = 2 * pi, rank_tolerance: float = 1e-10,
                          tolerance: float = 1e-8) -> FourierJumpRecovery:
    """Recover a periodic jump packet from ``2*maximum_jumps`` consecutive modes.

    The requested modes start at ``background_degree + 1``. Their derivative
    moments feed a Hankel rank gate, a monic annihilator solve, polynomial root
    extraction, unit-circle projection, and an all-moment amplitude refit.
    The returned certificate checks the represented real, mean-zero model's
    numerical residual. The declared background and rank gap remain assumptions.
    """

    degree = _integer(background_degree, "background_degree")
    bound = _integer(maximum_jumps, "maximum_jumps", minimum=1)
    length = _period(period)
    rank_cutoff = _tolerance(rank_tolerance, "rank_tolerance")
    if rank_cutoff >= 1:
        raise ValidationError("rank_tolerance must be less than one")
    allowed = _tolerance(tolerance, "tolerance")
    order_hint = None if known_order is None else _integer(known_order, "known_order")
    if order_hint is not None and order_hint > bound:
        raise ValidationError("known_order cannot exceed maximum_jumps")
    start = degree + 1
    _signed_index(start + 2 * bound - 1)
    indices = np.arange(start, start + 2 * bound, dtype=np.int64)
    moments = edge_moments_from_fourier(measurements, indices)
    for number in indices:
        if -int(number) in measurements:
            opposite = complex(measurements[-int(number)])
            positive = complex(measurements[int(number)])
            if not np.isfinite(opposite) or abs(opposite - positive.conjugate()) > allowed * max(1.0, abs(positive)):
                raise AssumptionError("paired Fourier measurements do not satisfy real conjugacy")
    hankel = hankel_moment_matrix(moments, bound, bound)
    singular = np.linalg.svd(hankel, compute_uv=False)
    numerical_rank = 0 if singular[0] == 0 else int(np.count_nonzero(singular > rank_cutoff * singular[0]))
    order = numerical_rank if order_hint is None else order_hint
    if order > numerical_rank:
        raise AssumptionError("known order exceeds the numerical Hankel rank at the supplied threshold")
    scale = max(1.0, float(np.max(np.abs(moments))))
    hankel_condition = 0.0
    vandermonde_condition = 0.0
    radial_error = 0.0
    imaginary_amplitudes = 0.0
    amplitude_sum = 0.0
    if order == 0:
        packet = PeriodicJumpPacket([], [], period=length)
        annihilator = np.array([1 + 0j])
        predicted = np.zeros_like(moments)
        recurrence = float(np.max(np.abs(moments)))
    else:
        leading = hankel_moment_matrix(moments, order, order)
        hankel_condition = float(np.linalg.cond(leading))
        if not np.isfinite(hankel_condition):
            raise AssumptionError("the leading recovery Hankel matrix is singular")
        annihilator = np.r_[np.linalg.solve(leading, -moments[order:2 * order]), 1 + 0j]
        roots = np.roots(annihilator[::-1])
        if not np.all(np.isfinite(roots)) or np.any(np.abs(roots) == 0):
            raise AssumptionError("annihilator root extraction produced nonfinite or zero nodes")
        radial_error = float(np.max(np.abs(np.abs(roots) - 1)))
        roots /= np.abs(roots)
        locations = np.mod((-np.angle(roots) / (2 * pi)) * length, length)
        permutation = np.argsort(locations)
        locations, roots = locations[permutation], roots[permutation]
        design = roots[None, :] ** indices[:, None]
        vandermonde_condition = float(np.linalg.cond(design))
        amplitudes, _, fitted_rank, _ = np.linalg.lstsq(design, moments, rcond=None)
        if fitted_rank != order or not np.isfinite(vandermonde_condition):
            raise AssumptionError("recovered nodes do not give a full-rank moment design")
        imaginary_amplitudes = float(np.max(np.abs(amplitudes.imag)))
        amplitude_sum = abs(complex(np.sum(amplitudes)))
        real_amplitudes = amplitudes.real - float(np.mean(amplitudes.real))
        packet = PeriodicJumpPacket(locations, real_amplitudes, period=length, tolerance=allowed)
        predicted = packet.moments(indices)
        recurrence = float(max(abs(np.dot(annihilator, moments[offset:offset + order + 1]))
                               for offset in range(moments.size - order)))
    residual = float(np.max(np.abs(predicted - moments)))
    amplitude_scale = max(1.0, float(np.max(np.abs(packet.amplitudes))) if order else 0.0)
    finite = all(np.isfinite(value) for value in (scale, residual, recurrence, radial_error,
                                                imaginary_amplitudes, amplitude_sum, amplitude_scale))
    valid = bool(finite and residual <= allowed * scale and recurrence <= allowed * scale
                 and imaginary_amplitudes <= allowed * amplitude_scale
                 and amplitude_sum <= allowed * max(1, order) * amplitude_scale and radial_error <= allowed)
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={"moment_fit": residual, "annihilator_recurrence": recurrence,
                   "node_circle": radial_error, "imaginary_amplitudes": imaginary_amplitudes,
                   "amplitude_sum_before_projection": amplitude_sum},
        tolerance=allowed,
        assumptions_checked=("finite consecutive high modes", "supplied numerical Hankel rank gate",
                             "real zero-sum amplitude consistency", "canonical mean-zero periodic integration"),
        assumptions_unchecked=("unknown smooth background is bandlimited by background_degree",
                               "the unknown target belongs to the distinct-node finite jump model",
                               "the supplied rank threshold identifies the target order",
                               "conditioning and noise are suitable for parameter recovery"),
        diagnostics={"numerical_hankel_rank": numerical_rank, "recovered_order": order,
                     "hankel_condition": hankel_condition, "vandermonde_condition": vandermonde_condition,
                     "minimum_sample_count": 2 * bound, "period": length},
        theorem_id="KM-SIGNAL-PRONY-01",
        message="finite Fourier jump model consistency verified" if valid else "finite Fourier jump model consistency failed",
    )
    return FourierJumpRecovery(packet, _frozen(annihilator), _frozen(singular), _frozen(indices),
                               _frozen(moments), _frozen(predicted), certificate)


__all__ = ["PeriodicJumpPacket", "FourierJumpRecovery", "edge_moments_from_fourier",
           "hankel_moment_matrix", "recover_fourier_jumps"]
