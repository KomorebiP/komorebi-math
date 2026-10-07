"""One dimensional singular packets and exact reconstruction identities."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, inf
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented


def _finite_real(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _positive_exponent(value: float) -> float:
    exponent = float(value)
    if exponent == inf:
        return exponent
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValidationError("p must be positive or infinity")
    return exponent


@documented(
    ProvenanceRecord(
        identifier="PSC-V-D15",
        title="Truncated power singular packet",
        status=ClaimStatus.DEFINITION,
        source="Paper V",
        assumptions=("nonnegative derivative order",),
        verification_suite="signals.singular_packets",
    )
)
@dataclass(frozen=True, slots=True)
class TruncatedPowerTerm:
    """The term ``amplitude * (x - location)_+^order / order!``."""

    location: float
    amplitude: complex
    order: int = 0

    def __post_init__(self) -> None:
        location = _finite_real(self.location, "location")
        order = int(self.order)
        if order != self.order or order < 0:
            raise ValidationError("packet order must be a nonnegative integer")
        amplitude = complex(self.amplitude)
        if not np.isfinite(amplitude.real) or not np.isfinite(amplitude.imag):
            raise ValidationError("packet amplitude must be finite")
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "amplitude", amplitude)

    def evaluate(self, points: Any) -> np.ndarray:
        x = np.asarray(points, dtype=float)
        positive = np.maximum(x - self.location, 0.0)
        if self.order == 0:
            basis = (x >= self.location).astype(float)
        else:
            basis = positive**self.order / factorial(self.order)
        return np.real_if_close(self.amplitude * basis, tol=1000)

    def location_bound(
        self,
        estimated_location: float,
        *,
        p: float = 2.0,
        interval: tuple[float, float] = (0.0, 1.0),
    ) -> float:
        """Return an ``L^p`` location perturbation bound on a finite interval.

        For order zero the expression is exact. For positive order it is the
        mean value bound used for truncated power packets.
        """

        estimated = _finite_real(estimated_location, "estimated_location")
        left, right = map(float, interval)
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            raise ValidationError("interval must have finite increasing endpoints")
        exponent = _positive_exponent(p)
        displacement = abs(self.location - estimated)
        magnitude = abs(self.amplitude)
        if displacement == 0:
            return 0.0
        if self.order == 0:
            lower = max(left, min(self.location, estimated))
            upper = min(right, max(self.location, estimated))
            overlap = max(0.0, upper - lower)
            return float(magnitude if exponent == inf and overlap > 0 else 0.0 if exponent == inf else magnitude * overlap ** (1.0 / exponent))
        support_scale = max(0.0, right - min(self.location, estimated))
        lipschitz = magnitude * support_scale ** (self.order - 1) / factorial(self.order - 1)
        measure_factor = 1.0 if exponent == inf else (right - left) ** (1.0 / exponent)
        return float(lipschitz * displacement * measure_factor)


TruncatedPowerPacket = TruncatedPowerTerm


@documented(
    ProvenanceRecord(
        identifier="PSC-V-D16",
        title="Finite one dimensional singular packet",
        status=ClaimStatus.DEFINITION,
        source="Paper V",
        assumptions=("finite packet family",),
        verification_suite="signals.singular_packets",
    )
)
@dataclass(frozen=True, slots=True)
class SingularPacket1D:
    """A finite sum of truncated power terms."""

    components: tuple[TruncatedPowerTerm, ...]

    def __init__(self, components: Iterable[TruncatedPowerTerm]) -> None:
        values = tuple(components)
        if any(not isinstance(value, TruncatedPowerTerm) for value in values):
            raise ValidationError("every component must be a TruncatedPowerTerm")
        object.__setattr__(self, "components", values)

    @classmethod
    def from_jumps(
        cls,
        locations: Iterable[float],
        amplitudes: Iterable[complex],
        *,
        orders: Iterable[int] | None = None,
    ) -> "SingularPacket1D":
        location_values = tuple(locations)
        amplitude_values = tuple(amplitudes)
        if len(location_values) != len(amplitude_values):
            raise DimensionError("jump locations and amplitudes must have equal length")
        order_values = (0,) * len(location_values) if orders is None else tuple(orders)
        if len(order_values) != len(location_values):
            raise DimensionError("orders must have the same length as jump locations")
        return cls(
            TruncatedPowerTerm(location, amplitude, order)
            for location, amplitude, order in zip(
                location_values,
                amplitude_values,
                order_values,
                strict=True,
            )
        )

    def evaluate(self, points: Any) -> np.ndarray:
        x = np.asarray(points, dtype=float)
        dtype = complex if any(np.iscomplexobj(component.amplitude) for component in self.components) else float
        result = np.zeros(x.shape, dtype=dtype)
        for component in self.components:
            result = result + component.evaluate(x)
        return np.real_if_close(result, tol=1000)

    def remove(self, signal: Any, points: Any) -> np.ndarray:
        value = np.asarray(signal)
        packet = self.evaluate(points)
        if value.shape != packet.shape:
            raise DimensionError("signal and packet grid shapes do not agree")
        return value - packet

    def restore(self, smooth: Any, points: Any) -> np.ndarray:
        value = np.asarray(smooth)
        packet = self.evaluate(points)
        if value.shape != packet.shape:
            raise DimensionError("smooth approximation and packet grid shapes do not agree")
        return value + packet

    def trace_jumps(self) -> dict[float, dict[int, complex | float]]:
        """Return the derivative jump amplitudes indexed by location and order."""

        result: dict[float, dict[int, complex]] = {}
        for component in self.components:
            by_order = result.setdefault(component.location, {})
            by_order[component.order] = by_order.get(component.order, 0.0) + component.amplitude
        return {
            location: {order: np.real_if_close(value, tol=1000).item() for order, value in by_order.items()}
            for location, by_order in result.items()
        }


@dataclass(frozen=True, slots=True)
class RetainedPacketResult:
    """A smooth approximation plus an exact or estimated singular packet."""

    smooth_component: np.ndarray
    smooth_approximation: np.ndarray
    packet: np.ndarray
    reconstruction: np.ndarray
    error: np.ndarray
    certificate: Certificate
    packet_error: np.ndarray | None = None


RetainedPacketReconstruction = RetainedPacketResult


@documented(
    ProvenanceRecord(
        identifier="PSC-V-T17",
        title="Exact retained packet reconstruction identity",
        status=ClaimStatus.PROVED,
        source="Paper V",
        assumptions=("signal equals smooth component plus retained packet",),
        verification_suite="signals.singular_packets",
    )
)
def retained_packet_reconstruct(
    signal: Any,
    packet: Any,
    approximator: Callable[[np.ndarray], Any],
    *,
    tolerance: float = 1e-11,
) -> RetainedPacketResult:
    """Approximate only the packet removed component and restore the packet."""

    f = np.asarray(signal)
    j = np.asarray(packet)
    if f.shape != j.shape:
        raise DimensionError("signal and retained packet must have the same shape")
    smooth = f - j
    approximation = np.asarray(approximator(smooth))
    if approximation.shape != smooth.shape:
        raise DimensionError("the smooth approximator changed the signal shape")
    reconstruction = approximation + j
    error = f - reconstruction
    identity_residual = float(np.linalg.norm((error - (smooth - approximation)).reshape(-1)))
    valid = identity_residual <= tolerance
    return RetainedPacketResult(
        smooth,
        approximation,
        j,
        reconstruction,
        error,
        Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"retained_packet_identity": identity_residual},
            tolerance=tolerance,
            assumptions_checked=("declared packet subtraction", "shape preserving smooth approximator"),
            theorem_id="PSC-V-T17",
            message="retained packet identity verified" if valid else "retained packet identity failed",
        ),
    )


def retained_packet_reconstruction(
    signal: Any,
    estimated_packet: Any,
    approximation_matrix: Any,
    *,
    true_smooth: Any | None = None,
    true_packet: Any | None = None,
    tolerance: float = 1e-10,
) -> RetainedPacketResult:
    """Apply ``Q(f - jhat) + jhat`` and certify its exact error identity."""

    f = np.asarray(signal)
    jhat = np.asarray(estimated_packet)
    matrix = np.asarray(approximation_matrix)
    if f.ndim != 1 or jhat.shape != f.shape:
        raise DimensionError("signal and estimated packet must be equal one dimensional arrays")
    if matrix.shape != (f.size, f.size):
        raise DimensionError("approximation matrix must be square on the signal space")
    smooth_estimate = f - jhat
    approximation = matrix @ smooth_estimate
    reconstruction = approximation + jhat
    error = f - reconstruction
    algebraic_formula = (np.eye(f.size, dtype=np.result_type(matrix, f)) - matrix) @ (f - jhat)
    residuals: dict[str, float] = {
        "estimated_packet_identity": float(np.linalg.norm(error - algebraic_formula))
    }
    assumptions_checked = ["finite linear approximation matrix", "estimated packet shape"]
    packet_error: np.ndarray | None = None
    smooth_component = smooth_estimate
    if true_smooth is not None or true_packet is not None:
        if true_smooth is None or true_packet is None:
            raise ValidationError("true_smooth and true_packet must be supplied together")
        smooth = np.asarray(true_smooth)
        packet = np.asarray(true_packet)
        if smooth.shape != f.shape or packet.shape != f.shape:
            raise DimensionError("true components must match the signal shape")
        decomposition_residual = float(np.linalg.norm(f - smooth - packet))
        packet_error = packet - jhat
        full_formula = (np.eye(f.size) - matrix) @ smooth + (np.eye(f.size) - matrix) @ packet_error
        residuals["declared_decomposition"] = decomposition_residual
        residuals["full_error_decomposition"] = float(np.linalg.norm(error - full_formula))
        assumptions_checked.append("declared smooth plus packet decomposition")
        smooth_component = smooth
    valid = max(residuals.values(), default=0.0) <= tolerance
    return RetainedPacketResult(
        smooth_component,
        approximation,
        jhat,
        reconstruction,
        error,
        Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=tolerance,
            assumptions_checked=tuple(assumptions_checked),
            theorem_id="PSC-V-T18",
            message="estimated packet reconstruction verified" if valid else "estimated packet identity failed",
        ),
        packet_error,
    )


def estimated_packet_reconstruct(
    signal: Any,
    estimated_packet: Any,
    approximation_matrix: Any,
    *,
    tolerance: float = 1e-10,
) -> ComputationResult[np.ndarray]:
    """Compatibility wrapper returning only the reconstructed signal."""

    result = retained_packet_reconstruction(
        signal,
        estimated_packet,
        approximation_matrix,
        tolerance=tolerance,
    )
    return ComputationResult(result.reconstruction, result.certificate)


def step_location_lp_error(
    amplitude: complex,
    location: float,
    estimated_location: float,
    *,
    p: float = 2.0,
) -> float:
    """Return the exact ``L^p`` distance between equal amplitude steps."""

    distance = abs(float(location) - float(estimated_location))
    magnitude = abs(complex(amplitude))
    exponent = _positive_exponent(p)
    return float(magnitude if exponent == inf and distance else 0.0 if exponent == inf else magnitude * distance ** (1.0 / exponent))


def scalar_region_lp_error(
    true_value: complex,
    estimated_value: complex,
    intersection_measure: float,
    true_only_measure: float,
    estimated_only_measure: float,
    *,
    p: float = 2.0,
) -> float:
    """Return the exact scalar region packet error from overlap measures."""

    measures = tuple(float(value) for value in (intersection_measure, true_only_measure, estimated_only_measure))
    if any(not np.isfinite(value) or value < 0 for value in measures):
        raise ValidationError("region measures must be finite and nonnegative")
    exponent = _positive_exponent(p)
    a = abs(complex(true_value))
    b = abs(complex(estimated_value))
    mismatch = abs(complex(true_value) - complex(estimated_value))
    if exponent == inf:
        terms = []
        if measures[0] > 0:
            terms.append(mismatch)
        if measures[1] > 0:
            terms.append(a)
        if measures[2] > 0:
            terms.append(b)
        return float(max(terms, default=0.0))
    value = mismatch**exponent * measures[0] + a**exponent * measures[1] + b**exponent * measures[2]
    return float(value ** (1.0 / exponent))


def overlap_partition_lp_error(
    true_values: Any,
    estimated_values: Any,
    overlap_matrix: Any,
    *,
    p: float = 2.0,
) -> float:
    """Return the exact piecewise constant error from a region overlap matrix."""

    a = np.asarray(true_values)
    b = np.asarray(estimated_values)
    overlap = np.asarray(overlap_matrix, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise DimensionError("region values must be one dimensional")
    if overlap.shape != (a.size, b.size):
        raise DimensionError("overlap matrix shape does not match the region value counts")
    if np.any(overlap < 0) or not np.all(np.isfinite(overlap)):
        raise ValidationError("overlap entries must be finite and nonnegative")
    exponent = _positive_exponent(p)
    differences = np.abs(a[:, None] - b[None, :])
    if exponent == inf:
        active = differences[overlap > 0]
        return float(np.max(active)) if active.size else 0.0
    return float(np.sum(overlap * differences**exponent) ** (1.0 / exponent))


def scalar_region_error(
    true_mask: Any,
    true_value: complex,
    estimated_mask: Any,
    estimated_value: complex,
    *,
    p: float = 2.0,
    tolerance: float = 1e-12,
) -> ComputationResult[float]:
    """Certify the exact empirical overlap formula for two scalar regions."""

    first = np.asarray(true_mask, dtype=bool)
    second = np.asarray(estimated_mask, dtype=bool)
    if first.shape != second.shape:
        raise DimensionError("region masks must have the same shape")
    exponent = _positive_exponent(p)
    true_array = complex(true_value) * first
    estimated_array = complex(estimated_value) * second
    difference = np.abs(true_array - estimated_array).reshape(-1)
    direct = float(np.max(difference)) if exponent == inf else float(np.sum(difference**exponent) ** (1.0 / exponent))
    formula = scalar_region_lp_error(
        true_value,
        estimated_value,
        float(np.count_nonzero(first & second)),
        float(np.count_nonzero(first & ~second)),
        float(np.count_nonzero(~first & second)),
        p=exponent,
    )
    residual = abs(direct - formula)
    return ComputationResult(
        formula,
        Certificate(
            valid=residual <= tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"region_overlap_formula": residual},
            tolerance=tolerance,
            assumptions_checked=("finite equal shaped masks",),
            theorem_id="PSC-V-T21",
            message="scalar region overlap identity verified" if residual <= tolerance else "region formula failed",
        ),
    )


def overlap_matrix_error(
    true_labels: Any,
    true_values: Sequence[complex],
    estimated_labels: Any,
    estimated_values: Sequence[complex],
    *,
    p: float = 2.0,
    tolerance: float = 1e-12,
) -> ComputationResult[float]:
    """Certify the finite overlap matrix formula for piecewise constant data."""

    labels_a = np.asarray(true_labels, dtype=int)
    labels_b = np.asarray(estimated_labels, dtype=int)
    if labels_a.shape != labels_b.shape:
        raise DimensionError("region label arrays must have the same shape")
    values_a = np.asarray(true_values)
    values_b = np.asarray(estimated_values)
    if labels_a.size and (np.min(labels_a) < 0 or np.max(labels_a) >= values_a.size):
        raise ValidationError("a true region label lies outside the value table")
    if labels_b.size and (np.min(labels_b) < 0 or np.max(labels_b) >= values_b.size):
        raise ValidationError("an estimated region label lies outside the value table")
    overlap = np.zeros((values_a.size, values_b.size), dtype=float)
    for first, second in zip(labels_a.reshape(-1), labels_b.reshape(-1), strict=True):
        overlap[first, second] += 1.0
    formula = overlap_partition_lp_error(values_a, values_b, overlap, p=p)
    exponent = _positive_exponent(p)
    direct_values = np.abs(values_a[labels_a] - values_b[labels_b]).reshape(-1)
    direct = float(np.max(direct_values)) if exponent == inf else float(np.sum(direct_values**exponent) ** (1.0 / exponent))
    residual = abs(formula - direct)
    return ComputationResult(
        formula,
        Certificate(
            valid=residual <= tolerance,
            exactness=Exactness.NUMERICAL,
            residuals={"overlap_matrix_formula": residual},
            tolerance=tolerance,
            assumptions_checked=("finite labeled partitions",),
            theorem_id="PSC-V-T22",
            message="overlap matrix identity verified" if residual <= tolerance else "overlap matrix identity failed",
        ),
    )


def finite_difference(signal: Any, order: int = 1) -> np.ndarray:
    """Return the forward finite difference of the declared order."""

    x = np.asarray(signal)
    value = int(order)
    if value != order or value < 0:
        raise ValidationError("difference order must be a nonnegative integer")
    return np.diff(x, n=value)


def difference_noise_bound(noise_bound: float, order: int) -> float:
    """Return the uniform amplification bound ``2^order * noise_bound``."""

    epsilon = float(noise_bound)
    value = int(order)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValidationError("noise bound must be finite and nonnegative")
    if value != order or value < 0:
        raise ValidationError("difference order must be a nonnegative integer")
    return float((2**value) * epsilon)


__all__ = [
    "RetainedPacketReconstruction",
    "RetainedPacketResult",
    "SingularPacket1D",
    "TruncatedPowerPacket",
    "TruncatedPowerTerm",
    "difference_noise_bound",
    "estimated_packet_reconstruct",
    "finite_difference",
    "overlap_matrix_error",
    "overlap_partition_lp_error",
    "retained_packet_reconstruct",
    "retained_packet_reconstruction",
    "scalar_region_error",
    "scalar_region_lp_error",
    "step_location_lp_error",
]
