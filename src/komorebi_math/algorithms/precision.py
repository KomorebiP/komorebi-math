"""Packetwise error certificates and precision allocation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import fsum
from typing import Any, Sequence

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import finite_float, strict_integer


def packet_error_bound(
    perturbation_norms: Sequence[float],
    source_packet_norms: Sequence[float],
) -> float:
    """Return the triangle inequality bound ``sum error_e source_norm_e``."""

    if len(perturbation_norms) != len(source_packet_norms):
        raise ValidationError("perturbation and source norm sequences must have equal length")
    perturbations = tuple(
        finite_float(value, "perturbation norm", minimum=0.0)
        for value in perturbation_norms
    )
    sources = tuple(
        finite_float(value, "source packet norm", minimum=0.0)
        for value in source_packet_norms
    )
    return float(fsum(error * source for error, source in zip(perturbations, sources, strict=True)))


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T26",
        title="Continuous packet precision allocation",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("positive sensitivities", "positive logarithmic costs", "positive error budget"),
        verification_suite="algorithms.precision",
    )
)
def optimal_precision_tolerances(
    sensitivities: Sequence[float],
    costs: Sequence[float],
    total_tolerance: float,
) -> np.ndarray:
    """Solve the unconstrained logarithmic precision allocation model."""

    tolerance_value = finite_float(total_tolerance, "total_tolerance", minimum=0.0, strict_minimum=True)
    sensitivity = np.asarray(sensitivities, dtype=float)
    cost = np.asarray(costs, dtype=float)
    if sensitivity.ndim != 1 or cost.ndim != 1 or sensitivity.size != cost.size or sensitivity.size == 0:
        raise ValidationError("sensitivities and costs must be nonempty vectors of equal length")
    if (
        np.any(~np.isfinite(sensitivity))
        or np.any(~np.isfinite(cost))
        or np.any(sensitivity <= 0)
        or np.any(cost <= 0)
    ):
        raise ValidationError("sensitivities, costs, and total_tolerance must be finite and positive")
    return tolerance_value * cost / (sensitivity * float(np.sum(cost)))


def clipped_precision_allocation(
    sensitivities: Sequence[float],
    costs: Sequence[float],
    total_tolerance: float,
    maxima: Sequence[float],
) -> np.ndarray:
    """Solve the logarithmic model with upper bounds on every tolerance."""

    tolerance_value = finite_float(total_tolerance, "total_tolerance", minimum=0.0, strict_minimum=True)
    sensitivity = np.asarray(sensitivities, dtype=float)
    cost = np.asarray(costs, dtype=float)
    maximum = np.asarray(maxima, dtype=float)
    if not (
        sensitivity.ndim == cost.ndim == maximum.ndim == 1
        and sensitivity.size == cost.size == maximum.size
        and sensitivity.size > 0
    ):
        raise ValidationError("inputs must be nonempty vectors of equal length")
    if (
        np.any(~np.isfinite(sensitivity))
        or np.any(~np.isfinite(cost))
        or np.any(~np.isfinite(maximum))
        or np.any(sensitivity <= 0)
        or np.any(cost <= 0)
        or np.any(maximum <= 0)
    ):
        raise ValidationError("all allocation parameters must be finite and positive")
    maximum_error = float(np.dot(sensitivity, maximum))
    if tolerance_value >= maximum_error:
        return maximum.copy()

    def weighted_error(multiplier: float) -> float:
        return float(
            np.dot(
                sensitivity,
                np.minimum(maximum, cost / (multiplier * sensitivity)),
            )
        )

    lower = 0.0
    upper = 1.0
    while weighted_error(upper) > tolerance_value:
        upper *= 2.0
    for _ in range(240):
        middle = 0.5 * (lower + upper)
        if weighted_error(middle) > tolerance_value:
            lower = middle
        else:
            upper = middle
    allocation = np.minimum(maximum, cost / (upper * sensitivity))
    scale = tolerance_value / float(np.dot(sensitivity, allocation))
    if scale < 1.0:
        allocation *= scale
    return allocation


@dataclass(frozen=True, slots=True)
class DTypeChoice:
    """One discrete precision option with additive cost and error weight."""

    name: str
    cost: float
    error_units: int
    dtype: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("a precision choice must have a nonempty string name")
        units = strict_integer(self.error_units, "precision error units", minimum=0)
        cost = finite_float(self.cost, "precision choice cost", minimum=0.0)
        if self.dtype is not None:
            try:
                np.dtype(self.dtype)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"unknown precision dtype: {self.dtype}") from exc
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "error_units", units)


@dataclass(frozen=True, slots=True)
class PrecisionAssignment:
    """Minimum cost precision choices under an integer error budget."""

    names: tuple[str, ...]
    total_cost: float
    error_units: int
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T28",
        title="Exact discrete precision assignment",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite option list for every packet", "integer error budget"),
        verification_suite="algorithms.precision",
    )
)
def assign_discrete_precisions(
    options: Sequence[Sequence[DTypeChoice]],
    budget: int,
) -> PrecisionAssignment | None:
    """Minimize total cost subject to an exact integer error budget."""

    budget_value = strict_integer(budget, "budget", minimum=0)
    if not options:
        raise ValidationError("at least one packet option list is required")
    budget = budget_value
    table: dict[int, tuple[Fraction, tuple[str, ...]]] = {0: (Fraction(0), ())}
    for packet_index, packet_options in enumerate(options):
        if not packet_options:
            raise ValidationError(f"packet {packet_index} has no precision options")
        names = tuple(choice.name for choice in packet_options)
        if len(set(names)) != len(names):
            raise ValidationError(f"packet {packet_index} precision option names must be unique")
        next_table: dict[int, tuple[Fraction, tuple[str, ...]]] = {}
        for used, (current_cost, current_names) in table.items():
            for choice in packet_options:
                new_error = used + choice.error_units
                if new_error > budget:
                    continue
                exact_cost = current_cost + Fraction.from_float(choice.cost)
                candidate = (exact_cost, current_names + (choice.name,))
                previous = next_table.get(new_error)
                if previous is None or candidate < previous:
                    next_table[new_error] = candidate
        table = next_table
        if not table:
            return None
    error_units, (total_cost, names) = min(
        table.items(),
        key=lambda item: (item[1][0], item[0], item[1][1]),
    )
    certificate = Certificate(
        valid=error_units <= budget,
        exactness=Exactness.EXACT,
        assumptions_checked=(
            "finite option lists",
            "integer additive error model",
            "exact dyadic comparison of the supplied finite floating costs",
        ),
        diagnostics={
            "packet_count": len(options),
            "budget": int(budget),
            "used_error_units": int(error_units),
            "exact_total_cost": str(total_cost),
        },
        theorem_id="PSC-VIII-T28",
        message="exact discrete precision assignment over the supplied binary floating costs",
    )
    return PrecisionAssignment(names, float(total_cost), int(error_units), certificate)


def _complex_partner(dtype: np.dtype[Any]) -> np.dtype[Any]:
    if dtype == np.dtype(np.float16) or dtype == np.dtype(np.float32):
        return np.dtype(np.complex64)
    if dtype == np.dtype(np.float64):
        return np.dtype(np.complex128)
    return dtype


def quantize_array(value: Any, dtype: Any) -> np.ndarray:
    """Cast an array while preserving complex values in a partner dtype."""

    array = np.asarray(value)
    try:
        target = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"unknown quantization dtype: {dtype}") from exc
    if np.iscomplexobj(array) and not np.issubdtype(target, np.complexfloating):
        target = _complex_partner(target)
    return np.asarray(array, dtype=target)


def quantize_by_packet(values: Sequence[Any], dtypes: Sequence[Any]) -> tuple[np.ndarray, ...]:
    """Quantize a packet sequence with one dtype per packet."""

    if len(values) != len(dtypes):
        raise ValidationError("values and dtypes must have equal length")
    return tuple(
        quantize_array(value, dtype)
        for value, dtype in zip(values, dtypes, strict=True)
    )


def quantization_error(value: Any, dtype: Any, *, norm_order: int | float = 2) -> float:
    """Return the norm of the error introduced by one dtype conversion."""

    array = np.asarray(value)
    quantized = quantize_array(array, dtype).astype(np.result_type(array, np.complex128), copy=False)
    return float(np.linalg.norm(quantized - array, ord=norm_order))


def precision_certificate(
    perturbation_norms: Sequence[float],
    source_packet_norms: Sequence[float],
    observed_error: float,
    *,
    tolerance: float = 1e-10,
) -> Certificate:
    """Certify an observed error against the packet triangle bound."""

    observed = finite_float(observed_error, "observed_error", minimum=0.0)
    tolerance_value = finite_float(tolerance, "tolerance", minimum=0.0)
    bound = packet_error_bound(perturbation_norms, source_packet_norms)
    valid = observed <= bound + tolerance_value
    return Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={"observed_error": observed, "certified_bound": bound},
        tolerance=tolerance_value,
        assumptions_checked=("packetwise operator perturbation norms", "source packet norms"),
        theorem_id="PSC-VIII-T25",
        message="valid packet precision certificate" if valid else "observed error exceeds the packet bound",
    )


__all__ = [
    "DTypeChoice",
    "PrecisionAssignment",
    "assign_discrete_precisions",
    "clipped_precision_allocation",
    "optimal_precision_tolerances",
    "packet_error_bound",
    "precision_certificate",
    "quantization_error",
    "quantize_array",
    "quantize_by_packet",
]
