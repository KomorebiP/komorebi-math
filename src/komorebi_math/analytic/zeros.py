"""Numerical zero strings, Hadamard products, and zeta diagnostics."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from math import factorial, pi, sin
from typing import Any

import numpy as np

from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import BackendUnavailableError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._series import validate_order
from .sections import denominator, denominator_derivative


@dataclass(frozen=True, slots=True)
class ZeroStringResult:
    """Positive real zero representatives with numerical residuals."""

    order: int
    indices: tuple[int, ...]
    zeros: tuple[float, ...]
    residuals: tuple[float, ...]
    asymptotic_seeds: tuple[float, ...]
    certificate: Certificate

    def rotational_orbits(self) -> tuple[tuple[complex, ...], ...]:
        """Return the complete rotational orbit of each positive representative."""

        omega = cmath.exp(2j * pi / self.order)
        return tuple(tuple(root * omega**k for k in range(self.order)) for root in self.zeros)


def asymptotic_positive_zero(order: int, index: int) -> float:
    """Return the leading positive zero model ``A*(index+1/2)``."""

    n = validate_order(order, minimum=2)
    m = int(index)
    if isinstance(index, bool) or m != index or m < 0:
        raise ValidationError("zero index must be a nonnegative integer")
    return pi * (m + 0.5) / sin(pi / n)


def _scaled_positive_value(order: int, value: float) -> tuple[complex, complex]:
    """Return the exponentially scaled denominator and derivative on the positive ray."""

    growth = np.cos(np.pi / order)
    terms = tuple(cmath.exp((cmath.exp(1j * (2 * k + 1) * np.pi / order) - growth) * value) for k in range(order))
    function = sum(terms) / order
    derivative = sum(
        (cmath.exp(1j * (2 * k + 1) * np.pi / order) - growth) * terms[k]
        for k in range(order)
    ) / order
    return function, derivative


def _newton_positive_zero(order: int, seed: float, tolerance: float, max_iterations: int) -> tuple[float, float]:
    if order in {2, 4}:
        residual = abs(_scaled_positive_value(order, seed)[0])
        return float(seed), float(residual)
    value = float(seed)
    for _ in range(max_iterations):
        function, derivative = _scaled_positive_value(order, value)
        f_value = function.real
        d_value = derivative.real
        if abs(d_value) <= np.finfo(float).eps:
            break
        step = f_value / d_value
        if not np.isfinite(step) or abs(step) > np.pi / max(np.sin(np.pi / order), 1e-12):
            step = np.sign(step) * np.pi / (2 * max(np.sin(np.pi / order), 1e-12))
        value -= step
        if value <= 0:
            value = seed / 2
        if abs(step) <= tolerance * max(1.0, abs(value)):
            residual = abs(_scaled_positive_value(order, value)[0])
            return float(value), float(residual)
    residual = abs(_scaled_positive_value(order, value)[0])
    return float(value), float(residual)


@documented(
    ProvenanceRecord(
        identifier="PSC-IV-T54",
        title="Large cyclic zero strings",
        status=ClaimStatus.PROVED,
        source="Paper IV",
        assumptions=("cyclic order at least two", "numerical Newton convergence for requested representatives"),
        limitations=("the executable routine certifies residuals, not the full asymptotic theorem",),
        verification_suite="analytic.zeros",
    )
)
def positive_zero_string(
    order: int,
    count: int,
    *,
    start: int = 0,
    tolerance: float = 1e-12,
    max_iterations: int = 64,
) -> ZeroStringResult:
    """Compute positive real zero representatives from asymptotic seeds."""

    n = validate_order(order, minimum=2)
    total = int(count)
    first = int(start)
    if isinstance(count, bool) or total != count or total < 1:
        raise ValidationError("count must be a positive integer")
    if isinstance(start, bool) or first != start or first < 0:
        raise ValidationError("start must be a nonnegative integer")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValidationError("tolerance must be positive and finite")
    seeds = tuple(asymptotic_positive_zero(n, index) for index in range(first, first + total))
    solved = tuple(_newton_positive_zero(n, seed, tolerance, max_iterations) for seed in seeds)
    roots = tuple(item[0] for item in solved)
    residuals = tuple(item[1] for item in solved)
    ordered = all(roots[index] < roots[index + 1] for index in range(len(roots) - 1))
    maximum = max(residuals)
    valid = ordered and maximum <= tolerance * 20
    return ZeroStringResult(
        order=n,
        indices=tuple(range(first, first + total)),
        zeros=roots,
        residuals=residuals,
        asymptotic_seeds=seeds,
        certificate=Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"maximum_zero_residual": maximum},
            tolerance=tolerance * 20,
            assumptions_checked=("positive asymptotic seeds", "Newton residuals", "strict ordering"),
            assumptions_unchecked=("exhaustion of every large zero outside the computed range",),
            theorem_id="PSC-IV-T54",
            message="positive zero representatives were computed with certified residuals",
        ),
    )


def zero_count_asymptotic(order: int, radius: float, *, reduced: bool = False) -> float:
    """Return the leading asymptotic zero count."""

    n = validate_order(order, minimum=2)
    r = float(radius)
    if not np.isfinite(r) or r < 0:
        raise ValidationError("radius must be nonnegative and finite")
    if reduced:
        return sin(pi / n) * r ** (1 / n) / pi
    return n * sin(pi / n) * r / pi


def hadamard_partial_product(order: int, z: complex, positive_zeros: tuple[float, ...]) -> complex:
    """Evaluate a finite reduced Hadamard product from positive representatives."""

    n = validate_order(order, minimum=2)
    value = 1.0 + 0j
    for root in positive_zeros:
        if root <= 0:
            raise ValidationError("positive zero representatives must be strictly positive")
        value *= 1 - complex(z) ** n / root**n
    return value


def reciprocal_coefficient_from_zeros(
    order: int,
    degree: int,
    positive_zeros: tuple[float, ...],
) -> complex:
    """Approximate one reciprocal coefficient from finitely many zero orbits."""

    n = validate_order(order, minimum=2)
    m = int(degree)
    if isinstance(degree, bool) or m != degree or m < 0:
        raise ValidationError("degree must be a nonnegative integer")
    total = 0j
    for root in positive_zeros:
        derivative = denominator_derivative(n, root)
        total += -n * root ** (-n * m - 1) / derivative
    return factorial(n * m) * total


def reduced_zero_zeta(
    order: int,
    exponent: complex,
    *,
    terms: int = 64,
    use_computed_zeros: bool = True,
) -> ComputationResult[complex]:
    """Evaluate a truncated reduced zero zeta sum.

    When ``use_computed_zeros`` is false, the Hurwitz zeta leading model is
    returned and the entire correction from the paper is not included.
    """

    n = validate_order(order, minimum=2)
    count = int(terms)
    if isinstance(terms, bool) or count != terms or count < 1:
        raise ValidationError("terms must be a positive integer")
    s = complex(exponent)
    if use_computed_zeros:
        result = positive_zero_string(n, count)
        result.certificate.require()
        value = sum(root ** (-n * s) for root in result.zeros)
        tail_scale = result.zeros[-1] ** max(0.0, 1 - n * s.real)
        certificate = Certificate(
            valid=True,
            exactness=Exactness.NUMERICAL,
            residuals={"maximum_zero_residual": max(result.residuals)},
            assumptions_checked=("finite computed zero string",),
            assumptions_unchecked=("zeta tail beyond the computed zeros",),
            diagnostics={"terms": count, "tail_scale_diagnostic": float(tail_scale)},
            theorem_id="PSC-IV-T68",
            message="truncated reduced zero zeta was evaluated",
        )
        return ComputationResult(complex(value), certificate)
    try:
        import mpmath as mp
    except ImportError as exc:
        raise BackendUnavailableError("the Hurwitz zeta model requires the scientific extra") from exc
    scale = pi / sin(pi / n)
    value = scale ** (-n * s) * mp.zeta(n * s, 0.5)
    return ComputationResult(
        complex(value),
        Certificate(
            valid=True,
            exactness=Exactness.CONDITIONAL,
            assumptions_checked=("Hurwitz zeta leading model",),
            assumptions_unchecked=("entire correction term",),
            diagnostics={"terms": 0},
            theorem_id="PSC-IV-T68",
            message="the leading Hurwitz zeta model was evaluated without the entire correction",
        ),
    )


@dataclass(frozen=True, slots=True)
class CyclicDenominator:
    """Executable distinguished cyclic denominator family."""

    order: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", validate_order(self.order))

    def __call__(self, z: complex) -> complex:
        """Evaluate the denominator."""

        return denominator(self.order, z)

    def derivative(self, z: complex, derivative_order: int = 1) -> complex:
        """Evaluate a derivative."""

        return denominator_derivative(self.order, z, derivative_order)

    def positive_zeros(self, count: int, *, start: int = 0) -> ZeroStringResult:
        """Compute positive real zero representatives."""

        return positive_zero_string(self.order, count, start=start)

    def zero_count(self, radius: float, *, reduced: bool = False) -> float:
        """Return the leading zero count asymptotic."""

        return zero_count_asymptotic(self.order, radius, reduced=reduced)


__all__ = [
    "CyclicDenominator",
    "ZeroStringResult",
    "asymptotic_positive_zero",
    "hadamard_partial_product",
    "positive_zero_string",
    "reciprocal_coefficient_from_zeros",
    "reduced_zero_zeta",
    "zero_count_asymptotic",
]
