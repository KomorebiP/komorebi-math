"""Bounded exact rational stencils for arithmetic packet stress controls.

The existing scalar unit-mode controls require integer values. This module
also handles positive-address Euler packets, whose normalized values can be
nonintegral rational numbers, and mixed displacements from Paper IX.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from numbers import Integral

from .._number_theory import require_integer, require_prime
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import ValidationError
from .valuation import padic_valuation


def _difference_data(
    values: Iterable[int | Fraction],
    index: int,
    steps: Iterable[int],
    max_stencil_size: int,
    max_order: int,
) -> tuple[Fraction, tuple[int, ...], tuple[tuple[int, int], ...]]:
    start = require_integer(index, name="index")
    cap = require_integer(max_stencil_size, name="max_stencil_size")
    order_cap = require_integer(max_order, name="max_order")
    if start < 0 or cap < 1 or order_cap < 0:
        raise ValidationError("index and max_order must be nonnegative and max_stencil_size positive")
    normalized: list[Fraction] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
            raise ValidationError("packet differences require exact integer or Fraction coefficients")
        normalized.append(
            Fraction(int(value))
            if isinstance(value, Integral)
            else Fraction(int(value.numerator), int(value.denominator))
        )
    if start >= len(normalized):
        raise ValidationError("starting row is outside the supplied finite packet prefix")
    displacements = tuple(require_integer(step, name="step") for step in steps)
    if any(step < 0 for step in displacements):
        raise ValidationError("packet displacements must be nonnegative")
    if len(displacements) > order_cap:
        raise ValidationError("mixed difference order exceeds max_order")
    weights = {0: 1}
    for step in displacements:
        next_weights: dict[int, int] = {}
        for offset, coefficient in weights.items():
            next_weights[offset] = next_weights.get(offset, 0) - coefficient
            next_weights[offset + step] = next_weights.get(offset + step, 0) + coefficient
        weights = {offset: coefficient for offset, coefficient in next_weights.items() if coefficient}
        if len(weights) > cap:
            raise ValidationError("mixed difference stencil exceeds max_stencil_size")
    if any(start + offset >= len(normalized) for offset in weights):
        raise ValidationError("mixed difference stencil exceeds the supplied finite packet prefix")
    value = sum(
        (coefficient * normalized[start + offset] for offset, coefficient in weights.items()), Fraction(0)
    )
    return value, displacements, tuple(sorted(weights.items()))


def rational_mixed_difference(
    values: Iterable[int | Fraction],
    index: int,
    steps: Iterable[int],
    *,
    max_stencil_size: int = 10000,
    max_order: int = 1000,
) -> Fraction:
    """Evaluate ``Delta_steps[0] ... Delta_steps[-1]`` on a finite prefix.

    Repeated offsets are combined before evaluation, so a repeated step has
    ``order+1`` support points. The empty stencil returns its starting value.
    A zero step annihilates a positive-order difference.
    """
    return _difference_data(values, index, steps, max_stencil_size, max_order)[0]


def certify_packet_difference(
    values: Iterable[int | Fraction],
    index: int,
    steps: Iterable[int],
    *,
    prime: int,
    valuation_bound: int,
    max_stencil_size: int = 10000,
    max_order: int = 1000,
) -> ComputationResult[Fraction]:
    """Check one exact finite valuation bound, without extrapolating it."""
    p = require_prime(prime)
    bound = require_integer(valuation_bound, name="valuation_bound")
    value, displacements, weights = _difference_data(values, index, steps, max_stencil_size, max_order)
    valuation = padic_valuation(value, p)
    return ComputationResult(
        value,
        Certificate(
            valid=valuation >= bound,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "exact rational finite packet prefix",
                "declared finite mixed stencil",
                "prime valuation with v_p(0)=infinity",
            ),
            diagnostics={
                "prime": p,
                "valuation": valuation,
                "valuation_bound": bound,
                "index": int(index),
                "steps": displacements,
                "stencil": weights,
                "finite_statement_only": True,
            },
            message="the sampled packet difference meets its valuation bound"
            if valuation >= bound
            else "the sampled packet difference violates its valuation bound",
        ),
    )


__all__ = ["certify_packet_difference", "rational_mixed_difference"]
