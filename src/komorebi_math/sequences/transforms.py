"""Exact finite and lazy transformations of integer sequences."""

from __future__ import annotations

from math import comb
from numbers import Number
from typing import Iterable

from .._number_theory import require_integer
from ..exceptions import ValidationError
from .base import Sequence, _native_number


def binomial_transform_values(values: Iterable[Number]) -> tuple[Number, ...]:
    """Return the finite binomial transform of a value prefix."""

    data = tuple(_native_number(value) for value in values)
    return tuple(sum(comb(n, k) * data[k] for k in range(n + 1)) for n in range(len(data)))


def inverse_binomial_transform_values(values: Iterable[Number]) -> tuple[Number, ...]:
    """Return the inverse finite binomial transform."""

    data = tuple(_native_number(value) for value in values)
    return tuple(sum((-1) ** (n - k) * comb(n, k) * data[k] for k in range(n + 1)) for n in range(len(data)))


def binomial_transform(sequence: Sequence) -> Sequence:
    """Return the lazy binomial transform of ``sequence``."""

    return Sequence(
        lambda n: sum(comb(n, k) * sequence.term(k) for k in range(n + 1)),
        name=f"binomial({sequence.name})",
        exact=sequence.exact,
        metadata={"parent": sequence.name, "transform": "binomial"},
    )


def inverse_binomial_transform(sequence: Sequence) -> Sequence:
    """Return the lazy inverse binomial transform."""

    return Sequence(
        lambda n: sum((-1) ** (n - k) * comb(n, k) * sequence.term(k) for k in range(n + 1)),
        name=f"inverse_binomial({sequence.name})",
        exact=sequence.exact,
        metadata={"parent": sequence.name, "transform": "inverse binomial"},
    )


def finite_difference(sequence: Sequence, *, step: int = 1, order: int = 1) -> Sequence:
    """Return an exact lazy finite difference sequence."""

    h = require_integer(step, name="step")
    r = require_integer(order, name="order")
    if h < 0 or r < 0:
        raise ValidationError("step and order must be nonnegative")
    return Sequence(
        lambda n: sum((-1) ** (r - k) * comb(r, k) * sequence.term(n + k * h) for k in range(r + 1)),
        name=f"Delta[{h}]^{r}({sequence.name})",
        exact=sequence.exact,
        metadata={"parent": sequence.name, "transform": "finite difference", "step": h, "order": r},
    )


def partial_sums(sequence: Sequence) -> Sequence:
    """Return the sequence of inclusive partial sums."""

    cache: list[Number] = []

    def evaluate(index: int) -> Number:
        while len(cache) <= index:
            n = len(cache)
            cache.append(sequence.term(n) + (cache[-1] if cache else 0))
        return cache[index]

    return Sequence(
        evaluate,
        name=f"partial sums of {sequence.name}",
        exact=sequence.exact,
        metadata={"parent": sequence.name, "transform": "partial sums"},
    )


def cauchy_product(left: Sequence, right: Sequence) -> Sequence:
    """Return the ordinary Cauchy product of two sequences."""

    return Sequence(
        lambda n: sum(left.term(k) * right.term(n - k) for k in range(n + 1)),
        name=f"{left.name} * {right.name}",
        exact=left.exact and right.exact,
        metadata={"left": left.name, "right": right.name, "transform": "Cauchy product"},
    )


__all__ = [
    "binomial_transform",
    "binomial_transform_values",
    "cauchy_product",
    "finite_difference",
    "inverse_binomial_transform",
    "inverse_binomial_transform_values",
    "partial_sums",
]
