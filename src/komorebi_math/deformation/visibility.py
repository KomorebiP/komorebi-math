"""Prime-adic visibility partitions and cluster trees."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import inf
from typing import Iterable

from .._number_theory import require_integer, require_prime
from ..exceptions import ValidationError


def rational_valuation(value: Fraction | int, prime: int) -> int | float:
    """Return the prime valuation of an exact rational number."""

    p = require_prime(prime)
    rational = Fraction(value)
    if rational == 0:
        return inf
    numerator = abs(rational.numerator)
    denominator = rational.denominator
    valuation = 0
    while numerator % p == 0:
        numerator //= p
        valuation += 1
    while denominator % p == 0:
        denominator //= p
        valuation -= 1
    return valuation


@dataclass(frozen=True, slots=True)
class VisibilityPair:
    """One pairwise separation depth."""

    left: int
    right: int
    depth: int | float


class VisibilityTree:
    """Nested visibility partitions of finitely many exact local coordinates."""

    def __init__(self, values: Iterable[Fraction | int], *, prime: int) -> None:
        self.values = tuple(Fraction(value) for value in values)
        if not self.values:
            raise ValidationError("a visibility tree requires at least one value")
        self.prime = require_prime(prime)

    def separation_depth(self, left: int, right: int) -> int | float:
        """Return the first integral depth at which two coordinates separate."""

        i = require_integer(left, name="left")
        j = require_integer(right, name="right")
        if not 0 <= i < len(self.values) or not 0 <= j < len(self.values):
            raise ValidationError("visibility index is out of range")
        valuation = rational_valuation(self.values[i] - self.values[j], self.prime)
        return inf if valuation == inf else int(valuation) + 1

    def pairwise_depths(self) -> tuple[VisibilityPair, ...]:
        """Return all pairwise separation depths."""

        return tuple(
            VisibilityPair(i, j, self.separation_depth(i, j))
            for i in range(len(self.values))
            for j in range(i + 1, len(self.values))
        )

    def partition(self, depth: int) -> tuple[tuple[int, ...], ...]:
        """Return the indices indistinguishable modulo the declared depth."""

        m = require_integer(depth, name="depth")
        if m < 1:
            raise ValidationError("visibility depth must be positive")
        parent = list(range(len(self.values)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a = find(left)
            b = find(right)
            if a != b:
                parent[b] = a

        for i in range(len(self.values)):
            for j in range(i + 1, len(self.values)):
                separation = self.separation_depth(i, j)
                if separation == inf or separation > m:
                    union(i, j)
        groups: dict[int, list[int]] = {}
        for index in range(len(self.values)):
            groups.setdefault(find(index), []).append(index)
        return tuple(tuple(group) for _, group in sorted(groups.items(), key=lambda item: min(item[1])))

    def refinement_chain(self, max_depth: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Return the nested partitions through ``max_depth``."""

        maximum = require_integer(max_depth, name="max_depth")
        if maximum < 1:
            raise ValidationError("max_depth must be positive")
        return tuple(self.partition(depth) for depth in range(1, maximum + 1))

    def ultrametric_certificate(self) -> bool:
        """Check the nonarchimedean separation inequality on every triple."""

        for i in range(len(self.values)):
            for j in range(len(self.values)):
                for k in range(len(self.values)):
                    left = self.separation_depth(i, k)
                    right = min(self.separation_depth(i, j), self.separation_depth(j, k))
                    if left < right:
                        return False
        return True


__all__ = ["VisibilityPair", "VisibilityTree", "rational_valuation"]
