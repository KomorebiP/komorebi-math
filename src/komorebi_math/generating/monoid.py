"""Finite additive and multiplicative monoid series."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Generic, Hashable, Iterable, Mapping, TypeVar


Index = TypeVar("Index", bound=Hashable)


@dataclass(frozen=True, slots=True)
class FiniteMonoidSeries(Generic[Index]):
    r"""A finite coefficient map with a declared associative index product."""

    coefficients: Mapping[Index, Fraction]
    combine: Callable[[Index, Index], Index]
    identity: Index
    name: str = "monoid series"

    def coefficient(self, index: Index) -> Fraction:
        return Fraction(self.coefficients.get(index, 0))

    def multiply(
        self,
        other: "FiniteMonoidSeries[Index]",
        *,
        admissible: Callable[[Index], bool] | None = None,
    ) -> "FiniteMonoidSeries[Index]":
        if self.identity != other.identity:
            raise ValueError("monoid identities must agree")
        output: dict[Index, Fraction] = {}
        for left_index, left_value in self.coefficients.items():
            for right_index, right_value in other.coefficients.items():
                index = self.combine(left_index, right_index)
                if admissible is not None and not admissible(index):
                    continue
                output[index] = output.get(index, Fraction(0)) + Fraction(left_value) * Fraction(right_value)
        return FiniteMonoidSeries(output, self.combine, self.identity, f"{self.name} * {other.name}")


def additive_series(values: Iterable[int | Fraction]) -> FiniteMonoidSeries[int]:
    r"""Create a finite ordinary series whose index operation is addition."""

    coefficients = {n: Fraction(value) for n, value in enumerate(values) if value}
    return FiniteMonoidSeries(coefficients, lambda left, right: left + right, 0, "additive series")


def multiplicative_series(values: Iterable[int | Fraction]) -> FiniteMonoidSeries[int]:
    r"""Create coefficients indexed by positive integers, starting at one."""

    coefficients = {n + 1: Fraction(value) for n, value in enumerate(values) if value}
    return FiniteMonoidSeries(coefficients, lambda left, right: left * right, 1, "Dirichlet series")


def dirichlet_convolution(
    left: Iterable[int | Fraction],
    right: Iterable[int | Fraction],
    *,
    count: int,
) -> tuple[Fraction, ...]:
    r"""Return ``(a*b)(n)=sum_(d|n)a(d)b(n/d)`` for ``1 <= n <= count``."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    a = tuple(Fraction(value) for value in left)
    b = tuple(Fraction(value) for value in right)
    output = []
    for n in range(1, count + 1):
        total = Fraction(0)
        for divisor in range(1, n + 1):
            if n % divisor == 0 and divisor <= len(a) and n // divisor <= len(b):
                total += a[divisor - 1] * b[n // divisor - 1]
        output.append(total)
    return tuple(output)


def prime_valuation_packets(values: Iterable[int | Fraction], prime: int) -> dict[int, tuple[tuple[int, Fraction], ...]]:
    """Group multiplicative coefficients by the exact prime-adic index valuation."""

    if prime < 2:
        raise ValueError("prime must be at least two")
    packets: dict[int, list[tuple[int, Fraction]]] = {}
    for offset, value in enumerate(values):
        index = offset + 1
        valuation = 0
        residual = index
        while residual % prime == 0:
            valuation += 1
            residual //= prime
        packets.setdefault(valuation, []).append((residual, Fraction(value)))
    return {valuation: tuple(packet) for valuation, packet in packets.items()}


__all__ = [
    "FiniteMonoidSeries",
    "additive_series",
    "dirichlet_convolution",
    "multiplicative_series",
    "prime_valuation_packets",
]
