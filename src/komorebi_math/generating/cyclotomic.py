"""Cyclotomic orbit tests for exact Mahler-product collapse."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import isqrt
from typing import Any, Mapping

from ..exceptions import BackendUnavailableError


def _sympy() -> Any:
    try:
        return import_module("sympy")
    except ModuleNotFoundError as exc:
        raise BackendUnavailableError(
            "symbolic cyclotomic verification requires the optional SymPy dependency"
        ) from exc

from ..exceptions import ValidationError


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor <= isqrt(value):
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def p_free_orbit(index: int, prime: int) -> tuple[int, int]:
    """Return ``(u,k)`` with ``index=u*prime**k`` and ``p`` not dividing ``u``."""

    if index < 1:
        raise ValueError("index must be positive")
    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    level = 0
    unit = index
    while unit % prime == 0:
        unit //= prime
        level += 1
    return unit, level


@dataclass(frozen=True)
class CyclotomicCollapse:
    """Exact finite cyclotomic coboundary analysis."""

    prime: int
    mask_exponents: tuple[tuple[int, int], ...]
    closed_form_exponents: tuple[tuple[int, int], ...]
    obstruction_orbits: tuple[int, ...]

    @property
    def collapses(self) -> bool:
        """Whether the Mahler product collapses to a finite rational function."""

        return not self.obstruction_orbits

    def require(self) -> None:
        """Raise when a p-free cyclotomic obstruction remains."""

        if self.obstruction_orbits:
            raise ValidationError(
                f"p-free cyclotomic obstructions: {self.obstruction_orbits}"
            )


def cyclotomic_coboundary(
    prime: int, exponents: Mapping[int, int]
) -> CyclotomicCollapse:
    """Solve ``R(z)=Q(z)/Q(z**p)`` for a finite cyclotomic mask.

    Factors are normalized to constant coefficient one. The exact obstruction
    is a nonzero exponent at level zero of a p-power orbit.
    """

    if not _is_prime(prime):
        raise ValueError("prime must be prime")
    cleaned = {
        int(index): int(value) for index, value in exponents.items() if int(value)
    }
    if any(index < 1 for index in cleaned):
        raise ValueError("cyclotomic indices must be positive")
    by_orbit: dict[int, dict[int, int]] = {}
    for index, value in cleaned.items():
        unit, level = p_free_orbit(index, prime)
        by_orbit.setdefault(unit, {})[level] = value
    obstructions = tuple(
        sorted(unit for unit, levels in by_orbit.items() if levels.get(0, 0))
    )
    if obstructions:
        return CyclotomicCollapse(
            prime, tuple(sorted(cleaned.items())), (), obstructions
        )
    closed: dict[int, int] = {}
    for unit, levels in by_orbit.items():
        maximum = max(levels, default=0)
        tail_sum = 0
        for level in range(maximum, 0, -1):
            tail_sum += levels.get(level, 0)
            exponent = -tail_sum
            if exponent:
                closed[unit * prime ** (level - 1)] = exponent
    return CyclotomicCollapse(
        prime,
        tuple(sorted(cleaned.items())),
        tuple(sorted(closed.items())),
        (),
    )


def normalized_cyclotomic_expression(exponents: Mapping[int, int]) -> Any:
    """Return a SymPy expression for normalized cyclotomic factors."""

    sp = _sympy()

    z = sp.Symbol("z")
    expression = sp.Integer(1)
    for index, exponent in sorted(exponents.items()):
        polynomial = sp.cyclotomic_poly(int(index), z)
        normalized = sp.cancel(polynomial / polynomial.subs(z, 0))
        expression *= normalized ** int(exponent)
    return sp.cancel(expression)


def verify_cyclotomic_coboundary(collapse: CyclotomicCollapse) -> bool:
    """Symbolically verify the closed form when the obstruction vanishes."""

    if not collapse.collapses:
        return False
    sp = _sympy()

    z = sp.Symbol("z")
    mask = normalized_cyclotomic_expression(dict(collapse.mask_exponents))
    closed = normalized_cyclotomic_expression(dict(collapse.closed_form_exponents))
    return sp.cancel(mask - closed / closed.subs(z, z ** collapse.prime)) == 0


def verify_cyclotomic_exponent_relation(collapse: CyclotomicCollapse) -> bool:
    """Verify ``R=Q/Q(z**p)`` directly in the cyclotomic divisor lattice."""

    if not collapse.collapses:
        return False
    reconstructed: dict[int, int] = {}
    for index, exponent in collapse.closed_form_exponents:
        reconstructed[index] = reconstructed.get(index, 0) + exponent
        if index % collapse.prime:
            reconstructed[index] = reconstructed.get(index, 0) - exponent
            lifted = index * collapse.prime
            reconstructed[lifted] = reconstructed.get(lifted, 0) - exponent
        else:
            lifted = index * collapse.prime
            reconstructed[lifted] = reconstructed.get(lifted, 0) - exponent
    reconstructed = {index: value for index, value in reconstructed.items() if value}
    return reconstructed == dict(collapse.mask_exponents)
