"""Carmichael jumps, scalar local modes, and finite simultaneous congruences."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, gcd
from typing import Callable

from .._number_theory import require_integer, require_prime
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import ValidationError
from .mahler import UnitModeMahler
from .valuation import carmichael_exponent_prime_power, padic_valuation


@dataclass(frozen=True, slots=True)
class DifferenceCongruence:
    """A finite difference value with an exact divisibility certificate."""

    value: int
    modulus: int
    certificate: Certificate


def local_unit_jump(prime: int, depth: int) -> int:
    """Return a universal exponent for units modulo ``prime**depth``."""

    return carmichael_exponent_prime_power(prime, depth)


def unit_mode_for_precision(base: int, prime: int, depth: int, *, residue: int = 0) -> UnitModeMahler:
    """Return the local unit mode using a Carmichael jump at one precision."""

    p = require_prime(prime)
    u = require_integer(base, name="base")
    level = require_integer(depth, name="depth")
    if level < 1:
        raise ValidationError("depth must be positive")
    if gcd(u, p) != 1:
        raise ValidationError("base must be a local unit")
    return UnitModeMahler(u, local_unit_jump(p, level), require_integer(residue, name="residue"))


def finite_difference_value(function: Callable[[int], int], index: int, step: int, order: int) -> int:
    """Evaluate a scalar finite difference exactly."""

    n = require_integer(index, name="index")
    h = require_integer(step, name="step")
    r = require_integer(order, name="order")
    if min(n, h, r) < 0:
        raise ValidationError("index, step, and order must be nonnegative")
    total = 0
    for k in range(r + 1):
        value = require_integer(function(n + k * h), name="function value")
        total += (-1) ** (r - k) * comb(r, k) * value
    return total


def certify_difference_congruence(
    function: Callable[[int], int],
    index: int,
    step: int,
    order: int,
    modulus: int,
) -> DifferenceCongruence:
    """Certify exact divisibility of one finite difference by ``modulus``."""

    n = require_integer(index, name="index")
    h = require_integer(step, name="step")
    r = require_integer(order, name="order")
    mod = require_integer(modulus, name="modulus")
    if min(n, h, r) < 0:
        raise ValidationError("index, step, and order must be nonnegative")
    if mod < 1:
        raise ValidationError("modulus must be positive")
    value = finite_difference_value(function, n, h, r)
    valid = value % mod == 0
    return DifferenceCongruence(
        value,
        mod,
        Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("integer valued function on the sampled range",),
            diagnostics={"index": n, "step": h, "order": r},
            message="finite difference has the declared exact divisibility",
        ),
    )


def simultaneous_unit_difference(
    base: int,
    primes_and_depths: dict[int, int],
    *,
    index: int,
    order: int,
) -> ComputationResult[dict[int, int]]:
    """Return simultaneous local valuations for one exponential unit mode.

    A common jump is the least common multiple of the local Carmichael exponents.
    A zero difference has valuation ``float('inf')`` at every declared prime.
    """

    if not primes_and_depths:
        raise ValidationError("at least one local precision is required")
    jumps: list[int] = []
    normalized: dict[int, int] = {}
    for prime, depth in primes_and_depths.items():
        p = require_prime(prime)
        u = require_integer(base, name="base")
        level = require_integer(depth, name="depth")
        if level < 1:
            raise ValidationError("every local depth must be positive")
        if gcd(u, p) != 1:
            raise ValidationError("base must be a unit at every declared prime")
        jump = local_unit_jump(p, level)
        jumps.append(jump)
        normalized[p] = level
    from math import lcm

    common = 1
    for jump in jumps:
        common = lcm(common, jump)
    u = require_integer(base, name="base")
    n = require_integer(index, name="index")
    r = require_integer(order, name="order")
    if n < 0 or r < 0:
        raise ValidationError("index and order must be nonnegative")
    value = u**n * (u**common - 1) ** r
    valuations = {prime: padic_valuation(value, prime) for prime in normalized}
    valid = all(valuations[prime] >= normalized[prime] * r for prime in normalized)
    return ComputationResult(
        valuations,
        Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("base is a unit at every declared prime", "common Carmichael jump"),
            diagnostics={"common_jump": common, "difference_value": value},
            theorem_id="PSC-IX-T26",
            message="one common jump gives the declared simultaneous local valuation gains",
        ),
    )


__all__ = [
    "DifferenceCongruence",
    "certify_difference_congruence",
    "finite_difference_value",
    "local_unit_jump",
    "simultaneous_unit_difference",
    "unit_mode_for_precision",
]
