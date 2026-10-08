"""Bernoulli–Hurwitz interpolation and exact integer-order residue packets."""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Any

from ..exceptions import ResourceLimitError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import complex_number, context, integer, rational, real


def _interpolation(ctx, z, a):
    if z == 0:
        return ctx.mpf(1)
    # Retain the true nonzero order near the removable singularity. The source
    # reader's abs(z)<1e-20 shortcut incorrectly made this family locally flat.
    extra = max(0, int(ctx.ceil(-ctx.log10(abs(z))))) + 10 if abs(z) < 1 else 10
    if ctx.dps + extra > 10000:
        raise ResourceLimitError("near-zero Bernoulli evaluation requires more than 10000 decimal digits")
    work = ctx.clone()
    work.dps = ctx.dps + extra
    order = work.mpc(z)
    translation = work.mpf(a)
    return ctx.mpc(-order * work.zeta(1 - order, translation))


@documented(ProvenanceRecord(
    identifier="special-functions.bernoulli-hurwitz-interpolation",
    title="Entire-order Bernoulli–Hurwitz family",
    status=ClaimStatus.CLASSICAL,
    source="Special Functions Consolidation v0.1, arithmetic_core.tex, Bernoulli–Hurwitz refinement; Paper I",
    assumptions=("translation a is real and positive",),
    limitations=("mpmath evaluation is not a certified enclosure.", "Near-zero working precision is limited to 10000 digits."),
    verification_suite="tests/test_special_functions_bernoulli.py",
))
def bernoulli_hurwitz(order: Any, a: Any = 1, *, dps: int = 50):
    """Evaluate -order*zeta(1-order,a), with the removable value 1 at order=0.

    At positive integer n the result is the Bernoulli polynomial B_n(a).
    """
    ctx = context(dps)
    z = complex_number(ctx, order, "order")
    translation = real(ctx, a, "a", positive=True)
    return _interpolation(ctx, z, translation)


def bernoulli_hurwitz_packet(order: Any, a: Any, modulus: int, residue: int, *, dps: int = 50):
    """Evaluate M**(order-1)*B(order,(a+r)/M), with value 1/M at order=0."""
    ctx = context(dps)
    z = complex_number(ctx, order, "order")
    translation = real(ctx, a, "a", positive=True)
    m = integer(modulus, "modulus", minimum=1)
    r = integer(residue, "residue")
    if r >= m:
        raise ValidationError("residue must be less than modulus")
    return ctx.power(m, z - 1) * _interpolation(ctx, z, (translation + r) / m)


def bernoulli_packet_integer(degree: int, a: int | Fraction | str, modulus: int, residue: int) -> Fraction:
    """Return the exact rational integer sample M**(n-1)*B_n((a+r)/M).

    This polynomial sample supports every rational a, including a=0, unlike
    the complex-order Hurwitz family, which requires a>0.
    """
    from ..generating.sequences import bernoulli_numbers

    n = integer(degree, "degree")
    translation = rational(a, "a")
    m = integer(modulus, "modulus", minimum=1)
    r = integer(residue, "residue")
    if r >= m:
        raise ValidationError("residue must be less than modulus")
    point = (translation + r) / m
    polynomial = sum((comb(n, k) * value * point ** (n - k)
                      for k, value in enumerate(bernoulli_numbers(n + 1))), Fraction(0))
    return Fraction(m) ** (n - 1) * polynomial
