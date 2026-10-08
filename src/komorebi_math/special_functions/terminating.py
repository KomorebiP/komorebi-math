"""Exact rational terminating scalar Racah and q-Racah expressions.

Scalar termination supplies neither positive orthogonality nor a comparison
with the matrix recoupling convention. Rational inputs prevent floating-point
zero detection from becoming a false nonsingularity claim.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Any

from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import integer, rational


def _q(ctx_value, name="q"):
    value = rational(ctx_value, name)
    if not 0 < value < 1:
        raise ValidationError("q must be an exact rational strictly between zero and one")
    return value


def _pochhammer(a: Fraction, q: Fraction, n: int) -> Fraction:
    product = Fraction(1)
    for r in range(n):
        product *= 1 - a * q ** r
    return product


def q_pochhammer(a: Any, q: Any, degree: int) -> Fraction:
    """Return the finite exact product (a;q)_n for rational a and 0<q<1."""
    return _pochhammer(rational(a, "a"), _q(q), integer(degree, "degree"))


def q_racah_lattice(y: int, gamma: Any, delta: Any, q: Any) -> Fraction:
    """Return q**(-y)+gamma*delta*q**(y+1); distinct sites are not assumed."""
    site = integer(y, "y")
    g = rational(gamma, "gamma")
    d = rational(delta, "delta")
    base = _q(q)
    return base ** (-site) + g * d * base ** (site + 1)


@documented(ProvenanceRecord(
    identifier="special-functions.scalar-q-racah",
    title="Terminating exact scalar q-Racah expression",
    status=ClaimStatus.DEFINITION,
    source="Special Functions Consolidation v0.1, recoupling_core.tex, Scalar bridge; Paper VIII, corrected C11",
    assumptions=("n and y are nonnegative integers", "q is rational with 0<q<1", "denominators of retained terms are nonzero"),
    limitations=("No positive orthogonality claim.", "No scalar-to-matrix recoupling identification.", "Singular parameters are rejected without cancellation."),
    verification_suite="tests/test_special_functions_terminating.py",
))
def q_racah_scalar(n: int, y: int, alpha: Any, beta: Any, gamma: Any, delta: Any, q: Any) -> Fraction:
    """Return the terminating 4-phi-3 scalar sum, retaining k<=min(n,y).

    Exact rational parameters are required; integers, Fractions, and rational
    strings are accepted. Every denominator of the retained terms must be
    nonzero. This contract does not assign values to a singular retained term.
    """
    degree = integer(n, "n")
    site = integer(y, "y")
    a, b, g, d = (rational(value, name) for value, name in
                   ((alpha, "alpha"), (beta, "beta"), (gamma, "gamma"), (delta, "delta")))
    base = _q(q)
    numerator_parameters = (base ** (-degree), a * b * base ** (degree + 1),
                            base ** (-site), g * d * base ** (site + 1))
    denominator_parameters = (a * base, b * d * base, g * base, base)
    term = Fraction(1)
    total = term
    for k in range(min(degree, site)):
        numerator = Fraction(1)
        denominator = Fraction(1)
        for parameter in numerator_parameters:
            numerator *= 1 - parameter * base ** k
        for parameter in denominator_parameters:
            denominator *= 1 - parameter * base ** k
        if denominator == 0:
            raise ValidationError(f"q-Racah denominator vanishes in term {k + 1}")
        term *= base * numerator / denominator
        total += term
    return total


def _rising(a: Fraction, n: int) -> Fraction:
    product = Fraction(1)
    for r in range(n):
        product *= a + r
    return product


def racah_scalar(n: int, y: int, a: Any, b: Any, c: Any, d: Any) -> Fraction:
    """Return the exact terminating classical 4-F-3 expression.

    The numerator parameters are -n,n+a+b+1,-y,y+c+d+1; the denominator
    parameters are a+1,b+d+1,c+1. Singular retained terms are rejected.
    """
    degree = integer(n, "n")
    site = integer(y, "y")
    av, bv, cv, dv = (rational(value, name) for value, name in ((a, "a"), (b, "b"), (c, "c"), (d, "d")))
    total = Fraction(0)
    for k in range(min(degree, site) + 1):
        numerator = (_rising(Fraction(-degree), k) * _rising(degree + av + bv + 1, k)
                     * _rising(Fraction(-site), k) * _rising(site + cv + dv + 1, k))
        denominator = _rising(av + 1, k) * _rising(bv + dv + 1, k) * _rising(cv + 1, k) * factorial(k)
        if denominator == 0:
            raise ValidationError(f"Racah denominator vanishes in term {k}")
        total += numerator / denominator
    return total
