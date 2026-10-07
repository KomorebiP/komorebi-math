"""Finite Mahler coordinates, affine reindexing, and local unit modes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from numbers import Number, Rational
from typing import Iterable

import numpy as np

from .._number_theory import require_integer, require_prime
from ..certificates import Certificate, ComputationResult, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .valuation import padic_valuation

ExactNumber = int | Fraction


def generalized_binomial(value: Number, order: int) -> Number:
    """Evaluate the binomial polynomial ``binomial(value, order)`` exactly when possible."""

    r = require_integer(order, name="order")
    if r < 0:
        return 0
    if r == 0:
        return 1
    if isinstance(value, (int, Fraction)) and not isinstance(value, bool):
        result_exact = Fraction(1)
        for offset in range(r):
            result_exact *= Fraction(value) - offset
        return result_exact / factorial(r)
    result: Number = 1
    for offset in range(r):
        result *= value - offset
    return result / factorial(r)


def difference_table(values: Iterable[Number]) -> tuple[tuple[Number, ...], ...]:
    """Return the complete forward difference table of a finite value list."""

    first = tuple(values)
    if not first:
        raise ValidationError("at least one value is required")
    if any(not isinstance(value, Number) for value in first):
        raise ValidationError("difference values must be numeric")
    rows = [first]
    current = first
    while len(current) > 1:
        current = tuple(current[index + 1] - current[index] for index in range(len(current) - 1))
        rows.append(current)
    return tuple(rows)


def mahler_coefficients(values: Iterable[Number]) -> tuple[Number, ...]:
    """Return finite Mahler coefficients from consecutive values starting at zero."""

    return tuple(row[0] for row in difference_table(values))


@documented(
    ProvenanceRecord(
        identifier="PSC-IX-T28",
        title="Finite Mahler reconstruction",
        status=ClaimStatus.PROVED,
        source="Paper IX",
        assumptions=("finite consecutive value prefix",),
        verification_suite="arithmetic.mahler",
    )
)
@dataclass(frozen=True, slots=True)
class MahlerSeries:
    """A finite expansion in the binomial polynomial basis."""

    coefficients: tuple[Number, ...]

    def __init__(self, coefficients: Iterable[Number]) -> None:
        values = tuple(coefficients)
        if not values:
            raise ValidationError("MahlerSeries requires at least one coefficient")
        if any(not isinstance(value, Number) for value in values):
            raise ValidationError("Mahler coefficients must be numeric")
        object.__setattr__(self, "coefficients", values)

    @property
    def degree(self) -> int:
        """Return the highest stored Mahler degree."""

        return len(self.coefficients) - 1

    @classmethod
    def from_values(cls, values: Iterable[Number]) -> "MahlerSeries":
        """Interpolate one finite consecutive value prefix."""

        return cls(mahler_coefficients(values))

    def evaluate(self, value: Number, *, degree: int | None = None) -> Number:
        """Evaluate the finite Mahler polynomial."""

        nmax = self.degree if degree is None else min(require_integer(degree, name="degree"), self.degree)
        if nmax < 0:
            raise ValidationError("degree must be nonnegative")
        return sum(self.coefficients[index] * generalized_binomial(value, index) for index in range(nmax + 1))

    def first(self, count: int) -> tuple[Number, ...]:
        """Evaluate at the first ``count`` nonnegative integers."""

        total = require_integer(count, name="count")
        if total < 0:
            raise ValidationError("count must be nonnegative")
        return tuple(self.evaluate(index) for index in range(total))

    def difference(self, order: int = 1) -> "MahlerSeries":
        """Return the forward difference in Mahler coordinates."""

        r = require_integer(order, name="order")
        if r < 0:
            raise ValidationError("order must be nonnegative")
        if r > self.degree:
            return MahlerSeries((0,))
        return MahlerSeries(self.coefficients[r:])

    def affine_subsequence(self, multiplier: int, shift: int) -> "MahlerSeries":
        """Return the exact Mahler expansion of ``f(multiplier*x + shift)``."""

        m = require_integer(multiplier, name="multiplier")
        s = require_integer(shift, name="shift")
        if m < 1:
            raise ValidationError("multiplier must be positive")
        if s < 0:
            raise ValidationError("shift must be nonnegative")
        transformed: list[Number] = []
        for k in range(self.degree + 1):
            total: Number = 0
            for r, coefficient in enumerate(self.coefficients):
                total += coefficient * affine_mahler_kernel(m, s, r, k)
            transformed.append(total)
        while len(transformed) > 1 and transformed[-1] == 0:
            transformed.pop()
        return MahlerSeries(transformed)

    def valuation_ledger(self, prime: int) -> tuple[int | float, ...]:
        """Return exact valuations of the stored coefficients."""

        p = require_prime(prime)
        if any(isinstance(value, bool) or not isinstance(value, Rational) for value in self.coefficients):
            raise ValidationError("valuation ledgers require exact rational coefficients")
        return tuple(padic_valuation(Fraction(value), p) for value in self.coefficients)

    def reconstruction_certificate(self) -> ComputationResult[tuple[Number, ...]]:
        """Validate exact reconstruction of the defining prefix."""

        values = self.first(len(self.coefficients))
        recovered = MahlerSeries.from_values(values)
        valid = recovered.coefficients == self.coefficients
        exact = all(isinstance(value, (int, Fraction)) for value in self.coefficients)
        return ComputationResult(
            values,
            Certificate(
                valid=valid,
                exactness=Exactness.EXACT if exact else Exactness.NUMERICAL,
                assumptions_checked=("finite Mahler polynomial",),
                diagnostics={"degree": self.degree},
                theorem_id="PSC-IX-T28",
                message="finite Mahler analysis and synthesis agree",
            ),
        )


def affine_mahler_kernel(multiplier: int, shift: int, source_degree: int, target_degree: int) -> int:
    """Return the exact affine Mahler transfer kernel.

    This is the coefficient of ``binomial(q, target_degree)`` in
    ``binomial(multiplier*q + shift, source_degree)``.
    """

    m = require_integer(multiplier, name="multiplier")
    s = require_integer(shift, name="shift")
    r = require_integer(source_degree, name="source_degree")
    k = require_integer(target_degree, name="target_degree")
    if m < 1 or s < 0 or r < 0 or k < 0:
        raise ValidationError("multiplier must be positive and all indices must be nonnegative")
    return sum(
        (-1) ** (k - j) * comb(k, j) * (comb(m * j + s, r) if r <= m * j + s else 0)
        for j in range(k + 1)
    )


def affine_kernel_matrix(multiplier: int, shift: int, degree: int) -> tuple[tuple[int, ...], ...]:
    """Return the finite affine Mahler transfer matrix."""

    nmax = require_integer(degree, name="degree")
    if nmax < 0:
        raise ValidationError("degree must be nonnegative")
    return tuple(
        tuple(affine_mahler_kernel(multiplier, shift, r, k) for r in range(nmax + 1))
        for k in range(nmax + 1)
    )


def affine_kernel_composition_certificate(
    first_multiplier: int,
    first_shift: int,
    second_multiplier: int,
    second_shift: int,
    degree: int,
) -> Certificate:
    """Validate exact composition of two affine Mahler kernels."""

    m = require_integer(first_multiplier, name="first_multiplier")
    s = require_integer(first_shift, name="first_shift")
    n = require_integer(second_multiplier, name="second_multiplier")
    t = require_integer(second_shift, name="second_shift")
    nmax = require_integer(degree, name="degree")
    if min(m, n) < 1 or min(s, t, nmax) < 0:
        raise ValidationError("multipliers must be positive and shifts and degree nonnegative")
    valid = True
    for r in range(nmax + 1):
        for ell in range(nmax + 1):
            left = sum(
                affine_mahler_kernel(m, s, r, k) * affine_mahler_kernel(n, t, k, ell)
                for k in range(nmax + 1)
            )
            right = affine_mahler_kernel(m * n, m * t + s, r, ell)
            valid = valid and left == right
    return Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        assumptions_checked=("finite affine Mahler degree",),
        diagnostics={"degree": nmax},
        theorem_id="PSC-IX-T34",
        message="affine Mahler kernels compose exactly",
    )


def prime_power_kernel_residue(
    prime: int,
    exponent: int,
    shift: int,
    source_degree: int,
    target_degree: int,
) -> int:
    """Return the sparse prime power affine kernel residue modulo ``prime``."""

    p = require_prime(prime)
    a = require_integer(exponent, name="exponent")
    s = require_integer(shift, name="shift")
    r = require_integer(source_degree, name="source_degree")
    k = require_integer(target_degree, name="target_degree")
    if a < 1 or min(s, r, k) < 0:
        raise ValidationError("exponent must be positive and indices nonnegative")
    multiplier = p**a
    offset = r - multiplier * k
    return comb(s, offset) % p if 0 <= offset <= s else 0


def prime_power_kernel_certificate(
    prime: int,
    exponent: int,
    shift: int,
    degree: int,
) -> Certificate:
    """Validate residual sparsity of one prime power affine kernel."""

    p = require_prime(prime)
    a = require_integer(exponent, name="exponent")
    s = require_integer(shift, name="shift")
    nmax = require_integer(degree, name="degree")
    if a < 1 or s < 0 or nmax < 0:
        raise ValidationError("exponent must be positive and shift and degree nonnegative")
    multiplier = p**a
    valid = True
    for k in range(nmax + 1):
        for r in range(nmax + 1):
            actual = affine_mahler_kernel(multiplier, s, r, k) % p
            expected = prime_power_kernel_residue(p, a, s, r, k)
            valid = valid and actual == expected
    return Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        assumptions_checked=("prime power multiplier", "finite kernel range"),
        diagnostics={"prime": p, "exponent": a, "shift": s, "degree": nmax},
        theorem_id="PSC-IX-T36",
        message="prime power Mahler refinement has the declared sparse residue kernel",
    )


@dataclass(frozen=True, slots=True)
class UnitModeMahler:
    """Mahler coordinates for one local unit mode ``base**(jump*x + residue)``."""

    base: int
    jump: int
    residue: int = 0

    def __post_init__(self) -> None:
        base = require_integer(self.base, name="base")
        jump = require_integer(self.jump, name="jump")
        residue = require_integer(self.residue, name="residue")
        if base == 0:
            raise ValidationError("base must be nonzero")
        if jump < 1:
            raise ValidationError("jump must be positive")
        if residue < 0:
            raise ValidationError("residue must be nonnegative")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "jump", jump)
        object.__setattr__(self, "residue", residue)

    @property
    def increment(self) -> int:
        """Return ``base**jump - 1``."""

        return self.base**self.jump - 1

    def coefficients(self, degree: int) -> tuple[int, ...]:
        """Return exact Mahler coefficients through ``degree``."""

        nmax = require_integer(degree, name="degree")
        if nmax < 0:
            raise ValidationError("degree must be nonnegative")
        scale = self.base**self.residue
        return tuple(scale * self.increment**r for r in range(nmax + 1))

    def evaluate(self, value: int, *, degree: int | None = None) -> int | Fraction:
        """Evaluate a finite truncation or the exact integer mode."""

        x = require_integer(value, name="value")
        if x < 0:
            raise ValidationError("value must be nonnegative")
        if degree is None or degree >= x:
            return self.base ** (self.jump * x + self.residue)
        return MahlerSeries(self.coefficients(degree)).evaluate(x)

    def valuation_slope(self, prime: int) -> int | float:
        """Return the local valuation gained per Mahler degree."""

        return padic_valuation(self.increment, require_prime(prime))

    def certificate(self, maximum_value: int, degree: int | None = None) -> Certificate:
        """Validate the stored truncation on the full requested integer range."""

        nmax = require_integer(maximum_value, name="maximum_value")
        if nmax < 0:
            raise ValidationError("maximum_value must be nonnegative")
        truncation = nmax if degree is None else require_integer(degree, name="degree")
        if truncation < 0:
            raise ValidationError("degree must be nonnegative")
        series = MahlerSeries(self.coefficients(truncation))
        first_failure: int | None = None
        for x in range(nmax + 1):
            if series.evaluate(x) != self.base ** (self.jump * x + self.residue):
                first_failure = x
                break
        valid = first_failure is None
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("nonnegative integer evaluation range", "full requested range checked exactly"),
            diagnostics={
                "maximum_value": nmax,
                "degree": truncation,
                "first_failure": first_failure,
                "increment_zero": self.increment == 0,
            },
            theorem_id="PSC-IX-T25",
            message=(
                "unit mode Mahler truncation agrees on the full requested range"
                if valid
                else "unit mode Mahler truncation fails inside the requested range"
            ),
        )


def matrix_mahler_coefficients(
    matrix: np.ndarray,
    jump: int,
    residue: int,
    degree: int,
) -> tuple[np.ndarray, ...]:
    """Return exact object matrix Mahler coefficients for ``C**(jump*x + residue)``."""

    array = np.asarray(matrix, dtype=object)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise DimensionError("matrix must be square")
    h = require_integer(jump, name="jump")
    j = require_integer(residue, name="residue")
    nmax = require_integer(degree, name="degree")
    if h < 1 or j < 0 or nmax < 0:
        raise ValidationError("jump must be positive and residue and degree nonnegative")
    identity = np.eye(array.shape[0], dtype=object)
    increment = np.linalg.matrix_power(array, h) - identity
    prefix = np.linalg.matrix_power(array, j)
    coefficients = []
    power = identity.copy()
    for _ in range(nmax + 1):
        coefficients.append(prefix @ power)
        power = power @ increment
    return tuple(coefficients)


def evaluate_matrix_mahler(coefficients: Iterable[np.ndarray], value: int) -> np.ndarray:
    """Evaluate a finite matrix Mahler expansion at a nonnegative integer."""

    x = require_integer(value, name="value")
    if x < 0:
        raise ValidationError("value must be nonnegative")
    values = tuple(np.asarray(coefficient, dtype=object) for coefficient in coefficients)
    if not values:
        raise ValidationError("at least one matrix coefficient is required")
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise DimensionError("matrix Mahler coefficients must have one common shape")
    result = np.zeros(shape, dtype=object)
    for index, coefficient in enumerate(values):
        result += coefficient * comb(x, index)
    return result


__all__ = [
    "MahlerSeries",
    "UnitModeMahler",
    "affine_kernel_composition_certificate",
    "affine_kernel_matrix",
    "affine_mahler_kernel",
    "difference_table",
    "evaluate_matrix_mahler",
    "generalized_binomial",
    "mahler_coefficients",
    "matrix_mahler_coefficients",
    "prime_power_kernel_certificate",
    "prime_power_kernel_residue",
]
