"""Finite alternating forms, radicals, and dyadic quadratic refinements."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isqrt
from typing import Any, Iterable

import numpy as np

from ..._number_theory import require_integer, require_prime
from ...certificates import Certificate, Exactness
from ...exceptions import DimensionError, ValidationError
from ...deformation.complexes import row_rank_mod
from ...nonlinear.lifting import solve_mod_prime


def _matrix_mod(value: Any, prime: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=object)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DimensionError("alternating form matrix must be square")
    output = np.empty(matrix.shape, dtype=int)
    for index, entry in np.ndenumerate(matrix):
        integer = int(entry)
        if integer != entry:
            raise ValidationError("finite alternating forms require integral entries")
        output[index] = integer % prime
    return output


@dataclass(frozen=True, slots=True)
class AlternatingForm:
    """A finite alternating bilinear form over a prime field."""

    matrix: Any
    prime: int

    def __post_init__(self) -> None:
        p = require_prime(self.prime)
        matrix = _matrix_mod(self.matrix, p)
        if np.any(np.diag(matrix) % p):
            raise ValidationError("alternating form must have zero diagonal")
        if np.any((matrix + matrix.T) % p):
            raise ValidationError("alternating form matrix is not skew symmetric")
        object.__setattr__(self, "prime", p)
        object.__setattr__(self, "matrix", matrix)

    @property
    def dimension(self) -> int:
        """Return the ambient vector space dimension."""

        return int(self.matrix.shape[0])

    @property
    def rank(self) -> int:
        """Return the exact form rank."""

        return row_rank_mod(self.matrix, self.prime)

    @property
    def radical_dimension(self) -> int:
        """Return the radical dimension."""

        return self.dimension - self.rank

    @property
    def nondegenerate(self) -> bool:
        """Return whether the form has zero radical."""

        return self.rank == self.dimension

    def radical_basis(self) -> tuple[tuple[int, ...], ...]:
        """Return a basis of the radical over the prime field."""

        solution = solve_mod_prime(self.matrix, np.zeros(self.dimension, dtype=int), self.prime)
        return solution.kernel_basis

    def evaluate(self, left: Iterable[int], right: Iterable[int]) -> int:
        """Evaluate the bilinear form."""

        x = np.asarray(tuple(left), dtype=int) % self.prime
        y = np.asarray(tuple(right), dtype=int) % self.prime
        if x.shape != (self.dimension,) or y.shape != (self.dimension,):
            raise DimensionError("form vectors have the wrong dimension")
        return int(x @ self.matrix @ y) % self.prime

    def certificate(self) -> Certificate:
        """Return an exact form certificate."""

        valid = not np.any(np.diag(self.matrix) % self.prime) and not np.any((self.matrix + self.matrix.T) % self.prime)
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("prime field", "alternating matrix"),
            diagnostics={"dimension": self.dimension, "rank": self.rank, "radical_dimension": self.radical_dimension},
            theorem_id="PSC-XI-SYMPLECTIC-ENVELOPE",
            message="valid finite alternating form" if valid else "alternating form validation failed",
        )


@dataclass(frozen=True, slots=True)
class QuadraticRefinement:
    """A dyadic quadratic refinement of a finite alternating form."""

    form: AlternatingForm
    linear: tuple[int, ...] = ()
    max_dimension: int = 20

    def __post_init__(self) -> None:
        if self.form.prime != 2:
            raise ValidationError("quadratic refinement is implemented over the field of two elements")
        linear = self.linear or (0,) * self.form.dimension
        values = tuple(require_integer(value, name="linear coefficient") % 2 for value in linear)
        if len(values) != self.form.dimension:
            raise DimensionError("linear refinement vector has the wrong dimension")
        if self.form.dimension > require_integer(self.max_dimension, name="max_dimension"):
            raise ValidationError("quadratic refinement enumeration exceeds the safety limit")
        object.__setattr__(self, "linear", values)

    def evaluate(self, vector: Iterable[int]) -> int:
        """Evaluate the chosen quadratic refinement."""

        x = tuple(require_integer(value, name="vector entry") % 2 for value in vector)
        if len(x) != self.form.dimension:
            raise DimensionError("quadratic vector has the wrong dimension")
        value = sum(self.linear[index] * x[index] for index in range(self.form.dimension))
        for i in range(self.form.dimension):
            for j in range(i + 1, self.form.dimension):
                value += int(self.form.matrix[i, j]) * x[i] * x[j]
        return value % 2

    def polar_certificate(self) -> Certificate:
        """Check that the polar form equals the declared alternating form."""

        maximum = 0
        vectors = tuple(product((0, 1), repeat=self.form.dimension))
        for left in vectors:
            for right in vectors:
                summed = tuple((a + b) % 2 for a, b in zip(left, right, strict=True))
                polar = (self.evaluate(summed) - self.evaluate(left) - self.evaluate(right)) % 2
                maximum = max(maximum, (polar - self.form.evaluate(left, right)) % 2)
        return Certificate(
            valid=maximum == 0,
            exactness=Exactness.EXACT,
            residuals={"polar_mismatch": float(maximum)},
            assumptions_checked=("finite exhaustive dyadic vectors",),
            theorem_id="PSC-XI-QUADRATIC-REFINEMENT",
            message="quadratic polar form verified" if maximum == 0 else "quadratic polar form failed",
        )

    def gauss_sum(self) -> int:
        """Return the integral dyadic Gauss sum."""

        return sum((-1) ** self.evaluate(vector) for vector in product((0, 1), repeat=self.form.dimension))

    def arf_invariant(self) -> int:
        """Return the Arf invariant of a nondegenerate even-dimensional form."""

        if not self.form.nondegenerate or self.form.dimension % 2:
            raise ValidationError("Arf invariant requires a nondegenerate even-dimensional form")
        gauss = self.gauss_sum()
        expected = 2 ** (self.form.dimension // 2)
        if abs(gauss) != expected:
            raise ArithmeticError("quadratic Gauss sum does not have the expected magnitude")
        return 0 if gauss > 0 else 1

    def normalized_gauss_sign(self) -> int:
        """Return the normalized dyadic Gauss sign."""

        return 1 if self.arf_invariant() == 0 else -1


__all__ = ["AlternatingForm", "QuadraticRefinement"]
