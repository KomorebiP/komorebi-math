"""Exact one-digit polynomial correction fibers over prime fields.

This implements the finite part of Paper IX, sections 4.1--4.4 and 4.6.
Finite towers are residue computations, not representations of p-adic limits.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product

from .._number_theory import require_integer, require_prime
from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError


def _point(values: Iterable[int], dimension: int) -> tuple[int, ...]:
    result = tuple(require_integer(value, name="point coordinate") for value in values)
    if len(result) != dimension:
        raise DimensionError("point has the wrong polynomial variable count")
    return result


@dataclass(frozen=True, slots=True, init=False)
class SparseIntegerPolynomial:
    """A frozen integer polynomial with explicit monomial exponent tuples."""

    variable_count: int
    terms: tuple[tuple[tuple[int, ...], int], ...]

    def __init__(
        self,
        variable_count: int,
        terms: Mapping[tuple[int, ...], int] | Iterable[tuple[tuple[int, ...], int]],
    ) -> None:
        dimension = require_integer(variable_count, name="variable_count")
        if dimension < 1:
            raise ValidationError("variable_count must be positive")
        combined: dict[tuple[int, ...], int] = {}
        for exponents, coefficient in terms.items() if isinstance(terms, Mapping) else terms:
            powers = _point(exponents, dimension)
            if any(power < 0 for power in powers):
                raise ValidationError("monomial exponents must be nonnegative")
            value = require_integer(coefficient, name="polynomial coefficient")
            combined[powers] = combined.get(powers, 0) + value
        object.__setattr__(self, "variable_count", dimension)
        object.__setattr__(
            self, "terms", tuple(sorted((powers, value) for powers, value in combined.items() if value))
        )

    def evaluate(self, point: Iterable[int]) -> int:
        """Evaluate with Python integers and no fixed-width arithmetic."""
        values = _point(point, self.variable_count)
        total = 0
        for powers, coefficient in self.terms:
            monomial = coefficient
            for value, power in zip(values, powers, strict=True):
                monomial *= value**power
            total += monomial
        return total

    def gradient(self, point: Iterable[int]) -> tuple[int, ...]:
        """Evaluate every formal partial derivative exactly."""
        values = _point(point, self.variable_count)
        result = [0] * self.variable_count
        for powers, coefficient in self.terms:
            for coordinate, exponent in enumerate(powers):
                if exponent:
                    monomial = coefficient * exponent
                    for index, (value, power) in enumerate(zip(values, powers, strict=True)):
                        monomial *= value ** (power - (index == coordinate))
                    result[coordinate] += monomial
        return tuple(result)


def _solve_mod(
    rows: tuple[tuple[int, ...], ...],
    rhs: tuple[int, ...],
    columns: int,
    prime: int,
) -> tuple[int, tuple[int, ...] | None, tuple[tuple[int, ...], ...]]:
    """RREF with an exact particular solution and full nullspace basis."""
    matrix = [
        [value % prime for value in row] + [value % prime] for row, value in zip(rows, rhs, strict=True)
    ]
    pivots: list[int] = []
    for column in range(columns):
        pivot = next((index for index in range(len(pivots), len(matrix)) if matrix[index][column]), None)
        if pivot is None:
            continue
        rank = len(pivots)
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inverse = pow(matrix[rank][column], -1, prime)
        matrix[rank] = [(value * inverse) % prime for value in matrix[rank]]
        for index in range(len(matrix)):
            if index != rank and matrix[index][column]:
                factor = matrix[index][column]
                matrix[index] = [
                    (value - factor * pivot_value) % prime
                    for value, pivot_value in zip(matrix[index], matrix[rank], strict=True)
                ]
        pivots.append(column)
    free = tuple(column for column in range(columns) if column not in pivots)
    basis: list[tuple[int, ...]] = []
    for column in free:
        vector = [0] * columns
        vector[column] = 1
        for index, pivot in enumerate(pivots):
            vector[pivot] = (-matrix[index][column]) % prime
        basis.append(tuple(vector))
    if any(not any(row[:columns]) and row[columns] for row in matrix):
        return len(pivots), None, tuple(basis)
    particular = [0] * columns
    for index, pivot in enumerate(pivots):
        particular[pivot] = matrix[index][columns]
    return len(pivots), tuple(particular), tuple(basis)


@dataclass(frozen=True, slots=True)
class PolynomialLiftFiber:
    """A finite next-residue fiber and its correction torsor data.

    ``certificate`` approves existence of a next lift only when the exact
    correction system is consistent. An empty fiber remains inspectable.
    """

    prime: int
    depth: int
    root: tuple[int, ...]
    jacobian: tuple[tuple[int, ...], ...]
    residual: tuple[int, ...]
    fixed_digits: tuple[tuple[int, int], ...]
    solved_jacobian_rank: int
    particular: tuple[int, ...] | None
    kernel_basis: tuple[tuple[int, ...], ...]
    certificate: Certificate

    @property
    def solvable(self) -> bool:
        return self.particular is not None

    @property
    def count(self) -> int:
        """Count distinct next residue roots, without enumerating them."""
        return self.prime ** len(self.kernel_basis) if self.solvable else 0

    @property
    def obstruction_dimension(self) -> int:
        """Return the cokernel dimension of the free-coordinate Jacobian."""
        return len(self.jacobian) - self.solved_jacobian_rank

    def correction_digits(self, *, max_count: int = 10000) -> tuple[tuple[int, ...], ...]:
        """Enumerate the full correction fiber within an explicit size cap."""
        limit = require_integer(max_count, name="max_count")
        if limit < 0:
            raise ValidationError("max_count must be nonnegative")
        if self.count > limit:
            raise ValidationError(f"correction fiber has {self.count} elements, exceeding max_count={limit}")
        if self.particular is None:
            return ()
        return tuple(
            tuple(
                (
                    self.particular[index]
                    + sum(
                        value * vector[index]
                        for value, vector in zip(coefficients, self.kernel_basis, strict=True)
                    )
                )
                % self.prime
                for index in range(len(self.root))
            )
            for coefficients in product(range(self.prime), repeat=len(self.kernel_basis))
        )

    def lifts(self, *, max_count: int = 10000) -> tuple[tuple[int, ...], ...]:
        """Return canonical roots modulo ``prime**(depth+1)`` above ``root``."""
        scale = self.prime**self.depth
        modulus = scale * self.prime
        return tuple(
            tuple((value + scale * digit) % modulus for value, digit in zip(self.root, digits, strict=True))
            for digits in self.correction_digits(max_count=max_count)
        )


@dataclass(frozen=True, slots=True, init=False)
class IntegerPolynomialSystem:
    """A frozen, nonempty integer polynomial equation list."""

    polynomials: tuple[SparseIntegerPolynomial, ...]

    def __init__(self, polynomials: Iterable[SparseIntegerPolynomial]) -> None:
        values = tuple(polynomials)
        if not values or any(not isinstance(value, SparseIntegerPolynomial) for value in values):
            raise ValidationError("a system requires integer polynomials")
        if len({value.variable_count for value in values}) != 1:
            raise DimensionError("all equations must use the same variables")
        object.__setattr__(self, "polynomials", values)

    @property
    def variable_count(self) -> int:
        return self.polynomials[0].variable_count

    def evaluate(self, point: Iterable[int]) -> tuple[int, ...]:
        values = _point(point, self.variable_count)
        return tuple(polynomial.evaluate(values) for polynomial in self.polynomials)

    def jacobian(self, point: Iterable[int]) -> tuple[tuple[int, ...], ...]:
        values = _point(point, self.variable_count)
        return tuple(polynomial.gradient(values) for polynomial in self.polynomials)

    def stack(self, other: IntegerPolynomialSystem) -> IntegerPolynomialSystem:
        """Require a common correcting digit for all equations in two systems."""
        if self.variable_count != other.variable_count:
            raise DimensionError("simultaneous systems must use the same variables")
        return IntegerPolynomialSystem((*self.polynomials, *other.polynomials))

    def lift_fiber(
        self,
        root: Iterable[int],
        *,
        prime: int,
        depth: int,
        fixed_digits: Mapping[int, int] | None = None,
    ) -> PolynomialLiftFiber:
        """Solve the exact one-digit obstruction, allowing prescribed digits.

        Polynomial depth is positive. Prime-field rank formulas do not apply
        to composite residue rings, which this API rejects explicitly.
        """
        p = require_prime(prime)
        level = require_integer(depth, name="depth")
        if level < 1:
            raise ValidationError("polynomial lifting depth must be positive")
        values = _point(root, self.variable_count)
        scale = p**level
        evaluation = self.evaluate(values)
        if any(value % scale for value in evaluation):
            raise ValidationError("supplied point is not a root at the declared lower depth")
        fixed: dict[int, int] = {}
        for coordinate, digit in (fixed_digits or {}).items():
            index = require_integer(coordinate, name="fixed coordinate")
            value = require_integer(digit, name="fixed digit")
            if not 0 <= index < self.variable_count or not 0 <= value < p:
                raise ValidationError("fixed digits require a valid coordinate and a digit in [0, prime)")
            fixed[index] = value
        free = tuple(index for index in range(self.variable_count) if index not in fixed)
        jacobian = tuple(tuple(value % p for value in row) for row in self.jacobian(values))
        residual = tuple((value // scale) % p for value in evaluation)
        rows = tuple(tuple(row[index] for index in free) for row in jacobian)
        rhs = tuple(
            (-value - sum(row[index] * digit for index, digit in fixed.items())) % p
            for row, value in zip(jacobian, residual, strict=True)
        )
        rank, solution, basis = _solve_mod(rows, rhs, len(free), p)

        def embed(vector: tuple[int, ...], *, particular: bool = False) -> tuple[int, ...]:
            result = [0] * self.variable_count
            for index, value in zip(free, vector, strict=True):
                result[index] = value
            if particular:
                for index, value in fixed.items():
                    result[index] = value
            return tuple(result)

        particular = None if solution is None else embed(solution, particular=True)
        kernel = tuple(embed(vector) for vector in basis)
        return PolynomialLiftFiber(
            p,
            level,
            values,
            jacobian,
            residual,
            tuple(sorted(fixed.items())),
            rank,
            particular,
            kernel,
            Certificate(
                valid=particular is not None,
                exactness=Exactness.EXACT,
                assumptions_checked=(
                    "integer polynomial coefficients",
                    "prime field",
                    "positive polynomial depth",
                    "supplied lower residue root",
                    "exact Jacobian correction equation",
                ),
                diagnostics={
                    "arithmetic": "Python integers",
                    "rank": rank,
                    "free_coordinate_count": len(free),
                    "kernel_dimension": len(kernel),
                    "lift_count": p ** len(kernel) if particular is not None else 0,
                    "scope": "one finite next residue level",
                },
                theorem_id="PSC-IX-FINITE-POLYNOMIAL-LIFT",
                message="next polynomial residue lifts exist"
                if particular is not None
                else "the exact finite polynomial obstruction is nonzero",
            ),
        )

    def finite_tower(
        self,
        root: Iterable[int],
        *,
        prime: int,
        depth: int,
        target_depth: int,
        max_steps: int = 1000,
    ) -> tuple[tuple[int, ...], ...]:
        """Choose zero free-coordinate digits to build a finite residue tower.

        A later obstruction raises. Previous success alone does not imply
        that a singular system lifts indefinitely.
        """
        p = require_prime(prime)
        initial = require_integer(depth, name="depth")
        target = require_integer(target_depth, name="target_depth")
        cap = require_integer(max_steps, name="max_steps")
        if initial < 1 or target < initial or cap < 0 or target - initial > cap:
            raise ValidationError(
                "finite tower depths must satisfy 1 <= depth <= target_depth within max_steps"
            )
        values = _point(root, self.variable_count)
        if any(value % (p**initial) for value in self.evaluate(values)):
            raise ValidationError("supplied point is not a lower residue root")
        result = [tuple(value % (p**initial) for value in values)]
        for level in range(initial, target):
            fiber = self.lift_fiber(result[-1], prime=p, depth=level)
            fiber.certificate.require()
            assert fiber.particular is not None
            result.append(
                tuple(
                    (value + p**level * digit) % (p ** (level + 1))
                    for value, digit in zip(result[-1], fiber.particular, strict=True)
                )
            )
        return tuple(result)


__all__ = ["IntegerPolynomialSystem", "PolynomialLiftFiber", "SparseIntegerPolynomial"]
