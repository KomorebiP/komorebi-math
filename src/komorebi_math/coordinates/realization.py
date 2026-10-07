"""Finite dimensional realizations, observability, and intertwiners."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .budget import DEFAULT_COORDINATE_BUDGET, CoordinateBudget

from .exact import (
    Matrix,
    Scalar,
    Vector,
    as_matrix,
    as_vector,
    determinant,
    identity_matrix,
    inverse_matrix,
    matrix_add,
    matrix_from_columns,
    matrix_multiply,
    matrix_power,
    matrix_rank,
    matrix_shape,
    matrix_vector_multiply,
    nullspace,
    solve_linear_system,
    zero_matrix,
)
from .recurrence import recurrence_polynomial


@dataclass(frozen=True, slots=True)
class LinearRealization:
    r"""A scalar sequence realization ``a_n=h^T A^n v``."""

    transition: Matrix
    initial: Vector
    output: Vector
    name: str = "linear realization"

    def __post_init__(self) -> None:
        transition = as_matrix(self.transition)
        initial = as_vector(self.initial)
        output = as_vector(self.output)
        rows, columns = matrix_shape(transition)
        if rows == 0 or rows != columns:
            raise DimensionError("the transition matrix must be nonempty and square")
        if len(initial) != rows or len(output) != rows:
            raise DimensionError("state and output dimensions must match the transition")
        if not self.name.strip():
            raise ValidationError("realization name must be nonempty")
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "name", str(self.name))

    @property
    def dimension(self) -> int:
        """Return the state dimension."""

        return len(self.initial)

    def state(self, index: int) -> Vector:
        """Return the exact state at a nonnegative index."""

        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValidationError("index must be a nonnegative integer")
        return matrix_vector_multiply(matrix_power(self.transition, index), self.initial)

    def term(self, index: int) -> Fraction:
        """Return one exact output term."""

        state = self.state(index)
        return sum((a * b for a, b in zip(self.output, state, strict=True)), Fraction(0))

    def prefix(self, count: int) -> Vector:
        """Return an exact output prefix."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationError("count must be a nonnegative integer")
        return tuple(self.term(index) for index in range(count))


@dataclass(frozen=True, slots=True)
class StateRecovery:
    """Finite horizon state recovery result."""

    exact: bool
    unique: bool
    rank: int
    horizon: int
    state: Vector | None
    predicted: Vector
    residual: Vector

    def certificate(self) -> Certificate:
        """Return an exact finite recovery certificate."""

        valid = self.state is not None and all(value == 0 for value in self.residual)
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite rational realization", "declared output horizon"),
            diagnostics={
                "exact": self.exact,
                "unique": self.unique,
                "rank": self.rank,
                "horizon": self.horizon,
            },
            theorem_id="CC.S2",
            message=(
                "initial state recovered exactly"
                if valid
                else "finite prefix is not exactly realized by the recovered state"
            ),
        )


@dataclass(frozen=True, slots=True)
class IntertwinerSpace:
    r"""Solutions of ``B S=S A`` from a source to a target realization."""

    source_dimension: int
    target_dimension: int
    basis: tuple[Matrix, ...]

    @property
    def dimension(self) -> int:
        """Return the dimension of the intertwiner vector space."""

        return len(self.basis)

    def contains_invertible_basis_element(self) -> bool:
        """Return whether a displayed basis element is invertible."""

        if self.source_dimension != self.target_dimension:
            return False
        return any(determinant(matrix) != 0 for matrix in self.basis)

    def certificate(self, source: Matrix, target: Matrix) -> Certificate:
        """Verify every displayed basis matrix against ``target*S=S*source``."""

        a = as_matrix(source)
        b = as_matrix(target)
        valid = all(
            matrix_multiply(b, matrix) == matrix_multiply(matrix, a)
            for matrix in self.basis
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite rational transition matrices",),
            diagnostics={
                "dimension": self.dimension,
                "source_dimension": self.source_dimension,
                "target_dimension": self.target_dimension,
                "invertible_basis_element": self.contains_invertible_basis_element(),
            },
            theorem_id="CC.S3",
            message="intertwiner basis satisfies the exact matrix equations",
        )


def observability_matrix(
    transition: Matrix,
    output: Sequence[Scalar],
    horizon: int,
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> Matrix:
    r"""Return rows ``h^T A^n`` for ``0 <= n < horizon``."""

    matrix = as_matrix(transition)
    rows, columns = matrix_shape(matrix)
    if rows == 0 or rows != columns:
        raise DimensionError("the transition matrix must be nonempty and square")
    h = as_vector(output)
    if len(h) != rows:
        raise DimensionError("output dimension does not match the transition")
    budget.require_horizon(horizon, operation="observability construction")
    budget.require_matrix(horizon, rows, operation="observability construction")
    current = identity_matrix(rows)
    result: list[Vector] = []
    for _ in range(horizon):
        row = tuple(
            sum((h[index] * current[index][column] for index in range(rows)), Fraction(0))
            for column in range(columns)
        )
        result.append(row)
        current = matrix_multiply(current, matrix)
    return tuple(result)


def _independent_row_indices(matrix: Matrix, count: int) -> tuple[int, ...]:
    rows, columns = matrix_shape(matrix)
    if count > min(rows, columns):
        return ()
    for indices in combinations(range(rows), count):
        submatrix = tuple(matrix[index] for index in indices)
        if matrix_rank(submatrix) == count:
            return tuple(indices)
    return ()


def recover_initial_state(
    prefix: Sequence[Scalar],
    transition: Matrix,
    output: Sequence[Scalar],
    *,
    budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
) -> StateRecovery:
    """Recover or test a state against a finite output prefix."""

    data = as_vector(prefix)
    observable = observability_matrix(transition, output, len(data), budget=budget)
    rows, dimension = matrix_shape(observable)
    system = solve_linear_system(observable, data)
    if system.consistent and system.solution is not None:
        predicted = matrix_vector_multiply(observable, system.solution)
        residual = tuple(a - b for a, b in zip(data, predicted, strict=True))
        return StateRecovery(
            True,
            system.rank == dimension,
            system.rank,
            rows,
            system.solution,
            predicted,
            residual,
        )
    independent = _independent_row_indices(observable, min(system.rank, dimension))
    candidate: Vector | None = None
    if independent and len(independent) == dimension:
        square = tuple(observable[index] for index in independent)
        rhs = tuple(data[index] for index in independent)
        candidate = matrix_vector_multiply(inverse_matrix(square), rhs)
    predicted = matrix_vector_multiply(observable, candidate) if candidate is not None else tuple(Fraction(0) for _ in data)
    residual = tuple(a - b for a, b in zip(data, predicted, strict=True))
    return StateRecovery(
        False,
        system.rank == dimension,
        system.rank,
        rows,
        candidate,
        predicted,
        residual,
    )


def cyclic_krylov_matrix(transition: Matrix, vector: Sequence[Scalar]) -> Matrix:
    """Return the square cyclic Krylov matrix ``[v,Av,...]``."""

    matrix = as_matrix(transition)
    rows, columns = matrix_shape(matrix)
    if rows == 0 or rows != columns:
        raise DimensionError("the transition matrix must be nonempty and square")
    current = as_vector(vector)
    if len(current) != rows:
        raise DimensionError("vector dimension does not match the transition")
    basis: list[Vector] = []
    for _ in range(rows):
        basis.append(current)
        current = matrix_vector_multiply(matrix, current)
    return matrix_from_columns(basis)


def intertwiner_space(source: Matrix, target: Matrix) -> IntertwinerSpace:
    r"""Return an exact basis of matrices satisfying ``target*S=S*source``."""

    a = as_matrix(source)
    b = as_matrix(target)
    source_rows, source_columns = matrix_shape(a)
    target_rows, target_columns = matrix_shape(b)
    if source_rows == 0 or source_rows != source_columns:
        raise DimensionError("source transition must be nonempty and square")
    if target_rows == 0 or target_rows != target_columns:
        raise DimensionError("target transition must be nonempty and square")
    variable_count = target_rows * source_rows
    equations: list[list[Fraction]] = []
    for row in range(target_rows):
        for column in range(source_rows):
            equation = [Fraction(0) for _ in range(variable_count)]
            for index in range(target_rows):
                equation[index * source_rows + column] += b[row][index]
            for index in range(source_rows):
                equation[row * source_rows + index] -= a[index][column]
            equations.append(equation)
    basis_vectors = nullspace(tuple(tuple(row) for row in equations))
    basis = tuple(
        tuple(
            tuple(vector[row * source_rows + column] for column in range(source_rows))
            for row in range(target_rows)
        )
        for vector in basis_vectors
    )
    return IntertwinerSpace(source_rows, target_rows, basis)


def similarity_from_cyclic_vectors(
    source: Matrix,
    target: Matrix,
    source_vector: Sequence[Scalar],
    target_vector: Sequence[Scalar],
) -> Matrix:
    r"""Construct ``S`` with ``target*S=S*source`` from cyclic vectors."""

    a = as_matrix(source)
    b = as_matrix(target)
    if matrix_shape(a) != matrix_shape(b):
        raise DimensionError("similar transitions must have equal dimension")
    source_krylov = cyclic_krylov_matrix(a, source_vector)
    target_krylov = cyclic_krylov_matrix(b, target_vector)
    if determinant(source_krylov) == 0 or determinant(target_krylov) == 0:
        raise ValidationError("both selected vectors must be cyclic")
    transform = matrix_multiply(target_krylov, inverse_matrix(source_krylov))
    if matrix_multiply(b, transform) != matrix_multiply(transform, a):
        raise ValidationError("the cyclic data do not define an intertwiner")
    return transform


def _polynomial_at_matrix(polynomial: Sequence[Scalar], matrix: Matrix) -> Matrix:
    coefficients = as_vector(polynomial)
    rows, columns = matrix_shape(matrix)
    if rows == 0 or rows != columns:
        raise DimensionError("matrix polynomial evaluation needs a square matrix")
    result = zero_matrix(rows, columns)
    power = identity_matrix(rows)
    for coefficient in coefficients:
        scaled = tuple(tuple(coefficient * value for value in row) for row in power)
        result = matrix_add(result, scaled)
        power = matrix_multiply(power, matrix)
    return result


def realization_defect(
    realization: LinearRealization,
    target_coefficients: Iterable[Scalar],
) -> LinearRealization:
    r"""Return a realization of the target recurrence defect sequence.

    If ``a_n=h^T A^n v`` and the target recurrence polynomial is ``q``,
    then ``delta_n=h^T A^n q(A)v``.
    """

    polynomial = recurrence_polynomial(target_coefficients)
    defect_initial = matrix_vector_multiply(
        _polynomial_at_matrix(polynomial, realization.transition),
        realization.initial,
    )
    return LinearRealization(
        realization.transition,
        defect_initial,
        realization.output,
        f"{realization.name} defect realization",
    )


__all__ = [
    "IntertwinerSpace",
    "LinearRealization",
    "StateRecovery",
    "cyclic_krylov_matrix",
    "intertwiner_space",
    "observability_matrix",
    "realization_defect",
    "recover_initial_state",
    "similarity_from_cyclic_vectors",
]
