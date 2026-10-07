"""Small exact linear algebra over the rational numbers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from ..exceptions import DimensionError, ValidationError

Scalar = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[tuple[Fraction, ...], ...]


def _fraction(value: Scalar) -> Fraction:
    if isinstance(value, bool):
        raise ValidationError("boolean values are not scalar coefficients")
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValidationError("value is not an exact rational coefficient") from exc


def as_vector(values: Iterable[Scalar]) -> Vector:
    """Return an immutable rational vector."""

    return tuple(_fraction(value) for value in values)


def as_matrix(rows: Iterable[Iterable[Scalar]]) -> Matrix:
    """Return an immutable rectangular rational matrix."""

    data = tuple(tuple(_fraction(value) for value in row) for row in rows)
    if not data:
        return ()
    width = len(data[0])
    if any(len(row) != width for row in data):
        raise DimensionError("matrix rows must have a common length")
    return data


def matrix_shape(matrix: Matrix) -> tuple[int, int]:
    """Return the matrix shape."""

    return len(matrix), len(matrix[0]) if matrix else 0


def zero_matrix(rows: int, columns: int) -> Matrix:
    """Return a rational zero matrix."""

    if rows < 0 or columns < 0:
        raise DimensionError("matrix dimensions must be nonnegative")
    return tuple(tuple(Fraction(0) for _ in range(columns)) for _ in range(rows))


def identity_matrix(size: int) -> Matrix:
    """Return a rational identity matrix."""

    if size < 0:
        raise DimensionError("matrix size must be nonnegative")
    return tuple(
        tuple(Fraction(int(row == column)) for column in range(size))
        for row in range(size)
    )


def transpose(matrix: Matrix) -> Matrix:
    """Return the matrix transpose."""

    rows, columns = matrix_shape(matrix)
    return tuple(tuple(matrix[row][column] for row in range(rows)) for column in range(columns))


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    """Add matrices with equal shape."""

    if matrix_shape(left) != matrix_shape(right):
        raise DimensionError("matrix shapes do not agree")
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def matrix_subtract(left: Matrix, right: Matrix) -> Matrix:
    """Subtract matrices with equal shape."""

    if matrix_shape(left) != matrix_shape(right):
        raise DimensionError("matrix shapes do not agree")
    return tuple(
        tuple(a - b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    """Return the exact matrix product."""

    left_rows, left_columns = matrix_shape(left)
    right_rows, right_columns = matrix_shape(right)
    if left_columns != right_rows:
        raise DimensionError("inner matrix dimensions do not agree")
    if left_rows == 0:
        return ()
    if right_columns == 0:
        return tuple(() for _ in range(left_rows))
    return tuple(
        tuple(
            sum((left[row][index] * right[index][column] for index in range(left_columns)), Fraction(0))
            for column in range(right_columns)
        )
        for row in range(left_rows)
    )


def matrix_vector_multiply(matrix: Matrix, vector: Sequence[Scalar]) -> Vector:
    """Return an exact matrix vector product."""

    data = as_vector(vector)
    rows, columns = matrix_shape(matrix)
    if columns != len(data):
        raise DimensionError("matrix and vector dimensions do not agree")
    return tuple(
        sum((matrix[row][column] * data[column] for column in range(columns)), Fraction(0))
        for row in range(rows)
    )


def matrix_power(matrix: Matrix, exponent: int) -> Matrix:
    """Return a nonnegative exact matrix power."""

    rows, columns = matrix_shape(matrix)
    if rows != columns:
        raise DimensionError("matrix powers require a square matrix")
    if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent < 0:
        raise ValidationError("exponent must be a nonnegative integer")
    result = identity_matrix(rows)
    base = matrix
    power = exponent
    while power:
        if power & 1:
            result = matrix_multiply(result, base)
        base = matrix_multiply(base, base)
        power >>= 1
    return result


def augment(left: Matrix, right: Matrix) -> Matrix:
    """Concatenate two matrices horizontally."""

    if len(left) != len(right):
        raise DimensionError("matrix row counts do not agree")
    return tuple(a + b for a, b in zip(left, right, strict=True))


def rref(matrix: Matrix) -> tuple[Matrix, tuple[int, ...]]:
    """Return reduced row echelon form and pivot columns."""

    rows, columns = matrix_shape(matrix)
    work = [list(row) for row in matrix]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        selected = next((row for row in range(pivot_row, rows) if work[row][column] != 0), None)
        if selected is None:
            continue
        work[pivot_row], work[selected] = work[selected], work[pivot_row]
        pivot = work[pivot_row][column]
        work[pivot_row] = [value / pivot for value in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = work[row][column]
            if factor != 0:
                work[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(work[row], work[pivot_row], strict=True)
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    return as_matrix(work), tuple(pivot_columns)


def matrix_rank(matrix: Matrix) -> int:
    """Return exact rational rank."""

    return len(rref(matrix)[1])


def determinant(matrix: Matrix) -> Fraction:
    """Return the exact determinant."""

    rows, columns = matrix_shape(matrix)
    if rows != columns:
        raise DimensionError("determinants require a square matrix")
    if rows == 0:
        return Fraction(1)
    work = [list(row) for row in matrix]
    result = Fraction(1)
    sign = 1
    for column in range(columns):
        selected = next((row for row in range(column, rows) if work[row][column] != 0), None)
        if selected is None:
            return Fraction(0)
        if selected != column:
            work[column], work[selected] = work[selected], work[column]
            sign *= -1
        pivot = work[column][column]
        result *= pivot
        for row in range(column + 1, rows):
            factor = work[row][column] / pivot
            for index in range(column + 1, columns):
                work[row][index] -= factor * work[column][index]
    return result * sign


def inverse_matrix(matrix: Matrix) -> Matrix:
    """Return the exact inverse of a nonsingular square matrix."""

    rows, columns = matrix_shape(matrix)
    if rows != columns:
        raise DimensionError("matrix inversion requires a square matrix")
    reduced, pivots = rref(augment(matrix, identity_matrix(rows)))
    if pivots[:rows] != tuple(range(rows)):
        raise ValidationError("matrix is singular")
    left = tuple(tuple(reduced[row][column] for column in range(rows)) for row in range(rows))
    if left != identity_matrix(rows):
        raise ValidationError("matrix is singular")
    return tuple(tuple(reduced[row][rows + column] for column in range(rows)) for row in range(rows))


@dataclass(frozen=True, slots=True)
class ExactLinearSystem:
    """Exact result for a possibly rectangular linear system."""

    consistent: bool
    rank: int
    solution: Vector | None
    nullspace_basis: tuple[Vector, ...]
    pivot_columns: tuple[int, ...]
    residual: Vector


def solve_linear_system(matrix: Matrix, right_hand_side: Sequence[Scalar]) -> ExactLinearSystem:
    """Solve a rational linear system and return one affine representative."""

    rows, columns = matrix_shape(matrix)
    rhs = as_vector(right_hand_side)
    if rows != len(rhs):
        raise DimensionError("right hand side length does not match the matrix")
    rhs_matrix = tuple((value,) for value in rhs)
    reduced, pivots_augmented = rref(augment(matrix, rhs_matrix))
    inconsistent = any(
        all(reduced[row][column] == 0 for column in range(columns))
        and reduced[row][columns] != 0
        for row in range(rows)
    )
    reduced_left, pivots = rref(matrix)
    rank = len(pivots)
    basis = nullspace(matrix)
    if inconsistent:
        return ExactLinearSystem(False, rank, None, basis, pivots, rhs)
    solution = [Fraction(0) for _ in range(columns)]
    pivot_row = 0
    for pivot_column in pivots_augmented:
        if pivot_column >= columns:
            continue
        solution[pivot_column] = reduced[pivot_row][columns]
        pivot_row += 1
    candidate = tuple(solution)
    predicted = matrix_vector_multiply(matrix, candidate)
    residual = tuple(rhs_value - predicted_value for rhs_value, predicted_value in zip(rhs, predicted, strict=True))
    return ExactLinearSystem(True, rank, candidate, basis, pivots, residual)


def nullspace(matrix: Matrix) -> tuple[Vector, ...]:
    """Return an exact basis for the right nullspace."""

    rows, columns = matrix_shape(matrix)
    if columns == 0:
        return ()
    reduced, pivots = rref(matrix)
    pivot_set = set(pivots)
    free_columns = [column for column in range(columns) if column not in pivot_set]
    basis: list[Vector] = []
    for free in free_columns:
        vector = [Fraction(0) for _ in range(columns)]
        vector[free] = Fraction(1)
        for pivot_row, pivot_column in enumerate(pivots):
            vector[pivot_column] = -reduced[pivot_row][free]
        basis.append(tuple(vector))
    return tuple(basis)


def matrix_from_columns(columns: Sequence[Sequence[Scalar]]) -> Matrix:
    """Build a matrix from column vectors."""

    if not columns:
        return ()
    data = tuple(as_vector(column) for column in columns)
    height = len(data[0])
    if any(len(column) != height for column in data):
        raise DimensionError("column lengths do not agree")
    return tuple(tuple(data[column][row] for column in range(len(data))) for row in range(height))


def columns(matrix: Matrix) -> tuple[Vector, ...]:
    """Return matrix columns."""

    return tuple(tuple(row[column] for row in matrix) for column in range(matrix_shape(matrix)[1]))


def kronecker(left: Matrix, right: Matrix) -> Matrix:
    """Return the exact Kronecker product."""

    left_rows, left_columns = matrix_shape(left)
    right_rows, right_columns = matrix_shape(right)
    return tuple(
        tuple(
            left[i][j] * right[r][c]
            for j in range(left_columns)
            for c in range(right_columns)
        )
        for i in range(left_rows)
        for r in range(right_rows)
    )


__all__ = [
    "ExactLinearSystem",
    "Matrix",
    "Scalar",
    "Vector",
    "as_matrix",
    "as_vector",
    "columns",
    "determinant",
    "identity_matrix",
    "inverse_matrix",
    "kronecker",
    "matrix_add",
    "matrix_from_columns",
    "matrix_multiply",
    "matrix_power",
    "matrix_rank",
    "matrix_shape",
    "matrix_subtract",
    "matrix_vector_multiply",
    "nullspace",
    "rref",
    "solve_linear_system",
    "transpose",
    "zero_matrix",
]
