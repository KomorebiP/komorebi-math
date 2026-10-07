"""Coordinate transforms over finite prime fields."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .exact import Scalar
from .transform import CoordinateTransform

ModularVector = tuple[int, ...]
ModularMatrix = tuple[tuple[int, ...], ...]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _matrix(values: Iterable[Iterable[int]], prime: int) -> ModularMatrix:
    rows = tuple(tuple(int(value) % prime for value in row) for row in values)
    if not rows:
        raise DimensionError("finite-field coordinate matrices must be nonempty")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise DimensionError("finite-field coordinate matrices must be nonempty and rectangular")
    return rows


def _shape(matrix: ModularMatrix) -> tuple[int, int]:
    return len(matrix), len(matrix[0])


def _multiply(left: ModularMatrix, right: ModularMatrix, prime: int) -> ModularMatrix:
    left_rows, left_columns = _shape(left)
    right_rows, right_columns = _shape(right)
    if left_columns != right_rows:
        raise DimensionError("finite-field matrix dimensions do not agree")
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(left_columns)) % prime
            for column in range(right_columns)
        )
        for row in range(left_rows)
    )


def _identity(size: int) -> ModularMatrix:
    return tuple(tuple(int(row == column) for column in range(size)) for row in range(size))


def _inverse(matrix: ModularMatrix, prime: int) -> ModularMatrix:
    rows, columns = _shape(matrix)
    if rows != columns:
        raise DimensionError("finite-field inversion requires a square matrix")
    work = [list(matrix[row] + _identity(rows)[row]) for row in range(rows)]
    for column in range(columns):
        selected = next((row for row in range(column, rows) if work[row][column] % prime), None)
        if selected is None:
            raise ValidationError("finite-field coordinate matrix is singular")
        work[column], work[selected] = work[selected], work[column]
        inverse_pivot = pow(work[column][column] % prime, -1, prime)
        work[column] = [(value * inverse_pivot) % prime for value in work[column]]
        for row in range(rows):
            if row == column:
                continue
            factor = work[row][column] % prime
            if factor:
                work[row] = [
                    (value - factor * pivot_value) % prime
                    for value, pivot_value in zip(work[row], work[column], strict=True)
                ]
    return tuple(tuple(work[row][rows + column] for column in range(rows)) for row in range(rows))


@dataclass(frozen=True, slots=True)
class FiniteFieldCoordinateTransform:
    """An exact linear coordinate transform over ``F_prime``."""

    source: str
    target: str
    matrix: ModularMatrix
    prime: int
    theorem: str = "CC.M1"
    description: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.prime, bool) or not isinstance(self.prime, int) or not _is_prime(self.prime):
            raise ValidationError("finite-field coordinates require a prime modulus")
        matrix = _matrix(self.matrix, self.prime)
        if not str(self.source).strip() or not str(self.target).strip():
            raise ValidationError("finite-field coordinate names must be nonempty")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "theorem", str(self.theorem))
        object.__setattr__(self, "description", str(self.description))

    @property
    def source_dimension(self) -> int:
        """Return the source dimension."""

        return _shape(self.matrix)[1]

    @property
    def target_dimension(self) -> int:
        """Return the target dimension."""

        return _shape(self.matrix)[0]

    @property
    def square(self) -> bool:
        """Return whether the transform is square."""

        return self.source_dimension == self.target_dimension

    def apply(self, coordinates: Sequence[int]) -> ModularVector:
        """Apply the transform modulo its prime."""

        values = tuple(int(value) % self.prime for value in coordinates)
        if len(values) != self.source_dimension:
            raise DimensionError("finite-field coordinate vector has the wrong length")
        return tuple(
            sum(self.matrix[row][column] * values[column] for column in range(self.source_dimension))
            % self.prime
            for row in range(self.target_dimension)
        )

    def inverse(self) -> "FiniteFieldCoordinateTransform":
        """Return the exact inverse over the prime field."""

        if not self.square:
            raise DimensionError("only square finite-field transforms can be inverted")
        return FiniteFieldCoordinateTransform(
            self.target,
            self.source,
            _inverse(self.matrix, self.prime),
            self.prime,
            f"inverse({self.theorem})",
            f"inverse of {self.description or self.source + ' to ' + self.target}",
        )

    def then(self, following: "FiniteFieldCoordinateTransform") -> "FiniteFieldCoordinateTransform":
        """Compose finite-field coordinate transforms."""

        if not isinstance(following, FiniteFieldCoordinateTransform):
            raise ValidationError("following must be a FiniteFieldCoordinateTransform")
        if self.prime != following.prime:
            raise ValidationError("finite-field coordinate transforms need a common prime")
        if self.target != following.source:
            raise ValidationError("finite-field coordinate-system names do not compose")
        return FiniteFieldCoordinateTransform(
            self.source,
            following.target,
            _multiply(following.matrix, self.matrix, self.prime),
            self.prime,
            f"{self.theorem} then {following.theorem}",
            "composed finite-field coordinate transform",
        )

    def certificate(self) -> Certificate:
        """Return an exact finite-field map certificate.

        Rectangular maps are valid finite-field maps but are not invertibility
        witnesses.  Square maps additionally verify two-sided invertibility.
        """

        if not self.square:
            return Certificate(
                valid=True,
                exactness=Exactness.EXACT,
                assumptions_checked=("prime modulus", "finite field arithmetic"),
                diagnostics={
                    "prime": self.prime,
                    "source_dimension": self.source_dimension,
                    "target_dimension": self.target_dimension,
                    "square": False,
                },
                theorem_id=self.theorem,
                message="rectangular finite-field coordinate map validated",
            )
        try:
            inverse = self.inverse()
        except ValidationError:
            return Certificate(
                valid=False,
                exactness=Exactness.EXACT,
                assumptions_checked=("prime modulus", "finite field arithmetic"),
                diagnostics={"prime": self.prime, "singular": True},
                theorem_id=self.theorem,
                message="finite-field coordinate transform is singular",
            )
        identity = _identity(self.source_dimension)
        valid = (
            _multiply(inverse.matrix, self.matrix, self.prime) == identity
            and _multiply(self.matrix, inverse.matrix, self.prime) == identity
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("prime modulus", "exact modular Gauss-Jordan elimination"),
            diagnostics={"prime": self.prime, "dimension": self.source_dimension},
            theorem_id=self.theorem,
            message="finite-field coordinate transform validated",
        )


def reduce_exact_transform_mod_prime(
    transform: CoordinateTransform,
    prime: int,
) -> FiniteFieldCoordinateTransform:
    """Reduce a rational coordinate transform modulo a prime.

    Every denominator must be invertible modulo the selected prime.
    """

    if isinstance(prime, bool) or not isinstance(prime, int) or not _is_prime(prime):
        raise ValidationError("modular reduction requires a prime")
    rows: list[tuple[int, ...]] = []
    for row in transform.matrix:
        reduced: list[int] = []
        for value in row:
            fraction = Fraction(value)
            denominator = fraction.denominator % prime
            if denominator == 0:
                raise ValidationError("coordinate denominator is not invertible modulo the prime")
            reduced.append((fraction.numerator % prime) * pow(denominator, -1, prime) % prime)
        rows.append(tuple(reduced))
    return FiniteFieldCoordinateTransform(
        transform.source,
        transform.target,
        tuple(rows),
        prime,
        "CC.M2",
        "prime-field reduction of an exact rational coordinate transform",
    )


__all__ = [
    "FiniteFieldCoordinateTransform",
    "ModularMatrix",
    "ModularVector",
    "reduce_exact_transform_mod_prime",
]
