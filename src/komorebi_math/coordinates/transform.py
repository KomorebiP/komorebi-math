"""Composable exact coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .exact import (
    Matrix,
    Scalar,
    Vector,
    as_matrix,
    determinant,
    identity_matrix,
    inverse_matrix,
    matrix_multiply,
    matrix_shape,
    matrix_vector_multiply,
)


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    """An exact matrix that maps source coordinates to target coordinates."""

    source: str
    target: str
    matrix: Matrix
    theorem: str
    description: str = ""

    def __post_init__(self) -> None:
        matrix = as_matrix(self.matrix)
        rows, columns = matrix_shape(matrix)
        if rows == 0 or columns == 0:
            raise DimensionError("coordinate transforms must be nonempty")
        if not self.source.strip() or not self.target.strip():
            raise ValidationError("coordinate-system names must be nonempty")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "theorem", str(self.theorem))
        object.__setattr__(self, "description", str(self.description))

    @property
    def source_dimension(self) -> int:
        """Return the source coordinate dimension."""

        return matrix_shape(self.matrix)[1]

    @property
    def target_dimension(self) -> int:
        """Return the target coordinate dimension."""

        return matrix_shape(self.matrix)[0]

    @property
    def square(self) -> bool:
        """Return whether the transform is square."""

        return self.source_dimension == self.target_dimension

    @property
    def determinant(self) -> Fraction:
        """Return the determinant of a square transform."""

        if not self.square:
            raise DimensionError("only square transforms have determinants")
        return determinant(self.matrix)

    def apply(self, coordinates: Sequence[Scalar]) -> Vector:
        """Map one coordinate vector to the target system."""

        return matrix_vector_multiply(self.matrix, coordinates)

    def inverse(self, *, theorem: str | None = None) -> "CoordinateTransform":
        """Return the exact inverse of a nonsingular square transform."""

        if not self.square:
            raise DimensionError("only square coordinate transforms can be inverted")
        return CoordinateTransform(
            self.target,
            self.source,
            inverse_matrix(self.matrix),
            theorem or self.theorem,
            f"inverse of {self.description or self.source + ' to ' + self.target}",
        )

    def then(
        self,
        following: "CoordinateTransform",
        *,
        theorem: str | None = None,
    ) -> "CoordinateTransform":
        """Compose this transform with a following target transform."""

        if not isinstance(following, CoordinateTransform):
            raise ValidationError("following must be a CoordinateTransform")
        if self.target != following.source:
            raise ValidationError("coordinate system names do not compose")
        if self.target_dimension != following.source_dimension:
            raise DimensionError("coordinate dimensions do not compose")
        return CoordinateTransform(
            self.source,
            following.target,
            matrix_multiply(following.matrix, self.matrix),
            theorem or f"{self.theorem} then {following.theorem}",
            (
                f"{self.description or self.source + ' to ' + self.target}, then "
                f"{following.description or following.source + ' to ' + following.target}"
            ),
        )

    def roundtrip_valid(self) -> bool:
        """Return whether exact inversion gives the identity."""

        if not self.square or self.determinant == 0:
            return False
        inverse = inverse_matrix(self.matrix)
        return (
            matrix_multiply(inverse, self.matrix) == identity_matrix(self.source_dimension)
            and matrix_multiply(self.matrix, inverse) == identity_matrix(self.target_dimension)
        )

    def certificate(self) -> Certificate:
        """Return an exact finite map certificate with invertibility diagnostics.

        Certificate validity means the declared finite map is structurally
        well formed.  The ``invertible`` diagnostic records whether it is a
        square isomorphism.
        """

        determinant_value = self.determinant if self.square else None
        valid = bool(self.matrix) and (
            not self.square or determinant_value is not None
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite rational matrix", "declared coordinate dimensions"),
            diagnostics={
                "source_dimension": self.source_dimension,
                "target_dimension": self.target_dimension,
                "square": self.square,
                "determinant": determinant_value,
                "invertible": self.square and determinant_value != 0,
            },
            theorem_id=self.theorem or None,
            message="exact coordinate transform validated",
        )


ExactCoordinateTransform = CoordinateTransform


__all__ = ["CoordinateTransform", "ExactCoordinateTransform"]
