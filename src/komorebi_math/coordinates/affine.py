"""Exact affine, homogeneous, and barycentric coordinate changes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .exact import (
    Matrix,
    Scalar,
    Vector,
    as_matrix,
    as_vector,
    identity_matrix,
    inverse_matrix,
    matrix_from_columns,
    matrix_multiply,
    matrix_shape,
    matrix_vector_multiply,
    solve_linear_system,
)


@dataclass(frozen=True, slots=True)
class AffineCoordinateTransform:
    r"""An exact affine map ``y=A x+b`` between equal-dimensional spaces."""

    source: str
    target: str
    matrix: Matrix
    offset: Vector
    theorem: str = "CC.F1"
    description: str = ""

    def __post_init__(self) -> None:
        matrix = as_matrix(self.matrix)
        rows, columns = matrix_shape(matrix)
        if rows == 0 or rows != columns:
            raise DimensionError("affine coordinate transforms require a nonempty square matrix")
        offset = as_vector(self.offset)
        if len(offset) != rows:
            raise DimensionError("affine offset length must match the matrix dimension")
        if not str(self.source).strip() or not str(self.target).strip():
            raise ValidationError("affine coordinate-system names must be nonempty")
        if not self.invertible:
            raise ValidationError("affine coordinate matrices must be invertible")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "theorem", str(self.theorem))
        object.__setattr__(self, "description", str(self.description))

    @property
    def dimension(self) -> int:
        """Return the affine-space dimension."""

        return len(self.offset)

    @property
    def invertible(self) -> bool:
        """Return whether the linear part is nonsingular."""

        try:
            inverse_matrix(self.matrix)
        except ValidationError:
            return False
        return True

    def apply(self, coordinates: Sequence[Scalar]) -> Vector:
        """Apply the affine coordinate change."""

        values = as_vector(coordinates)
        if len(values) != self.dimension:
            raise DimensionError("coordinate length does not match the affine dimension")
        linear = matrix_vector_multiply(self.matrix, values)
        return tuple(value + shift for value, shift in zip(linear, self.offset, strict=True))

    def inverse(self) -> "AffineCoordinateTransform":
        """Return the exact inverse affine coordinate change."""

        inverse_linear = inverse_matrix(self.matrix)
        inverse_offset = tuple(-value for value in matrix_vector_multiply(inverse_linear, self.offset))
        return AffineCoordinateTransform(
            self.target,
            self.source,
            inverse_linear,
            inverse_offset,
            f"inverse({self.theorem})",
            f"inverse of {self.description or self.source + ' to ' + self.target}",
        )

    def then(self, following: "AffineCoordinateTransform") -> "AffineCoordinateTransform":
        """Compose this affine map with a following map."""

        if not isinstance(following, AffineCoordinateTransform):
            raise ValidationError("following must be an AffineCoordinateTransform")
        if self.target != following.source:
            raise ValidationError("affine coordinate-system names do not compose")
        if self.dimension != following.dimension:
            raise DimensionError("affine dimensions do not compose")
        matrix = matrix_multiply(following.matrix, self.matrix)
        mapped_offset = matrix_vector_multiply(following.matrix, self.offset)
        offset = tuple(
            value + shift
            for value, shift in zip(mapped_offset, following.offset, strict=True)
        )
        return AffineCoordinateTransform(
            self.source,
            following.target,
            matrix,
            offset,
            f"{self.theorem} then {following.theorem}",
            "composed affine coordinate change",
        )

    def homogeneous_matrix(self) -> Matrix:
        """Return the homogeneous matrix of the affine coordinate change."""

        n = self.dimension
        return tuple(
            tuple(self.matrix[row][column] for column in range(n)) + (self.offset[row],)
            for row in range(n)
        ) + (tuple(Fraction(0) for _ in range(n)) + (Fraction(1),),)

    def certificate(self) -> Certificate:
        """Return an exact finite affine-invertibility certificate."""

        inverse = self.inverse()
        probes = identity_matrix(self.dimension)
        valid = all(
            inverse.apply(self.apply(tuple(probes[row][column] for row in range(self.dimension))))
            == tuple(probes[row][column] for row in range(self.dimension))
            for column in range(self.dimension)
        ) and inverse.apply(self.apply(tuple(Fraction(0) for _ in range(self.dimension)))) == tuple(
            Fraction(0) for _ in range(self.dimension)
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("exact rational linear part", "nonsingular affine chart"),
            diagnostics={"dimension": self.dimension, "homogeneous_matrix": self.homogeneous_matrix()},
            theorem_id=self.theorem,
            message="exact affine coordinate transform validated",
        )


@dataclass(frozen=True, slots=True)
class BarycentricCoordinateSystem:
    """Barycentric coordinates of an exact nondegenerate simplex."""

    name: str
    vertices: tuple[Vector, ...]

    def __post_init__(self) -> None:
        vertices = tuple(as_vector(vertex) for vertex in self.vertices)
        if len(vertices) < 2:
            raise DimensionError("a simplex needs at least two vertices")
        ambient = len(vertices[0])
        if ambient + 1 != len(vertices):
            raise DimensionError("a full simplex in dimension d needs d+1 vertices")
        if any(len(vertex) != ambient for vertex in vertices):
            raise DimensionError("simplex vertices must share an ambient dimension")
        augmented = tuple(vertex + (Fraction(1),) for vertex in vertices)
        matrix = matrix_from_columns(augmented)
        try:
            inverse_matrix(matrix)
        except ValidationError as exc:
            raise ValidationError("simplex vertices must be affinely independent") from exc
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "name", str(self.name).strip() or "barycentric simplex")

    @property
    def dimension(self) -> int:
        """Return the ambient simplex dimension."""

        return len(self.vertices) - 1

    @property
    def augmented_vertex_matrix(self) -> Matrix:
        """Return columns ``(vertex,1)`` used for barycentric analysis."""

        return matrix_from_columns(tuple(vertex + (Fraction(1),) for vertex in self.vertices))

    def to_cartesian(self, weights: Sequence[Scalar], *, require_affine: bool = True) -> Vector:
        """Map barycentric weights to one Cartesian point."""

        values = as_vector(weights)
        if len(values) != len(self.vertices):
            raise DimensionError("barycentric weight count must match the simplex vertices")
        if require_affine and sum(values, Fraction(0)) != 1:
            raise ValidationError("barycentric weights must sum to one")
        return tuple(
            sum(
                (values[index] * self.vertices[index][coordinate] for index in range(len(values))),
                Fraction(0),
            )
            for coordinate in range(self.dimension)
        )

    def from_cartesian(self, point: Sequence[Scalar]) -> Vector:
        """Return exact barycentric coordinates of a Cartesian point."""

        value = as_vector(point)
        if len(value) != self.dimension:
            raise DimensionError("Cartesian point dimension does not match the simplex")
        solution = solve_linear_system(self.augmented_vertex_matrix, value + (Fraction(1),))
        if not solution.consistent or solution.solution is None:
            raise ValidationError("point does not admit barycentric coordinates")
        return solution.solution

    def certificate(self) -> Certificate:
        """Return an exact vertex round-trip certificate."""

        recovered = tuple(self.from_cartesian(vertex) for vertex in self.vertices)
        expected = identity_matrix(len(self.vertices))
        expected_columns = tuple(
            tuple(expected[row][column] for row in range(len(self.vertices)))
            for column in range(len(self.vertices))
        )
        return Certificate(
            valid=recovered == expected_columns,
            exactness=Exactness.EXACT,
            assumptions_checked=("affine independence", "exact augmented solve"),
            diagnostics={"dimension": self.dimension, "vertex_count": len(self.vertices)},
            theorem_id="CC.F3",
            message="barycentric coordinate system validated",
        )


def affine_basis_transform(
    origin: Sequence[Scalar],
    basis_vectors: Sequence[Sequence[Scalar]],
    *,
    local_name: str = "local affine coordinates",
    ambient_name: str = "ambient Cartesian coordinates",
) -> AffineCoordinateTransform:
    """Construct an affine chart from an origin and a square vector basis."""

    offset = as_vector(origin)
    vectors = tuple(as_vector(vector) for vector in basis_vectors)
    if len(vectors) != len(offset):
        raise DimensionError("an affine basis needs one vector per ambient dimension")
    if any(len(vector) != len(offset) for vector in vectors):
        raise DimensionError("affine basis vectors must match the origin dimension")
    return AffineCoordinateTransform(
        local_name,
        ambient_name,
        matrix_from_columns(vectors),
        offset,
        "CC.F2",
        "affine chart from an origin and basis vectors",
    )


def translation_transform(
    offset: Sequence[Scalar],
    *,
    source: str = "original affine coordinates",
    target: str = "translated affine coordinates",
) -> AffineCoordinateTransform:
    """Return an exact translation coordinate change."""

    shift = as_vector(offset)
    if not shift:
        raise DimensionError("translation offsets must be nonempty")
    return AffineCoordinateTransform(source, target, identity_matrix(len(shift)), shift, "CC.F4", "translation")


__all__ = [
    "AffineCoordinateTransform",
    "BarycentricCoordinateSystem",
    "affine_basis_transform",
    "translation_transform",
]
