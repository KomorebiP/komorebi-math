"""Arbitrary finite bases and exact coordinate atlases."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
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
    identity_matrix,
    inverse_matrix,
    kronecker,
    matrix_from_columns,
    matrix_multiply,
    matrix_rank,
    matrix_shape,
    matrix_vector_multiply,
    solve_linear_system,
    transpose,
    zero_matrix,
)
from .transform import CoordinateTransform


@dataclass(frozen=True, slots=True)
class CoordinateSystem:
    """Metadata for one finite exact coordinate system."""

    name: str
    dimension: int
    kind: str = "exact"
    description: str = ""

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValidationError("coordinate-system names must be nonempty")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension < 1:
            raise DimensionError("coordinate-system dimensions must be positive integers")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", str(self.kind).strip() or "exact")
        object.__setattr__(self, "description", str(self.description))


@dataclass(frozen=True, slots=True)
class FiniteBasis:
    """A linearly independent finite basis inside a rational ambient space.

    Basis vectors are stored as columns.  A rectangular basis is permitted,
    so the basis may describe a proper subspace of the ambient vector space.
    """

    name: str
    vectors: tuple[Vector, ...]
    labels: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValidationError("basis names must be nonempty")
        vectors = tuple(as_vector(vector) for vector in self.vectors)
        if not vectors:
            raise DimensionError("a finite basis must contain at least one vector")
        ambient = len(vectors[0])
        if ambient == 0:
            raise DimensionError("basis vectors must be nonempty")
        if any(len(vector) != ambient for vector in vectors):
            raise DimensionError("basis vectors must share an ambient dimension")
        matrix = matrix_from_columns(vectors)
        if matrix_rank(matrix) != len(vectors):
            raise ValidationError("basis vectors must be linearly independent")
        labels = tuple(str(label) for label in self.labels)
        if labels and len(labels) != len(vectors):
            raise DimensionError("basis labels must match the basis dimension")
        if not labels:
            labels = tuple(f"b{index}" for index in range(len(vectors)))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "description", str(self.description))

    @property
    def dimension(self) -> int:
        """Return the basis dimension."""

        return len(self.vectors)

    @property
    def ambient_dimension(self) -> int:
        """Return the ambient vector-space dimension."""

        return len(self.vectors[0])

    @property
    def matrix(self) -> Matrix:
        """Return the synthesis matrix whose columns are basis vectors."""

        return matrix_from_columns(self.vectors)

    @property
    def square(self) -> bool:
        """Return whether the basis spans the entire ambient space."""

        return self.dimension == self.ambient_dimension

    def synthesize(self, coordinates: Sequence[Scalar]) -> Vector:
        """Reconstruct an ambient vector from basis coordinates."""

        values = as_vector(coordinates)
        if len(values) != self.dimension:
            raise DimensionError("coordinate length does not match the basis dimension")
        return matrix_vector_multiply(self.matrix, values)

    def analyze(self, vector: Sequence[Scalar]) -> Vector:
        """Return the unique coordinates of an ambient vector in this basis."""

        value = as_vector(vector)
        if len(value) != self.ambient_dimension:
            raise DimensionError("ambient vector length does not match the basis")
        solution = solve_linear_system(self.matrix, value)
        if not solution.consistent or solution.solution is None:
            raise ValidationError("ambient vector does not lie in the basis span")
        if solution.rank != self.dimension:
            raise ValidationError("basis coordinates are not unique")
        return solution.solution

    def contains(self, vector: Sequence[Scalar]) -> bool:
        """Return whether a vector lies in the basis span."""

        try:
            self.analyze(vector)
        except (DimensionError, ValidationError):
            return False
        return True

    def connection_to(
        self,
        target: "FiniteBasis",
        *,
        theorem: str = "CC.B1",
        budget: CoordinateBudget = DEFAULT_COORDINATE_BUDGET,
    ) -> CoordinateTransform:
        """Return target coordinates of every source basis vector.

        The two bases must have the same dimension and span the same subspace
        of a common ambient space.
        """

        if not isinstance(target, FiniteBasis):
            raise ValidationError("target must be a FiniteBasis")
        if self.ambient_dimension != target.ambient_dimension:
            raise DimensionError("basis ambient dimensions do not agree")
        if self.dimension != target.dimension:
            raise DimensionError("basis dimensions do not agree")
        budget.require_matrix(
            self.dimension,
            self.dimension,
            operation="finite basis connection",
        )
        columns = tuple(target.analyze(vector) for vector in self.vectors)
        transform = CoordinateTransform(
            self.name,
            target.name,
            matrix_from_columns(columns),
            theorem,
            "exact connection between arbitrary finite bases",
        )
        if any(not self.contains(vector) for vector in target.vectors):
            raise ValidationError("source and target bases do not span the same subspace")
        return transform

    def dual_rows(self) -> Matrix:
        """Return dual coordinate functionals for a square basis."""

        if not self.square:
            raise DimensionError("dual rows require a square ambient basis")
        return inverse_matrix(self.matrix)

    def gram_matrix(self) -> Matrix:
        """Return the standard exact Gram matrix."""

        return matrix_multiply(transpose(self.matrix), self.matrix)

    def orthogonal_projector(self) -> Matrix:
        """Return the exact standard-inner-product projector onto the span."""

        basis = self.matrix
        gram_inverse = inverse_matrix(self.gram_matrix())
        return matrix_multiply(matrix_multiply(basis, gram_inverse), transpose(basis))

    def certificate(self) -> Certificate:
        """Return a certificate for independence and dimensions."""

        rank = matrix_rank(self.matrix)
        return Certificate(
            valid=rank == self.dimension,
            exactness=Exactness.EXACT,
            assumptions_checked=("finite rational vectors", "exact row reduction"),
            diagnostics={
                "dimension": self.dimension,
                "ambient_dimension": self.ambient_dimension,
                "rank": rank,
                "square": self.square,
            },
            theorem_id="CC.B0",
            message="finite basis validated",
        )


@dataclass(frozen=True, slots=True)
class AtlasPath:
    """One composed path through a coordinate atlas."""

    systems: tuple[str, ...]
    transform: CoordinateTransform

    @property
    def length(self) -> int:
        """Return the number of edges in the path."""

        return max(len(self.systems) - 1, 0)


@dataclass(frozen=True, slots=True)
class CommutingDiagramCheck:
    """Exact comparison of two paths with common endpoints."""

    left: AtlasPath
    right: AtlasPath
    valid: bool
    difference: Matrix

    def certificate(self) -> Certificate:
        """Return an exact commuting-diagram certificate."""

        return Certificate(
            valid=self.valid,
            exactness=Exactness.EXACT,
            assumptions_checked=("common source", "common target", "exact matrix products"),
            diagnostics={
                "left_path": self.left.systems,
                "right_path": self.right.systems,
                "difference": self.difference,
            },
            theorem_id="CC.A2",
            message="coordinate atlas paths commute" if self.valid else "coordinate atlas paths differ",
        )


@dataclass(slots=True)
class CoordinateAtlas:
    """A graph of exact coordinate systems and composable transforms."""

    name: str = "coordinate atlas"
    _systems: dict[str, CoordinateSystem] = field(default_factory=dict, init=False, repr=False)
    _edges: dict[tuple[str, str], CoordinateTransform] = field(default_factory=dict, init=False, repr=False)

    def register_system(self, system: CoordinateSystem) -> None:
        """Register one coordinate system, rejecting incompatible duplicates."""

        if not isinstance(system, CoordinateSystem):
            raise ValidationError("system must be a CoordinateSystem")
        existing = self._systems.get(system.name)
        if existing is not None and existing != system:
            raise ValidationError("coordinate system is already registered with different metadata")
        self._systems[system.name] = system

    def register_basis(self, basis: FiniteBasis) -> None:
        """Register one finite basis as a coordinate system."""

        if not isinstance(basis, FiniteBasis):
            raise ValidationError("basis must be a FiniteBasis")
        self.register_system(
            CoordinateSystem(
                basis.name,
                basis.dimension,
                kind="finite_basis",
                description=basis.description,
            )
        )

    def register_transform(
        self,
        transform: CoordinateTransform,
        *,
        include_inverse: bool = False,
        replace: bool = False,
    ) -> None:
        """Register one directed transform and optionally its exact inverse."""

        if not isinstance(transform, CoordinateTransform):
            raise ValidationError("transform must be a CoordinateTransform")
        source = self._systems.get(transform.source)
        target = self._systems.get(transform.target)
        if source is None or target is None:
            raise ValidationError("both coordinate systems must be registered first")
        if source.dimension != transform.source_dimension:
            raise DimensionError("source system dimension does not match the transform")
        if target.dimension != transform.target_dimension:
            raise DimensionError("target system dimension does not match the transform")
        key = (transform.source, transform.target)
        if key in self._edges and not replace and self._edges[key] != transform:
            raise ValidationError("a different transform is already registered for this edge")
        self._edges[key] = transform
        if include_inverse:
            inverse = transform.inverse(theorem=f"inverse({transform.theorem})")
            inverse_key = (inverse.source, inverse.target)
            if inverse_key in self._edges and not replace and self._edges[inverse_key] != inverse:
                raise ValidationError("a different inverse transform is already registered")
            self._edges[inverse_key] = inverse

    @property
    def systems(self) -> tuple[CoordinateSystem, ...]:
        """Return registered systems in deterministic name order."""

        return tuple(self._systems[name] for name in sorted(self._systems))

    @property
    def transforms(self) -> tuple[CoordinateTransform, ...]:
        """Return registered transforms in deterministic endpoint order."""

        return tuple(self._edges[key] for key in sorted(self._edges))

    def direct_transform(self, source: str, target: str) -> CoordinateTransform | None:
        """Return a directly registered transform when present."""

        return self._edges.get((source, target))

    def resolve(self, source: str, target: str) -> AtlasPath:
        """Return a shortest composable path between two systems."""

        if source not in self._systems or target not in self._systems:
            raise ValidationError("source and target must be registered systems")
        if source == target:
            dimension = self._systems[source].dimension
            return AtlasPath(
                (source,),
                CoordinateTransform(
                    source,
                    target,
                    identity_matrix(dimension),
                    "CC.A0",
                    "identity atlas path",
                ),
            )
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, (source,))])
        visited = {source}
        found: tuple[str, ...] | None = None
        while queue:
            current, path = queue.popleft()
            neighbors = sorted(edge_target for edge_source, edge_target in self._edges if edge_source == current)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                candidate = path + (neighbor,)
                if neighbor == target:
                    found = candidate
                    queue.clear()
                    break
                visited.add(neighbor)
                queue.append((neighbor, candidate))
        if found is None:
            raise ValidationError("no coordinate path connects the requested systems")
        transform = self._edges[(found[0], found[1])]
        for left, right in zip(found[1:-1], found[2:], strict=True):
            transform = transform.then(self._edges[(left, right)], theorem="CC.A1")
        return AtlasPath(found, transform)

    def compose_path(self, systems: Sequence[str]) -> AtlasPath:
        """Compose an explicitly declared atlas path."""

        names = tuple(str(name) for name in systems)
        if not names:
            raise ValidationError("an atlas path must contain at least one system")
        if len(names) == 1:
            return self.resolve(names[0], names[0])
        edges: list[CoordinateTransform] = []
        for source, target in zip(names[:-1], names[1:], strict=True):
            transform = self._edges.get((source, target))
            if transform is None:
                raise ValidationError(f"atlas edge {source!r} to {target!r} is not registered")
            edges.append(transform)
        composed = edges[0]
        for transform in edges[1:]:
            composed = composed.then(transform, theorem="CC.A1")
        return AtlasPath(names, composed)

    def compare_paths(self, left: Sequence[str], right: Sequence[str]) -> CommutingDiagramCheck:
        """Check whether two explicit atlas paths define the same map."""

        left_path = self.compose_path(left)
        right_path = self.compose_path(right)
        if left_path.transform.source != right_path.transform.source:
            raise ValidationError("paths must have a common source")
        if left_path.transform.target != right_path.transform.target:
            raise ValidationError("paths must have a common target")
        if matrix_shape(left_path.transform.matrix) != matrix_shape(right_path.transform.matrix):
            raise DimensionError("path transform shapes do not agree")
        difference = tuple(
            tuple(a - b for a, b in zip(left_row, right_row, strict=True))
            for left_row, right_row in zip(
                left_path.transform.matrix,
                right_path.transform.matrix,
                strict=True,
            )
        )
        zero = zero_matrix(*matrix_shape(difference))
        return CommutingDiagramCheck(left_path, right_path, difference == zero, difference)


def standard_basis(dimension: int, *, name: str = "standard") -> FiniteBasis:
    """Return the standard basis of a rational vector space."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise DimensionError("dimension must be a positive integer")
    matrix = identity_matrix(dimension)
    vectors = tuple(tuple(matrix[row][column] for row in range(dimension)) for column in range(dimension))
    return FiniteBasis(name, vectors, tuple(f"e{index}" for index in range(dimension)))


def basis_from_matrix(
    name: str,
    matrix: Iterable[Iterable[Scalar]],
    *,
    labels: Sequence[str] = (),
    description: str = "",
) -> FiniteBasis:
    """Construct a basis from a matrix whose columns are basis vectors."""

    data = as_matrix(matrix)
    rows, columns = matrix_shape(data)
    if rows == 0 or columns == 0:
        raise DimensionError("basis matrices must be nonempty")
    vectors = tuple(tuple(data[row][column] for row in range(rows)) for column in range(columns))
    return FiniteBasis(name, vectors, tuple(labels), description)


def direct_sum_basis(left: FiniteBasis, right: FiniteBasis, *, name: str | None = None) -> FiniteBasis:
    """Return the external direct sum of two finite bases."""

    if not isinstance(left, FiniteBasis) or not isinstance(right, FiniteBasis):
        raise ValidationError("direct sums require FiniteBasis objects")
    zero_left = tuple(Fraction(0) for _ in range(left.ambient_dimension))
    zero_right = tuple(Fraction(0) for _ in range(right.ambient_dimension))
    vectors = tuple(vector + zero_right for vector in left.vectors) + tuple(
        zero_left + vector for vector in right.vectors
    )
    labels = tuple(f"L:{label}" for label in left.labels) + tuple(f"R:{label}" for label in right.labels)
    return FiniteBasis(name or f"{left.name} direct sum {right.name}", vectors, labels)


def tensor_product_basis(left: FiniteBasis, right: FiniteBasis, *, name: str | None = None) -> FiniteBasis:
    """Return the tensor-product basis in lexicographic factor order."""

    if not isinstance(left, FiniteBasis) or not isinstance(right, FiniteBasis):
        raise ValidationError("tensor products require FiniteBasis objects")
    vectors: list[Vector] = []
    labels: list[str] = []
    for left_label, left_vector in zip(left.labels, left.vectors, strict=True):
        for right_label, right_vector in zip(right.labels, right.vectors, strict=True):
            column = kronecker(
                tuple((value,) for value in left_vector),
                tuple((value,) for value in right_vector),
            )
            vectors.append(tuple(row[0] for row in column))
            labels.append(f"{left_label}⊗{right_label}")
    return FiniteBasis(name or f"{left.name} tensor {right.name}", tuple(vectors), tuple(labels))


def permutation_transform(
    permutation: Sequence[int],
    *,
    source: str = "original order",
    target: str = "permuted order",
) -> CoordinateTransform:
    """Return an exact coordinate permutation.

    The target entry at position ``i`` is the source entry at
    ``permutation[i]``.
    """

    order = tuple(permutation)
    size = len(order)
    if sorted(order) != list(range(size)):
        raise ValidationError("permutation must contain every index exactly once")
    matrix = tuple(
        tuple(Fraction(int(column == order[row])) for column in range(size))
        for row in range(size)
    )
    return CoordinateTransform(source, target, matrix, "CC.B2", "exact coordinate permutation")


def scaling_transform(
    scales: Sequence[Scalar],
    *,
    source: str = "unscaled coordinates",
    target: str = "scaled coordinates",
) -> CoordinateTransform:
    """Return an invertible diagonal coordinate scaling."""

    values = as_vector(scales)
    if not values:
        raise DimensionError("coordinate scaling must be nonempty")
    if any(value == 0 for value in values):
        raise ValidationError("coordinate scaling factors must be nonzero")
    matrix = tuple(
        tuple(values[row] if row == column else Fraction(0) for column in range(len(values)))
        for row in range(len(values))
    )
    return CoordinateTransform(source, target, matrix, "CC.B3", "exact diagonal coordinate scaling")


__all__ = [
    "AtlasPath",
    "CommutingDiagramCheck",
    "CoordinateAtlas",
    "CoordinateSystem",
    "FiniteBasis",
    "basis_from_matrix",
    "direct_sum_basis",
    "permutation_transform",
    "scaling_transform",
    "standard_basis",
    "tensor_product_basis",
]
