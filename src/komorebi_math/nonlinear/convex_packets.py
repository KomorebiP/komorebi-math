"""Experimental convex/Finsler packets for finite-dimensional linear dynamics.

Block 11 widens the research-tier geometric packet surface beyond quadratic
metrics.  The first implemented class is a centrally symmetric parallelotope
norm

    ||x||_{F,inf} = ||F x||_inf,

with invertible real coordinate map ``F``.  Its John ellipsoid is explicit:

    E_F = {x : ||F x||_2 <= 1},

so its quadratic metric is ``Q = F.T @ F`` and

    E_F subset K_F subset sqrt(n) E_F.

The module is intentionally not re-exported from ``komorebi_math.nonlinear``.
It is an experimental bridge between convex-body packets and the metric
packets introduced in Blocks 9--10.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Iterable

import numpy as np

from ..exceptions import DimensionError, ValidationError


def _real_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) and np.any(np.abs(np.imag(array)) > 0.0):
        raise ValidationError(f"{name} must be real")
    array = np.asarray(np.real(array), dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise DimensionError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values")
    return array


def _real_vector(value: Any, *, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) and np.any(np.abs(np.imag(array)) > 0.0):
        raise ValidationError(f"{name} must be real")
    array = np.asarray(np.real(array), dtype=float)
    if array.ndim != 1 or array.shape[0] != dimension:
        raise DimensionError(f"{name} must have shape ({dimension},)")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values")
    return array


def _invertible_square(value: Any, *, name: str, tolerance: float) -> np.ndarray:
    matrix = _real_matrix(value, name=name)
    if matrix.shape[0] != matrix.shape[1]:
        raise DimensionError(f"{name} must be square")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if float(singular[-1]) <= tolerance * max(1.0, float(singular[0])):
        raise ValidationError(f"{name} must be invertible and numerically resolved")
    return matrix


def _positive_metric(value: Any, *, dimension: int, tolerance: float) -> np.ndarray:
    metric = _real_matrix(value, name="metric")
    if metric.shape != (dimension, dimension):
        raise DimensionError("metric has the wrong dimension")
    scale = max(1.0, float(np.linalg.norm(metric, ord=2)))
    symmetry_error = float(np.linalg.norm(metric - metric.T, ord="fro"))
    if symmetry_error > tolerance * scale:
        raise ValidationError("metric must be symmetric within tolerance")
    metric = 0.5 * (metric + metric.T)
    eigenvalues = np.linalg.eigvalsh(metric)
    if float(eigenvalues[0]) <= tolerance * scale:
        raise ValidationError("metric must be positive definite and numerically resolved")
    return metric


def quadratic_operator_norm(matrix: Any, metric: Any, *, tolerance: float = 1e-12) -> float:
    """Return the operator norm induced by a positive-definite quadratic metric.

    Unlike ``metric_distortion_packet``, this routine permits singular maps and
    measures one-sided growth rather than invertible anisotropy.
    """

    operator = _real_matrix(matrix, name="operator")
    if operator.shape[0] != operator.shape[1]:
        raise DimensionError("operator must be square")
    q = _positive_metric(metric, dimension=operator.shape[0], tolerance=tolerance)
    factor = np.linalg.cholesky(q).T
    representative = factor @ operator @ np.linalg.inv(factor)
    return float(np.linalg.svd(representative, compute_uv=False)[0])


@dataclass(frozen=True, slots=True)
class ConvexQuadraticCompression:
    """One operator viewed in a convex packet and its John ellipsoid."""

    dimension: int
    convex_operator_norm: float
    john_quadratic_operator_norm: float
    john_theorem_upper_bound: float

    @property
    def compression_ratio(self) -> float:
        if self.convex_operator_norm == 0.0:
            return 1.0 if self.john_quadratic_operator_norm == 0.0 else float("inf")
        return self.john_quadratic_operator_norm / self.convex_operator_norm


@dataclass(frozen=True, slots=True)
class ParallelotopeNormPacket:
    """A norm packet ``||x|| = ||F x||_inf`` with explicit John ellipsoid."""

    coordinate_map: np.ndarray
    dimension: int
    coordinate_condition_number: float

    def vector_norm(self, vector: Any) -> float:
        x = _real_vector(vector, dimension=self.dimension, name="vector")
        return float(np.max(np.abs(self.coordinate_map @ x)))

    def operator_norm(self, matrix: Any) -> float:
        operator = _real_matrix(matrix, name="operator")
        if operator.shape != (self.dimension, self.dimension):
            raise DimensionError("operator has the wrong dimension")
        representative = self.coordinate_map @ operator @ np.linalg.inv(self.coordinate_map)
        return float(np.max(np.sum(np.abs(representative), axis=1)))

    @property
    def john_metric(self) -> np.ndarray:
        metric = np.array(self.coordinate_map.T @ self.coordinate_map, copy=True)
        metric.setflags(write=False)
        return metric

    @property
    def john_factor(self) -> float:
        return sqrt(float(self.dimension))

    def john_operator_norm(self, matrix: Any, *, tolerance: float = 1e-12) -> float:
        return quadratic_operator_norm(matrix, self.john_metric, tolerance=tolerance)

    def compress_operator(
        self,
        matrix: Any,
        *,
        tolerance: float = 1e-12,
    ) -> ConvexQuadraticCompression:
        convex = self.operator_norm(matrix)
        quadratic = self.john_operator_norm(matrix, tolerance=tolerance)
        upper = self.john_factor * convex
        return ConvexQuadraticCompression(
            dimension=self.dimension,
            convex_operator_norm=convex,
            john_quadratic_operator_norm=quadratic,
            john_theorem_upper_bound=upper,
        )

    def transport_coordinates(
        self,
        gauge: Any,
        *,
        tolerance: float = 1e-12,
    ) -> "ParallelotopeNormPacket":
        """Transport the norm packet under ``x' = G x``.

        The new coordinate map is ``F G^{-1}``, so represented vector and
        operator norms remain unchanged when the vector/operator is transformed
        covariantly.
        """

        g = _invertible_square(gauge, name="gauge", tolerance=tolerance)
        if g.shape != (self.dimension, self.dimension):
            raise DimensionError("gauge has the wrong dimension")
        return parallelotope_norm_packet(self.coordinate_map @ np.linalg.inv(g), tolerance=tolerance)


def parallelotope_norm_packet(
    coordinate_map: Any,
    *,
    tolerance: float = 1e-12,
) -> ParallelotopeNormPacket:
    """Create a real parallelotope norm packet from an invertible map ``F``."""

    f = _invertible_square(coordinate_map, name="coordinate map", tolerance=tolerance)
    frozen = np.array(f, copy=True)
    frozen.setflags(write=False)
    condition = float(np.linalg.cond(frozen, p=2))
    if not isfinite(condition):
        raise ValidationError("coordinate map condition number must be finite")
    return ParallelotopeNormPacket(
        coordinate_map=frozen,
        dimension=frozen.shape[0],
        coordinate_condition_number=condition,
    )


def joint_operator_bound(
    packet: ParallelotopeNormPacket,
    matrices: Iterable[Any],
) -> float:
    """Return ``max_i ||A_i||`` in one parallelotope norm packet."""

    values = [packet.operator_norm(matrix) for matrix in matrices]
    if not values:
        raise ValidationError("at least one operator is required")
    return max(values)


__all__ = [
    "BranchSupportQuery",
    "ConvexQuadraticCompression",
    "ParallelotopeNormPacket",
    "PolyhedralSupportPacket",
    "SupportQuery",
    "joint_operator_bound",
    "parallelotope_norm_packet",
    "polyhedral_support_packet",
    "quadratic_operator_norm",
]


@dataclass(frozen=True, slots=True)
class SupportQuery:
    """Support value together with active atoms and retained provenance."""

    value: float
    active_indices: tuple[int, ...]
    active_provenance: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class BranchSupportQuery:
    """One Bellman/support query before geometric branch information is merged."""

    value: float
    branch_values: tuple[float, ...]
    active_branches: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PolyhedralSupportPacket:
    """Finite support-function packet ``h_C(x)=max_v <v,x>``.

    Rows of ``atoms`` generate the compact polytope ``C = conv(atoms)``.  The
    optional ``provenance`` words do not change the convex geometry; they retain
    which branch history produced an atom.  This separates *value semantics*
    (the support function) from *policy/provenance semantics* (argmax history).

    Bellman transport by primal matrices ``A_i`` acts contravariantly on atoms:

        C -> conv_i A_i^T C,
        h(x) -> max_i h(A_i x).

    Provenance words are stored in chronological primal execution order.
    """

    atoms: np.ndarray
    ambient_dimension: int
    provenance: tuple[tuple[int, ...], ...]

    @property
    def atom_count(self) -> int:
        return int(self.atoms.shape[0])

    def support(self, direction: Any) -> float:
        x = _real_vector(direction, dimension=self.ambient_dimension, name="direction")
        return float(np.max(self.atoms @ x))

    def query(self, direction: Any, *, tolerance: float = 1e-12) -> SupportQuery:
        x = _real_vector(direction, dimension=self.ambient_dimension, name="direction")
        values = self.atoms @ x
        value = float(np.max(values))
        scale = max(1.0, abs(value))
        active = tuple(int(i) for i in np.flatnonzero(values >= value - tolerance * scale))
        seen: set[tuple[int, ...]] = set()
        labels: list[tuple[int, ...]] = []
        for index in active:
            label = self.provenance[index]
            if label not in seen:
                seen.add(label)
                labels.append(label)
        return SupportQuery(value=value, active_indices=active, active_provenance=tuple(labels))

    def exposed_atoms(self, direction: Any, *, tolerance: float = 1e-12) -> np.ndarray:
        query = self.query(direction, tolerance=tolerance)
        result = np.array(self.atoms[list(query.active_indices)], copy=True)
        result.setflags(write=False)
        return result

    def linear_image(self, matrix: Any) -> "PolyhedralSupportPacket":
        """Return the support packet of ``A C`` for a possibly rectangular ``A``."""

        operator = _real_matrix(matrix, name="operator")
        if operator.shape[1] != self.ambient_dimension:
            raise DimensionError("operator has incompatible source dimension")
        transformed = self.atoms @ operator.T
        return polyhedral_support_packet(transformed, provenance=self.provenance)

    def branch_query(
        self,
        matrices: Iterable[Any],
        direction: Any,
        *,
        tolerance: float = 1e-12,
    ) -> BranchSupportQuery:
        """Evaluate ``max_i h_C(A_i x)`` without discarding branch identity."""

        x = _real_vector(direction, dimension=self.ambient_dimension, name="direction")
        values: list[float] = []
        for matrix in matrices:
            operator = _real_matrix(matrix, name="operator")
            if operator.shape != (self.ambient_dimension, self.ambient_dimension):
                raise DimensionError("branch operator has the wrong dimension")
            values.append(self.support(operator @ x))
        if not values:
            raise ValidationError("at least one branch operator is required")
        value = max(values)
        scale = max(1.0, abs(value))
        active = tuple(i for i, branch_value in enumerate(values) if branch_value >= value - tolerance * scale)
        return BranchSupportQuery(value=value, branch_values=tuple(values), active_branches=active)

    def bellman_envelope(self, matrices: Iterable[Any]) -> "PolyhedralSupportPacket":
        """Return the labeled dual packet for ``x -> max_i h_C(A_i x)``.

        The returned atoms are the union of ``A_i^T C`` generators.  No convex
        hull reduction is performed: redundant atoms are harmless for support
        values and may be necessary when downstream tasks ask for *all* branch
        provenance rather than only the value.
        """

        transformed_blocks: list[np.ndarray] = []
        labels: list[tuple[int, ...]] = []
        matrices_list = list(matrices)
        if not matrices_list:
            raise ValidationError("at least one branch operator is required")
        for branch_index, matrix in enumerate(matrices_list):
            operator = _real_matrix(matrix, name="operator")
            if operator.shape != (self.ambient_dimension, self.ambient_dimension):
                raise DimensionError("branch operator has the wrong dimension")
            transformed_blocks.append(self.atoms @ operator)
            # B_B(B_A h)(x)=max_{j,i} h(A_i B_j x), so a newly applied
            # outer Bellman branch acts first on the primal state and is
            # prepended to the existing execution word.
            labels.extend((branch_index,) + word for word in self.provenance)
        return polyhedral_support_packet(np.vstack(transformed_blocks), provenance=tuple(labels))


def polyhedral_support_packet(
    atoms: Any,
    *,
    provenance: Iterable[Iterable[int]] | None = None,
) -> PolyhedralSupportPacket:
    """Create a finite polyhedral support packet from generating atoms."""

    points = _real_matrix(atoms, name="atoms")
    frozen = np.array(points, copy=True)
    frozen.setflags(write=False)
    if provenance is None:
        labels = tuple(() for _ in range(frozen.shape[0]))
    else:
        labels = tuple(tuple(int(index) for index in word) for word in provenance)
        if len(labels) != frozen.shape[0]:
            raise DimensionError("provenance must contain one word per atom")
        if any(index < 0 for word in labels for index in word):
            raise ValidationError("provenance branch indices must be nonnegative")
    return PolyhedralSupportPacket(
        atoms=frozen,
        ambient_dimension=frozen.shape[1],
        provenance=labels,
    )
