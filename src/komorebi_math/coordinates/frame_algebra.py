"""Finite coefficient-space closure for analysis--synthesis systems (Block 18).

For presentations X=(A_X,S_X) and Y=(A_Y,S_Y), a physical map T has
canonical coefficient lift L=A_Y T S_X.  This module implements the resulting
gauge, bridge, block, adjoint, and exact rational algebra without assuming
that the generally oblique projection A S is orthogonal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .analysis_synthesis import AnalysisSynthesisSystem
from .exact import (
    Matrix,
    Vector,
    identity_matrix,
    inverse_matrix,
    kronecker,
    matrix_add,
    matrix_multiply,
    matrix_rank,
    matrix_shape,
    matrix_subtract,
    matrix_vector_multiply,
    transpose,
)


def _readonly_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise DimensionError(f"{name} must be a matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DimensionError(f"{name} must be nonempty")
    try:
        finite = bool(np.all(np.isfinite(array)))
    except TypeError as exc:
        raise ValidationError(f"{name} must contain numerical scalars") from exc
    if not finite:
        raise ValidationError(f"{name} must contain finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _numeric_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != length:
        raise DimensionError(f"{name} must have length {length}")
    try:
        finite = bool(np.all(np.isfinite(array)))
    except TypeError as exc:
        raise ValidationError(f"{name} must contain numerical scalars") from exc
    if not finite:
        raise ValidationError(f"{name} must contain finite values")
    return np.array(array, copy=False)


def _readonly_product(*factors: np.ndarray) -> np.ndarray:
    if not factors:
        raise ValidationError("matrix product requires at least one factor")
    result = np.asarray(factors[0])
    for factor in factors[1:]:
        result = result @ factor
    answer = np.array(result, copy=True)
    answer.setflags(write=False)
    return answer


def _compatible(
    left: AnalysisSynthesisSystem,
    right: AnalysisSynthesisSystem,
) -> bool:
    return left.source_dimension == right.source_dimension and left.source == right.source


@dataclass(frozen=True, slots=True)
class CoefficientPartition:
    """A strict ordered partition of finite coefficient indices."""

    dimension: int
    groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, Integral):
            raise ValidationError("coefficient dimension must be an integer")
        dimension = int(self.dimension)
        if dimension <= 0:
            raise DimensionError("coefficient dimension must be positive")
        try:
            raw_groups = tuple(tuple(group) for group in self.groups)
        except TypeError as exc:
            raise ValidationError("coefficient groups must be iterable") from exc
        if not raw_groups or any(not group for group in raw_groups):
            raise ValidationError("coefficient groups must be nonempty")
        normalized: list[tuple[int, ...]] = []
        for group in raw_groups:
            values: list[int] = []
            for index in group:
                if isinstance(index, bool) or not isinstance(index, Integral):
                    raise ValidationError("coefficient indices must be integers")
                value = int(index)
                if value < 0 or value >= dimension:
                    raise ValidationError("coefficient index is outside the declared dimension")
                values.append(value)
            if len(values) != len(set(values)):
                raise ValidationError("coordinates in a group must be unique")
            normalized.append(tuple(sorted(values)))
        flattened = tuple(index for group in normalized for index in group)
        if len(flattened) != len(set(flattened)):
            raise ValidationError("coefficient groups must be disjoint")
        if set(flattened) != set(range(dimension)):
            raise ValidationError("coefficient groups must partition every coordinate")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "groups", tuple(normalized))

    @classmethod
    def singletons(cls, dimension: int) -> "CoefficientPartition":
        return cls(dimension, tuple((index,) for index in range(dimension)))

    @classmethod
    def one_block(cls, dimension: int) -> "CoefficientPartition":
        return cls(dimension, (tuple(range(dimension)),))

    def refines(self, other: "CoefficientPartition") -> bool:
        if not isinstance(other, CoefficientPartition) or self.dimension != other.dimension:
            return False
        coarse = tuple(frozenset(group) for group in other.groups)
        return all(
            any(frozenset(group) <= container for container in coarse)
            for group in self.groups
        )

    def coarsens(self, other: "CoefficientPartition") -> bool:
        return isinstance(other, CoefficientPartition) and other.refines(self)


@dataclass(frozen=True, slots=True)
class AnalysisSynthesisGauge:
    """Gauge and quotient calculus for one numerical presentation."""

    system: AnalysisSynthesisSystem

    def __post_init__(self) -> None:
        if not isinstance(self.system, AnalysisSynthesisSystem):
            raise ValidationError("gauge requires an AnalysisSynthesisSystem")
        self.system.certificate().require()

    @property
    def analysis(self) -> np.ndarray:
        return self.system.analysis

    @property
    def synthesis(self) -> np.ndarray:
        return self.system.synthesis

    @property
    def source(self) -> str:
        return self.system.source

    @property
    def coordinates(self) -> str:
        return self.system.coordinates

    @property
    def source_dimension(self) -> int:
        return self.system.source_dimension

    @property
    def coordinate_dimension(self) -> int:
        return self.system.coordinate_dimension

    @property
    def tolerance(self) -> float:
        return self.system.tolerance

    def coefficient_projection(self) -> np.ndarray:
        return _readonly_product(self.analysis, self.synthesis)

    def gauge_projection(self) -> np.ndarray:
        projection = self.coefficient_projection()
        result = np.eye(self.coordinate_dimension, dtype=projection.dtype) - projection
        result.setflags(write=False)
        return result

    def coefficient_metric(self) -> np.ndarray:
        return _readonly_product(self.synthesis.conj().T, self.synthesis)

    def canonical_coefficients(self, value: Any) -> np.ndarray:
        vector = _numeric_vector(
            value, length=self.coordinate_dimension, name="coefficient vector"
        )
        return self.coefficient_projection() @ vector

    def gauge_part(self, value: Any) -> np.ndarray:
        vector = _numeric_vector(
            value, length=self.coordinate_dimension, name="coefficient vector"
        )
        return self.gauge_projection() @ vector

    def gauge_equivalent(self, left: Any, right: Any) -> bool:
        first = _numeric_vector(
            left, length=self.coordinate_dimension, name="left coefficient vector"
        )
        second = _numeric_vector(
            right, length=self.coordinate_dimension, name="right coefficient vector"
        )
        residual = float(np.linalg.norm(self.synthesis @ (first - second)))
        scale = max(
            1.0,
            float(np.linalg.norm(self.synthesis @ first)),
            float(np.linalg.norm(self.synthesis @ second)),
        )
        return residual <= self.tolerance * scale

    def coefficient_seminorm(self, value: Any) -> float:
        vector = _numeric_vector(
            value, length=self.coordinate_dimension, name="coefficient vector"
        )
        return float(np.linalg.norm(self.synthesis @ vector))

    def dual(self) -> "AnalysisSynthesisGauge":
        system = AnalysisSynthesisSystem(
            self.synthesis.conj().T,
            self.analysis.conj().T,
            source=f"{self.source}*",
            coordinates=f"{self.coordinates}*",
            tolerance=self.tolerance,
            theorem="KM-AS-GAUGE-DUAL-01",
        )
        return AnalysisSynthesisGauge(system)

    def transition_to(
        self,
        other: "AnalysisSynthesisGauge | AnalysisSynthesisSystem",
    ) -> np.ndarray:
        target = other if isinstance(other, AnalysisSynthesisGauge) else AnalysisSynthesisGauge(other)
        if not _compatible(self.system, target.system):
            raise ValidationError(
                "transition requires presentations of the same physical source space"
            )
        return _readonly_product(target.analysis, self.synthesis)

    def direct_sum(self, other: "AnalysisSynthesisGauge") -> "AnalysisSynthesisGauge":
        if not isinstance(other, AnalysisSynthesisGauge):
            raise ValidationError("direct sum requires another gauge")
        rows = self.coordinate_dimension + other.coordinate_dimension
        columns = self.source_dimension + other.source_dimension
        dtype = np.result_type(
            self.analysis.dtype,
            self.synthesis.dtype,
            other.analysis.dtype,
            other.synthesis.dtype,
        )
        analysis = np.zeros((rows, columns), dtype=dtype)
        analysis[: self.coordinate_dimension, : self.source_dimension] = self.analysis
        analysis[self.coordinate_dimension :, self.source_dimension :] = other.analysis
        synthesis = np.zeros((columns, rows), dtype=dtype)
        synthesis[: self.source_dimension, : self.coordinate_dimension] = self.synthesis
        synthesis[self.source_dimension :, self.coordinate_dimension :] = other.synthesis
        system = AnalysisSynthesisSystem(
            analysis,
            synthesis,
            source=f"{self.source} direct-sum {other.source}",
            coordinates=f"{self.coordinates} direct-sum {other.coordinates}",
            tolerance=max(self.tolerance, other.tolerance),
            theorem="KM-AS-GAUGE-SUM-01",
        )
        return AnalysisSynthesisGauge(system)

    def tensor_product(self, other: "AnalysisSynthesisGauge") -> "AnalysisSynthesisGauge":
        if not isinstance(other, AnalysisSynthesisGauge):
            raise ValidationError("tensor product requires another gauge")
        system = AnalysisSynthesisSystem(
            np.kron(self.analysis, other.analysis),
            np.kron(self.synthesis, other.synthesis),
            source=f"{self.source} tensor {other.source}",
            coordinates=f"{self.coordinates} tensor {other.coordinates}",
            tolerance=max(self.tolerance, other.tolerance),
            theorem="KM-AS-GAUGE-TENSOR-01",
        )
        return AnalysisSynthesisGauge(system)

    def certificate(self) -> Certificate:
        projection = self.coefficient_projection()
        gauge = self.gauge_projection()
        metric = self.coefficient_metric()
        identity = np.eye(self.coordinate_dimension, dtype=projection.dtype)
        residuals = {
            "projection_idempotence": float(np.linalg.norm(projection @ projection - projection)),
            "gauge_idempotence": float(np.linalg.norm(gauge @ gauge - gauge)),
            "projection_complement": float(np.linalg.norm(projection + gauge - identity)),
            "analysis_support": float(np.linalg.norm(projection @ self.analysis - self.analysis)),
            "synthesis_support": float(np.linalg.norm(self.synthesis @ projection - self.synthesis)),
            "metric_left_support": float(np.linalg.norm(projection.conj().T @ metric - metric)),
            "metric_right_support": float(np.linalg.norm(metric @ projection - metric)),
        }
        scale = max(1.0, float(np.linalg.norm(projection)), float(np.linalg.norm(metric)))
        valid = self.system.certificate().valid and all(
            residual <= self.tolerance * scale for residual in residuals.values()
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=self.tolerance,
            assumptions_checked=(
                "finite analysis--synthesis presentation",
                "analysis--synthesis left inverse",
                "coefficient quotient by synthesis kernel",
            ),
            diagnostics={
                "source_dimension": self.source_dimension,
                "coordinate_dimension": self.coordinate_dimension,
                "redundant": self.system.redundant,
            },
            theorem_id="KM-AS-GAUGE-01",
            message="coefficient gauge algebra verified" if valid else "coefficient gauge algebra failed",
        )


@dataclass(frozen=True, slots=True)
class AnalysisSynthesisMap:
    """A physical linear map and its canonical coefficient lift."""

    domain: AnalysisSynthesisSystem
    codomain: AnalysisSynthesisSystem
    operator: np.ndarray
    theorem: str = "KM-AS-MAP-01"

    def __post_init__(self) -> None:
        if not isinstance(self.domain, AnalysisSynthesisSystem) or not isinstance(
            self.codomain, AnalysisSynthesisSystem
        ):
            raise ValidationError("map endpoints must be analysis--synthesis systems")
        self.domain.certificate().require()
        self.codomain.certificate().require()
        operator = _readonly_matrix(self.operator, name="physical operator")
        expected = (self.codomain.source_dimension, self.domain.source_dimension)
        if operator.shape != expected:
            raise DimensionError(
                "operator shape must be (codomain source dimension, domain source dimension)"
            )
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "theorem", str(self.theorem))

    @classmethod
    def identity(cls, system: AnalysisSynthesisSystem) -> "AnalysisSynthesisMap":
        return cls(system, system, np.eye(system.source_dimension), "KM-AS-MAP-ID-01")

    @property
    def tolerance(self) -> float:
        return max(self.domain.tolerance, self.codomain.tolerance)

    def coefficient_lift(self) -> np.ndarray:
        return _readonly_product(
            self.codomain.analysis, self.operator, self.domain.synthesis
        )

    def reconstruct(self) -> np.ndarray:
        return _readonly_product(
            self.codomain.synthesis,
            self.coefficient_lift(),
            self.domain.analysis,
        )

    def apply_source(self, value: Any) -> np.ndarray:
        vector = _numeric_vector(
            value, length=self.domain.source_dimension, name="source vector"
        )
        return self.operator @ vector

    def apply_coefficients(self, value: Any) -> np.ndarray:
        vector = _numeric_vector(
            value, length=self.domain.coordinate_dimension, name="coefficient vector"
        )
        return self.coefficient_lift() @ vector

    def then(self, after: "AnalysisSynthesisMap") -> "AnalysisSynthesisMap":
        if not isinstance(after, AnalysisSynthesisMap):
            raise ValidationError("composition requires another analysis--synthesis map")
        if not _compatible(self.codomain, after.domain):
            raise ValidationError("middle physical presentations are incompatible")
        return AnalysisSynthesisMap(
            self.domain,
            after.codomain,
            after.operator @ self.operator,
            theorem="KM-AS-MAP-COMPOSE-01",
        )

    def compose(self, before: "AnalysisSynthesisMap") -> "AnalysisSynthesisMap":
        return before.then(self)

    def transition_bridge(self, after: "AnalysisSynthesisMap") -> np.ndarray:
        if not _compatible(self.codomain, after.domain):
            raise ValidationError("middle physical presentations are incompatible")
        return AnalysisSynthesisGauge(self.codomain).transition_to(after.domain)

    def adjoint(self) -> "AnalysisSynthesisMap":
        return AnalysisSynthesisMap(
            self.codomain,
            self.domain,
            self.operator.conj().T,
            theorem="KM-AS-MAP-ADJOINT-01",
        )

    def dual_adjoint(self) -> "AnalysisSynthesisMap":
        return AnalysisSynthesisMap(
            AnalysisSynthesisGauge(self.codomain).dual().system,
            AnalysisSynthesisGauge(self.domain).dual().system,
            self.operator.conj().T,
            theorem="KM-AS-MAP-DUAL-ADJOINT-01",
        )

    def scaled(self, scalar: complex) -> "AnalysisSynthesisMap":
        value = np.asarray(scalar)
        if value.ndim != 0 or not bool(np.isfinite(value)):
            raise ValidationError("map scalar must be finite")
        return AnalysisSynthesisMap(
            self.domain, self.codomain, scalar * self.operator, "KM-AS-MAP-SCALE-01"
        )

    def add(self, other: "AnalysisSynthesisMap") -> "AnalysisSynthesisMap":
        if (
            not isinstance(other, AnalysisSynthesisMap)
            or self.domain is not other.domain
            or self.codomain is not other.codomain
        ):
            raise ValidationError("addition requires identical endpoint presentations")
        return AnalysisSynthesisMap(
            self.domain, self.codomain, self.operator + other.operator, "KM-AS-MAP-ADD-01"
        )

    def blocks(
        self,
        source_partition: CoefficientPartition | None = None,
        target_partition: CoefficientPartition | None = None,
    ) -> "AnalysisSynthesisCoefficientBlocks":
        source = source_partition or CoefficientPartition.singletons(
            self.domain.coordinate_dimension
        )
        target = target_partition or CoefficientPartition.singletons(
            self.codomain.coordinate_dimension
        )
        return AnalysisSynthesisCoefficientBlocks(self, source, target)

    def certificate(self) -> Certificate:
        lift = self.coefficient_lift()
        domain_projection = AnalysisSynthesisGauge(self.domain).coefficient_projection()
        codomain_projection = AnalysisSynthesisGauge(self.codomain).coefficient_projection()
        residuals = {
            "coefficient_sandwich": float(
                np.linalg.norm(codomain_projection @ lift @ domain_projection - lift)
            ),
            "physical_reconstruction": float(np.linalg.norm(self.reconstruct() - self.operator)),
        }
        scale = max(1.0, float(np.linalg.norm(lift)), float(np.linalg.norm(self.operator)))
        valid = all(residual <= self.tolerance * scale for residual in residuals.values())
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals=residuals,
            tolerance=self.tolerance,
            assumptions_checked=(
                "finite analysis--synthesis endpoints",
                "endpoint left-inverse laws",
                "physical operator dimensions",
            ),
            diagnostics={
                "domain_source_dimension": self.domain.source_dimension,
                "domain_coordinate_dimension": self.domain.coordinate_dimension,
                "codomain_source_dimension": self.codomain.source_dimension,
                "codomain_coordinate_dimension": self.codomain.coordinate_dimension,
            },
            theorem_id=self.theorem or None,
            message="coefficient map verified" if valid else "coefficient map failed",
        )


@dataclass(frozen=True, slots=True)
class AnalysisSynthesisCoefficientBlocks:
    """Small target-by-source blocks derived from a coefficient lift."""

    mapping: AnalysisSynthesisMap
    source_partition: CoefficientPartition
    target_partition: CoefficientPartition
    blocks: tuple[tuple[np.ndarray, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, AnalysisSynthesisMap):
            raise ValidationError("coefficient blocks require a map")
        if self.source_partition.dimension != self.mapping.domain.coordinate_dimension:
            raise DimensionError("source partition dimension does not match the map")
        if self.target_partition.dimension != self.mapping.codomain.coordinate_dimension:
            raise DimensionError("target partition dimension does not match the map")
        lift = self.mapping.coefficient_lift()
        rows: list[tuple[np.ndarray, ...]] = []
        for target_group in self.target_partition.groups:
            row: list[np.ndarray] = []
            for source_group in self.source_partition.groups:
                block = np.array(lift[np.ix_(target_group, source_group)], copy=True)
                block.setflags(write=False)
                row.append(block)
            rows.append(tuple(row))
        object.__setattr__(self, "blocks", tuple(rows))

    def assemble(self) -> np.ndarray:
        result = np.zeros(
            (self.target_partition.dimension, self.source_partition.dimension),
            dtype=self.mapping.coefficient_lift().dtype,
        )
        for target_index, target_group in enumerate(self.target_partition.groups):
            for source_index, source_group in enumerate(self.source_partition.groups):
                result[np.ix_(target_group, source_group)] = self.blocks[target_index][
                    source_index
                ]
        result.setflags(write=False)
        return result

    def reblock(
        self,
        source_partition: CoefficientPartition,
        target_partition: CoefficientPartition,
    ) -> "AnalysisSynthesisCoefficientBlocks":
        return AnalysisSynthesisCoefficientBlocks(
            self.mapping, source_partition, target_partition
        )

    def physical_block(self, target_index: int, source_index: int) -> np.ndarray:
        try:
            target_group = self.target_partition.groups[target_index]
            source_group = self.source_partition.groups[source_index]
            block = self.blocks[target_index][source_index]
        except IndexError as exc:
            raise ValidationError("coefficient block index is outside the partition") from exc
        return _readonly_product(
            self.mapping.codomain.synthesis[:, target_group],
            block,
            self.mapping.domain.analysis[source_group, :],
        )

    def then(
        self,
        after: "AnalysisSynthesisCoefficientBlocks",
        *,
        source_partition: CoefficientPartition | None = None,
        target_partition: CoefficientPartition | None = None,
    ) -> "AnalysisSynthesisCoefficientBlocks":
        if not isinstance(after, AnalysisSynthesisCoefficientBlocks):
            raise ValidationError("block composition requires coefficient blocks")
        return self.mapping.then(after.mapping).blocks(
            source_partition or self.source_partition,
            target_partition or after.target_partition,
        )

    def certificate(self) -> Certificate:
        residual = float(np.linalg.norm(self.assemble() - self.mapping.coefficient_lift()))
        scale = max(1.0, float(np.linalg.norm(self.mapping.coefficient_lift())))
        valid = self.mapping.certificate().valid and residual <= self.mapping.tolerance * scale
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"coefficient_block_assembly": residual},
            tolerance=self.mapping.tolerance,
            assumptions_checked=(
                "source coefficient partition",
                "target coefficient partition",
                "canonical coefficient lift",
            ),
            diagnostics={
                "source_block_count": len(self.source_partition.groups),
                "target_block_count": len(self.target_partition.groups),
            },
            theorem_id="KM-AS-BLOCK-01",
            message="coefficient blocks verified" if valid else "coefficient blocks failed",
        )


def _exact_scalar(value: Any, *, name: str = "exact coefficient") -> Fraction:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must not be boolean")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    raise ValidationError(f"{name} must be an integer or Fraction")


def _exact_matrix(value: Any, *, name: str) -> Matrix:
    try:
        raw_rows = tuple(tuple(row) for row in value)
    except TypeError as exc:
        raise ValidationError(f"{name} must be an iterable matrix") from exc
    if not raw_rows or not raw_rows[0]:
        raise DimensionError(f"{name} must be nonempty")
    width = len(raw_rows[0])
    if any(len(row) != width for row in raw_rows):
        raise DimensionError(f"{name} rows must have a common length")
    return tuple(
        tuple(_exact_scalar(entry, name=f"{name} entry") for entry in row)
        for row in raw_rows
    )


def _exact_vector(value: Any, *, length: int, name: str) -> Vector:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValidationError(f"{name} must be iterable") from exc
    if len(raw) != length:
        raise DimensionError(f"{name} must have length {length}")
    return tuple(_exact_scalar(entry, name=f"{name} entry") for entry in raw)


def _exact_scale(matrix: Matrix, scalar: Fraction) -> Matrix:
    return tuple(tuple(scalar * entry for entry in row) for row in matrix)


def _exact_direct_sum(left: Matrix, right: Matrix) -> Matrix:
    left_rows, left_columns = matrix_shape(left)
    right_rows, right_columns = matrix_shape(right)
    return tuple(
        tuple(
            (
                left[row][column]
                if row < left_rows and column < left_columns
                else right[row - left_rows][column - left_columns]
                if row >= left_rows and column >= left_columns
                else Fraction(0)
            )
            for column in range(left_columns + right_columns)
        )
        for row in range(left_rows + right_rows)
    )


@dataclass(frozen=True, slots=True)
class ExactAnalysisSynthesisSystem:
    """An exact rational analysis--synthesis presentation."""

    analysis: Matrix
    synthesis: Matrix
    source: str = "source"
    coordinates: str = "coordinates"
    theorem: str = "KM-AS-EXACT-01"

    def __post_init__(self) -> None:
        analysis = _exact_matrix(self.analysis, name="exact analysis matrix")
        synthesis = _exact_matrix(self.synthesis, name="exact synthesis matrix")
        if matrix_shape(synthesis) != (
            matrix_shape(analysis)[1],
            matrix_shape(analysis)[0],
        ):
            raise DimensionError(
                "exact synthesis shape must be (source dimension, coordinate dimension)"
            )
        if matrix_multiply(synthesis, analysis) != identity_matrix(
            matrix_shape(analysis)[1]
        ):
            raise ValidationError("exact synthesis must be a left inverse of analysis")
        if not str(self.source).strip() or not str(self.coordinates).strip():
            raise ValidationError("exact system space names must be nonempty")
        object.__setattr__(self, "analysis", analysis)
        object.__setattr__(self, "synthesis", synthesis)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "coordinates", str(self.coordinates))
        object.__setattr__(self, "theorem", str(self.theorem))

    @classmethod
    def from_invertible(
        cls,
        analysis: Any,
        *,
        source: str = "source",
        coordinates: str = "coordinates",
        theorem: str = "KM-AS-EXACT-01",
    ) -> "ExactAnalysisSynthesisSystem":
        matrix = _exact_matrix(analysis, name="exact analysis matrix")
        rows, columns = matrix_shape(matrix)
        if rows != columns:
            raise DimensionError("an exact invertible analysis matrix must be square")
        return cls(matrix, inverse_matrix(matrix), source, coordinates, theorem)

    @classmethod
    def canonical_dual(
        cls,
        analysis: Any,
        *,
        source: str = "source",
        coordinates: str = "coordinates",
        theorem: str = "KM-AS-EXACT-01",
    ) -> "ExactAnalysisSynthesisSystem":
        matrix = _exact_matrix(analysis, name="exact analysis matrix")
        rows, columns = matrix_shape(matrix)
        if rows < columns or matrix_rank(matrix) != columns:
            raise ValidationError("exact analysis matrix must have full column rank")
        adjoint = transpose(matrix)
        gram = matrix_multiply(adjoint, matrix)
        synthesis = matrix_multiply(inverse_matrix(gram), adjoint)
        return cls(matrix, synthesis, source, coordinates, theorem)

    @property
    def source_dimension(self) -> int:
        return matrix_shape(self.analysis)[1]

    @property
    def coordinate_dimension(self) -> int:
        return matrix_shape(self.analysis)[0]

    @property
    def redundant(self) -> bool:
        return self.coordinate_dimension > self.source_dimension

    def frame_operator(self) -> Matrix:
        return matrix_multiply(self.synthesis, self.analysis)

    def coefficient_projection(self) -> Matrix:
        return matrix_multiply(self.analysis, self.synthesis)

    def gauge_projection(self) -> Matrix:
        return matrix_subtract(
            identity_matrix(self.coordinate_dimension),
            self.coefficient_projection(),
        )

    def coefficient_metric(self) -> Matrix:
        return matrix_multiply(transpose(self.synthesis), self.synthesis)

    def analyze(self, value: Sequence[Any]) -> Vector:
        vector = _exact_vector(
            value, length=self.source_dimension, name="exact source vector"
        )
        return matrix_vector_multiply(self.analysis, vector)

    def synthesize(self, value: Sequence[Any]) -> Vector:
        vector = _exact_vector(
            value, length=self.coordinate_dimension, name="exact coefficient vector"
        )
        return matrix_vector_multiply(self.synthesis, vector)

    def canonical_coefficients(self, value: Sequence[Any]) -> Vector:
        vector = _exact_vector(
            value, length=self.coordinate_dimension, name="exact coefficient vector"
        )
        return matrix_vector_multiply(self.coefficient_projection(), vector)

    def gauge_part(self, value: Sequence[Any]) -> Vector:
        vector = _exact_vector(
            value, length=self.coordinate_dimension, name="exact coefficient vector"
        )
        return matrix_vector_multiply(self.gauge_projection(), vector)

    def gauge_equivalent(self, left: Sequence[Any], right: Sequence[Any]) -> bool:
        first = _exact_vector(
            left, length=self.coordinate_dimension, name="left exact coefficient vector"
        )
        second = _exact_vector(
            right, length=self.coordinate_dimension, name="right exact coefficient vector"
        )
        difference = tuple(a - b for a, b in zip(first, second, strict=True))
        return matrix_vector_multiply(self.synthesis, difference) == tuple(
            Fraction(0) for _ in range(self.source_dimension)
        )

    def coefficient_seminorm_squared(self, value: Sequence[Any]) -> Fraction:
        represented = self.synthesize(value)
        return sum((entry * entry for entry in represented), Fraction(0))

    def dual(self) -> "ExactAnalysisSynthesisSystem":
        return ExactAnalysisSynthesisSystem(
            transpose(self.synthesis),
            transpose(self.analysis),
            source=f"{self.source}*",
            coordinates=f"{self.coordinates}*",
            theorem="KM-AS-EXACT-DUAL-01",
        )

    def transition_to(self, other: "ExactAnalysisSynthesisSystem") -> Matrix:
        if (
            not isinstance(other, ExactAnalysisSynthesisSystem)
            or self.source_dimension != other.source_dimension
            or self.source != other.source
        ):
            raise ValidationError(
                "exact transition requires the same physical source space"
            )
        return matrix_multiply(other.analysis, self.synthesis)

    def direct_sum(
        self,
        other: "ExactAnalysisSynthesisSystem",
    ) -> "ExactAnalysisSynthesisSystem":
        if not isinstance(other, ExactAnalysisSynthesisSystem):
            raise ValidationError("exact direct sum requires another exact system")
        return ExactAnalysisSynthesisSystem(
            _exact_direct_sum(self.analysis, other.analysis),
            _exact_direct_sum(self.synthesis, other.synthesis),
            source=f"{self.source} direct-sum {other.source}",
            coordinates=f"{self.coordinates} direct-sum {other.coordinates}",
            theorem="KM-AS-EXACT-SUM-01",
        )

    def tensor_product(
        self,
        other: "ExactAnalysisSynthesisSystem",
    ) -> "ExactAnalysisSynthesisSystem":
        if not isinstance(other, ExactAnalysisSynthesisSystem):
            raise ValidationError("exact tensor product requires another exact system")
        return ExactAnalysisSynthesisSystem(
            kronecker(self.analysis, other.analysis),
            kronecker(self.synthesis, other.synthesis),
            source=f"{self.source} tensor {other.source}",
            coordinates=f"{self.coordinates} tensor {other.coordinates}",
            theorem="KM-AS-EXACT-TENSOR-01",
        )

    def certificate(self) -> Certificate:
        projection = self.coefficient_projection()
        metric = self.coefficient_metric()
        valid = (
            self.frame_operator() == identity_matrix(self.source_dimension)
            and matrix_multiply(projection, projection) == projection
            and matrix_multiply(transpose(projection), metric) == metric
            and matrix_multiply(metric, projection) == metric
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals={},
            tolerance=None,
            assumptions_checked=(
                "exact rational coefficients",
                "exact analysis--synthesis left inverse",
                "exact coefficient quotient",
            ),
            diagnostics={
                "source_dimension": self.source_dimension,
                "coordinate_dimension": self.coordinate_dimension,
                "redundant": self.redundant,
            },
            theorem_id=self.theorem or None,
            message="exact system verified" if valid else "exact system failed",
        )


@dataclass(frozen=True, slots=True)
class ExactAnalysisSynthesisMap:
    """An exact rational physical map and coefficient lift."""

    domain: ExactAnalysisSynthesisSystem
    codomain: ExactAnalysisSynthesisSystem
    operator: Matrix
    theorem: str = "KM-AS-EXACT-MAP-01"

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ExactAnalysisSynthesisSystem) or not isinstance(
            self.codomain, ExactAnalysisSynthesisSystem
        ):
            raise ValidationError("exact map endpoints must be exact systems")
        operator = _exact_matrix(self.operator, name="exact physical operator")
        if matrix_shape(operator) != (
            self.codomain.source_dimension,
            self.domain.source_dimension,
        ):
            raise DimensionError(
                "exact operator shape must be (codomain source dimension, domain source dimension)"
            )
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "theorem", str(self.theorem))

    @classmethod
    def identity(
        cls,
        system: ExactAnalysisSynthesisSystem,
    ) -> "ExactAnalysisSynthesisMap":
        return cls(system, system, identity_matrix(system.source_dimension))

    def coefficient_lift(self) -> Matrix:
        return matrix_multiply(
            matrix_multiply(self.codomain.analysis, self.operator),
            self.domain.synthesis,
        )

    def reconstruct(self) -> Matrix:
        return matrix_multiply(
            matrix_multiply(self.codomain.synthesis, self.coefficient_lift()),
            self.domain.analysis,
        )

    def apply_coefficients(self, value: Sequence[Any]) -> Vector:
        vector = _exact_vector(
            value, length=self.domain.coordinate_dimension, name="exact coefficient vector"
        )
        return matrix_vector_multiply(self.coefficient_lift(), vector)

    def then(self, after: "ExactAnalysisSynthesisMap") -> "ExactAnalysisSynthesisMap":
        if not isinstance(after, ExactAnalysisSynthesisMap):
            raise ValidationError("exact composition requires another exact map")
        if (
            self.codomain.source_dimension != after.domain.source_dimension
            or self.codomain.source != after.domain.source
        ):
            raise ValidationError("exact middle physical presentations are incompatible")
        return ExactAnalysisSynthesisMap(
            self.domain,
            after.codomain,
            matrix_multiply(after.operator, self.operator),
            theorem="KM-AS-EXACT-MAP-COMPOSE-01",
        )

    def compose(
        self,
        before: "ExactAnalysisSynthesisMap",
    ) -> "ExactAnalysisSynthesisMap":
        return before.then(self)

    def adjoint(self) -> "ExactAnalysisSynthesisMap":
        return ExactAnalysisSynthesisMap(
            self.codomain,
            self.domain,
            transpose(self.operator),
            theorem="KM-AS-EXACT-MAP-ADJOINT-01",
        )

    def dual_adjoint(self) -> "ExactAnalysisSynthesisMap":
        return ExactAnalysisSynthesisMap(
            self.codomain.dual(),
            self.domain.dual(),
            transpose(self.operator),
            theorem="KM-AS-EXACT-MAP-DUAL-ADJOINT-01",
        )

    def scaled(self, scalar: Any) -> "ExactAnalysisSynthesisMap":
        coefficient = _exact_scalar(scalar, name="exact map scalar")
        return ExactAnalysisSynthesisMap(
            self.domain,
            self.codomain,
            _exact_scale(self.operator, coefficient),
            theorem="KM-AS-EXACT-MAP-SCALE-01",
        )

    def add(self, other: "ExactAnalysisSynthesisMap") -> "ExactAnalysisSynthesisMap":
        if (
            not isinstance(other, ExactAnalysisSynthesisMap)
            or self.domain is not other.domain
            or self.codomain is not other.codomain
        ):
            raise ValidationError("exact addition requires identical endpoints")
        return ExactAnalysisSynthesisMap(
            self.domain,
            self.codomain,
            matrix_add(self.operator, other.operator),
            theorem="KM-AS-EXACT-MAP-ADD-01",
        )

    def blocks(
        self,
        source_partition: CoefficientPartition,
        target_partition: CoefficientPartition,
    ) -> tuple[tuple[Matrix, ...], ...]:
        if source_partition.dimension != self.domain.coordinate_dimension:
            raise DimensionError("exact source partition dimension does not match")
        if target_partition.dimension != self.codomain.coordinate_dimension:
            raise DimensionError("exact target partition dimension does not match")
        lift = self.coefficient_lift()
        return tuple(
            tuple(
                tuple(
                    tuple(lift[row][column] for column in source_group)
                    for row in target_group
                )
                for source_group in source_partition.groups
            )
            for target_group in target_partition.groups
        )

    def certificate(self) -> Certificate:
        lift = self.coefficient_lift()
        valid = (
            matrix_multiply(
                matrix_multiply(self.codomain.coefficient_projection(), lift),
                self.domain.coefficient_projection(),
            )
            == lift
            and self.reconstruct() == self.operator
        )
        return Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            residuals={},
            tolerance=None,
            assumptions_checked=(
                "exact rational endpoint systems",
                "exact endpoint left-inverse laws",
                "exact physical operator dimensions",
            ),
            diagnostics={
                "domain_source_dimension": self.domain.source_dimension,
                "domain_coordinate_dimension": self.domain.coordinate_dimension,
                "codomain_source_dimension": self.codomain.source_dimension,
                "codomain_coordinate_dimension": self.codomain.coordinate_dimension,
            },
            theorem_id=self.theorem or None,
            message="exact coefficient map verified" if valid else "exact coefficient map failed",
        )


__all__ = [
    "AnalysisSynthesisCoefficientBlocks",
    "AnalysisSynthesisGauge",
    "AnalysisSynthesisMap",
    "CoefficientPartition",
    "ExactAnalysisSynthesisMap",
    "ExactAnalysisSynthesisSystem",
]
