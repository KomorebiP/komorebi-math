"""General finite analysis--synthesis systems.

This module turns the common reconstruction law ``S A = I`` into one explicit
runtime object.  It covers invertible coordinate transforms, compatible
redundant frames, finite sampling left inverses, and packetized operator
blocks without pretending that approximate or rank-deficient systems are
exact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .._number_theory import require_integer


def _numeric_array(value: Any, *, name: str) -> np.ndarray:
    """Promote integer input before arithmetic in this numerical API."""

    try:
        dtype = np.complex128 if np.iscomplexobj(value) else np.float64
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must contain finite numerical values") from exc


def _norm(value: Any) -> float:
    """A display norm that avoids intermediate squared-magnitude overflow."""

    array = np.asarray(value)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        magnitudes = np.hypot(array.real, array.imag)
        result = float(np.hypot.reduce(magnitudes.ravel(), initial=0.0))
    return result if np.isfinite(result) else float("inf")


def _within_tolerance(residual: float, scale: float, tolerance: float) -> bool:
    # Nonfinite intermediate arithmetic never grants a numerical certificate.
    return bool(np.isfinite(residual) and np.isfinite(scale)
                and residual / max(1.0, scale) <= tolerance)


def _matrix(value: Any, *, name: str) -> np.ndarray:
    array = _numeric_array(value, name=name)
    if array.ndim != 2:
        raise DimensionError(f"{name} must be a matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DimensionError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = _numeric_array(value, name=name)
    if array.ndim != 1 or array.shape[0] != length:
        raise DimensionError(f"{name} must have length {length}")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values")
    return array


def _normalized_groups(
    groups: Iterable[Iterable[int]] | None,
    coordinate_dimension: int,
) -> tuple[tuple[int, ...], ...]:
    if groups is None:
        return tuple((index,) for index in range(coordinate_dimension))
    normalized = tuple(tuple(require_integer(index, name="coordinate index") for index in group)
                       for group in groups)
    if not normalized or any(not group for group in normalized):
        raise ValidationError("analysis--synthesis groups must be nonempty")
    flattened = tuple(index for group in normalized for index in group)
    if len(flattened) != len(set(flattened)):
        raise ValidationError("analysis--synthesis groups must be disjoint")
    if set(flattened) != set(range(coordinate_dimension)):
        raise ValidationError("analysis--synthesis groups must partition all coordinates")
    return normalized


@dataclass(frozen=True, slots=True)
class AnalysisSynthesisBlocks:
    """Packet blocks of one operator under a declared analysis--synthesis atlas."""

    system: "AnalysisSynthesisSystem"
    groups: tuple[tuple[int, ...], ...]
    blocks: tuple[tuple[np.ndarray, ...], ...]
    operator: np.ndarray

    def __post_init__(self) -> None:
        groups = _normalized_groups(self.groups, self.system.coordinate_dimension)
        size = len(groups)
        if len(self.blocks) != size or any(len(row) != size for row in self.blocks):
            raise DimensionError("operator block grid does not match the packet groups")
        expected = (self.system.source_dimension, self.system.source_dimension)
        operator = _matrix(self.operator, name="operator")
        if operator.shape != expected:
            raise DimensionError("operator has the wrong source-space shape")
        normalized_rows: list[tuple[np.ndarray, ...]] = []
        for row in self.blocks:
            normalized_row: list[np.ndarray] = []
            for block in row:
                value = _matrix(block, name="operator block")
                if value.shape != expected:
                    raise DimensionError("operator block has the wrong source-space shape")
                normalized_row.append(value)
            normalized_rows.append(tuple(normalized_row))
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "blocks", tuple(normalized_rows))
        object.__setattr__(self, "groups", groups)

    def reconstruct(self) -> np.ndarray:
        """Sum every transfer block back into the represented operator."""

        dtype = np.result_type(self.operator.dtype, *(block.dtype for row in self.blocks for block in row))
        result = np.zeros(self.operator.shape, dtype=dtype)
        for row in self.blocks:
            for block in row:
                result = result + block
        return result

    def certificate(self) -> Certificate:
        """Certify numerical reconstruction of the operator block expansion."""

        reconstructed = self.reconstruct()
        with np.errstate(over="ignore", invalid="ignore"):
            residual = _norm(reconstructed - self.operator)
            components = self.system.components(self.groups)
            definition_residual = max(
                _norm(block - target @ self.operator @ source)
                for target, row in zip(components, self.blocks, strict=True)
                for source, block in zip(components, row, strict=True)
            )
        scale = max(1.0, _norm(self.operator))
        tolerance = self.system.tolerance
        valid = (self.system.certificate().valid
                 and _within_tolerance(residual, scale, tolerance)
                 and _within_tolerance(definition_residual, scale, tolerance))
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"operator_block_reconstruction": residual,
                       "operator_block_definition": definition_residual},
            tolerance=tolerance,
            assumptions_checked=(
                "finite analysis and synthesis matrices",
                "coordinate groups form a partition",
                "analysis--synthesis left inverse",
                "each block equals its declared reconstruction components",
            ),
            theorem_id="KM-AS-02",
            message=(
                "analysis--synthesis operator blocks reconstruct"
                if valid
                else "analysis--synthesis operator blocks do not reconstruct"
            ),
        )


@dataclass(frozen=True, slots=True)
class AnalysisSynthesisSystem:
    """A finite analysis map ``A`` and synthesis map ``S`` with ``S A = I``.

    ``analysis`` has shape ``(coordinate_dimension, source_dimension)`` and
    ``synthesis`` has the transposed rectangular shape.  A square invertible
    transform is one case.  A redundant frame or finite sampling system is
    also supported when the supplied synthesis is a valid left inverse.
    """

    analysis: np.ndarray
    synthesis: np.ndarray
    source: str = "source"
    coordinates: str = "coordinates"
    tolerance: float = 1e-12
    theorem: str = "KM-AS-01"

    def __post_init__(self) -> None:
        analysis = _matrix(self.analysis, name="analysis matrix")
        synthesis = _matrix(self.synthesis, name="synthesis matrix")
        if synthesis.shape != (analysis.shape[1], analysis.shape[0]):
            raise DimensionError(
                "synthesis shape must be (source_dimension, coordinate_dimension)"
            )
        if not str(self.source).strip() or not str(self.coordinates).strip():
            raise ValidationError("analysis--synthesis space names must be nonempty")
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValidationError("analysis--synthesis tolerance must be positive and finite")
        object.__setattr__(self, "analysis", analysis)
        object.__setattr__(self, "synthesis", synthesis)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "coordinates", str(self.coordinates))
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "theorem", str(self.theorem))

    @classmethod
    def from_invertible(
        cls,
        matrix: Any,
        *,
        source: str = "source",
        coordinates: str = "coordinates",
        tolerance: float = 1e-12,
        theorem: str = "KM-AS-01",
    ) -> "AnalysisSynthesisSystem":
        """Construct a square system from an invertible analysis matrix."""

        analysis = _matrix(matrix, name="analysis matrix")
        if analysis.shape[0] != analysis.shape[1]:
            raise DimensionError("an invertible analysis matrix must be square")
        try:
            synthesis = np.linalg.inv(analysis)
        except np.linalg.LinAlgError as exc:
            raise ValidationError("analysis matrix is singular") from exc
        return cls(analysis, synthesis, source, coordinates, tolerance, theorem)

    @classmethod
    def canonical_dual(
        cls,
        analysis: Any,
        *,
        source: str = "source",
        coordinates: str = "coordinates",
        tolerance: float = 1e-12,
        theorem: str = "KM-AS-01",
    ) -> "AnalysisSynthesisSystem":
        """Construct the canonical left inverse of a full-column-rank analysis map."""

        matrix = _matrix(analysis, name="analysis matrix")
        relative_cutoff = float(tolerance)
        if not np.isfinite(relative_cutoff) or relative_cutoff <= 0:
            raise ValidationError("analysis--synthesis tolerance must be positive and finite")
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        largest = float(singular_values[0]) if singular_values.size else 0.0
        rank = int(np.count_nonzero(singular_values > relative_cutoff * largest))
        if rank != matrix.shape[1]:
            raise ValidationError(
                "analysis matrix must have full column rank at the declared tolerance"
            )
        synthesis = np.linalg.pinv(matrix, rcond=relative_cutoff)
        system = cls(matrix, synthesis, source, coordinates, tolerance, theorem)
        system.certificate().require()
        return system

    @property
    def source_dimension(self) -> int:
        return int(self.analysis.shape[1])

    @property
    def coordinate_dimension(self) -> int:
        return int(self.analysis.shape[0])

    @property
    def redundant(self) -> bool:
        return self.coordinate_dimension > self.source_dimension

    @property
    def square(self) -> bool:
        return self.coordinate_dimension == self.source_dimension

    def frame_operator(self) -> np.ndarray:
        """Return the source-space reconstruction operator ``S A``."""

        return self.synthesis @ self.analysis

    def analyze(self, value: Any) -> np.ndarray:
        """Map one source vector into analysis coordinates."""

        source = _vector(value, length=self.source_dimension, name="source vector")
        return self.analysis @ source

    def synthesize(self, value: Any) -> np.ndarray:
        """Map one coordinate vector back into the source space."""

        coordinates = _vector(
            value,
            length=self.coordinate_dimension,
            name="coordinate vector",
        )
        return self.synthesis @ coordinates

    def reconstruct(self, value: Any) -> np.ndarray:
        """Analyze and synthesize one source vector."""

        return self.synthesize(self.analyze(value))

    def component(self, indices: int | Sequence[int]) -> np.ndarray:
        """Return the reconstruction component ``S E_I A``.

        The components of a complete coordinate partition always sum to the
        source-space identity when ``S A = I``.  In a square invertible atlas
        they are algebraic projectors.  In a redundant frame they need not be
        idempotent or mutually annihilating, so ``component`` is the canonical
        term used by the 1.0-candidate API.
        """

        if isinstance(indices, (int, np.integer)):
            normalized = (require_integer(indices, name="component coordinate"),)
        else:
            normalized = tuple(require_integer(index, name="component coordinate") for index in indices)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValidationError("component coordinates must be nonempty and unique")
        if min(normalized) < 0 or max(normalized) >= self.coordinate_dimension:
            raise ValidationError("component coordinate is outside the analysis range")
        selector = np.zeros(
            (self.coordinate_dimension, self.coordinate_dimension),
            dtype=np.result_type(self.analysis.dtype, self.synthesis.dtype),
        )
        selector[normalized, normalized] = 1
        return self.synthesis @ selector @ self.analysis

    def components(
        self,
        groups: Iterable[Iterable[int]] | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Return reconstruction components for a coordinate partition."""

        normalized = _normalized_groups(groups, self.coordinate_dimension)
        return tuple(self.component(group) for group in normalized)

    def projector(self, indices: int | Sequence[int]) -> np.ndarray:
        """Compatibility alias for :meth:`component`.

        The returned map is a genuine projector for square invertible systems.
        For redundant systems it is only an additive reconstruction component.
        New code should use :meth:`component` unless idempotence has been
        established independently.
        """

        return self.component(indices)

    def projectors(
        self,
        groups: Iterable[Iterable[int]] | None = None,
    ) -> tuple[np.ndarray, ...]:
        """Compatibility alias for :meth:`components`."""

        return self.components(groups)

    def operator_blocks(
        self,
        operator: Any,
        *,
        groups: Iterable[Iterable[int]] | None = None,
    ) -> AnalysisSynthesisBlocks:
        """Resolve ``T`` into blocks ``C_beta T C_alpha``."""

        matrix = _matrix(operator, name="operator")
        expected = (self.source_dimension, self.source_dimension)
        if matrix.shape != expected:
            raise DimensionError("operator has the wrong source-space shape")
        normalized = _normalized_groups(groups, self.coordinate_dimension)
        components = tuple(self.component(group) for group in normalized)
        blocks = tuple(
            tuple(target @ matrix @ source for source in components)
            for target in components
        )
        return AnalysisSynthesisBlocks(self, normalized, blocks, matrix)

    def certificate(self) -> Certificate:
        """Certify the finite left-inverse reconstruction law ``S A = I``."""

        operator = self.frame_operator()
        identity = np.eye(self.source_dimension, dtype=operator.dtype)
        residual = _norm(operator - identity)
        scale = max(1.0, _norm(identity))
        rank = int(np.linalg.matrix_rank(self.analysis))
        valid = (_within_tolerance(residual, scale, self.tolerance)
                 and rank == self.source_dimension)
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"analysis_synthesis_left_inverse": residual},
            tolerance=self.tolerance,
            assumptions_checked=(
                "finite matrices",
                "declared matrix dimensions",
                "full source-space rank",
            ),
            diagnostics={
                "source_dimension": self.source_dimension,
                "coordinate_dimension": self.coordinate_dimension,
                "analysis_rank": rank,
                "square": self.square,
                "redundant": self.redundant,
            },
            theorem_id=self.theorem or None,
            message=(
                "analysis--synthesis reconstruction verified"
                if valid
                else "analysis--synthesis reconstruction failed"
            ),
        )

    def reconstruction_certificate(self, value: Any) -> Certificate:
        """Certify one concrete analysis--synthesis round trip."""

        source = _vector(value, length=self.source_dimension, name="source vector")
        reconstructed = self.reconstruct(source)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = _norm(reconstructed - source)
        scale = max(1.0, _norm(source))
        valid = self.certificate().valid and _within_tolerance(residual, scale, self.tolerance)
        return Certificate(
            valid=valid,
            exactness=Exactness.NUMERICAL,
            residuals={"analysis_synthesis_roundtrip": residual},
            tolerance=self.tolerance,
            assumptions_checked=("analysis--synthesis left inverse",),
            theorem_id=self.theorem or None,
            message=(
                "analysis--synthesis round trip verified"
                if valid
                else "analysis--synthesis round trip failed"
            ),
        )


__all__ = ["AnalysisSynthesisBlocks", "AnalysisSynthesisSystem"]
