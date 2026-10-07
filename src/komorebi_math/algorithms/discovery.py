"""Restricted discovery of packet support and block diagonal plus low rank form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from .ir import Address, PacketIR
from .layout import PacketLayout
from .low_rank import BlockDiagonalLowRankOperator, LowRankOperator
from .structured import BlockDiagonalOperator, DenseOperator


@dataclass(frozen=True, slots=True)
class SupportDiscoveryResult:
    """Thresholded packet support and its discarded norm diagnostic."""

    packet_ir: PacketIR
    block_norms: dict[Address, float]
    threshold: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-H01",
        title="Threshold packet support discovery",
        status=ClaimStatus.HEURISTIC,
        source="Paper VIII",
        assumptions=("fixed packet layout", "declared norm threshold"),
        limitations=("a thresholded zero is not an exact algebraic zero",),
        verification_suite="algorithms.discovery",
    )
)
def discover_packet_support(
    matrix: Any,
    layout: PacketLayout,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 1e-12,
) -> SupportDiscoveryResult:
    """Discover a heuristic block support inside a fixed packet layout."""

    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape != (layout.dimension, layout.dimension):
        raise DimensionError("matrix shape does not match the packet layout")
    if (
        absolute_tolerance < 0
        or relative_tolerance < 0
        or not np.isfinite(absolute_tolerance)
        or not np.isfinite(relative_tolerance)
    ):
        raise ValidationError("support tolerances must be finite and nonnegative")
    matrix_norm = float(np.linalg.norm(array, 2))
    threshold = float(absolute_tolerance) + float(relative_tolerance) * matrix_norm
    block_norms: dict[Address, float] = {}
    blocks: dict[Address, DenseOperator] = {}
    for target, target_slice in enumerate(layout.slices):
        for source, source_slice in enumerate(layout.slices):
            block = np.array(array[target_slice, source_slice], copy=True)
            norm = float(np.linalg.norm(block, 2))
            block_norms[(target, source)] = norm
            if norm > threshold:
                blocks[(target, source)] = DenseOperator(block)
    packet_ir = PacketIR.square(
        layout,
        blocks,
        metadata={
            "discovery": "threshold support",
            "absolute_tolerance": float(absolute_tolerance),
            "relative_tolerance": float(relative_tolerance),
        },
    )
    discarded = array - packet_ir.materialize()
    discarded_norm = float(np.linalg.norm(discarded, 2))
    certificate = Certificate(
        valid=True,
        exactness=Exactness.HEURISTIC,
        residuals={"discarded_spectral_norm": discarded_norm},
        tolerance=threshold,
        assumptions_checked=("fixed packet layout", "declared threshold"),
        assumptions_unchecked=("discarded blocks are mathematically zero or negligible for the application",),
        diagnostics={"active_blocks": len(packet_ir.support), "matrix_norm": matrix_norm},
        theorem_id="PSC-VIII-H01",
        message="threshold packet support discovered",
    )
    return SupportDiscoveryResult(packet_ir, block_norms, threshold, certificate)


@dataclass(frozen=True, slots=True)
class BDLRDiscoveryResult:
    """Optimal fixed diagonal, rank constrained residual approximation."""

    operator: BlockDiagonalLowRankOperator
    singular_values: np.ndarray
    spectral_error: float
    frobenius_error: float
    certificate: Certificate


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T13",
        title="Optimal fixed layout block diagonal plus low rank approximation",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("fixed packet layout", "rank constrained approximation of the off diagonal residual"),
        verification_suite="algorithms.discovery",
    )
)
def discover_block_diagonal_low_rank(
    matrix: Any,
    layout: PacketLayout,
    rank: int,
    *,
    tolerance: float = 1e-10,
) -> BDLRDiscoveryResult:
    """Project onto a fixed block diagonal part and truncate its residual SVD."""

    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape != (layout.dimension, layout.dimension):
        raise DimensionError("matrix shape does not match the packet layout")
    if isinstance(rank, bool) or int(rank) != rank or not 0 <= int(rank) <= layout.dimension:
        raise ValidationError("rank must be an integer inside the matrix dimension")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValidationError("tolerance must be finite and nonnegative")
    diagonal_blocks = tuple(
        DenseOperator(np.array(array[packet_slice, packet_slice], copy=True))
        for packet_slice in layout.slices
    )
    diagonal = BlockDiagonalOperator(diagonal_blocks, layout.names)
    residual = array - diagonal.materialize()
    left_vectors, singular_values, right_vectors = np.linalg.svd(residual, full_matrices=False)
    retained = min(int(rank), singular_values.size)
    roots = np.sqrt(singular_values[:retained])
    left = left_vectors[:, :retained] * roots[None, :]
    right = right_vectors.conj().T[:, :retained] * roots[None, :]
    correction = LowRankOperator(left, right)
    operator = BlockDiagonalLowRankOperator(diagonal, correction)
    spectral_error = float(singular_values[retained]) if retained < singular_values.size else 0.0
    frobenius_error = float(np.linalg.norm(singular_values[retained:]))
    actual_error = float(np.linalg.norm(array - operator.materialize(), 2))
    valid = actual_error <= spectral_error + tolerance * max(1.0, float(np.linalg.norm(array, 2)))
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={
            "actual_spectral_error": actual_error,
            "optimal_spectral_error": spectral_error,
            "optimal_frobenius_error": frobenius_error,
        },
        tolerance=tolerance,
        assumptions_checked=("fixed packet diagonal projection", "singular value truncation of residual"),
        diagnostics={"requested_rank": int(rank), "retained_rank": retained},
        theorem_id="PSC-VIII-T13",
        message="valid fixed layout block diagonal plus low rank approximation" if valid else "BDLR discovery certificate failed",
    )
    return BDLRDiscoveryResult(operator, singular_values, spectral_error, frobenius_error, certificate)


__all__ = [
    "BDLRDiscoveryResult",
    "SupportDiscoveryResult",
    "discover_block_diagonal_low_rank",
    "discover_packet_support",
]
