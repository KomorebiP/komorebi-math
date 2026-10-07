"""Low rank algebra, Woodbury solves, and certified recompression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import finite_float, strict_integer
from .structured import BlockDiagonalOperator, DenseOperator, StructuredOperator

Array = np.ndarray


@dataclass(frozen=True, slots=True)
class LowRankOperator(StructuredOperator):
    """A finite operator represented as ``left @ right.conj().T``."""

    left: Array
    right: Array

    def __post_init__(self) -> None:
        left = np.asarray(self.left)
        right = np.asarray(self.right)
        if left.ndim != 2 or right.ndim != 2:
            raise DimensionError("low rank factors must be matrices")
        if left.shape[1] != right.shape[1]:
            raise DimensionError("low rank factors must have the same column count")
        object.__setattr__(self, "left", np.array(left, copy=True))
        object.__setattr__(self, "right", np.array(right, copy=True))

    @property
    def rank_bound(self) -> int:
        return int(self.left.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.left.shape[0]), int(self.right.shape[0]))

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.result_type(self.left, self.right)

    def apply(self, value: Any) -> Array:
        array = np.asarray(value)
        vector = array.ndim == 1
        if vector:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] != self.shape[1]:
            raise DimensionError("right hand side has an incompatible shape")
        result = self.left @ (self.right.conj().T @ array)
        return result[:, 0] if vector else result

    def materialize(self) -> Array:
        return self.left @ self.right.conj().T

    def adjoint(self) -> "LowRankOperator":
        return LowRankOperator(self.right, self.left)

    def storage_entries(self) -> int:
        return int(self.left.size + self.right.size)

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        rows, columns = self.shape
        rank = self.rank_bound
        return int(2 * rank * (rows + columns) * rhs_count)


@dataclass(frozen=True, slots=True)
class RecompressionResult:
    """Rank truncated factors together with optimal tail certificates."""

    operator: LowRankOperator
    singular_values: Array
    retained_rank: int
    spectral_error: float
    frobenius_error: float
    certificate: Certificate

    @property
    def left(self) -> Array:
        return self.operator.left

    @property
    def right(self) -> Array:
        return self.operator.right

    def materialize(self) -> Array:
        return self.operator.materialize()


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T12",
        title="Optimal small core recompression",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite factor matrices", "rank constrained spectral or Frobenius approximation"),
        verification_suite="algorithms.low_rank",
    )
)
def recompress_core(
    left: Any,
    core: Any,
    right: Any,
    rank: int,
    *,
    tolerance: float = 1e-10,
) -> RecompressionResult:
    """Compress ``left @ core @ right.conj().T`` through a small SVD.

    The left and right bases may have different column counts. Rank deficient
    factors are handled by the reduced QR decompositions.
    """

    left_array = np.asarray(left)
    core_array = np.asarray(core)
    right_array = np.asarray(right)
    if left_array.ndim != 2 or right_array.ndim != 2 or core_array.ndim != 2:
        raise DimensionError("left, core, and right must be matrices")
    if core_array.shape != (left_array.shape[1], right_array.shape[1]):
        raise DimensionError("core shape must match the factor column counts")
    maximum_rank = min(left_array.shape[0], right_array.shape[0], *core_array.shape)
    rank_value = strict_integer(rank, "rank", minimum=0, maximum=maximum_rank)
    tolerance_value = finite_float(tolerance, "tolerance", minimum=0.0)

    if left_array.shape[1] == 0 or right_array.shape[1] == 0 or core_array.size == 0:
        empty = LowRankOperator(
            np.zeros((left_array.shape[0], 0), dtype=np.result_type(left_array, core_array, right_array)),
            np.zeros((right_array.shape[0], 0), dtype=np.result_type(left_array, core_array, right_array)),
        )
        certificate = Certificate(
            valid=True,
            exactness=Exactness.NUMERICAL,
            residuals={"spectral_error": 0.0, "frobenius_error": 0.0},
            tolerance=tolerance_value,
            assumptions_checked=("empty low rank correction",),
            theorem_id="PSC-VIII-T12",
            message="empty low rank correction",
        )
        return RecompressionResult(empty, np.zeros(0), 0, 0.0, 0.0, certificate)

    q_left, r_left = np.linalg.qr(left_array, mode="reduced")
    q_right, r_right = np.linalg.qr(right_array, mode="reduced")
    small_core = r_left @ core_array @ r_right.conj().T
    u, singular_values, vh = np.linalg.svd(small_core, full_matrices=False)
    retained_rank = min(rank_value, int(singular_values.size))
    roots = np.sqrt(singular_values[:retained_rank])
    compressed_left = (q_left @ u[:, :retained_rank]) * roots[None, :]
    compressed_right = (q_right @ vh.conj().T[:, :retained_rank]) * roots[None, :]
    operator = LowRankOperator(compressed_left, compressed_right)
    spectral_error = (
        float(singular_values[retained_rank])
        if retained_rank < singular_values.size
        else 0.0
    )
    frobenius_error = float(np.linalg.norm(singular_values[retained_rank:]))
    exact_matrix = left_array @ core_array @ right_array.conj().T
    actual_error = float(np.linalg.norm(exact_matrix - operator.materialize(), 2))
    valid = actual_error <= spectral_error + tolerance_value * max(1.0, float(np.linalg.norm(exact_matrix, 2)))
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.NUMERICAL,
        residuals={
            "actual_spectral_error": actual_error,
            "certified_spectral_error": spectral_error,
            "certified_frobenius_error": frobenius_error,
        },
        tolerance=tolerance_value,
        assumptions_checked=("reduced QR", "small core singular value decomposition"),
        diagnostics={
            "left_columns": left_array.shape[1],
            "right_columns": right_array.shape[1],
            "retained_rank": retained_rank,
        },
        theorem_id="PSC-VIII-T12",
        message="valid optimal small core recompression" if valid else "recompression certificate failed",
    )
    return RecompressionResult(
        operator,
        singular_values,
        retained_rank,
        spectral_error,
        frobenius_error,
        certificate,
    )


def recompress_factors(
    left: Any,
    right: Any,
    rank: int,
    *,
    tolerance: float = 1e-10,
) -> RecompressionResult:
    """Recompress a two factor product through an identity small core."""

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise DimensionError("low rank factors must be matrices")
    if left_array.shape[1] != right_array.shape[1]:
        raise DimensionError("low rank factors must have the same column count")
    core = np.eye(left_array.shape[1], dtype=np.result_type(left_array, right_array))
    return recompress_core(left_array, core, right_array, rank, tolerance=tolerance)


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T10",
        title="Block diagonal plus low rank algebra",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite square block diagonal part", "finite low rank correction"),
        verification_suite="algorithms.low_rank",
    )
)
@dataclass(frozen=True, slots=True)
class BlockDiagonalLowRankOperator(StructuredOperator):
    """A square block diagonal operator plus one low rank correction."""

    diagonal: BlockDiagonalOperator
    correction: LowRankOperator

    def __post_init__(self) -> None:
        if self.correction.shape != self.diagonal.shape:
            raise DimensionError("low rank correction must match the block diagonal shape")

    @classmethod
    def from_factors(
        cls,
        diagonal: BlockDiagonalOperator,
        left: Any,
        right: Any,
    ) -> "BlockDiagonalLowRankOperator":
        return cls(diagonal, LowRankOperator(np.asarray(left), np.asarray(right)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.diagonal.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.result_type(self.diagonal.dtype, self.correction.dtype)

    @property
    def rank_bound(self) -> int:
        return self.correction.rank_bound

    def apply(self, value: Any) -> Array:
        return self.diagonal.apply(value) + self.correction.apply(value)

    def materialize(self) -> Array:
        return self.diagonal.materialize() + self.correction.materialize()

    def adjoint(self) -> "BlockDiagonalLowRankOperator":
        return BlockDiagonalLowRankOperator(self.diagonal.adjoint(), self.correction.adjoint())

    def storage_entries(self) -> int:
        return self.diagonal.storage_entries() + self.correction.storage_entries()

    def flop_count(self, rhs: int = 1) -> int:
        return self.diagonal.flop_count(rhs) + self.correction.flop_count(rhs)

    def solve(self, value: Any) -> Array:
        """Apply the Woodbury formula when the required small inverse exists."""

        right_hand_side = np.asarray(value)
        vector = right_hand_side.ndim == 1
        if vector:
            right_hand_side = right_hand_side[:, None]
        if right_hand_side.ndim != 2 or right_hand_side.shape[0] != self.shape[0]:
            raise DimensionError("right hand side has an incompatible shape")
        d_inverse_rhs = self.diagonal.solve(right_hand_side)
        if self.rank_bound == 0:
            return d_inverse_rhs[:, 0] if vector else d_inverse_rhs
        d_inverse_left = self.diagonal.solve(self.correction.left)
        middle = (
            np.eye(self.rank_bound, dtype=self.dtype)
            + self.correction.right.conj().T @ d_inverse_left
        )
        correction = d_inverse_left @ np.linalg.solve(
            middle,
            self.correction.right.conj().T @ d_inverse_rhs,
        )
        result = d_inverse_rhs - correction
        return result[:, 0] if vector else result

    def exact_product(self, right: "BlockDiagonalLowRankOperator") -> "BlockDiagonalLowRankOperator":
        """Return an exact block diagonal plus low rank representation of a product."""

        if self.diagonal.layout != right.diagonal.layout:
            raise DimensionError("block diagonal layouts must agree")
        diagonal_blocks = tuple(
            DenseOperator(left.materialize() @ right_block.materialize())
            for left, right_block in zip(
                self.diagonal.blocks,
                right.diagonal.blocks,
                strict=True,
            )
        )
        diagonal_product = BlockDiagonalOperator(diagonal_blocks, self.diagonal.layout.names)
        ua = self.correction.left
        va = self.correction.right
        ub = right.correction.left
        vb = right.correction.right
        pieces_left: list[Array] = []
        pieces_right: list[Array] = []
        if ub.shape[1]:
            interaction = va.conj().T @ ub
            pieces_left.append(self.diagonal.apply(ub) + ua @ interaction)
            pieces_right.append(vb)
        if ua.shape[1]:
            pieces_left.append(ua)
            pieces_right.append(right.diagonal.adjoint().apply(va))
        if pieces_left:
            left_factor = np.concatenate(pieces_left, axis=1)
            right_factor = np.concatenate(pieces_right, axis=1)
        else:
            left_factor = np.zeros((self.shape[0], 0), dtype=self.dtype)
            right_factor = np.zeros((self.shape[1], 0), dtype=self.dtype)
        return BlockDiagonalLowRankOperator.from_factors(
            diagonal_product,
            left_factor,
            right_factor,
        )

    def recompress(self, rank: int, *, tolerance: float = 1e-10) -> tuple["BlockDiagonalLowRankOperator", RecompressionResult]:
        result = recompress_factors(
            self.correction.left,
            self.correction.right,
            rank,
            tolerance=tolerance,
        )
        return BlockDiagonalLowRankOperator(self.diagonal, result.operator), result


def product_chain_error_bound(
    exact_factors: Sequence[Any],
    approximate_factors: Sequence[Any],
) -> float:
    """Telescoping spectral norm bound for a product of perturbed factors."""

    if len(exact_factors) != len(approximate_factors) or not exact_factors:
        raise ValidationError("factor lists must have the same positive length")
    exact = [np.asarray(factor) for factor in exact_factors]
    approximate = [np.asarray(factor) for factor in approximate_factors]
    if any(factor.ndim != 2 for factor in (*exact, *approximate)):
        raise DimensionError("all product factors must be matrices")
    if any(true.shape != estimate.shape for true, estimate in zip(exact, approximate, strict=True)):
        raise DimensionError("each approximate factor must match its exact factor")
    if any(exact[index].shape[1] != exact[index - 1].shape[0] for index in range(1, len(exact))):
        raise DimensionError("product factors are not composable in list order")
    total = 0.0
    for index, (true_factor, approximate_factor) in enumerate(
        zip(exact, approximate, strict=True)
    ):
        left = float(
            np.prod(
                [np.linalg.norm(factor, 2) for factor in approximate[index + 1 :]],
                dtype=float,
            )
        ) if index + 1 < len(exact) else 1.0
        right = float(
            np.prod(
                [np.linalg.norm(factor, 2) for factor in exact[:index]],
                dtype=float,
            )
        ) if index else 1.0
        total += left * float(np.linalg.norm(approximate_factor - true_factor, 2)) * right
    return float(total)


def accumulated_recompression_bound(
    operator_norms: Sequence[float],
    local_errors: Sequence[float],
    *,
    initial_error: float = 0.0,
) -> float:
    """Propagate an inhomogeneous repeated recompression error recurrence.

    The recurrence is ``E_j <= norm_j E_{j-1} + error_j``.
    """

    if len(operator_norms) != len(local_errors):
        raise ValidationError("operator_norms and local_errors must have equal length")
    initial = finite_float(initial_error, "initial_error", minimum=0.0)
    norms = tuple(finite_float(value, "operator norm", minimum=0.0) for value in operator_norms)
    errors = tuple(finite_float(value, "local error", minimum=0.0) for value in local_errors)
    error = initial
    for norm, local in zip(norms, errors, strict=True):
        error = norm * error + local
    return error


__all__ = [
    "BlockDiagonalLowRankOperator",
    "LowRankOperator",
    "RecompressionResult",
    "accumulated_recompression_bound",
    "product_chain_error_bound",
    "recompress_core",
    "recompress_factors",
]
