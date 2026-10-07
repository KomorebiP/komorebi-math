"""Structured finite linear operators used by the packet compiler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..exceptions import DimensionError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import strict_integer
from .layout import PacketLayout

Array = np.ndarray


def _rhs(value: Any, expected_rows: int) -> tuple[Array, bool]:
    array = np.asarray(value)
    vector = array.ndim == 1
    if vector:
        array = array[:, None]
    if array.ndim != 2:
        raise DimensionError("a right hand side must be a vector or a matrix")
    if array.shape[0] != expected_rows:
        raise DimensionError(
            f"right hand side leading dimension must be {expected_rows}, received {array.shape[0]}"
        )
    return array, vector


def _restore_rhs(value: Array, vector: bool) -> Array:
    return value[:, 0] if vector else value


class StructuredOperator(ABC):
    """Abstract finite operator with explicit storage and cost semantics."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]: ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype[Any]: ...

    @abstractmethod
    def apply(self, value: Any) -> Array: ...

    @abstractmethod
    def materialize(self) -> Array: ...

    @abstractmethod
    def adjoint(self) -> "StructuredOperator": ...

    @abstractmethod
    def storage_entries(self) -> int: ...

    @abstractmethod
    def flop_count(self, rhs: int = 1) -> int: ...

    def solve(self, value: Any) -> Array:
        rows, columns = self.shape
        if rows != columns:
            raise DimensionError("only square structured operators can be solved")
        right, vector = _rhs(value, rows)
        solution = np.linalg.solve(self.materialize(), right)
        return _restore_rhs(solution, vector)

    def certificate_residual(self, value: Any, result: Any) -> float:
        right, _ = _rhs(value, self.shape[1])
        candidate, _ = _rhs(result, self.shape[0])
        return float(np.linalg.norm(self.apply(right) - candidate))


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-D02",
        title="Dense operator contract",
        status=ClaimStatus.DEFINITION,
        source="Paper VIII",
        assumptions=("finite matrix",),
        verification_suite="algorithms.structured",
    )
)
@dataclass(frozen=True, slots=True)
class DenseOperator(StructuredOperator):
    """Explicit dense matrix implementation of the structured contract."""

    matrix: Array

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix)
        if matrix.ndim != 2:
            raise DimensionError("a dense operator must be two dimensional")
        object.__setattr__(self, "matrix", np.array(matrix, copy=True))

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.matrix.dtype

    def apply(self, value: Any) -> Array:
        right, vector = _rhs(value, self.shape[1])
        return _restore_rhs(self.matrix @ right, vector)

    def materialize(self) -> Array:
        return np.array(self.matrix, copy=True)

    def adjoint(self) -> "DenseOperator":
        return DenseOperator(self.matrix.conj().T)

    def storage_entries(self) -> int:
        return int(self.matrix.size)

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        rows, columns = self.shape
        return 2 * rows * columns * rhs_count


@dataclass(frozen=True, slots=True)
class IdentityOperator(StructuredOperator):
    """Scalar multiple of the identity with constant storage."""

    dimension: int
    scalar: complex = 1.0
    _dtype: np.dtype[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dimension = strict_integer(self.dimension, "identity dimension", minimum=1)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_dtype", np.asarray(self.scalar).dtype)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def apply(self, value: Any) -> Array:
        right, vector = _rhs(value, self.dimension)
        return _restore_rhs(self.scalar * right, vector)

    def materialize(self) -> Array:
        return self.scalar * np.eye(self.dimension, dtype=self.dtype)

    def adjoint(self) -> "IdentityOperator":
        return IdentityOperator(self.dimension, np.conjugate(self.scalar))

    def storage_entries(self) -> int:
        return 1

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        return self.dimension * rhs_count

    def solve(self, value: Any) -> Array:
        if self.scalar == 0:
            raise np.linalg.LinAlgError("zero scalar identity is singular")
        right, vector = _rhs(value, self.dimension)
        return _restore_rhs(right / self.scalar, vector)


@dataclass(frozen=True, slots=True)
class DiagonalOperator(StructuredOperator):
    """Square diagonal operator stored by its diagonal entries."""

    diagonal: Array

    def __post_init__(self) -> None:
        diagonal = np.asarray(self.diagonal)
        if diagonal.ndim != 1 or diagonal.size == 0:
            raise DimensionError("diagonal entries must be a nonempty vector")
        object.__setattr__(self, "diagonal", np.array(diagonal, copy=True))

    @property
    def shape(self) -> tuple[int, int]:
        n = int(self.diagonal.size)
        return (n, n)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.diagonal.dtype

    def apply(self, value: Any) -> Array:
        right, vector = _rhs(value, self.shape[1])
        result = self.diagonal[:, None] * right
        return _restore_rhs(result, vector)

    def materialize(self) -> Array:
        return np.diag(self.diagonal)

    def adjoint(self) -> "DiagonalOperator":
        return DiagonalOperator(np.conjugate(self.diagonal))

    def storage_entries(self) -> int:
        return int(self.diagonal.size)

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        return int(self.diagonal.size) * rhs_count

    def solve(self, value: Any) -> Array:
        if np.any(self.diagonal == 0):
            raise np.linalg.LinAlgError("diagonal operator is singular")
        right, vector = _rhs(value, self.shape[0])
        result = right / self.diagonal[:, None]
        return _restore_rhs(result, vector)


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T03",
        title="Block diagonal packet operator",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite square packet blocks",),
        verification_suite="algorithms.structured",
    )
)
@dataclass(frozen=True, slots=True)
class BlockDiagonalOperator(StructuredOperator):
    """Square direct sum of independently executable structured blocks."""

    blocks: tuple[StructuredOperator | Array, ...]
    names: tuple[str, ...] | None = None
    layout: PacketLayout = field(init=False)
    _blocks: tuple[StructuredOperator, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValidationError("a block diagonal operator must have at least one block")
        normalized = tuple(as_structured_operator(block) for block in self.blocks)
        if any(block.shape[0] != block.shape[1] for block in normalized):
            raise DimensionError("block diagonal entries must be square")
        layout = PacketLayout(tuple(block.shape[0] for block in normalized), self.names)
        object.__setattr__(self, "_blocks", normalized)
        object.__setattr__(self, "blocks", normalized)
        object.__setattr__(self, "layout", layout)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.layout.dimension, self.layout.dimension)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.result_type(*(block.dtype for block in self._blocks))

    def apply(self, value: Any) -> Array:
        right, vector = _rhs(value, self.shape[1])
        parts = self.layout.split(right)
        outputs = tuple(block.apply(part) for block, part in zip(self._blocks, parts, strict=True))
        result = self.layout.merge(outputs)
        return _restore_rhs(result, vector)

    def solve(self, value: Any) -> Array:
        right, vector = _rhs(value, self.shape[0])
        parts = self.layout.split(right)
        outputs = tuple(block.solve(part) for block, part in zip(self._blocks, parts, strict=True))
        result = self.layout.merge(outputs)
        return _restore_rhs(result, vector)

    def materialize(self) -> Array:
        result = np.zeros(self.shape, dtype=self.dtype)
        for packet_slice, block in zip(self.layout.slices, self._blocks, strict=True):
            result[packet_slice, packet_slice] = block.materialize()
        return result

    def adjoint(self) -> "BlockDiagonalOperator":
        return BlockDiagonalOperator(tuple(block.adjoint() for block in self._blocks), self.layout.names)

    def storage_entries(self) -> int:
        return int(sum(block.storage_entries() for block in self._blocks))

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        return int(sum(block.flop_count(rhs_count) for block in self._blocks))

    def block(self, packet: int | str) -> StructuredOperator:
        return self._blocks[self.layout.index(packet)]


@dataclass(frozen=True, slots=True)
class CirculantOperator(StructuredOperator):
    """Circulant matrix applied through a discrete Fourier transform."""

    first_column: Array

    def __post_init__(self) -> None:
        column = np.asarray(self.first_column)
        if column.ndim != 1 or column.size == 0:
            raise DimensionError("first_column must be a nonempty vector")
        object.__setattr__(self, "first_column", np.array(column, copy=True))

    @property
    def shape(self) -> tuple[int, int]:
        n = int(self.first_column.size)
        return (n, n)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.first_column.dtype

    def apply(self, value: Any) -> Array:
        right, vector = _rhs(value, self.shape[1])
        spectrum = np.fft.fft(self.first_column)
        result = np.fft.ifft(spectrum[:, None] * np.fft.fft(right, axis=0), axis=0)
        if not np.issubdtype(np.result_type(self.first_column, right), np.complexfloating):
            result = np.real_if_close(result)
        return _restore_rhs(np.asarray(result), vector)

    def materialize(self) -> Array:
        n = self.shape[0]
        return np.asarray(
            [[self.first_column[(row - column) % n] for column in range(n)] for row in range(n)],
            dtype=self.dtype,
        )

    def adjoint(self) -> "CirculantOperator":
        matrix = self.materialize().conj().T
        return CirculantOperator(matrix[:, 0])

    def storage_entries(self) -> int:
        return self.shape[0]

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        n = self.shape[0]
        stages = int(np.ceil(np.log2(max(2, n))))
        return int(10 * n * stages * rhs_count)


@dataclass(frozen=True, slots=True)
class ToeplitzOperator(StructuredOperator):
    """Rectangular Toeplitz matrix applied through circulant embedding."""

    first_column: Array
    first_row: Array

    def __post_init__(self) -> None:
        column = np.asarray(self.first_column)
        row = np.asarray(self.first_row)
        if column.ndim != 1 or row.ndim != 1 or column.size == 0 or row.size == 0:
            raise DimensionError("Toeplitz boundary data must be nonempty vectors")
        if not np.allclose(column[0], row[0]):
            raise ValidationError("Toeplitz corner entries must agree")
        object.__setattr__(self, "first_column", np.array(column, copy=True))
        object.__setattr__(self, "first_row", np.array(row, copy=True))

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.first_column.size), int(self.first_row.size))

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.result_type(self.first_column, self.first_row)

    def apply(self, value: Any) -> Array:
        right, vector = _rhs(value, self.shape[1])
        rows, columns = self.shape
        embedding = np.concatenate(
            (
                self.first_column,
                np.zeros(1, dtype=np.result_type(self.dtype, right.dtype)),
                self.first_row[:0:-1],
            )
        )
        size = embedding.size
        padded = np.zeros((size, right.shape[1]), dtype=np.result_type(embedding, right))
        padded[:columns] = right
        result = np.fft.ifft(
            np.fft.fft(embedding)[:, None] * np.fft.fft(padded, axis=0),
            axis=0,
        )[:rows]
        if not np.issubdtype(np.result_type(self.dtype, right.dtype), np.complexfloating):
            result = np.real_if_close(result)
        return _restore_rhs(np.asarray(result), vector)

    def materialize(self) -> Array:
        rows, columns = self.shape
        result = np.empty((rows, columns), dtype=self.dtype)
        for row in range(rows):
            for column in range(columns):
                offset = row - column
                result[row, column] = (
                    self.first_column[offset] if offset >= 0 else self.first_row[-offset]
                )
        return result

    def adjoint(self) -> "ToeplitzOperator":
        matrix = self.materialize().conj().T
        return ToeplitzOperator(matrix[:, 0], matrix[0, :])

    def storage_entries(self) -> int:
        return self.shape[0] + self.shape[1] - 1

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        length = self.shape[0] + self.shape[1]
        stages = int(np.ceil(np.log2(max(2, length))))
        return int(10 * length * stages * rhs_count)


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T07",
        title="Kronecker modewise application",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("two finite matrix factors",),
        verification_suite="algorithms.structured",
    )
)
@dataclass(frozen=True, slots=True)
class KroneckerOperator(StructuredOperator):
    """Two factor Kronecker product with modewise application."""

    left: Array
    right: Array

    def __post_init__(self) -> None:
        left = np.asarray(self.left)
        right = np.asarray(self.right)
        if left.ndim != 2 or right.ndim != 2:
            raise DimensionError("Kronecker factors must be matrices")
        object.__setattr__(self, "left", np.array(left, copy=True))
        object.__setattr__(self, "right", np.array(right, copy=True))

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self.left.shape[0] * self.right.shape[0],
            self.left.shape[1] * self.right.shape[1],
        )

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.result_type(self.left, self.right)

    def apply(self, value: Any) -> Array:
        right_hand_side, vector = _rhs(value, self.shape[1])
        outputs: list[Array] = []
        for column in range(right_hand_side.shape[1]):
            matrix = right_hand_side[:, column].reshape(
                (self.right.shape[1], self.left.shape[1]), order="F"
            )
            transformed = self.right @ matrix @ self.left.T
            outputs.append(transformed.reshape(-1, order="F"))
        result = np.column_stack(outputs)
        return _restore_rhs(result, vector)

    def materialize(self) -> Array:
        return np.kron(self.left, self.right)

    def adjoint(self) -> "KroneckerOperator":
        return KroneckerOperator(self.left.conj().T, self.right.conj().T)

    def storage_entries(self) -> int:
        return int(self.left.size + self.right.size)

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        a_rows, a_columns = self.left.shape
        b_rows, b_columns = self.right.shape
        first = 2 * b_rows * b_columns * a_columns
        second = 2 * b_rows * a_columns * a_rows
        return int((first + second) * rhs_count)


@dataclass(frozen=True, slots=True)
class WalshHadamardOperator(StructuredOperator):
    """Fast Walsh Hadamard transform in power of two dimension."""

    dimension: int
    normalized: bool = False
    dtype_value: Any = float

    def __post_init__(self) -> None:
        n = strict_integer(self.dimension, "Walsh Hadamard dimension", minimum=1)
        if n & (n - 1):
            raise ValidationError("Walsh Hadamard dimension must be a positive power of two")
        object.__setattr__(self, "dimension", n)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.dtype_value)

    def apply(self, value: Any) -> Array:
        right, vector = _rhs(value, self.dimension)
        result_dtype = np.result_type(right.dtype, self.dtype)
        if self.normalized and not np.issubdtype(result_dtype, np.inexact):
            result_dtype = np.dtype(float)
        result = np.array(right, dtype=result_dtype, copy=True)
        width = 1
        while width < self.dimension:
            for start in range(0, self.dimension, 2 * width):
                upper = result[start : start + width].copy()
                lower = result[start + width : start + 2 * width].copy()
                result[start : start + width] = upper + lower
                result[start + width : start + 2 * width] = upper - lower
            width *= 2
        if self.normalized:
            result /= np.sqrt(self.dimension)
        return _restore_rhs(result, vector)

    def materialize(self) -> Array:
        return self.apply(np.eye(self.dimension, dtype=self.dtype))

    def adjoint(self) -> "WalshHadamardOperator":
        return self

    def storage_entries(self) -> int:
        return 0

    def flop_count(self, rhs: int = 1) -> int:
        rhs_count = strict_integer(rhs, "rhs", minimum=1)
        return int(self.dimension * int(np.log2(self.dimension)) * rhs_count)

    def solve(self, value: Any) -> Array:
        transformed = self.apply(value)
        if self.normalized:
            return transformed
        return transformed / self.dimension


def as_structured_operator(value: StructuredOperator | Any) -> StructuredOperator:
    """Return an existing structured operator or wrap a dense matrix."""

    if isinstance(value, StructuredOperator):
        return value
    return DenseOperator(np.asarray(value))


__all__ = [
    "BlockDiagonalOperator",
    "CirculantOperator",
    "DenseOperator",
    "DiagonalOperator",
    "IdentityOperator",
    "KroneckerOperator",
    "StructuredOperator",
    "ToeplitzOperator",
    "WalshHadamardOperator",
    "as_structured_operator",
]
