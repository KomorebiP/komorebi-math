"""NumPy backend."""

from __future__ import annotations

from .._residuals import finite_tolerance

from typing import Any

import numpy as np

from ..exceptions import DimensionError
from .base import LinearAlgebraBackend


class NumpyBackend(LinearAlgebraBackend):
    name = "numpy"
    exact = False

    def __init__(self, *, dtype: Any = np.complex128) -> None:
        self.dtype = dtype

    def as_matrix(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=self.dtype)
        if array.ndim != 2:
            raise DimensionError("a matrix must be two dimensional")
        return array

    def as_vector(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=self.dtype)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise DimensionError("a vector must be one dimensional")
        return array

    def shape(self, value: Any) -> tuple[int, ...]:
        return tuple(np.shape(value))

    def zeros(self, rows: int, cols: int) -> np.ndarray:
        return np.zeros((rows, cols), dtype=self.dtype)

    def eye(self, size: int) -> np.ndarray:
        return np.eye(size, dtype=self.dtype)

    def matmul(self, left: Any, right: Any) -> np.ndarray:
        return np.asarray(left) @ np.asarray(right)

    def add(self, left: Any, right: Any) -> np.ndarray:
        return np.asarray(left) + np.asarray(right)

    def subtract(self, left: Any, right: Any) -> np.ndarray:
        return np.asarray(left) - np.asarray(right)

    def scale(self, scalar: Any, value: Any) -> np.ndarray:
        return scalar * np.asarray(value)

    def transpose(self, value: Any) -> np.ndarray:
        return np.asarray(value).T

    def adjoint(self, value: Any) -> np.ndarray:
        return np.asarray(value).conj().T

    def inverse(self, value: Any) -> np.ndarray:
        return np.linalg.inv(np.asarray(value))

    def rank(self, value: Any, *, tolerance: float | None = None) -> int:
        return int(np.linalg.matrix_rank(np.asarray(value), tol=tolerance))

    def norm(self, value: Any) -> float:
        # Hypot reduction avoids squaring a tiny nonzero value into zero (or
        # overflowing large squares). This remains a floating-point diagnostic.
        array = np.asarray(value)
        magnitudes = np.hypot(array.real, array.imag)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            return float(np.hypot.reduce(magnitudes.ravel(), initial=0.0))

    def is_zero(self, value: Any, *, tolerance: float | None = None) -> bool:
        tol = finite_tolerance(1e-10 if tolerance is None else tolerance)
        array = np.asarray(value)
        if not bool(np.all(np.isfinite(array))):
            return False
        if tol == 0.0:
            return bool(np.all(array == 0))
        return self.norm(array) <= tol

    def column_stack(self, values: list[Any]) -> np.ndarray:
        if not values:
            return np.zeros((0, 0), dtype=self.dtype)
        columns = [self.as_vector(value) for value in values]
        return np.column_stack(columns)

    def nullspace(self, value: Any, *, tolerance: float | None = None) -> list[np.ndarray]:
        matrix = self.as_matrix(value)
        if matrix.size == 0:
            return [self.eye(matrix.shape[1])[:, i] for i in range(matrix.shape[1])]
        _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
        if tolerance is None:
            tolerance = max(matrix.shape, default=0) * np.finfo(float).eps * (
                float(singular_values[0]) if singular_values.size else 1.0
            )
        rank = int(np.sum(singular_values > tolerance))
        return [vh.conj().T[:, i] for i in range(rank, vh.shape[0])]

    def kron(self, left: Any, right: Any) -> np.ndarray:
        return np.kron(np.asarray(left), np.asarray(right))

    def copy(self, value: Any) -> np.ndarray:
        return np.array(value, copy=True)
