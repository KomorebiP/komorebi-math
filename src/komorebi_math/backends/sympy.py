"""Optional exact SymPy backend."""

from __future__ import annotations

from typing import Any

from ..exceptions import BackendUnavailableError, DimensionError
from .base import LinearAlgebraBackend


def _sympy() -> Any:
    try:
        import sympy as sp
    except ImportError as exc:
        raise BackendUnavailableError(
            "the SymPy backend requires the 'exact' optional dependency"
        ) from exc
    return sp


class SympyBackend(LinearAlgebraBackend):
    name = "sympy"
    exact = True

    def as_matrix(self, value: Any) -> Any:
        sp = _sympy()
        matrix = value if isinstance(value, sp.MatrixBase) else sp.Matrix(value)
        if len(matrix.shape) != 2:
            raise DimensionError("a matrix must be two dimensional")
        return matrix

    def as_vector(self, value: Any) -> Any:
        sp = _sympy()
        if isinstance(value, sp.MatrixBase):
            matrix = value
        else:
            items = list(value)
            matrix = sp.Matrix(items)
        if matrix.cols == 1:
            return matrix
        if matrix.rows == 1:
            return matrix.T
        raise DimensionError("a vector must have one row or one column")

    def shape(self, value: Any) -> tuple[int, ...]:
        return tuple(value.shape)

    def zeros(self, rows: int, cols: int) -> Any:
        return _sympy().zeros(rows, cols)

    def eye(self, size: int) -> Any:
        return _sympy().eye(size)

    def matmul(self, left: Any, right: Any) -> Any:
        return left * right

    def add(self, left: Any, right: Any) -> Any:
        return left + right

    def subtract(self, left: Any, right: Any) -> Any:
        return left - right

    def scale(self, scalar: Any, value: Any) -> Any:
        return scalar * value

    def transpose(self, value: Any) -> Any:
        return value.T

    def adjoint(self, value: Any) -> Any:
        return value.conjugate().T

    def inverse(self, value: Any) -> Any:
        return value.inv()

    def rank(self, value: Any, *, tolerance: float | None = None) -> int:
        del tolerance
        return int(value.rank())

    def norm(self, value: Any) -> float:
        sp = _sympy()
        if value == sp.zeros(*value.shape):
            return 0.0
        return float(sp.N(value.norm()))

    def is_zero(self, value: Any, *, tolerance: float | None = None) -> bool:
        del tolerance
        sp = _sympy()
        return bool(value == sp.zeros(*value.shape))

    def column_stack(self, values: list[Any]) -> Any:
        sp = _sympy()
        if not values:
            return sp.zeros(0, 0)
        return sp.Matrix.hstack(*(self.as_vector(value) for value in values))

    def nullspace(self, value: Any, *, tolerance: float | None = None) -> list[Any]:
        del tolerance
        return list(value.nullspace())

    def kron(self, left: Any, right: Any) -> Any:
        return _sympy().kronecker_product(left, right)

    def copy(self, value: Any) -> Any:
        return value.copy()
