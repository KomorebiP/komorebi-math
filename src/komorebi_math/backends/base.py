"""Linear algebra backend interface used by the finite dimensional core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LinearAlgebraBackend(ABC):
    name: str
    exact: bool

    @abstractmethod
    def as_matrix(self, value: Any) -> Any: ...

    @abstractmethod
    def as_vector(self, value: Any) -> Any: ...

    @abstractmethod
    def shape(self, value: Any) -> tuple[int, ...]: ...

    @abstractmethod
    def zeros(self, rows: int, cols: int) -> Any: ...

    @abstractmethod
    def eye(self, size: int) -> Any: ...

    @abstractmethod
    def matmul(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def add(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def subtract(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def scale(self, scalar: Any, value: Any) -> Any: ...

    @abstractmethod
    def transpose(self, value: Any) -> Any: ...

    @abstractmethod
    def adjoint(self, value: Any) -> Any: ...

    @abstractmethod
    def inverse(self, value: Any) -> Any: ...

    @abstractmethod
    def rank(self, value: Any, *, tolerance: float | None = None) -> int: ...

    @abstractmethod
    def norm(self, value: Any) -> float: ...

    @abstractmethod
    def is_zero(self, value: Any, *, tolerance: float | None = None) -> bool: ...

    @abstractmethod
    def column_stack(self, values: list[Any]) -> Any: ...

    @abstractmethod
    def nullspace(self, value: Any, *, tolerance: float | None = None) -> list[Any]: ...

    @abstractmethod
    def kron(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def copy(self, value: Any) -> Any: ...
