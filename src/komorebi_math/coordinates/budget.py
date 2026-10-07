"""Explicit resource contracts for finite coordinate computations."""

from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import ResourceLimitError, ValidationError


@dataclass(frozen=True, slots=True)
class CoordinateBudget:
    """Finite limits applied before coordinate matrices are materialized."""

    maximum_matrix_dimension: int = 512
    maximum_polynomial_degree: int = 512
    maximum_horizon: int = 4096
    maximum_exact_entries: int = 1_000_000

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_matrix_dimension", self.maximum_matrix_dimension),
            ("maximum_polynomial_degree", self.maximum_polynomial_degree),
            ("maximum_horizon", self.maximum_horizon),
            ("maximum_exact_entries", self.maximum_exact_entries),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValidationError(f"{name} must be a positive integer")

    def require_matrix(self, rows: int, columns: int, *, operation: str) -> None:
        """Reject a matrix allocation that exceeds the declared contract."""

        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (rows, columns)):
            raise ValidationError("matrix dimensions must be nonnegative integers")
        if rows > self.maximum_matrix_dimension or columns > self.maximum_matrix_dimension:
            raise ResourceLimitError(
                f"{operation} requires a {rows} by {columns} matrix, exceeding the "
                f"coordinate dimension limit {self.maximum_matrix_dimension}"
            )
        if rows * columns > self.maximum_exact_entries:
            raise ResourceLimitError(
                f"{operation} requires {rows * columns} exact entries, exceeding the "
                f"coordinate entry limit {self.maximum_exact_entries}"
            )

    def require_degree(self, degree: int, *, operation: str) -> None:
        """Reject a polynomial degree outside the declared contract."""

        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValidationError("degree must be a nonnegative integer")
        if degree > self.maximum_polynomial_degree:
            raise ResourceLimitError(
                f"{operation} degree {degree} exceeds the coordinate degree limit "
                f"{self.maximum_polynomial_degree}"
            )

    def require_horizon(self, horizon: int, *, operation: str) -> None:
        """Reject a finite horizon outside the declared contract."""

        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValidationError("horizon must be a nonnegative integer")
        if horizon > self.maximum_horizon:
            raise ResourceLimitError(
                f"{operation} horizon {horizon} exceeds the coordinate horizon limit "
                f"{self.maximum_horizon}"
            )


DEFAULT_COORDINATE_BUDGET = CoordinateBudget()


__all__ = ["CoordinateBudget", "DEFAULT_COORDINATE_BUDGET"]
