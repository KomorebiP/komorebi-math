"""Validation helpers for finite quantum models."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._residuals import finite_tolerance
from ..exceptions import DimensionError, ValidationError


def as_matrix(value: Any, *, name: str = "matrix") -> np.ndarray:
    """Return a finite two dimensional complex matrix."""

    matrix = np.asarray(value, dtype=complex)
    if matrix.ndim != 2:
        raise DimensionError(f"{name} must be two dimensional")
    if matrix.size and not np.all(np.isfinite(matrix)):
        raise ValidationError(f"{name} must contain finite values")
    return matrix


def as_square_matrix(value: Any, *, name: str = "matrix") -> np.ndarray:
    """Return a finite square complex matrix."""

    matrix = as_matrix(value, name=name)
    if matrix.shape[0] != matrix.shape[1]:
        raise DimensionError(f"{name} must be square")
    return matrix


def as_vector(value: Any, *, dimension: int | None = None, name: str = "vector") -> np.ndarray:
    """Return a finite complex vector."""

    vector = np.asarray(value, dtype=complex)
    if vector.ndim != 1:
        raise DimensionError(f"{name} must be one dimensional")
    if dimension is not None and vector.shape != (dimension,):
        raise DimensionError(f"{name} must have dimension {dimension}")
    if vector.size and not np.all(np.isfinite(vector)):
        raise ValidationError(f"{name} must contain finite values")
    return vector


def require_hermitian(matrix: Any, *, tolerance: float = 1e-10, name: str = "matrix") -> np.ndarray:
    """Validate Hermitian symmetry."""

    tolerance = finite_tolerance(tolerance)
    value = as_square_matrix(matrix, name=name)
    if not np.allclose(value, value.conj().T, atol=tolerance, rtol=0.0):
        raise ValidationError(f"{name} must be Hermitian")
    return value


def require_unitary(matrix: Any, *, tolerance: float = 1e-10, name: str = "matrix") -> np.ndarray:
    """Validate unitarity."""

    tolerance = finite_tolerance(tolerance)
    value = as_square_matrix(matrix, name=name)
    identity = np.eye(value.shape[0], dtype=complex)
    if not np.allclose(value.conj().T @ value, identity, atol=tolerance, rtol=0.0):
        raise ValidationError(f"{name} must be unitary")
    return value


def require_projector(
    matrix: Any,
    *,
    tolerance: float = 1e-10,
    orthogonal: bool = True,
    name: str = "projector",
) -> np.ndarray:
    """Validate idempotence and optionally orthogonal self adjointness."""

    tolerance = finite_tolerance(tolerance)
    value = as_square_matrix(matrix, name=name)
    if not np.allclose(value @ value, value, atol=tolerance, rtol=0.0):
        raise ValidationError(f"{name} must be idempotent")
    if orthogonal and not np.allclose(value, value.conj().T, atol=tolerance, rtol=0.0):
        raise ValidationError(f"{name} must be an orthogonal projector")
    return value


def density_matrix(value: Any, *, tolerance: float = 1e-10, normalize: bool = False) -> np.ndarray:
    """Validate a finite density matrix."""

    tolerance = finite_tolerance(tolerance)
    matrix = require_hermitian(value, tolerance=tolerance, name="density matrix")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValidationError("density matrix spectrum exceeds finite floating-point range")
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValidationError("density matrix must be positive semidefinite")
    trace = complex(np.trace(matrix))
    if not np.isfinite(trace):
        raise ValidationError("density matrix trace exceeds finite floating-point range")
    if normalize:
        if abs(trace) <= tolerance:
            raise ValidationError("density matrix trace must be nonzero")
        matrix = matrix / trace
    elif abs(trace - 1) > tolerance:
        raise ValidationError("density matrix must have trace one")
    return np.real_if_close(matrix, tol=1000)


def orthogonal_packet_family(projectors: tuple[np.ndarray, ...], *, tolerance: float = 1e-10) -> tuple[np.ndarray, ...]:
    """Validate a complete finite family of orthogonal projectors."""

    if not projectors:
        raise ValidationError("at least one packet projector is required")
    values = tuple(require_projector(projector, tolerance=tolerance) for projector in projectors)
    dimension = values[0].shape[0]
    if any(projector.shape != (dimension, dimension) for projector in values):
        raise DimensionError("all packet projectors must act on one common space")
    for left_index, left in enumerate(values):
        for right_index, right in enumerate(values):
            expected = left if left_index == right_index else np.zeros_like(left)
            if not np.allclose(left @ right, expected, atol=tolerance, rtol=0.0):
                raise ValidationError("packet projectors must be pairwise orthogonal")
    if not np.allclose(sum(values), np.eye(dimension), atol=tolerance, rtol=0.0):
        raise ValidationError("packet projectors must sum to the identity")
    return values


__all__ = [
    "as_matrix",
    "as_square_matrix",
    "as_vector",
    "density_matrix",
    "orthogonal_packet_family",
    "require_hermitian",
    "require_projector",
    "require_unitary",
]
