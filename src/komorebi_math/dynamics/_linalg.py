"""Small matrix functions used by the finite dynamics layer."""

from __future__ import annotations

from math import ceil, log2
from typing import Any, Literal

import numpy as np

from ..exceptions import BackendUnavailableError, DimensionError, ValidationError


def as_square_matrix(matrix: Any, *, name: str = "matrix") -> np.ndarray:
    """Validate and return a finite square matrix."""

    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise DimensionError(f"{name} must be square")
    if value.size and not np.all(np.isfinite(value)):
        raise ValidationError(f"{name} must contain finite values")
    return value


def matrix_exponential(
    matrix: Any,
    *,
    tolerance: float = 1e-15,
    max_terms: int = 256,
    method: Literal["auto", "scipy", "taylor"] = "auto",
) -> np.ndarray:
    """Return a finite matrix exponential.

    ``auto`` uses SciPy when it is installed and otherwise uses the NumPy only
    scaling and Taylor implementation. ``taylor`` forces the base installation
    path so that it can be tested independently of optional dependencies.
    """

    value = as_square_matrix(matrix, name="matrix exponential input")
    tol = float(tolerance)
    terms = int(max_terms)
    if not np.isfinite(tol) or tol <= 0:
        raise ValidationError("matrix exponential tolerance must be positive and finite")
    if isinstance(max_terms, bool) or terms != max_terms or terms < 1:
        raise ValidationError("max_terms must be a positive integer")
    if method not in {"auto", "scipy", "taylor"}:
        raise ValidationError("method must be auto, scipy, or taylor")

    if method in {"auto", "scipy"}:
        try:
            from scipy.linalg import expm
        except ImportError:
            if method == "scipy":
                raise BackendUnavailableError("SciPy is required for method='scipy'") from None
        else:
            return np.asarray(expm(value))

    dtype = np.result_type(value.dtype, np.complex128)
    value = value.astype(dtype, copy=False)
    size = value.shape[0]
    norm_one = float(np.linalg.norm(value, 1))
    scale_power = max(0, int(ceil(log2(norm_one / 0.5)))) if norm_one > 0.5 else 0
    scaled = value / (2**scale_power)
    result = np.eye(size, dtype=dtype)
    term = np.eye(size, dtype=dtype)
    for index in range(1, terms + 1):
        term = term @ scaled / index
        result = result + term
        if np.linalg.norm(term, 1) <= tol * max(1.0, np.linalg.norm(result, 1)):
            break
    else:
        raise ArithmeticError("matrix exponential Taylor fallback did not converge")
    for _ in range(scale_power):
        result = result @ result
    return np.real_if_close(result, tol=1000)


def hermitian_fractional_power(
    matrix: Any,
    exponent: float,
    *,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a Hermitian fractional power with its spectral decomposition."""

    value = as_square_matrix(matrix, name="fractional operator")
    if not np.allclose(value, value.conj().T, atol=tolerance, rtol=0.0):
        raise ValidationError("fractional spectral calculus requires a Hermitian matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValidationError("fractional spectral calculus requires a positive semidefinite matrix")
    clipped = np.maximum(eigenvalues, 0.0)
    powered = eigenvectors @ np.diag(clipped ** float(exponent)) @ eigenvectors.conj().T
    return np.real_if_close(powered, tol=1000), clipped, eigenvectors


def block_partition(
    matrix: Any,
    retained: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Partition a square matrix into retained and hidden coordinate blocks."""

    value = as_square_matrix(matrix, name="partitioned operator")
    n = value.shape[0]
    retained = tuple(int(index) for index in retained)
    if len(set(retained)) != len(retained) or any(index < 0 or index >= n for index in retained):
        raise DimensionError("retained indices must be distinct valid matrix indices")
    hidden = tuple(index for index in range(n) if index not in retained)
    if not retained or not hidden:
        raise DimensionError("both retained and hidden sectors must be nonempty")
    pp = value[np.ix_(retained, retained)]
    pq = value[np.ix_(retained, hidden)]
    qp = value[np.ix_(hidden, retained)]
    qq = value[np.ix_(hidden, hidden)]
    return pp, pq, qp, qq, hidden


__all__ = ["as_square_matrix", "block_partition", "hermitian_fractional_power", "matrix_exponential"]
