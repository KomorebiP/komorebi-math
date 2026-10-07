"""Private ownership helpers for certified finite objects.

Constructors that retain matrices must own a snapshot of caller data.  NumPy
arrays are additionally marked read-only so cached derived data cannot be
invalidated by mutating a public array reference.  This module is private and
does not add a public API surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def numpy_snapshot(value: Any, *, dtype: Any = complex) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def backend_matrix_snapshot(backend: Any, value: Any) -> Any:
    matrix = backend.copy(backend.as_matrix(value))
    if isinstance(matrix, np.ndarray):
        matrix.setflags(write=False)
    return matrix
