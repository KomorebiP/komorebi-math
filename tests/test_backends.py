from __future__ import annotations

import numpy as np
import sympy as sp

from komorebi_math.backends import available_backends, get_backend


def test_numpy_backend_linear_algebra() -> None:
    backend = get_backend("numpy")
    matrix = backend.as_matrix([[1, 2], [3, 4]])
    assert backend.shape(matrix) == (2, 2)
    assert backend.rank(matrix) == 2
    assert np.allclose(backend.matmul(matrix, backend.eye(2)), matrix)
    assert np.allclose(backend.inverse(matrix) @ matrix, np.eye(2))


def test_numpy_backend_nullspace() -> None:
    backend = get_backend("numpy")
    matrix = backend.as_matrix([[1, 1, 0], [0, 0, 1]])
    basis = backend.nullspace(matrix)
    assert len(basis) == 1
    assert np.linalg.norm(matrix @ basis[0]) < 1e-10


def test_sympy_backend_exact() -> None:
    backend = get_backend("sympy")
    matrix = backend.as_matrix([[1, 2], [2, 4]])
    assert backend.exact
    assert backend.rank(matrix) == 1
    assert backend.is_zero(matrix - sp.Matrix([[1, 2], [2, 4]]))
    nullspace = backend.nullspace(matrix)
    assert len(nullspace) == 1
    assert matrix * nullspace[0] == sp.zeros(2, 1)


def test_backend_availability_report() -> None:
    report = available_backends()
    assert report["numpy"]
    assert report["sympy"]
    assert set(report) == {"numpy", "sympy", "scipy", "mpmath", "cupy"}
