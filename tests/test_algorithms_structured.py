from __future__ import annotations

import numpy as np

from komorebi_math.algorithms import (
    BlockDiagonalOperator,
    CirculantOperator,
    DenseOperator,
    DiagonalOperator,
    IdentityOperator,
    KroneckerOperator,
    ToeplitzOperator,
    WalshHadamardOperator,
)


def test_dense_diagonal_identity() -> None:
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    dense = DenseOperator(matrix)
    x = np.array([1.0, 2.0])
    assert np.allclose(dense.apply(x), matrix @ x)
    diagonal = DiagonalOperator(np.array([2.0, 3.0]))
    assert np.allclose(diagonal.solve(diagonal.apply(x)), x)
    identity = IdentityOperator(2, 4.0)
    assert np.allclose(identity.solve(identity.apply(x)), x)


def test_block_diagonal_apply_solve_and_adjoint() -> None:
    blocks = BlockDiagonalOperator((np.array([[2.0]]), np.array([[3.0, 1.0], [1.0, 2.0]])))
    matrix = blocks.materialize()
    x = np.arange(1.0, 4.0)
    assert np.allclose(blocks.apply(x), matrix @ x)
    assert np.allclose(blocks.solve(matrix @ x), x)
    assert np.allclose(blocks.adjoint().materialize(), matrix.conj().T)


def test_circulant_matches_dense_for_vector_and_matrix() -> None:
    rng = np.random.default_rng(2)
    column = rng.normal(size=7)
    operator = CirculantOperator(column)
    x = rng.normal(size=7)
    rhs = rng.normal(size=(7, 3))
    assert np.allclose(operator.apply(x), operator.materialize() @ x)
    assert np.allclose(operator.apply(rhs), operator.materialize() @ rhs)


def test_toeplitz_matches_dense_rectangular() -> None:
    column = np.array([1.0, 2.0, 3.0, 4.0])
    row = np.array([1.0, -1.0, 0.5])
    operator = ToeplitzOperator(column, row)
    x = np.array([2.0, 3.0, 4.0])
    assert operator.shape == (4, 3)
    assert np.allclose(operator.apply(x), operator.materialize() @ x)
    assert np.allclose(operator.adjoint().materialize(), operator.materialize().conj().T)


def test_kronecker_modewise_apply() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 2))
    b = rng.normal(size=(4, 3))
    operator = KroneckerOperator(a, b)
    x = rng.normal(size=6)
    assert np.allclose(operator.apply(x), np.kron(a, b) @ x)


def test_walsh_hadamard() -> None:
    rng = np.random.default_rng(4)
    operator = WalshHadamardOperator(8, normalized=True)
    x = rng.normal(size=8)
    y = operator.apply(x)
    assert np.allclose(operator.apply(y), x)
    assert np.allclose(operator.materialize().T @ operator.materialize(), np.eye(8))
    assert operator.flop_count() == 24

def test_walsh_normalized_integer_input_and_strict_dimension() -> None:
    operator = WalshHadamardOperator(4, normalized=True, dtype_value=int)
    value = np.array([1, 2, 3, 4], dtype=int)
    transformed = operator.apply(value)
    assert np.issubdtype(transformed.dtype, np.inexact)
    assert np.allclose(operator.apply(transformed), value)
    matrix = operator.materialize()
    assert np.issubdtype(matrix.dtype, np.inexact)
    assert np.allclose(matrix.T.conj() @ matrix, np.eye(4))

    import pytest
    from komorebi_math.exceptions import ValidationError

    with pytest.raises(ValidationError):
        IdentityOperator(2.5)
    with pytest.raises(ValidationError):
        WalshHadamardOperator(4.0)
    with pytest.raises(ValidationError):
        WalshHadamardOperator(True)

