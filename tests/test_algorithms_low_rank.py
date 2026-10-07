from __future__ import annotations

import numpy as np

from komorebi_math.algorithms import (
    BlockDiagonalLowRankOperator,
    BlockDiagonalOperator,
    LowRankOperator,
    accumulated_recompression_bound,
    product_chain_error_bound,
    recompress_core,
    recompress_factors,
)


def test_low_rank_apply_and_adjoint() -> None:
    rng = np.random.default_rng(5)
    left = rng.normal(size=(7, 3))
    right = rng.normal(size=(5, 3))
    operator = LowRankOperator(left, right)
    x = rng.normal(size=5)
    assert np.allclose(operator.apply(x), operator.materialize() @ x)
    assert np.allclose(operator.adjoint().materialize(), operator.materialize().conj().T)


def test_recompress_factors_matches_truncated_svd() -> None:
    rng = np.random.default_rng(6)
    left = rng.normal(size=(12, 5))
    right = rng.normal(size=(10, 5))
    result = recompress_factors(left, right, 2)
    result.certificate.require()
    matrix = left @ right.T
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert np.isclose(result.spectral_error, singular_values[2])
    assert np.isclose(np.linalg.norm(matrix - result.materialize(), 2), singular_values[2])


def test_recompress_rectangular_core_and_rank_deficiency() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(size=(9, 4))
    left[:, 3] = left[:, 0] + left[:, 1]
    right = rng.normal(size=(8, 3))
    core = rng.normal(size=(4, 3))
    result = recompress_core(left, core, right, 3)
    result.certificate.require()
    assert np.allclose(result.materialize(), left @ core @ right.T, atol=1e-10)


def test_bdlr_apply_solve_and_product() -> None:
    rng = np.random.default_rng(8)
    diagonal_a = BlockDiagonalOperator((np.array([[3.0, 0.2], [0.2, 2.0]]), np.array([[4.0]])))
    diagonal_b = BlockDiagonalOperator((np.array([[2.0, 0.1], [0.1, 3.0]]), np.array([[5.0]])))
    ua = rng.normal(size=(3, 1)) * 0.1
    va = rng.normal(size=(3, 1)) * 0.1
    ub = rng.normal(size=(3, 2)) * 0.1
    vb = rng.normal(size=(3, 2)) * 0.1
    a = BlockDiagonalLowRankOperator.from_factors(diagonal_a, ua, va)
    b = BlockDiagonalLowRankOperator.from_factors(diagonal_b, ub, vb)
    x = rng.normal(size=3)
    assert np.allclose(a.apply(x), a.materialize() @ x)
    assert np.allclose(a.solve(a.materialize() @ x), x)
    product = a.exact_product(b)
    assert np.allclose(product.materialize(), a.materialize() @ b.materialize())


def test_bdlr_recompress() -> None:
    rng = np.random.default_rng(9)
    diagonal = BlockDiagonalOperator((np.eye(3), np.eye(2)))
    left = rng.normal(size=(5, 4))
    right = rng.normal(size=(5, 4))
    operator = BlockDiagonalLowRankOperator.from_factors(diagonal, left, right)
    compressed, result = operator.recompress(2)
    result.certificate.require()
    assert np.linalg.norm(operator.materialize() - compressed.materialize(), 2) <= result.spectral_error + 1e-9


def test_product_error_bound() -> None:
    rng = np.random.default_rng(10)
    exact = [rng.normal(size=(3, 3)) for _ in range(4)]
    approximate = [matrix + 1e-4 * rng.normal(size=(3, 3)) for matrix in exact]
    exact_product = np.eye(3)
    approximate_product = np.eye(3)
    for matrix in exact:
        exact_product = matrix @ exact_product
    for matrix in approximate:
        approximate_product = matrix @ approximate_product
    bound = product_chain_error_bound(exact, approximate)
    assert np.linalg.norm(approximate_product - exact_product, 2) <= bound + 1e-10


def test_accumulated_recompression_bound() -> None:
    bound = accumulated_recompression_bound([2.0, 3.0, 4.0], [0.1, 0.2, 0.3], initial_error=0.5)
    assert np.isclose(bound, 4.0 * (3.0 * (2.0 * 0.5 + 0.1) + 0.2) + 0.3)

def test_low_rank_strict_rank_and_chain_validation() -> None:
    import pytest
    from komorebi_math.exceptions import DimensionError, ValidationError

    left = np.eye(3)
    right = np.eye(3)
    with pytest.raises(ValidationError):
        recompress_factors(left, right, 1.5)
    with pytest.raises(ValidationError):
        recompress_factors(left, right, True)
    with pytest.raises(DimensionError):
        product_chain_error_bound(
            [np.zeros((2, 3)), np.zeros((4, 5))],
            [np.zeros((2, 3)), np.zeros((4, 5))],
        )

