from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from komorebi_math import Exactness
from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.nonlinear.autodiff import (
    AutomaticDifferentiationProvider,
    JaxDerivativeProvider,
    TorchDerivativeProvider,
    autodiff_provider,
    available_ad_frameworks,
)
from komorebi_math.nonlinear.derivatives import PolynomialDerivativeProvider
from komorebi_math.nonlinear.sparse import (
    SparsePolynomialDerivativeProvider,
    SparseSymmetricTensor,
    SparseTensorEntry,
)
from tests.optional_backend_support import run_isolated_ad_backend_smoke


def _symmetric_tensor(order: int, source: int, target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tensor = np.zeros((target, *(source for _ in range(order))), dtype=float)
    from itertools import combinations_with_replacement, permutations

    for output in range(target):
        for indices in combinations_with_replacement(range(source), order):
            value = float(rng.normal())
            for permutation in set(permutations(indices)):
                tensor[(output, *permutation)] = value
    return tensor


def test_sparse_symmetric_tensor_round_trip_and_contraction() -> None:
    dense = _symmetric_tensor(3, 3, 2, seed=11)
    dense[np.abs(dense) < 0.4] = 0.0
    sparse = SparseSymmetricTensor.from_dense(dense)

    assert np.array_equal(sparse.to_dense(), dense)
    sparse.certificate().require()
    assert sparse.stored_entry_count <= sparse.expanded_nonzero_count
    assert sparse.storage_ratio < 1.0

    directions = (
        np.array([0.5, -0.2, 0.8]),
        np.array([-0.4, 0.9, 0.1]),
        np.array([0.3, 0.7, -0.6]),
    )
    expected = dense
    for direction in directions:
        expected = np.tensordot(expected, direction, axes=(1, 0))
    assert np.allclose(sparse.contract(directions), expected)


def test_sparse_tensor_validates_orbits_and_dense_symmetry() -> None:
    with pytest.raises(ValidationError, match="nondecreasing"):
        SparseSymmetricTensor(2, 2, 1, (SparseTensorEntry(0, (1, 0), 3.0),))

    nonsymmetric = np.zeros((1, 2, 2))
    nonsymmetric[0, 0, 1] = 1.0
    with pytest.raises(ValidationError, match="not symmetric"):
        SparseSymmetricTensor.from_dense(nonsymmetric)


def test_sparse_provider_matches_dense_provider_through_order_three() -> None:
    tensors = {
        0: np.array([1.5, -0.25]),
        1: _symmetric_tensor(1, 3, 2, seed=12),
        2: _symmetric_tensor(2, 3, 2, seed=13),
        3: _symmetric_tensor(3, 3, 2, seed=14),
    }
    dense = PolynomialDerivativeProvider(tensors, source_dimension=3, target_dimension=2)
    sparse = SparsePolynomialDerivativeProvider.from_dense_provider(dense)
    point = np.array([0.2, -0.4, 0.7])
    directions = (
        np.array([0.5, 0.1, -0.3]),
        np.array([-0.2, 0.8, 0.4]),
        np.array([0.6, -0.7, 0.2]),
    )

    assert np.allclose(sparse.value(point), dense.value(point))
    for order in range(1, 4):
        assert np.allclose(
            sparse.derivative(order, point, directions[:order]),
            dense.derivative(order, point, directions[:order]),
        )
    sparse.storage_certificate((point,), directions).require()
    assert sparse.to_dense_provider().source_dimension == 3


def test_sparse_provider_infers_constant_source_dimension_from_higher_order() -> None:
    quadratic = np.zeros((2, 3, 3))
    quadratic[0, 0, 0] = 2.0
    quadratic[1, 2, 2] = -4.0
    provider = SparsePolynomialDerivativeProvider(
        {0: np.array([1.0, 2.0]), 2: quadratic}
    )
    assert provider.source_dimension == 3
    assert provider.target_dimension == 2
    assert np.allclose(provider.value([2.0, 0.0, 3.0]), [5.0, -16.0])


def test_sparse_exact_fraction_provider_preserves_exactness() -> None:
    tensor = np.zeros((1, 2, 2), dtype=object)
    tensor[0, 0, 0] = Fraction(2, 3)
    tensor[0, 0, 1] = tensor[0, 1, 0] = Fraction(1, 5)
    provider = SparsePolynomialDerivativeProvider(
        {2: tensor},
        source_dimension=2,
        exactness=Exactness.EXACT,
    )
    value = provider.derivative(
        2,
        np.array([0, 0], dtype=object),
        (
            np.array([Fraction(1, 2), Fraction(2, 3)], dtype=object),
            np.array([Fraction(3, 4), Fraction(1, 7)], dtype=object),
        ),
    )
    assert value.dtype == object
    assert value[0] == Fraction(51, 140)


def test_exact_dense_realization_preserves_structural_zero_terms() -> None:
    zero_constant = SparseSymmetricTensor.from_mapping(0, 2, 1, {})
    linear = SparseSymmetricTensor.from_mapping(
        1,
        2,
        1,
        {
            (0, 0): Fraction(3, 5),
            (0, 1): Fraction(-7, 9),
        },
    )
    sparse = SparsePolynomialDerivativeProvider(
        {0: zero_constant, 1: linear},
        exactness=Exactness.EXACT,
    )
    dense = sparse.to_dense_provider()
    point = np.array([Fraction(2, 3), Fraction(-4, 7)], dtype=object)

    assert all(tensor.dtype == object for tensor in dense.tensors.values())
    assert np.array_equal(sparse.value(point), dense.value(point))
    assert dense.value(point)[0] == Fraction(266, 315)


def test_framework_neutral_first_order_jvp_adapter() -> None:
    def function(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + 3 * x[1]])

    def central_jvp(function_, point, direction):
        step = 1e-6
        primal = function_(point)
        tangent = (function_(point + step * direction) - function_(point - step * direction)) / (
            2 * step
        )
        return primal, tangent

    provider = AutomaticDifferentiationProvider(
        function,
        central_jvp,
        source_dimension=2,
        target_dimension=1,
        max_order=1,
    )
    point = np.array([0.7, -0.4])
    direction = np.array([0.2, 0.9])
    assert np.allclose(provider.value(point), function(point))
    assert np.allclose(
        provider.derivative(1, point, (direction,)),
        [2 * point[0] * direction[0] + 3 * direction[1]],
        atol=1e-8,
    )


def _expected_polynomial_derivative(
    order: int,
    point: np.ndarray,
    directions: tuple[np.ndarray, ...],
) -> np.ndarray:
    x, y = point
    if order == 1:
        (u,) = directions
        return np.array([3 * x * x * u[0] + y * u[0] + x * u[1], 3 * y * y * u[1]])
    if order == 2:
        u, v = directions
        return np.array(
            [6 * x * u[0] * v[0] + u[0] * v[1] + u[1] * v[0], 6 * y * u[1] * v[1]]
        )
    if order == 3:
        u, v, w = directions
        return np.array([6 * u[0] * v[0] * w[0], 6 * u[1] * v[1] * w[1]])
    raise AssertionError(order)


@pytest.mark.optional_backend
def test_jax_adapter_matches_polynomial_through_order_three() -> None:
    run_isolated_ad_backend_smoke("jax", "nonlinear")


@pytest.mark.optional_backend
def test_torch_adapter_matches_polynomial_through_order_three() -> None:
    run_isolated_ad_backend_smoke("torch", "nonlinear")



def test_autodiff_factory_and_availability_contract() -> None:
    availability = available_ad_frameworks()
    assert set(availability) == {"jax", "torch"}
    assert isinstance(availability["jax"], bool)
    assert isinstance(availability["torch"], bool)

    provider = autodiff_provider(
        "torch",
        lambda x: x,
        source_dimension=2,
        target_dimension=2,
        max_order=1,
    )
    assert isinstance(provider, TorchDerivativeProvider)
    with pytest.raises(ValidationError, match="unknown"):
        autodiff_provider("unknown", lambda x: x, 1, 1)
    with pytest.raises(DimensionError):
        provider.derivative(1, [0.0], ([1.0],))
