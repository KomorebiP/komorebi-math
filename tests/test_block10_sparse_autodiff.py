from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from komorebi_math import Exactness
from komorebi_math.exceptions import BackendUnavailableError, ValidationError
from komorebi_math.nonlinear import (
    AutomaticDifferentiationProvider,
    JaxDerivativeProvider,
    SparsePolynomialDerivativeProvider,
    SparseSymmetricTensor,
    TorchDerivativeProvider,
    autodiff_provider,
    available_ad_frameworks,
)
from tests.optional_backend_support import run_isolated_ad_backend_smoke


def symmetric_cubic_tensor() -> np.ndarray:
    tensor = np.zeros((2, 2, 2, 2), dtype=float)
    tensor[0, 0, 0, 0] = 2.0
    for indices in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        tensor[(0, *indices)] = 3.0
    for indices in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        tensor[(1, *indices)] = -4.0
    tensor[1, 1, 1, 1] = 5.0
    return tensor


def test_sparse_symmetric_tensor_roundtrip_and_contraction() -> None:
    dense = symmetric_cubic_tensor()
    sparse = SparseSymmetricTensor.from_dense(dense)
    sparse.certificate().require()
    assert sparse.order == 3
    assert sparse.stored_entry_count == 4
    assert sparse.dense_entry_count == 16
    assert sparse.storage_ratio == pytest.approx(0.25)
    assert np.array_equal(sparse.to_dense(), dense)

    u = np.array([0.5, -1.0])
    v = np.array([2.0, 0.25])
    w = np.array([-0.75, 3.0])
    expected = np.einsum("oijk,i,j,k->o", dense, u, v, w)
    assert np.allclose(sparse.contract((u, v, w)), expected)


def test_sparse_tensor_rejects_nonsymmetric_dense_input() -> None:
    dense = np.zeros((1, 2, 2), dtype=float)
    dense[0, 0, 1] = 1.0
    with pytest.raises(ValidationError, match="not symmetric"):
        SparseSymmetricTensor.from_dense(dense)


def test_sparse_order_zero_tensor_exact_contraction() -> None:
    sparse = SparseSymmetricTensor.from_dense(
        np.array([Fraction(2, 3), Fraction(-5, 7)], dtype=object),
        source_dimension=3,
    )
    assert sparse.order == 0
    assert sparse.source_dimension == 3
    value = sparse.contract((), exact=True)
    assert tuple(value) == (Fraction(2, 3), Fraction(-5, 7))


def test_sparse_polynomial_provider_matches_dense_exactly() -> None:
    tensors = {
        0: np.array([Fraction(1, 2), Fraction(-1, 3)], dtype=object),
        1: np.array(
            [
                [Fraction(2), Fraction(-1)],
                [Fraction(3, 2), Fraction(4)],
            ],
            dtype=object,
        ),
        2: np.array(
            [
                [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(-3)]],
                [[Fraction(0), Fraction(5)], [Fraction(5), Fraction(4)]],
            ],
            dtype=object,
        ),
    }
    from komorebi_math.nonlinear import PolynomialDerivativeProvider

    dense = PolynomialDerivativeProvider(tensors, exactness=Exactness.EXACT)
    sparse = SparsePolynomialDerivativeProvider.from_dense_provider(dense)
    point = np.array([Fraction(2, 5), Fraction(-3, 7)], dtype=object)
    directions = (
        np.array([Fraction(1, 3), Fraction(4, 5)], dtype=object),
        np.array([Fraction(-2, 9), Fraction(7, 11)], dtype=object),
    )
    assert np.array_equal(sparse.value(point), dense.value(point))
    for order in (1, 2):
        assert np.array_equal(
            sparse.derivative(order, point, directions[:order]),
            dense.derivative(order, point, directions[:order]),
        )
    sparse.storage_certificate((point,), directions).require()


def test_exact_dense_provider_normalizes_zero_tensor_entries() -> None:
    from komorebi_math.nonlinear import PolynomialDerivativeProvider

    provider = PolynomialDerivativeProvider(
        {
            0: np.zeros(1),
            1: np.array([[Fraction(3, 5)]], dtype=object),
            2: np.zeros((1, 1, 1)),
        },
        exactness=Exactness.EXACT,
    )
    point = np.array([Fraction(2, 7)], dtype=object)
    direction = np.array([Fraction(-4, 9)], dtype=object)

    value = provider.value(point)[0]
    derivative = provider.derivative(1, point, (direction,))[0]
    assert value == Fraction(6, 35)
    assert derivative == Fraction(-4, 15)
    assert isinstance(value, Fraction)
    assert isinstance(derivative, Fraction)


def test_sparse_mapping_constructor_and_storage_counts() -> None:
    tensor = SparseSymmetricTensor.from_mapping(
        2,
        3,
        1,
        {
            (0, 0, 0): 2.0,
            (0, 0, 2): -1.0,
            (0, 1, 2): 4.0,
        },
    )
    assert tensor.stored_entry_count == 3
    assert tensor.expanded_nonzero_count == 1 + 2 + 2
    with pytest.raises(ValidationError, match="nondecreasing"):
        SparseSymmetricTensor.from_mapping(2, 3, 1, {(0, 2, 0): 1.0})


def test_framework_neutral_ad_provider_first_directional_derivative() -> None:
    def function(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 3 + x[0] * x[1], np.sin(x[1])])

    def finite_jvp(function_, point, direction):
        step = 1e-6
        primal = function_(point)
        tangent = (function_(point + step * direction) - function_(point - step * direction)) / (2 * step)
        return primal, tangent

    provider = AutomaticDifferentiationProvider(
        function,
        finite_jvp,
        source_dimension=2,
        target_dimension=2,
        max_order=1,
    )
    point = np.array([0.7, -0.2])
    direction = np.array([0.4, 1.3])
    expected = np.array(
        [
            (3 * point[0] ** 2 + point[1]) * direction[0] + point[0] * direction[1],
            np.cos(point[1]) * direction[1],
        ]
    )
    assert np.allclose(provider.derivative(1, point, (direction,)), expected, atol=1e-8)
    provider.validation_certificate(point, (direction,), tolerance=1e-7).require()


def test_autodiff_factory_and_availability_contract() -> None:
    availability = available_ad_frameworks()
    assert set(availability) == {"jax", "torch"}
    with pytest.raises(ValidationError, match="unknown"):
        autodiff_provider("unknown", lambda x: x, 1, 1)


@pytest.mark.optional_backend
def test_jax_adapter_through_third_order() -> None:
    run_isolated_ad_backend_smoke("jax", "block10")


@pytest.mark.optional_backend
def test_torch_adapter_through_third_order() -> None:
    run_isolated_ad_backend_smoke("torch", "block10")



def test_unavailable_framework_raises_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JaxDerivativeProvider, "available", staticmethod(lambda: False))
    provider = JaxDerivativeProvider(lambda x: x, 1, 1)
    with pytest.raises(BackendUnavailableError, match="not installed"):
        provider.value(np.array([1.0]))
