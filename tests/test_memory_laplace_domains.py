"""Independent finite integrals and domain tests for finite memory transforms."""
from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.dynamics import MemoryKernel
from komorebi_math.exceptions import ValidationError


def test_infinite_laplace_rejects_a_divergent_integral_but_continuation_is_explicit() -> None:
    kernel = MemoryKernel([[1.]], [[1.]], [[1.]])
    with pytest.raises(ValidationError, match="spectral abscissa"):
        kernel.laplace(0.)
    certificate = kernel.certificate(0., horizon=1.)
    assert not certificate.valid
    assert certificate.assumptions_unchecked
    assert kernel.rational_transform(0.)[0, 0] == pytest.approx(-1.)
    assert kernel.laplace(2.)[0, 0] == pytest.approx(1.)
    kernel.certificate(2., horizon=1.).require()


@pytest.mark.parametrize("horizon", [0., .1, 2., 10.])
def test_finite_laplace_at_scalar_resonance_is_horizon(horizon: float) -> None:
    kernel = MemoryKernel([[1.]], [[1.]], [[1.]])
    assert kernel.finite_laplace(1., horizon)[0, 0] == pytest.approx(horizon)


def test_finite_laplace_at_jordan_resonance_integrates_polynomial() -> None:
    kernel = MemoryKernel([[1., 0.]], [[1., 1.], [0., 1.]], [[0.], [1.]])
    # exp(-t) K(t) = t, so the finite integral is h^2/2.
    assert kernel.finite_laplace(1., 2.)[0, 0] == pytest.approx(2.)


@pytest.mark.parametrize("parameter", [2., 2. + .7j, -.4 + .2j])
def test_finite_laplace_matches_independent_scalar_antiderivative(parameter: complex) -> None:
    kernel = MemoryKernel([[2.]], [[1.]], [[3.]])
    horizon = .73
    exponent = 1. - parameter
    expected = 6. * np.expm1(exponent * horizon) / exponent
    assert kernel.finite_laplace(parameter, horizon)[0, 0] == pytest.approx(expected)


def test_memory_matrices_are_snapshots_of_constructor_inputs() -> None:
    b, d, c = (np.array([[value]]) for value in (1., -1., 2.))
    kernel = MemoryKernel(b, d, c)
    b[:] = np.nan
    d[:] = np.nan
    c[:] = np.nan
    assert kernel.at(0.)[0, 0] == pytest.approx(2.)
    kernel.certificate(1.).require()


@pytest.mark.parametrize("parameter", [float("nan"), float("inf"), complex(1., float("inf"))])
def test_memory_transform_rejects_nonfinite_parameter(parameter: complex) -> None:
    kernel = MemoryKernel([[1.]], [[-1.]], [[1.]])
    for method in (kernel.laplace, kernel.rational_transform, kernel.certificate):
        with pytest.raises(ValidationError, match="finite"):
            method(parameter)
    with pytest.raises(ValidationError, match="finite"):
        kernel.finite_laplace(parameter, 1.)


@pytest.mark.parametrize("horizon", [-1., float("nan"), float("inf")])
def test_memory_certificate_validates_horizon(horizon: float) -> None:
    kernel = MemoryKernel([[1.]], [[-1.]], [[1.]])
    with pytest.raises(ValidationError, match="horizon"):
        kernel.certificate(1., horizon=horizon)
    with pytest.raises(ValidationError, match="horizon"):
        kernel.finite_laplace(1., horizon)
