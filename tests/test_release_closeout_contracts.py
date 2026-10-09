"""Small controls for exact evidence and nested portable scalar boundaries."""

from fractions import Fraction

import numpy as np
import pytest

from komorebi_math.certificates import Certificate, Exactness
from komorebi_math.exceptions import ValidationError
from komorebi_math.export.persistence import (
    dumps_portable,
    from_portable_data,
    loads_portable,
    portable_digest,
    to_portable_data,
)
from komorebi_math.nonlinear.autodiff import AutomaticDifferentiationProvider


@pytest.mark.parametrize("integer_type", [np.int8, np.int64, np.uint64])
@pytest.mark.parametrize("as_certificate", [False, True])
def test_numpy_fraction_components_roundtrip_as_native_exact_integers(integer_type, as_certificate):
    fraction = Fraction(integer_type(5), integer_type(3))
    original = (Certificate(True, Exactness.EXACT, {"identity": fraction})
                if as_certificate else fraction)
    encoded = to_portable_data(original)
    for restored in (from_portable_data(encoded), loads_portable(dumps_portable(original))):
        actual = restored.residuals["identity"] if as_certificate else restored
        assert actual == Fraction(5, 3)
        assert type(actual.numerator) is int
        assert type(actual.denominator) is int
        if as_certificate:
            assert restored.recorded_valid
            assert not restored.current_valid


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
@pytest.mark.parametrize("cell", [[], [1], [1, 2], (), (1,), (1, 2),
                                np.array([1]), np.array([1, 2])])
def test_nested_sequence_is_one_object_array_cell(shape, cell):
    original = np.empty(shape, dtype=object)
    original.flat[0] = cell
    for restored in (from_portable_data(to_portable_data(original)),
                     loads_portable(dumps_portable(original))):
        assert restored.shape == shape
        assert restored.dtype == np.dtype(object)
        actual = restored.flat[0]
        assert type(actual) is type(cell)
        if isinstance(cell, np.ndarray):
            assert actual.dtype == cell.dtype
            assert np.array_equal(actual, cell)
        else:
            assert actual == cell
        assert portable_digest(restored) == portable_digest(original)


def _asymmetric_provider(scale, exactness):
    # This deliberately faulty primitive has a hand-computable commutator.
    # At x=0, applying directions (1,2) gives 2*scale while (2,1) gives
    # 4*scale. The certificate must reject this finite symmetry mismatch.
    def jvp(function, point, direction):
        return (function(point), function(point + direction) - function(point)
                + scale * direction * direction * point)
    return AutomaticDifferentiationProvider(
        lambda point: point * 0,
        jvp,
        1,
        1,
        max_order=2,
        exactness=exactness,
    )


@pytest.mark.parametrize("scale", [Fraction(1, 10**400), Fraction(10**400)])
def test_exact_symmetry_rejects_nonzero_rational_difference_outside_float_range(scale):
    provider = _asymmetric_provider(scale, Exactness.EXACT)
    point = np.array([Fraction(0)], dtype=object)
    directions = (np.array([Fraction(1)], dtype=object),
                  np.array([Fraction(2)], dtype=object))
    left = provider.derivative(2, point, directions)[0]
    right = provider.derivative(2, point, directions[::-1])[0]
    assert left == 2 * scale
    assert right == 4 * scale
    assert left - right == -2 * scale != 0
    certificate = provider.validation_certificate(point, directions)
    assert not certificate.current_valid
    assert certificate.residuals["directional_symmetry"] == 1.0
    with pytest.raises(ValidationError):
        certificate.require()


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_numerical_symmetry_norm_preserves_tiny_and_large_finite_differences(scale):
    provider = _asymmetric_provider(scale, Exactness.NUMERICAL)
    point = np.array([0.0])
    directions = (np.array([1.0]), np.array([2.0]))
    certificate = provider.validation_certificate(point, directions, tolerance=0.0)
    assert not certificate.current_valid
    residual = certificate.residuals["directional_symmetry"]
    assert np.isfinite(residual)
    assert residual == pytest.approx(2 * scale, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("exactness", [Exactness.EXACT, Exactness.NUMERICAL])
def test_zero_symmetry_discrepancy_still_passes(exactness):
    provider = _asymmetric_provider(Fraction(0), exactness)
    provider.validation_certificate(
        np.array([Fraction(0)], dtype=object),
        (np.array([Fraction(1)], dtype=object), np.array([Fraction(2)], dtype=object)),
        tolerance=0.0,
    ).require()
