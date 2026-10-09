"""Independent controls for numerical defects found in the release stress run."""
from fractions import Fraction
from math import factorial

import mpmath
import numpy as np
import pytest

from komorebi_math.exceptions import DimensionError, ResourceLimitError, ValidationError
from komorebi_math.quantum.channels import QuantumChannel
from komorebi_math.quantum.dirac import central_mode_momentum
from komorebi_math.quantum.lindblad import LindbladGenerator
from komorebi_math.quantum.spectral import (
    heat_trace, low_mode_dispersion_bound, periodic_laplacian, periodic_lattice_eigenvalues,
)
from komorebi_math.special_functions.shape import (
    beta_binomial_refinement, gamma_sech_ambiguity, log_gamma_window,
    meixner_characteristic, meixner_density, shape_affinity,
    shape_refinement_coefficient, shape_transfer_singular_value,
)


def rising_exact(value, degree):
    result = Fraction(1)
    for step in range(degree):
        result *= value + step
    return result


def assert_mpf_relative(actual, expected, tolerance="1e-45"):
    ref = mpmath.mp.clone()
    ref.dps = 90
    assert abs(ref.mpf(str(actual)) / expected - 1) < ref.mpf(tolerance)


@pytest.mark.parametrize("exponent", [0, 10, 30, 50, 100, 300])
@pytest.mark.parametrize("orders", [(1, 0), (2, 3), (0, 0), (4, 5)])
def test_huge_shape_refinement_matches_exact_rational_product(exponent, orders):
    scale = 10 ** exponent
    shapes = (scale, 3 * scale)
    n = sum(orders)
    square = Fraction(factorial(n), 1) / rising_exact(sum(shapes), n)
    for shape, degree in zip(shapes, orders, strict=True):
        square *= rising_exact(shape, degree) / factorial(degree)
    ref = mpmath.mp.clone()
    ref.dps = 90
    expected = ref.sqrt(ref.mpf(square.numerator) / square.denominator)
    assert_mpf_relative(shape_refinement_coefficient(shapes, orders), expected)


@pytest.mark.parametrize("exponent", [0, 10, 30, 50, 100, 300])
@pytest.mark.parametrize("degree", [0, 1, 4, 12])
def test_huge_shape_transfer_matches_exact_rational_product(exponent, degree):
    shape = 10 ** exponent
    square = rising_exact(shape, degree) / rising_exact(4 * shape, degree)
    ref = mpmath.mp.clone()
    ref.dps = 90
    expected = ref.sqrt(ref.mpf(square.numerator) / square.denominator)
    assert_mpf_relative(shape_transfer_singular_value(shape, 3 * shape, degree), expected)


@pytest.mark.parametrize("exponent", [10, 30, 50, 100, 300])
def test_huge_meixner_peak_matches_independent_duplication_gamma_ratio(exponent):
    ref = mpmath.mp.clone()
    ref.dps = exponent + 90
    half = ref.mpf(10) ** exponent / 2
    # Duplication reduces the peak to one ratio; no subtraction of O(s log s).
    expected = ref.gammaprod([half], [half + ref.mpf(".5")]) / ref.sqrt(ref.pi)
    assert_mpf_relative(meixner_density("1e" + str(exponent), 0), expected)


@pytest.mark.parametrize("frequency", [0, ".125", "3", "30"])
def test_meixner_known_density_controls(frequency):
    ref = mpmath.mp.clone()
    ref.dps = 90
    omega = ref.mpf(frequency)
    assert_mpf_relative(meixner_density(1, frequency), 1 / ref.cosh(ref.pi * omega))
    expected = 2 / ref.pi if omega == 0 else 2 * omega / ref.sinh(ref.pi * omega)
    assert_mpf_relative(meixner_density(2, frequency), expected)


def test_large_beta_binomial_coefficients_normalize_and_match_binomial_limit():
    ref = mpmath.mp.clone()
    ref.dps = 90
    coefficients = beta_binomial_refinement(12, "1e100", "1e100")
    assert abs(sum(ref.mpf(str(value)) ** 2 for value in coefficients) - 1) < ref.mpf("1e-45")
    # The finite-shape correction is O(1e-100), below the requested precision.
    from math import comb
    for index, value in enumerate(coefficients):
        assert_mpf_relative(value, ref.sqrt(ref.mpf(comb(12, index)) / 2 ** 12))


def test_huge_shape_tiny_shift_characteristic_retains_nontrivial_limit():
    ref = mpmath.mp.clone()
    ref.dps = 90
    # log cosh(h/2)=h^2/8+O(h^4), leaving an error O(1e-200).
    assert_mpf_relative(meixner_characteristic("1e200", "1e-100"), ref.exp(-ref.mpf(1) / 8))


@pytest.mark.parametrize("operation", [
    lambda: shape_refinement_coefficient(["1e10000", 1], [1, 0]),
    lambda: shape_transfer_singular_value("1e10000", 1, 1),
    lambda: meixner_density("1e10000", 0),
    lambda: shape_affinity("1e10000", "1e10000"),
    lambda: gamma_sech_ambiguity("1e10000", 1, 0, 0),
    lambda: log_gamma_window(1, "1e100"),
])
def test_extreme_work_demands_reject_before_unbounded_precision(operation):
    with pytest.raises(ResourceLimitError, match="10000"):
        operation()


def test_shape_work_context_does_not_change_global_mpmath_precision():
    original = mpmath.mp.dps
    shape_refinement_coefficient(["1e300", "1e300"], [1, 0])
    assert mpmath.mp.dps == original


def test_huge_represented_central_phase_avoids_sinc_pi_round_trip():
    mesh = 3.1473428762850648e187
    mode = np.array([-60, -87, -35])
    phase = mesh * (2 * np.pi * mode)
    np.testing.assert_allclose(central_mode_momentum(mode, mesh) * mesh, np.sin(phase), rtol=1e-15, atol=1e-15)
    # This checks the represented phase, not a claim of exact huge-angle data.


@pytest.mark.parametrize("mesh", [1e-300, 1e-150, 1e-8, -.125, 1e10, 3.1473428762850648e187])
def test_central_symbol_zero_mesh_limit_and_even_mesh_metamorphism(mesh):
    mode = np.array([0, -2, 3])
    actual = central_mode_momentum(mode, mesh)
    np.testing.assert_allclose(actual, central_mode_momentum(mode, -mesh), rtol=1e-15, atol=0)
    if abs(mesh) < 1e-8:
        np.testing.assert_allclose(actual, 2 * np.pi * mode, rtol=1e-15, atol=0)


@pytest.mark.parametrize("spacing", [1e-200, 1e200])
@pytest.mark.parametrize("operation", [
    lambda h: periodic_laplacian((3,), spacing=h),
    lambda h: periodic_lattice_eigenvalues((3,), spacing=h),
    lambda h: low_mode_dispersion_bound((3,), (1,), spacing=h),
])
def test_unrepresentable_lattice_scales_have_typed_rejection(spacing, operation):
    with pytest.raises(ValidationError, match="spacing"):
        operation(spacing)


@pytest.mark.parametrize("spacing", [1e-80, 1e80])
def test_representable_extreme_dispersion_avoids_fourth_power_overflow(spacing):
    discrete, continuum, bound = low_mode_dispersion_bound((8, 9), (2, 3), spacing=spacing)
    angles = 2 * np.pi * np.array([2 / 8, 3 / 9])
    scale = (1 / spacing) / spacing
    np.testing.assert_allclose(np.array([discrete, continuum, bound]) / scale,
        [sum(4 * np.sin(angles / 2) ** 2), sum(angles ** 2), sum(angles ** 4) / 12], rtol=2e-15)
    assert 0 <= continuum - discrete <= bound


def test_heat_trace_rejects_eigenvalue_overflow_instead_of_silent_wrong_value():
    with pytest.raises(ValidationError, match="eigenvalues"):
        heat_trace([[1e308, 1e308], [1e308, 1e308]], 1e-308)


def test_channel_owns_readonly_kraus_snapshot_and_revalidates_reassigned_state():
    kraus = np.eye(2, dtype=complex)
    channel = QuantumChannel((kraus,))
    kraus[:] = np.nan
    np.testing.assert_array_equal(channel.apply(np.eye(2)), np.eye(2))
    assert not channel.kraus[0].flags.writeable
    channel.kraus = (np.full((2, 2), np.nan),)
    with pytest.raises(ValidationError, match="finite"):
        channel.apply(np.eye(2))


@pytest.mark.parametrize("tolerance", [float("inf"), float("nan"), -1])
def test_channel_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValidationError):
        QuantumChannel((np.eye(2),), tolerance=tolerance)


@pytest.mark.parametrize("operation", [
    lambda channel: channel.apply(np.eye(2)),
    lambda channel: channel.adjoint(np.eye(2)),
    lambda channel: channel.superoperator(),
    lambda channel: channel.is_trace_preserving(),
    lambda channel: channel.certificate(),
])
def test_channel_product_overflow_cannot_escape_as_nonfinite_output(operation):
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValidationError, match="finite"):
            operation(QuantumChannel((np.eye(2) * 1e308,)))


def test_lindblad_evolution_wrong_dimension_has_public_dimension_error():
    generator = LindbladGenerator(np.eye(2), ())
    with pytest.raises(DimensionError, match="dimension"):
        generator.evolve(np.eye(3) / 3, 1)


def test_channel_choi_averaging_does_not_overflow_finite_entries():
    with np.errstate(over="ignore", invalid="ignore"):
        certificate = QuantumChannel((np.array([[1e154], [0.]]),)).certificate()
    assert not certificate.valid
    assert np.isfinite(certificate.residuals["trace_preservation"])
    assert certificate.residuals["negative_choi_part"] == 0


def test_channel_unrepresentable_choi_eigenvalues_have_typed_rejection():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValidationError, match="spectrum"):
            QuantumChannel((np.eye(2) * 1e154,)).certificate()
