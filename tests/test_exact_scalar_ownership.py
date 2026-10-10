"""Public exact constructions must own fixed-width scalar components."""

from fractions import Fraction

import numpy as np
import pytest

from komorebi_math.arithmetic.packet_differences import rational_mixed_difference
from komorebi_math.arithmetic.polynomial_lifting import SparseIntegerPolynomial
from komorebi_math.core.paper_machinery import rational_spectral_projectors
from komorebi_math.dynamics.gaussian_verifier import (
    FactorGaussianModel,
    exact_gaussian_posterior,
    outward_dyadic,
)
from komorebi_math.dynamics.hermite import gaussian_polynomial_inner_product
from komorebi_math.exceptions import ResourceLimitError, ValidationError
from komorebi_math.generating.cartier_prime_power import compile_prime_power_cartier_representation
from komorebi_math.langlands.local.tame import FrobeniusOrbit
from komorebi_math.langlands.local.tame_tensor import tame_tensor_character_packets
from komorebi_math.signals.foundations import compression_closure
from komorebi_math.special_functions.paper_machinery import feedback_odd_derivatives


def owned(value):
    assert type(value.numerator) is int
    assert type(value.denominator) is int


@pytest.mark.parametrize("component_type", [np.int64, np.uint64])
@pytest.mark.parametrize("numerator,denominator", [(2**32, 1), (1, 2**32 + 1), (2**62, 3)])
def test_gaussian_inner_product_uses_arbitrary_precision_fraction_components(
    component_type, numerator, denominator
):
    value = Fraction(component_type(numerator), component_type(denominator))
    expected = Fraction(numerator**2, denominator**2)
    with np.errstate(all="raise"):
        actual = gaussian_polynomial_inner_product((value,), (value,))
    assert actual == expected
    owned(actual)


def test_hermite_degree_cap_is_owned_before_doubling():
    # No high-degree polynomial is allocated: the limit itself used to wrap
    # when computing the moment limit, despite the constant input.
    assert gaussian_polynomial_inner_product((2,), (3,), max_degree=np.int64(2**62)) == 6


@pytest.mark.parametrize("scalar", [np.int64(2**32), Fraction(np.int64(2**32), np.int64(1))])
def test_gaussian_gain_and_dyadic_enclosure_accept_owned_exact_integer_scalars(scalar):
    size = 2**64
    model = FactorGaussianModel(
        prior_factor=[[scalar]],
        observation_operator=[[1]],
        noise_factor=[[]],
        noise_floor=1,
        prior_mean=[0],
        observation=[1],
    )
    posterior = exact_gaussian_posterior(model)
    assert posterior.mean == (Fraction(size, size + 1),)
    assert posterior.covariance == ((Fraction(size, size + 1),),)
    interval = outward_dyadic(scalar, 8)
    assert interval.lower == interval.upper == 2**32
    for value in (posterior.mean[0], posterior.covariance[0][0], interval.lower, interval.upper):
        owned(value)


def test_special_feedback_owns_fraction_components_and_keeps_explicit_string_domain():
    value = Fraction(np.int64(2**32), np.int64(1))
    with np.errstate(all="raise"):
        actual = feedback_odd_derivatives(((value, 1, 1),), last=2)
    assert actual == (1, 2**33, 2**68)
    assert actual == feedback_odd_derivatives(((str(2**32), "1", "1"),), last=2)
    for entry in actual:
        owned(entry)


def test_packet_stencil_sum_cannot_wrap_at_the_signed_integer_boundary():
    value = Fraction(np.int64(2**62), np.int64(1))
    with np.errstate(all="raise"):
        actual = rational_mixed_difference((value, 0, value), np.int64(0), (np.int64(1), np.int64(1)))
    assert actual == 2**63
    owned(actual)


def test_tame_tensor_closing_product_cannot_overflow_to_zero():
    value = Fraction(np.int64(2**32), np.int64(1))
    orbit = FrobeniusOrbit(np.int64(2), np.int64(1), np.int64(0))
    with np.errstate(all="raise"):
        result = tame_tensor_character_packets(orbit, orbit, left_scalar=value, right_scalar=value)
    assert result.packets[0].closing_scalar == 2**64
    assert result.local_polynomial_coefficients == (1, -(2**64))
    assert result.certificate.valid
    for entry in result.local_polynomial_coefficients:
        owned(entry)


def test_sparse_integer_polynomial_already_owns_fixed_width_inputs():
    polynomial = SparseIntegerPolynomial(np.int64(1), {(np.int64(2),): np.int64(2**32)})
    value = polynomial.evaluate((np.int64(2**32),))
    gradient = polynomial.gradient((np.int64(2**32),))
    assert value == 2**96 and type(value) is int
    assert gradient == (2**65,) and type(gradient[0]) is int


def test_prime_power_cartier_already_owns_fixed_width_coefficients():
    raw = (np.int64(1), np.int64(2**62))
    supplied = compile_prime_power_cartier_representation(raw, (np.int64(1),), np.int64(3), np.int64(2))
    reference = compile_prime_power_cartier_representation((1, 2**62), (1,), 3, 2)
    assert supplied.sequence(np.int64(24)) == reference.sequence(24)
    assert supplied.certificate(24).valid
    assert all(type(value) is int for value in supplied.numerator)


def test_sympy_spectral_and_signal_ingress_already_canonicalize_exact_scalars():
    sp = pytest.importorskip("sympy")
    value = Fraction(np.int64(2**32), np.int64(1))
    variable = sp.Symbol("x")
    packets = rational_spectral_projectors([[value, 0], [0, 1]], (variable - 2**32, variable - 1))
    assert tuple(packets.projectors) == (sp.diag(1, 0), sp.diag(0, 1))
    closure = compression_closure([[0, value], [value, 0]], sp.diag(1, 0))
    assert closure.composition_defect == sp.diag(2**64, 0)
    assert closure.certificate.valid


def test_signal_matrix_dimensions_cannot_overflow_the_pre_copy_resource_guard():
    pytest.importorskip("sympy")

    class Oversized:
        shape = (np.int64(2**32), np.int64(2**32))

        def tolist(self):
            raise AssertionError("oversized dimensions must be rejected before copying")

    with pytest.raises(ResourceLimitError):
        compression_closure(Oversized(), [[1]])


@pytest.mark.parametrize("bad", [1.0, True, np.bool_(True)])
def test_exact_fraction_ownership_does_not_promote_approximate_or_boolean_inputs(bad):
    for operation in (
        lambda: gaussian_polynomial_inner_product((bad,), (1,)),
        lambda: outward_dyadic(bad, 8),
        lambda: rational_mixed_difference((bad,), 0, ()),
        lambda: tame_tensor_character_packets(FrobeniusOrbit(2, 1, 0), FrobeniusOrbit(2, 1, 0), left_scalar=bad),
        lambda: feedback_odd_derivatives(((bad, 1, 1),), last=1),
    ):
        with pytest.raises(ValidationError):
            operation()
