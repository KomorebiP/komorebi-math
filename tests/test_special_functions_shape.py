"""Independent quadrature and polynomial identities for the new shape layer."""

from fractions import Fraction
import json
import subprocess
import sys

import pytest

from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.special_functions import (
    beta_binomial_refinement, beta_logit_density, gamma_sech_ambiguity,
    log_gamma_fourier, log_gamma_window, meixner_characteristic, meixner_density,
    meixner_pollaczek, shape_affinity, shape_refinement_coefficient,
    shape_transfer_singular_value,
)


def mp_context(dps=45):
    mp = pytest.importorskip("mpmath")
    ctx = mp.mp.clone()
    ctx.dps = dps
    return ctx


def assert_close(actual, expected, tolerance="1e-35"):
    ctx = mp_context()
    assert abs(actual - expected) <= ctx.mpf(tolerance) * max(1, abs(expected))


def test_import_keeps_optional_backends_lazy():
    code = "import sys,json; import komorebi_math.special_functions; print(json.dumps([k for k in ('mpmath','scipy','sympy') if k in sys.modules]))"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []


def test_fourier_transform_matches_direct_quadrature():
    ctx = mp_context()
    s = ctx.mpf("1.75")
    omega = ctx.mpf("0.6")
    # Integrate the defining window after u=exp(y); no special-function formula.
    integrand = lambda u: ctx.exp((s * ctx.log(u) - u) / 2) * ctx.power(u, -ctx.j * omega) / u
    integral = ctx.quad(integrand, [0, ctx.mpf(".25"), 1, 4, ctx.inf]) / ctx.sqrt(2 * ctx.pi * ctx.gamma(s))
    assert_close(log_gamma_fourier(s, omega, dps=45), integral, "1e-32")


def test_cross_ambiguity_matches_defining_integral_and_factorizes():
    ctx = mp_context()
    a, g, h, eta = map(ctx.mpf, ("1.25", "2.75", ".8", ".4"))
    def integral_term(u):
        y = ctx.log(u)
        window1 = ctx.exp((a * (y + h / 2) - ctx.exp(y + h / 2)) / 2) / ctx.sqrt(ctx.gamma(a))
        window2 = ctx.exp((g * (y - h / 2) - ctx.exp(y - h / 2)) / 2) / ctx.sqrt(ctx.gamma(g))
        return window1 * window2 * ctx.exp(ctx.j * eta * y) / u
    direct = ctx.quad(integral_term, [0, ctx.mpf(".25"), 1, 4, ctx.inf])
    ambiguity = gamma_sech_ambiguity(a, g, h, eta, dps=45)
    assert_close(ambiguity, direct, "1e-32")
    assert_close(abs(ambiguity) ** 2 / (2 * ctx.pi),
                 beta_logit_density(a, g, h, dps=45) * meixner_density(a + g, eta, dps=45))
    assert_close(gamma_sech_ambiguity(g, a, -h, -eta, dps=45), ctx.conj(ambiguity))


def test_window_density_and_affinity_normalizations():
    ctx = mp_context()
    for shape in (Fraction(1, 2), Fraction(7, 3)):
        assert_close(log_gamma_window(shape, 0, dps=45) ** 2,
                     ctx.exp(-1) / ctx.gamma(ctx.mpf(shape.numerator) / shape.denominator))
        assert_close(shape_affinity(shape, shape, dps=45), 1)
    overlap = shape_affinity(Fraction(1, 2), 3, dps=45)
    assert 0 < overlap < 1
    assert_close(meixner_density(1, "0.7", dps=45), 1 / ctx.cosh(ctx.pi * ctx.mpf("0.7")))
    assert_close(beta_logit_density(1, 1, ".8", dps=45),
                 ctx.exp(ctx.mpf(".8")) / (1 + ctx.exp(ctx.mpf(".8"))) ** 2)


def test_pollaczek_matches_generating_function_coefficients():
    ctx = mp_context()
    s, x = ctx.mpf("1.7"), ctx.mpf(".4")
    generating = lambda z: (1 - ctx.j * z) ** (-s / 2 + ctx.j * x) * (1 + ctx.j * z) ** (-s / 2 - ctx.j * x)
    coefficients = ctx.taylor(generating, 0, 7)
    for n, coefficient in enumerate(coefficients):
        normalized = coefficient * ctx.sqrt(ctx.factorial(n) / ctx.rf(s, n))
        assert_close(meixner_pollaczek(s, n, x, dps=45), normalized)


def test_two_child_addition_and_weight_normalization():
    ctx = mp_context()
    a, g, x, y = map(ctx.mpf, (".7", "2.3", "-.8", ".4"))
    for n in range(9):
        weights = beta_binomial_refinement(n, a, g, dps=45)
        assert_close(sum(c * c for c in weights), 1)
        right = sum(c * meixner_pollaczek(a, r, x, dps=45)
                    * meixner_pollaczek(g, n - r, y, dps=45) for r, c in enumerate(weights))
        assert_close(right, meixner_pollaczek(a + g, n, x + y, dps=45))


def test_multichild_tree_coassociativity():
    a, b, c = Fraction(3, 4), Fraction(5, 4), Fraction(7, 3)
    r, s, t = 2, 1, 3
    direct = shape_refinement_coefficient((a, b, c), (r, s, t))
    left_tree = shape_refinement_coefficient((a + b, c), (r + s, t)) * shape_refinement_coefficient((a, b), (r, s))
    right_tree = shape_refinement_coefficient((a, b + c), (r, s + t)) * shape_refinement_coefficient((b, c), (s, t))
    assert_close(direct, left_tree)
    assert_close(direct, right_tree)


def test_characteristic_shape_semigroup_and_transfer_composition():
    ctx = mp_context()
    a, b, c, h = map(ctx.mpf, (".7", "1.2", "2.3", "3.1"))
    assert_close(meixner_characteristic(a + b, h), meixner_characteristic(a, h) * meixner_characteristic(b, h))
    for n in range(6):
        assert_close(shape_transfer_singular_value(a, b + c, n),
                     shape_transfer_singular_value(a, b, n) * shape_transfer_singular_value(a + b, c, n))


def test_precision_isolation_and_large_shift():
    mp = pytest.importorskip("mpmath")
    original = mp.mp.dps
    value = gamma_sech_ambiguity(1, 1, 2000, 0, dps=70)
    assert value.real > 0 and value.imag == 0
    assert mp.mp.dps == original
    assert log_gamma_window(1, 20, dps=50) > 0


@pytest.mark.parametrize("shape", [0, -1, True, float("nan"), float("inf")])
def test_shapes_reject_invalid_inputs(shape):
    pytest.importorskip("mpmath")
    with pytest.raises(ValidationError):
        meixner_density(shape, 0)


def test_degree_dimension_and_coordinate_validation():
    pytest.importorskip("mpmath")
    with pytest.raises(ValidationError):
        meixner_pollaczek(1, 2.5, 0)
    with pytest.raises(ValidationError):
        gamma_sech_ambiguity(1, 1, float("nan"), 0)
    with pytest.raises(DimensionError):
        shape_refinement_coefficient((1, 2), (3,))
    with pytest.raises(DimensionError):
        shape_refinement_coefficient((), ())
