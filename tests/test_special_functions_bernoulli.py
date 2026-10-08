"""Exact residue refinement and near-removable-singularity regressions."""

from fractions import Fraction

import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.special_functions import bernoulli_hurwitz, bernoulli_hurwitz_packet, bernoulli_packet_integer


@pytest.mark.parametrize("n", range(10))
def test_exact_integer_packet_reconstruction_and_refinement(n):
    a = Fraction(3, 7)
    whole = bernoulli_packet_integer(n, a, 1, 0)
    assert sum(bernoulli_packet_integer(n, a, 5, r) for r in range(5)) == whole
    for r in range(3):
        parent = bernoulli_packet_integer(n, a, 3, r)
        assert sum(bernoulli_packet_integer(n, a, 12, r + 3 * child) for child in range(4)) == parent


def test_integer_samples_match_independent_symbolic_polynomials():
    sympy = pytest.importorskip("sympy")
    a = Fraction(-2, 3)
    for n in range(8):
        expected = sympy.Rational(5) ** (n - 1) * sympy.bernoulli(n, sympy.Rational(7, 15))
        assert sympy.Rational(bernoulli_packet_integer(n, a, 5, 3)) == expected


def test_numeric_complex_order_reconstruction_and_refinement():
    mp = pytest.importorskip("mpmath")
    ctx = mp.mp.clone(); ctx.dps = 60
    z = ctx.mpc(".7", ".35")
    whole = bernoulli_hurwitz(z, "1.25", dps=60)
    packets = tuple(bernoulli_hurwitz_packet(z, "1.25", 4, r, dps=60) for r in range(4))
    assert abs(sum(packets) - whole) < ctx.mpf("1e-55")
    children = tuple(bernoulli_hurwitz_packet(z, "1.25", 12, 2 + 4 * r, dps=60) for r in range(3))
    assert abs(sum(children) - packets[2]) < ctx.mpf("1e-55")


def test_exact_zero_values_and_positive_integer_agreement():
    pytest.importorskip("mpmath")
    assert bernoulli_hurwitz(0, "1.25") == 1
    assert bernoulli_hurwitz_packet(0, "1.25", 4, 3) == Fraction(1, 4)
    for n in range(1, 8):
        exact = bernoulli_packet_integer(n, Fraction(5, 4), 4, 3)
        actual = bernoulli_hurwitz_packet(n, Fraction(5, 4), 4, 3)
        assert abs(complex(actual) - float(exact)) < 1e-12


def test_tiny_nonzero_orders_are_not_flattened_to_zero_order():
    mp = pytest.importorskip("mpmath")
    ctx = mp.mp.clone(); ctx.dps = 130
    z = ctx.mpf("1e-80")
    value = bernoulli_hurwitz(z, "1.25", dps=130)
    derivative = (value - 1) / z
    assert abs(derivative - ctx.digamma(ctx.mpf("1.25"))) < ctx.mpf("1e-45")
    assert value != 1


@pytest.mark.parametrize("a", [0, -1, True, float("nan"), float("inf")])
def test_hurwitz_positive_translation_domain(a):
    pytest.importorskip("mpmath")
    with pytest.raises(ValidationError):
        bernoulli_hurwitz(1, a)


def test_exact_sample_domain_and_bad_residue():
    assert bernoulli_packet_integer(1, 0, 1, 0) == Fraction(-1, 2)
    with pytest.raises(ValidationError):
        bernoulli_packet_integer(1, .25, 1, 0)
    with pytest.raises(ValidationError):
        bernoulli_packet_integer(1, 1, 3, 3)
    with pytest.raises(ValidationError):
        bernoulli_packet_integer(True, 1, 3, 0)
