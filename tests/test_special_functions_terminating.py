"""Exact finite scalar evaluations and the corrected q-Racah negative control."""

from fractions import Fraction as F
import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.special_functions import q_pochhammer, q_racah_lattice, q_racah_scalar, racah_scalar


def direct_q_sum(n, y, a, b, g, d, q):
    """Independent products, using the original full n-terminating formula."""
    def poch(value, k):
        product = F(1)
        for r in range(k):
            product *= 1 - value * q ** r
        return product
    total = F(0)
    for k in range(n + 1):
        numerator = poch(q ** (-n), k) * poch(a * b * q ** (n + 1), k) * poch(q ** (-y), k) * poch(g * d * q ** (y + 1), k)
        denominator = poch(a * q, k) * poch(b * d * q, k) * poch(g * q, k) * poch(q, k)
        total += numerator * q ** k / denominator
    return total


@pytest.mark.parametrize("n,y", [(0, 4), (1, 3), (3, 1), (4, 5)])
def test_q_racah_exact_full_sum_and_duality(n, y):
    a, b, g, d, q = F(2, 7), F(3, 5), F(4, 9), F(5, 8), F(2, 3)
    actual = q_racah_scalar(n, y, a, b, g, d, q)
    assert actual == direct_q_sum(n, y, a, b, g, d, q)
    assert actual == q_racah_scalar(y, n, g, d, a, b, q)
    assert isinstance(actual, F)


def test_c11_termination_does_not_imply_positive_orthogonality():
    q, a, b, g, d = F(1, 2), F(1, 2), F(1, 2), F(4), F(1)
    assert g * q == q ** (-1)
    assert [q_racah_lattice(y, g, d, q) for y in range(2)] == [3, 3]
    table = tuple(tuple(q_racah_scalar(n, y, a, b, g, d, q) for y in range(2)) for n in range(2))
    assert table == ((1, 1), (1, 1))


def test_finite_q_pochhammer_exact_decimal_strings():
    assert q_pochhammer("0.25", "1/2", 3) == F(3, 4) * F(7, 8) * F(15, 16)
    assert q_pochhammer(2, "1/2", 2) == 0
    assert q_pochhammer(2, "1/2", 0) == 1


def test_classical_racah_agrees_with_independent_hypergeometric_evaluation():
    sympy = pytest.importorskip("sympy")
    n, y, a, b, c, d = 3, 4, F(1, 2), F(1, 3), F(2, 5), F(3, 7)
    expected = sympy.hyper((-n, n + sympy.Rational(a) + sympy.Rational(b) + 1, -y, y + sympy.Rational(c) + sympy.Rational(d) + 1),
                          (sympy.Rational(a) + 1, sympy.Rational(b) + sympy.Rational(d) + 1, sympy.Rational(c) + 1), 1).expand(func=True)
    assert sympy.Rational(racah_scalar(n, y, a, b, c, d)) == expected


def test_singular_retained_terms_are_rejected_before_cancellation():
    with pytest.raises(ValidationError, match="denominator"):
        q_racah_scalar(2, 2, 2, F(1, 2), 3, 1, F(1, 2))
    with pytest.raises(ValidationError, match="denominator"):
        racah_scalar(1, 1, -1, 2, 3, 4)


@pytest.mark.parametrize("q", [0, 1, -1, F(3, 2), .5, True])
def test_q_domain_is_exact_and_nonsingular(q):
    with pytest.raises(ValidationError):
        q_racah_scalar(1, 1, 1, 1, 1, 1, q)


def test_strict_integer_and_rational_domains():
    with pytest.raises(ValidationError):
        q_racah_scalar(1.0, 1, 1, 1, 1, 1, F(1, 2))
    with pytest.raises(ValidationError):
        racah_scalar(1, 1, .5, 1, 1, 1)


def test_numpy_boolean_domain_is_consistent_across_supported_versions():
    import numpy as np
    for value in (np.bool_(False), np.bool_(True)):
        with pytest.raises(ValidationError):
            q_pochhammer(1, "1/2", value)
        with pytest.raises(ValidationError):
            racah_scalar(1, 1, value, 1, 1, 1)
