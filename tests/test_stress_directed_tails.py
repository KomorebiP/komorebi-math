"""Bounded cross-boundary stress controls for directed quadratic-form tails.

Exact dyadic algebra and separately constructed directed Erlang formulas are
controls. General dense eigensolver comparisons are numerical checks, not proofs.
"""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")
from komorebi_math import quadratic_forms as qf
from komorebi_math.exceptions import ValidationError


SMALL_CAPS = dict(initial_K=2, max_K=2, initial_L="1", max_L=1,
                  initial_interval_dps=15, max_interval_dps=15, working_dps=30)
CONTROL_CAPS = dict(initial_K=16, max_K=128, initial_L="8", max_L=12,
                    initial_interval_dps=35, max_interval_dps=35, working_dps=80)


def _literal_endpoints(text):
    return tuple(Fraction(Decimal(x.strip())) for x in text[1:-1].split(","))


def _binary_fraction(raw):
    sign, mantissa, exponent, bitcount = raw
    assert bitcount >= 0
    value = Fraction(int(mantissa)) * Fraction(2) ** int(exponent)
    return -value if sign else value


def independent_erlang_log10_enclosure(q, coefficient, df):
    """Exact rational polynomial followed by separate directed log evaluation."""
    x = Fraction(Decimal(q)) / (2 * Fraction(Decimal(coefficient)))
    term = total = Fraction(1)
    for k in range(1, df // 2):
        term *= x / k
        total += term
    previous = mp.iv.prec
    try:
        mp.iv.dps = 100
        xiv = mp.iv.mpf(x.numerator) / x.denominator
        totaliv = mp.iv.mpf(total.numerator) / total.denominator
        result = (-xiv + mp.iv.log(totaliv)) / mp.iv.log(10)
        return tuple(_binary_fraction(raw) for raw in result._mpi_)
    finally:
        mp.iv.prec = previous


@pytest.mark.parametrize("scale", [-300, -100, 0, 100, 300])
@pytest.mark.parametrize("df", [4, 8, 12])
@pytest.mark.parametrize("multiple", [2, 10])
def test_scaled_tails_enclose_independent_directed_erlang(scale, df, multiple):
    coefficient, q = f"1e{scale}", f"{df * multiple}e{scale}"
    result = qf.certified_positive_logsf(q, (coefficient for _ in range(df)),
        relative_accuracy="0.02", **CONTROL_CAPS)
    assert result.K <= CONTROL_CAPS["max_K"]
    assert result.interval_dps <= CONTROL_CAPS["max_interval_dps"]
    assert Decimal(result.L) <= CONTROL_CAPS["max_L"]
    assert result.log10_interval != "[-inf,+inf]"
    lower, upper = _literal_endpoints(result.log10_interval)
    reference_lower, reference_upper = independent_erlang_log10_enclosure(q, coefficient, df)
    assert lower <= reference_lower <= reference_upper <= upper
    if result.certified:
        assert Decimal(result.achieved_relative_ratio_minus_one) <= Decimal("0.02")
    else:
        assert result.failure_reason


@pytest.mark.parametrize("exponents", [
    (-500, -250, 0, 250, 500), (-10, -5, 0, 5, 10),
    (-500, -500, -500, -500), (500, 500, 500, 500),
    (-1, 0, 1, 2), (0, 0, 0, 0),
])
@pytest.mark.parametrize("digits", [15, 50])
def test_exact_dyadic_diagonal_spectrum_extreme_scales(exponents, digits):
    stored = [math.ldexp(1.0, exponent) for exponent in exponents]
    result = qf.certify_float64_factor_spectrum(np.diag(stored),
        candidate_digits=digits, interval_dps=digits + 10)
    exact = sorted((Fraction(x) ** 2 for x in stored), reverse=True)
    for lower, upper, value in zip(result.lower, result.upper, exact):
        assert Fraction(Decimal(lower)) <= value <= Fraction(Decimal(upper))
    assert result.candidate_digits == digits


@pytest.mark.parametrize("factor", [np.zeros((4, 4)), np.ones((4, 1)), np.diag([1., 1., 0., 0.])])
def test_unresolved_spectral_rank_is_an_explicit_failure(factor):
    point, interval = mp.mp.prec, mp.iv.prec
    with pytest.raises(RuntimeError, match="rank <= 2"):
        qf.certified_float64_factor_logsf("20", factor, tail_kwargs=SMALL_CAPS)
    assert (mp.mp.prec, mp.iv.prec) == (point, interval)


@pytest.mark.parametrize("q", ["4." + "0" * 59 + "1", "1e300"])
def test_planning_precision_exhaustion_returns_uncertified_result(q):
    result = qf.certified_positive_logsf(q, ["1"] * 4, **SMALL_CAPS)
    assert not result.certified
    assert "planning precision budget exhausted" in result.failure_reason
    assert result.log10_interval == "[-inf,+inf]"
    assert result.refinements == 0


def test_interval_domain_margin_refines_only_to_supplied_cap():
    result = qf.certified_positive_logsf("1e50", ["1"] * 4,
        **(SMALL_CAPS | dict(working_dps=160, max_interval_dps=45)))
    assert not result.certified
    assert "interval precision budget exhausted" in result.failure_reason
    assert result.interval_dps == 45
    assert result.refinements == 2


def test_conservative_matrix_mean_exhaustion_returns_uncertified_result():
    factor = np.diag([1.1, 1.2, 1.3, 1.4])
    exact_q = sum((Fraction(float(x)) ** 2 for x in factor.flat), Fraction(0)) + Fraction(1, 10 ** 150)
    with localcontext() as context:
        context.prec = 300
        q = str(Decimal(exact_q.numerator) / Decimal(exact_q.denominator))
    result = qf.certified_float64_factor_logsf(q, factor,
        initial_candidate_digits=15, max_candidate_digits=45, candidate_step=30,
        interval_dps=15, tail_kwargs=SMALL_CAPS | dict(working_dps=120))
    assert not result.certified
    assert "planning domain was resolved" in result.failure_reason
    assert result.candidate_digits == result.spectral_enclosure.candidate_digits == 45
    assert result.spectral_refinements == 2


def test_reported_spectral_digits_are_the_last_attempt_not_the_unvisited_cap():
    result = qf.certified_float64_factor_logsf("20", np.eye(4), relative_accuracy="1e-50",
        initial_candidate_digits=15, max_candidate_digits=16, candidate_step=30,
        interval_dps=20, tail_kwargs=SMALL_CAPS)
    assert not result.certified
    assert result.candidate_digits == result.spectral_enclosure.candidate_digits == 15
    assert result.spectral_refinements == 1


@pytest.mark.parametrize("q", ["4", "3.9999999999999999999999999999999999999"])
def test_exact_domain_rejections_remain_validation_errors(q):
    with pytest.raises(ValidationError, match="above the mean"):
        qf.certified_positive_logsf(q, ["1"] * 4, **SMALL_CAPS)


def test_direct_design_keeps_a_specific_precision_failure_exception():
    with pytest.raises(ValueError, match="working precision"):
        qf.design_positive_tail_parameters("4." + "0" * 59 + "1", ["1"] * 4,
            working_dps=30)


def test_direct_interval_rejects_exact_invalid_margin_before_computation():
    with pytest.raises(ValueError, match="eta must exceed theta"):
        qf.interval_positive_logsf("20", ["1"] * 4, theta_decimal="0.2",
            eta_decimal="0.2", upper_decimal="30", K=2, interval_dps=30)


def test_public_concurrent_success_and_failure_restore_precision():
    before = mp.mp.prec, mp.iv.prec
    try:
        mp.mp.prec, mp.iv.prec = 79, 83
        def task(index):
            if index % 3 == 0:
                result = qf.certified_positive_logsf("4." + "0" * 59 + "1", ["1"] * 4,
                    **SMALL_CAPS)
                assert not result.certified
                return "precision_exhausted"
            if index % 3 == 1:
                result = qf.certify_float64_factor_spectrum(np.diag([0.5, 1., 2.]),
                    candidate_digits=20, interval_dps=30)
                assert len(result.lower) == 3
                return "spectrum"
            with pytest.raises(ValueError, match="upper"):
                qf.interval_positive_logsf("20", ["1"] * 4, theta_decimal="0.1",
                    eta_decimal="0.2", upper_decimal="20", K=2, interval_dps=30)
            return "domain_error"
        with ThreadPoolExecutor(max_workers=4) as executor:
            outcomes = list(executor.map(task, range(24)))
        assert len(outcomes) == 24
        assert (mp.mp.prec, mp.iv.prec) == (79, 83)
    finally:
        mp.mp.prec, mp.iv.prec = before


def test_matrix_failure_reason_tracks_the_last_resolved_domain_attempt():
    factor = np.diag([1.1, 1.2, 1.3, 1.4])
    exact_q = sum((Fraction(float(x)) ** 2 for x in factor.flat), Fraction(0)) + Fraction(1, 10 ** 30)
    with localcontext() as context:
        context.prec = 200
        q = str(Decimal(exact_q.numerator) / Decimal(exact_q.denominator))
    result = qf.certified_float64_factor_logsf(q, factor,
        initial_candidate_digits=15, max_candidate_digits=45, candidate_step=30,
        interval_dps=15, tail_kwargs=SMALL_CAPS | dict(working_dps=120))
    assert not result.certified
    assert result.spectral_refinements == 2
    assert result.failure_reason == "spectral/tail resource caps reached before requested end-to-end accuracy was proved"
