"""Independent numerical and public-domain checks for directed research tails."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from fractions import Fraction
import importlib
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from komorebi_math.exceptions import BackendUnavailableError, ValidationError
from komorebi_math import quadratic_forms as qf


def endpoints(text):
    return tuple(Decimal(x.strip()) for x in text[1:-1].split(",", 1))


def independent_even_gamma_log10(q, coefficient, df):
    x = Fraction(Decimal(q)) / (2 * Fraction(Decimal(coefficient)))
    term = total = Fraction(1)
    for k in range(1, df // 2):
        term *= x / k
        total += term
    with mp.workdps(150):
        logp = -mp.mpf(x.numerator) / x.denominator + mp.log(
            mp.mpf(total.numerator) / total.denominator
        )
        return Decimal(mp.nstr(logp / mp.log(10), 140))


@pytest.mark.parametrize("rank,q,coefficient,K", [
    (4, "8", "1", 256), (6, "12", "1", 256),
    (8, "16", "1", 128), (12, "36", "0.5", 128),
    (80, "4", "0.02", 64),
])
def test_equal_coefficient_tail_contains_independent_gamma_formula(rank, q, coefficient, K):
    values = [coefficient] * rank
    params = qf.design_positive_tail_parameters(q, iter(values), L="8", working_dps=130, output_digits=100)
    result = qf.interval_positive_logsf(q, iter(values), theta_decimal=params["theta"],
        upper_decimal=params["upper"], eta_decimal=params["eta"], K=K, interval_dps=60)
    lo, hi = endpoints(result.log10_interval)
    reference = independent_even_gamma_log10(q, coefficient, rank)
    assert result.passed_positive_sum
    assert lo <= reference <= hi
    scipy_stats = pytest.importorskip("scipy.stats")
    assert abs(float(reference) - scipy_stats.chi2.logsf(float(q) / float(coefficient), rank) / np.log(10)) < 1e-12


def test_certified_controller_materializes_one_shot_generator_and_accepts_directed_width():
    result = qf.certified_positive_logsf("30", ("0.25" for _ in range(32)),
        relative_accuracy="1e-9", initial_K=64, max_K=512, initial_L="10", max_L=14,
        initial_interval_dps=40, max_interval_dps=80, working_dps=120)
    assert result.certified
    assert Decimal(result.achieved_relative_ratio_minus_one) <= Decimal("1e-9")
    lo, hi = endpoints(result.log10_interval)
    assert lo <= independent_even_gamma_log10("30", "0.25", 32) <= hi


def test_resource_caps_fail_closed_and_support_cap_never_overshoots():
    result = qf.certified_positive_logsf("20", ["1"] * 8, relative_accuracy="1e-50",
        initial_K=16, max_K=16, initial_L="8", max_L=9,
        initial_interval_dps=20, max_interval_dps=20, working_dps=100)
    assert not result.certified
    assert result.failure_reason
    assert result.K <= 16 and Decimal(result.L) <= 9 and result.interval_dps <= 20


@pytest.mark.parametrize("kwargs", [
    dict(initial_K=0), dict(initial_K=1.5), dict(initial_K=True),
    dict(initial_K=64, max_K=32), dict(initial_L="8.5"),
    dict(initial_L="NaN"), dict(initial_L="10", max_L=9),
    dict(initial_interval_dps=0), dict(initial_interval_dps=80, max_interval_dps=50),
    dict(working_dps=10), dict(unknown_cap=10),
])
def test_invalid_controller_options_reject_before_execution(kwargs):
    with pytest.raises(ValidationError):
        qf.certified_positive_logsf("20", ["1"] * 8, **kwargs)


@pytest.mark.parametrize("q,values", [
    ("NaN", ["1"] * 8), ("Infinity", ["1"] * 8), ("-1", ["1"] * 8),
    ("8", ["1"] * 8), ("7.9", ["1"] * 8), ("20", ["1"] * 2),
    ("20", ["0"] * 8), ("20", ["-1"] * 8), ("20", ["NaN"] * 8),
    ("20", "11111111"),
])
def test_invalid_automatic_tail_domain_rejected(q, values):
    with pytest.raises(ValidationError):
        qf.certified_positive_logsf(q, values)


def test_above_mean_test_preserves_exact_decimal_literals_at_low_ambient_precision():
    with mp.workdps(5), pytest.raises(ValidationError):
        qf.design_positive_tail_parameters("8.0000000000000000001", ["1.0000000000000000001"] * 8)


@pytest.mark.parametrize("factor", [
    np.eye(3, dtype=complex) * (1 + 1j), [[float("nan")]], [[float("inf")]], [], [1, 2, 3],
])
def test_invalid_factor_domain_rejected(factor):
    with pytest.raises(ValidationError):
        qf.certify_float64_factor_spectrum(factor)


def test_ragged_factor_shape_is_a_public_validation_error():
    with pytest.raises(ValidationError, match="rectangular"):
        qf.certify_float64_factor_spectrum([[1, 2], [3]])


def test_interval_object_display_cannot_authorize_a_false_width_certificate():
    literal = "[-1,-0.9999999999999999999999999999999999999999]"
    before = mp.iv.prec
    try:
        mp.iv.dps = 120
        value = mp.iv.mpf(literal[1:-1].split(","))
        mp.iv.dps = 5
        assert qf.directed_relative_accuracy(str(value), "1e-50").accepted
        assert not qf.directed_relative_accuracy(literal, "1e-50").accepted
        with pytest.raises(ValidationError, match="literal string"):
            qf.directed_relative_accuracy(value, "1e-50")
    finally:
        mp.iv.prec = before


def test_verified_spectrum_contains_known_binary64_diagonal_spectrum():
    result = qf.certify_float64_factor_spectrum(np.diag([1., 1.5, 2., 2.5]), candidate_digits=40, interval_dps=50)
    for lower, upper, exact in zip(result.lower, result.upper, ["6.25", "4", "2.25", "1"]):
        assert Decimal(lower) <= Decimal(exact) <= Decimal(upper)


def test_nondyadic_stored_factor_spectrum_contains_higher_precision_reference():
    A = np.array([[0.1, 0.2, 0.3], [0.2, -0.1, 0.5], [0.7, 0.2, -0.4]])
    result = qf.certify_float64_factor_spectrum(A, candidate_digits=50, interval_dps=65)
    with mp.workdps(130):
        exact_A = mp.matrix([[mp.mpf(float(x).as_integer_ratio()[0]) / float(x).as_integer_ratio()[1] for x in row] for row in A])
        reference = sorted(mp.eigsy(exact_A * exact_A.T, eigvals_only=True), reverse=True)
        for lower, upper, value in zip(result.lower, result.upper, reference):
            assert mp.mpf(lower) <= value <= mp.mpf(upper)


def test_matrix_tail_contains_independent_gamma_and_accepts_whole_outer_width():
    result = qf.certified_float64_factor_logsf("20", np.eye(8), relative_accuracy="1e-3",
        initial_candidate_digits=25, max_candidate_digits=25, interval_dps=40,
        tail_accuracy_fraction="0.5", tail_kwargs=dict(initial_K=128, max_K=512,
            initial_L="10", max_L=12, initial_interval_dps=40, max_interval_dps=60, working_dps=100))
    assert result.certified
    assert Decimal(result.achieved_relative_ratio_minus_one) <= Decimal("1e-3")
    lo, hi = endpoints(result.log10_interval)
    assert lo <= independent_even_gamma_log10("20", "1", 8) <= hi


def test_shared_precision_is_restored_on_success_and_backend_domain_failure():
    point, interval = mp.mp.prec, mp.iv.prec
    try:
        mp.mp.prec, mp.iv.prec = 79, 83
        result = qf.even_df_scaled_chi_square_logsf_interval("20", coefficient_decimal="1", df=8, interval_dps=40)
        assert endpoints(result["log10_interval"])[0] < 0
        assert (mp.mp.prec, mp.iv.prec) == (79, 83)
        with pytest.raises(ValueError, match="upper"):
            qf.interval_positive_logsf("20", ["1"] * 8, theta_decimal="0.1",
                eta_decimal="0.2", upper_decimal="20", interval_dps=100)
        assert (mp.mp.prec, mp.iv.prec) == (79, 83)
    finally:
        mp.mp.prec, mp.iv.prec = point, interval


def test_namespace_serializes_entire_backend_operation(monkeypatch):
    implementation = importlib.import_module("komorebi_math.quadratic_forms._directed_accuracy")
    active = maximum = 0
    state_lock = threading.Lock()
    def probe(*args, **kwargs):
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        mp.iv.prec = 101
        time.sleep(0.015)
        with state_lock:
            active -= 1
        return "ok"
    monkeypatch.setattr(implementation, "directed_relative_accuracy", probe)
    before = mp.iv.prec
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: qf.directed_relative_accuracy("[-1,-1]", "0.1"), range(3)))
    assert results == ["ok"] * 3
    assert maximum == 1 and mp.iv.prec == before


def test_lazy_namespace_and_actionable_missing_dependency_in_fresh_interpreter():
    program = '''
import sys
sys.modules["mpmath"] = None
import komorebi_math
from komorebi_math import quadratic_forms
from komorebi_math.exceptions import BackendUnavailableError
try:
    quadratic_forms.directed_relative_accuracy("[-1,-1]", "0.1")
except BackendUnavailableError as error:
    assert "mpmath==1.3.0" in str(error)
else:
    raise AssertionError("missing mpmath was not diagnosed")
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_result_type_exports_are_available():
    assert qf.AccuracyDrivenTailResult.__name__ == "AccuracyDrivenTailResult"
    assert qf.VerifiedSpectrum.__name__ == "VerifiedSpectrum"


def test_unqualified_interval_library_version_rejected(monkeypatch):
    monkeypatch.setattr(mp, "__version__", "99.0.0")
    with pytest.raises(BackendUnavailableError, match="qualified for mpmath==1.3.0"):
        qf.directed_relative_accuracy("[-1,-1]", "0.1")


@pytest.mark.parametrize("kwargs", [
    dict(q_decimal="-1", coefficient_decimal="1", df=4),
    dict(q_decimal="1", coefficient_decimal="0", df=4),
    dict(q_decimal="1", coefficient_decimal="1", df=3),
    dict(q_decimal="1", coefficient_decimal="1", df=4.0),
])
def test_erlang_control_rejects_invalid_distribution_domain(kwargs):
    with pytest.raises(ValidationError):
        qf.even_df_scaled_chi_square_logsf_interval(**kwargs)


def test_erlang_control_at_zero_contains_probability_one():
    result = qf.even_df_scaled_chi_square_logsf_interval("0", coefficient_decimal="1", df=4)
    lo, hi = endpoints(result["log10_interval"])
    assert lo <= 0 <= hi
