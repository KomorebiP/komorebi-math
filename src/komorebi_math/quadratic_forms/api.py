"""Validated, serialized public boundary around the directed K-COS backend."""

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from importlib import import_module
from operator import index
from threading import RLock

import numpy as np

from ..exceptions import BackendUnavailableError, ValidationError

_LOCK = RLock()


def _require_mpmath():
    try:
        mp = import_module("mpmath")
    except ImportError as error:
        raise BackendUnavailableError(
            'quadratic_forms requires mpmath: install "komorebi-math[scientific]" '
            "or mpmath==1.3.0"
        ) from error
    if getattr(mp, "__version__", None) != "1.3.0":
        raise BackendUnavailableError(
            "quadratic_forms directed arithmetic is qualified for mpmath==1.3.0; "
            "install that version before requesting a certificate"
        )
    return mp


@contextmanager
def _context():
    with _LOCK:
        mp = _require_mpmath()
        point_precision, interval_precision = mp.mp.prec, mp.iv.prec
        try:
            yield
        finally:
            mp.mp.prec = point_precision
            mp.iv.prec = interval_precision


def _call(module, function, *args, **kwargs):
    with _context():
        implementation = import_module(f"{__package__}.{module}")
        return getattr(implementation, function)(*args, **kwargs)


def _decimal(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a finite decimal")
    text = str(value).strip()
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError) as error:
        raise ValidationError(f"{name} must be a finite decimal") from error
    if not number.is_finite():
        raise ValidationError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValidationError(f"{name} must be positive")
    if nonnegative and number < 0:
        raise ValidationError(f"{name} must be nonnegative")
    return text


def _integer(value, name, *, minimum=1):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = index(value)
    except TypeError as error:
        raise ValidationError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    return result


def _spectrum(values):
    if isinstance(values, (str, bytes)):
        raise ValidationError("eigenvalue_decimals must be an iterable of coefficients")
    try:
        result = tuple(_decimal(x, "eigenvalue", positive=True) for x in values)
    except TypeError as error:
        raise ValidationError("eigenvalue_decimals must be iterable") from error
    if len(result) <= 2:
        raise ValidationError("the directed COS tail bound requires rank > 2")
    return result


def _query(q, values, *, above_mean):
    q = _decimal(q, "q", nonnegative=True)
    if above_mean and Fraction(Decimal(q)) <= sum(
        (Fraction(Decimal(x)) for x in values), Fraction(0)
    ):
        raise ValidationError("automatic positive-tilt planning requires q above the mean")
    return q


def _tail_options(options):
    defaults = dict(
        initial_K=32, initial_L="8", initial_interval_dps=50,
        max_K=131072, max_L=24, max_interval_dps=180, working_dps=260,
    )
    unknown = set(options) - defaults.keys()
    if unknown:
        raise ValidationError(f"unknown tail options: {', '.join(sorted(unknown))}")
    defaults.update(options)
    for key in ("initial_K", "max_K", "max_L", "initial_interval_dps", "max_interval_dps", "working_dps"):
        defaults[key] = _integer(defaults[key], key)
    initial_l = Decimal(_decimal(defaults["initial_L"], "initial_L", positive=True))
    if initial_l != initial_l.to_integral_value():
        raise ValidationError("initial_L must be a positive integer")
    defaults["initial_L"] = str(initial_l)
    for initial, maximum in (("initial_K", "max_K"), ("initial_interval_dps", "max_interval_dps")):
        if defaults[initial] > defaults[maximum]:
            raise ValidationError(f"{initial} cannot exceed {maximum}")
    if initial_l > defaults["max_L"]:
        raise ValidationError("initial_L cannot exceed max_L")
    if defaults["working_dps"] < 30:
        raise ValidationError("working_dps must be >= 30 for automatic planning")
    return defaults


def design_positive_tail_parameters(q_decimal, eigenvalue_decimals, *, L="10", working_dps=220, output_digits=180):
    """Plan positive design points for an above-mean upper-tail query.

    Design points need no certified saddlepoint equality: the tail evaluator
    independently checks their admissibility with directed arithmetic.
    """
    values = _spectrum(eigenvalue_decimals)
    q = _query(q_decimal, values, above_mean=True)
    L = _decimal(L, "L", positive=True)
    working_dps = _integer(working_dps, "working_dps", minimum=30)
    output_digits = _integer(output_digits, "output_digits", minimum=15)
    return _call("_interval", "design_positive_tail_parameters", q, values,
                 L=L, working_dps=working_dps, output_digits=output_digits)


def interval_positive_logsf(q_decimal, eigenvalue_decimals, *, theta_decimal, upper_decimal,
                           eta_decimal, K=128, interval_dps=90):
    """Enclose log10 P(Q >= q) for exact decimal positive coefficients.

    ``log10_interval`` carries the enclosure. Float convenience fields are
    display summaries and must not authorize a certificate decision.
    """
    values = _spectrum(eigenvalue_decimals)
    q = _query(q_decimal, values, above_mean=False)
    theta = _decimal(theta_decimal, "theta", positive=True)
    upper = _decimal(upper_decimal, "upper", positive=True)
    eta = _decimal(eta_decimal, "eta", positive=True)
    return _call("_interval", "interval_positive_logsf", q, values,
                 theta_decimal=theta, upper_decimal=upper, eta_decimal=eta,
                 K=_integer(K, "K"), interval_dps=_integer(interval_dps, "interval_dps"))


def certified_positive_logsf(q_decimal, eigenvalue_decimals, *, relative_accuracy="1e-10", **options):
    """Prove a multiplicative tail-width request or return certified=False.

    Supports strictly positive spectra of rank > 2 and q above their mean.
    Resource caps constrain every attempted transform/precision refinement.
    """
    values = _spectrum(eigenvalue_decimals)
    q = _query(q_decimal, values, above_mean=True)
    tolerance = _decimal(relative_accuracy, "relative_accuracy", positive=True)
    return _call("_accuracy", "certified_positive_logsf", q, values,
                 relative_accuracy=tolerance, **_tail_options(options))


def directed_relative_accuracy(log10_interval, requested, *, dps=120):
    """Decide p_upper / p_lower - 1 <= requested from a literal string.

    Interval objects' ordinary string conversion can round inward or collapse
    their width at ambient precision, so they are rejected at this boundary.
    """
    if not isinstance(log10_interval, str):
        raise ValidationError("log10_interval must be an outward bracketed literal string")
    tolerance = _decimal(requested, "requested", positive=True)
    return _call("_directed_accuracy", "directed_relative_accuracy", log10_interval,
                 tolerance, dps=_integer(dps, "dps"))


def even_df_scaled_chi_square_logsf_interval(q_decimal, *, coefficient_decimal, df, interval_dps=90):
    """Independent finite Erlang-tail enclosure for positive even df."""
    df = _integer(df, "df")
    if df % 2:
        raise ValidationError("df must be even")
    return _call("_interval", "even_df_scaled_chi_square_logsf_interval",
                 _decimal(q_decimal, "q", nonnegative=True),
                 coefficient_decimal=_decimal(coefficient_decimal, "coefficient", positive=True),
                 df=df, interval_dps=_integer(interval_dps, "interval_dps"))


def _factor(A):
    try:
        raw = np.asarray(A)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("A must be a finite rectangular factor matrix") from error
    if np.iscomplexobj(raw):
        raise ValidationError("A must be a real binary64 factor matrix")
    try:
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("A must convert to a finite binary64 factor matrix") from error
    if result.ndim != 2 or not all(result.shape) or not np.all(np.isfinite(result)):
        raise ValidationError("A must be a finite nonempty matrix")
    return result


def certify_float64_factor_spectrum(A, *, candidate_digits=80, interval_dps=90, working_dps=None):
    """Enclose the sorted eigenvalues of the exact stored A @ A.T.

    This dense full-basis bridge checks the eigensolver candidate with directed
    residual and orthogonality bounds. Binary64 rounding of an earlier unknown
    matrix is outside the input contract.
    """
    factor = _factor(A)
    candidate_digits = _integer(candidate_digits, "candidate_digits", minimum=15)
    interval_dps = _integer(interval_dps, "interval_dps")
    if working_dps is not None:
        working_dps = _integer(working_dps, "working_dps", minimum=30)
    return _call("_spectrum", "certify_float64_factor_spectrum", factor,
                 candidate_digits=candidate_digits, interval_dps=interval_dps, working_dps=working_dps)


def certified_float64_factor_logsf(q_decimal, A, *, relative_accuracy="1e-10",
                                 initial_candidate_digits=50, max_candidate_digits=140,
                                 candidate_step=30, interval_dps=90,
                                 tail_accuracy_fraction="0.1", tail_kwargs=None):
    """Enclose a positive-tail query from an exact stored binary64 factor.

    Requires at least three positive lower spectral bounds. Inadmissible or
    unresolved spectral-rank cases raise an error, while exhausted valid
    spectral/tail accuracy budgets return certified=False.
    """
    factor = _factor(A)
    if factor.shape[0] <= 2:
        raise ValidationError("the directed matrix tail requires at least three rows")
    q = _decimal(q_decimal, "q", nonnegative=True)
    mean = sum((Fraction(float(x)) ** 2 for x in factor.flat), Fraction(0))
    if Fraction(Decimal(q)) <= mean:
        raise ValidationError("automatic matrix-tail planning requires q above the exact trace")
    initial = _integer(initial_candidate_digits, "initial_candidate_digits", minimum=15)
    maximum = _integer(max_candidate_digits, "max_candidate_digits", minimum=15)
    if initial > maximum:
        raise ValidationError("initial_candidate_digits cannot exceed max_candidate_digits")
    fraction = _decimal(tail_accuracy_fraction, "tail_accuracy_fraction", positive=True)
    if Decimal(fraction) >= 1:
        raise ValidationError("tail_accuracy_fraction must lie in (0,1)")
    return _call("_spectrum", "certified_float64_factor_logsf", q, factor,
                 relative_accuracy=_decimal(relative_accuracy, "relative_accuracy", positive=True),
                 initial_candidate_digits=initial, max_candidate_digits=maximum,
                 candidate_step=_integer(candidate_step, "candidate_step"),
                 interval_dps=_integer(interval_dps, "interval_dps"),
                 tail_accuracy_fraction=fraction, tail_kwargs=_tail_options(dict(tail_kwargs or {})))
