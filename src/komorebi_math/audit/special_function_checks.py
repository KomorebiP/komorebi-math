"""Installed exact stress checks for Discovery 1.1 paper machinery."""

from __future__ import annotations

from fractions import Fraction as F
from math import factorial, isqrt

from ..certificates import Exactness
from ..exceptions import ResourceLimitError, ValidationError
from .runner import CheckOutcome, StressCheck, StressContext


def _picard(scales, degree):
    """Independent dense ordinary-series integration, including even entries."""
    q, r, a, b = scales
    u = v = [F(0)] * (degree + 1)
    for _ in range(degree + 1):
        left, right = [F(0)] * degree, [F(0)] * degree
        left[0] = right[0] = F(1)
        for i, x in enumerate(u[:degree]):
            for j, y in enumerate(v[: degree - i]):
                left[i + j] += x * q**i * y * r**j
                right[i + j] += x * a**i * y * b**j
        u = [F(0)] + [x / (i + 1) for i, x in enumerate(left)]
        v = [F(0)] + [x / (i + 1) for i, x in enumerate(right)]
    return u, v


def _coupled(context: StressContext):
    from ..special_functions.paper_machinery import (
        coupled_odd_derivatives,
        coupled_reduction_discriminant,
        recover_coupled_rational_scales,
    )

    samples = context.parameters["samples"]
    records, decisions = [], []
    for _ in range(samples):
        scales = tuple(F(context.rng.randrange(1, 10), 10) for _ in range(4))
        rows = coupled_odd_derivatives(*scales, last=3)
        dense = _picard(scales, 7)
        matches = all(
            value == (row[n // 2] / factorial(n) if n % 2 else 0)
            for row, full in zip(rows, dense, strict=True)
            for n, value in enumerate(full)
        )
        jet = tuple(
            value
            for n in range(1, 4)
            for value in (rows[0][n] / factorial(2 * n + 1), rows[1][n] / factorial(2 * n + 1))
        )
        D = coupled_reduction_discriminant(*scales)
        if D:
            recovered = recover_coupled_rational_scales(*jet)
            inverse_ok = recovered == scales
            bad = (*jet[:-1], jet[-1] + F(1, 10**20))
            try:
                recover_coupled_rational_scales(*bad)
            except ValidationError:
                rejects_noise = True
            else:
                rejects_noise = False
        else:
            recovered = None
            try:
                recover_coupled_rational_scales(*jet)
            except ValidationError:
                inverse_ok = True
            else:
                inverse_ok = False
            rejects_noise = True  # Noise branch was not executed at a reduction point.
        decisions.extend((matches, inverse_ok, rejects_noise))
        records.append(
            {
                "scales": scales,
                "ordinary_jet": jet,
                "D": D,
                "dense_picard_match": matches,
                "recovered": recovered,
                "noise_control_executed": bool(D),
                "inverse_control": inverse_ok,
                "noise_rejected": rejects_noise if D else None,
            }
        )
    return CheckOutcome(
        all(decisions),
        {"cases": records},
        "dense ordinary Picard construction, exact inverse, and inconsistent-jet controls",
    )


def _tail(context: StressContext):
    from ..special_functions.paper_machinery import (
        coupled_certified_evaluation,
        coupled_integer_growth_anchor,
        coupled_odd_derivatives,
    )

    scales = (F(1, 5), F(2, 5), F(3, 5), F(7, 10))
    rows = coupled_odd_derivatives(*([F(1, 2)] * 4), last=12)
    anchor = all(row == tuple(F(1, 2**n) for n in range(13)) for row in rows)
    ratio = coupled_integer_growth_anchor(*scales)
    x = F(3, 2)
    cert = coupled_certified_evaluation(*([F(1, 2)] * 4), x, last=20, outer_radius=4)
    terms = [x ** (2 * n + 1) / F(2**n * factorial(2 * n + 1)) for n in range(41)]
    next_ratio = x * x / F(2 * 82 * 83)
    remainder = terms[-1] * next_ratio / (1 - next_ratio)
    encloses = cert.u_interval[0] <= sum(terms) and sum(terms) + remainder <= cert.u_interval[1]
    rejects_bad_anchor = False
    try:
        coupled_integer_growth_anchor(*scales, exponent=2)
    except ValidationError:
        rejects_bad_anchor = True
    return CheckOutcome(
        anchor and ratio == F(1, 2) and encloses and rejects_bad_anchor,
        {
            "sinh_derivative_anchor": anchor,
            "exact_perron_ratio": ratio,
            "sinh_interval": cert.u_interval,
            "uniform_tail": cert.tail_bound,
            "independent_sinh_remainder": remainder,
            "interval_contains_sinh": encloses,
            "invalid_characteristic_anchor_rejected": rejects_bad_anchor,
        },
        "known sinh solution and exact characteristic identity",
    )


def _evaluate(poly, x):
    value = F(0)
    for c in reversed(poly):
        value = value * x + c
    return value


def _sieve(context: StressContext):
    from ..special_functions.paper_machinery import (
        PaperMachineryBudget,
        sieve_interpolation_certificate,
        sieve_negative_detector,
    )

    delta = F(1, 10**12)
    positive = sieve_interpolation_certificate((1, 3, 5, 7, 11))
    monomials = all(
        tuple(
            sum(positive.nodes[i] ** d * positive.basis_polynomials[i][k] for i in range(5)) for k in range(5)
        )
        == tuple(F(k == d) for k in range(5))
        for d in range(5)
    )
    detector = sieve_negative_detector(delta)
    # Independent integer square-divisor predicate; omitted tail is bounded by
    # the detector's analytic envelope, not by this finite sum.
    norm = sum(
        F(1, n**4) * (1 if n % 2 else -delta) * _evaluate(detector.polynomial, (1 - F(1, n * n)) ** 2) ** 2
        for n in range(1, 101)
        if not any(n % (j * j) == 0 for j in range(2, isqrt(n) + 1))
    )
    segment_guard = threshold_guard = False
    try:
        sieve_negative_detector(delta, killed=(1, 3, 7))
    except ValidationError:
        segment_guard = True
    try:
        sieve_interpolation_certificate((1,), budget=PaperMachineryBudget(max_exact_bits=1))
    except ResourceLimitError:
        threshold_guard = True
    margin = positive.positivity_margin(delta)
    passed = (
        monomials
        and margin > 0
        and detector.degree == 7
        and detector.detects_negativity
        and norm < 0
        and norm <= detector.upper_bound
        and segment_guard
        and threshold_guard
    )
    return CheckOutcome(
        passed,
        {
            "interpolation_monomial_identity": monomials,
            "positive_degree": positive.degree,
            "positivity_margin": margin,
            "detector_degree": detector.degree,
            "norm_upper": detector.upper_bound,
            "finite_norm_cutoff_100": norm,
            "initial_segment_guard": segment_guard,
            "derived_threshold_budget_guard": threshold_guard,
        },
        "whole polynomial-space bound and normalized negative detector",
    )


def _compact(context: StressContext):
    from ..special_functions.paper_machinery import prime_exclusion_disk_bound, weak_tap_disk_bound

    prime_bounds = tuple((p, prime_exclusion_disk_bound(p)) for p in (101, 1009, 10007))
    derivative_identity = all(
        prime_exclusion_disk_bound(p, derivative_order=k) == value * F(2, p * p - 1) ** k
        for p, value in prime_bounds
        for k in range(4)
    )
    weak = weak_tap_disk_bound(F(1, 10**12))
    rejected = False
    try:
        prime_exclusion_disk_bound(9)
    except ValidationError:
        rejected = True
    return CheckOutcome(
        derivative_identity and weak < F(241890267, 10**21) and rejected,
        {
            "prime_disk_bounds": prime_bounds,
            "derivative_identity": derivative_identity,
            "weak_unit_disk_bound": weak,
            "composite_prime_rejected": rejected,
        },
        "rational disk envelopes; no truncated arithmetic sum or floating display",
    )


def special_function_checks(samples: int = 8) -> tuple[StressCheck, ...]:
    """Four bounded, isolated, exact checks; sampled denominators are fixed at ten."""
    if type(samples) is not int or not 1 <= samples <= 128:
        raise ValueError("samples must be an integer between 1 and 128")
    suite = "paper_special_functions"
    return (
        StressCheck(
            "paper.special.coupled_exact_inverse",
            suite,
            _coupled,
            Exactness.EXACT,
            {
                "rational_scale_numerators": [1, 9],
                "denominator": 10,
                "sampled_quadruples": samples,
                "dense_picard_degree": 7,
            },
            parameters={"samples": samples},
            assumptions=("Exact rational coefficients; no noisy-data approval.",),
        ),
        StressCheck(
            "paper.special.coupled_tail_anchor",
            suite,
            _tail,
            Exactness.EXACT,
            {
                "sinh_derivatives_odd_index": [0, 12],
                "tail_last_odd_index": 20,
                "independent_sinh_last_index": 40,
                "rational_x": "3/2",
                "outer_radius": 4,
            },
            assumptions=("The universal analytic tail bound uses the stated positive-contraction theorem.",),
        ),
        StressCheck(
            "paper.special.sieve_resolution",
            suite,
            _sieve,
            Exactness.EXACT,
            {
                "positive_labels": [1, 3, 5, 7, 11],
                "detector_degree": 7,
                "delta": "1/1000000000000",
                "finite_norm_cutoff": 100,
            },
            assumptions=("Finite norm is an independent check; the full-tail sign uses the exact envelope.",),
        ),
        StressCheck(
            "paper.special.compact_resolution",
            suite,
            _compact,
            Exactness.EXACT,
            {
                "primes": [101, 1009, 10007],
                "disk_radius": 5,
                "derivative_orders": [0, 3],
                "weak_tap_last_index": 40,
                "weak_tap_radius": 1,
                "weak_tap_epsilon": "1/1000000000000",
            },
            assumptions=("Bounds concern the stated arithmetic and weak-feedback families.",),
        ),
    )


__all__ = ["special_function_checks"]
