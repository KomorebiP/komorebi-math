"""Independent finite checks of the Discovery 1.1 production machinery."""

from fractions import Fraction as F
from itertools import product
from math import factorial

import pytest

from komorebi_math.exceptions import ResourceLimitError, ValidationError
from komorebi_math.special_functions.paper_machinery import (
    PaperMachineryBudget,
    asymmetric_odd_derivatives,
    coupled_certified_evaluation,
    coupled_cleared_derivatives,
    coupled_disk_majorant,
    coupled_integer_growth_anchor,
    coupled_odd_derivatives,
    coupled_reduction_discriminant,
    feedback_odd_derivatives,
    prime_exclusion_disk_bound,
    recover_coupled_rational_scales,
    recover_coupled_squared_scales,
    sieve_interpolation_certificate,
    sieve_negative_detector,
    weak_tap_disk_bound,
)

ANCHOR = tuple(map(F, ("1/5", "2/5", "3/5", "7/10")))


def dense_picard(scales, degree):
    """Ordinary dense power-series integration, independently of odd recurrence."""
    q, r, a, b = scales
    u = v = [F(0)] * (degree + 1)
    for _ in range(degree + 1):
        rhs_u, rhs_v = [F(0)] * degree, [F(0)] * degree
        rhs_u[0] = rhs_v[0] = F(1)
        for i, x in enumerate(u[:degree]):
            for j, y in enumerate(v[: degree - i]):
                rhs_u[i + j] += x * q**i * y * r**j
                rhs_v[i + j] += x * a**i * y * b**j
        u = [F(0)] + [x / (i + 1) for i, x in enumerate(rhs_u)]
        v = [F(0)] + [x / (i + 1) for i, x in enumerate(rhs_v)]
    return u, v


@pytest.mark.parametrize(
    "scales", [ANCHOR, (F(1, 3), F(2, 3), F(1, 2), F(3, 4)), (F(-1, 2), F(2), F(0), F(1, 3))]
)
def test_coupled_recurrence_against_dense_picard(scales):
    rows = coupled_odd_derivatives(*scales, last=7)
    for row, dense in zip(rows, dense_picard(scales, 15), strict=True):
        for n in range(16):
            assert dense[n] == (row[n // 2] / factorial(n) if n % 2 else 0)


def test_scalar_anchors_and_empty_feedback():
    assert feedback_odd_derivatives([], 4) == (F(1), F(0), F(0), F(0), F(0))
    # F'=1+F(z/2)^2 has F=sqrt(2)*sinh(z/sqrt(2)).
    expected = tuple(F(1, 2**n) for n in range(13))
    assert asymmetric_odd_derivatives("1/2", "1/2", 12) == expected
    assert coupled_odd_derivatives(*([F(1, 2)] * 4), 12) == (expected, expected)
    # An independent tangent construction divides sine's series by cosine's.
    sine, cosine = [], []
    for n in range(26):
        sine.append(F((-1) ** ((n - 1) // 2), factorial(n)) if n % 2 else F(0))
        cosine.append(F((-1) ** (n // 2), factorial(n)) if not n % 2 else F(0))
    quotient = []
    for n in range(26):
        quotient.append(sine[n] - sum(cosine[k] * quotient[n - k] for k in range(1, n + 1)))
    actual = asymmetric_odd_derivatives(1, 1, 12)
    assert tuple(actual[n] / factorial(2 * n + 1) for n in range(13)) == tuple(quotient[1::2])


def test_clearing_and_correct_component_swap():
    u, v = coupled_odd_derivatives(*ANCHOR, 12)
    assert coupled_odd_derivatives(*ANCHOR[::-1], 12) == (v, u)
    assert coupled_odd_derivatives(*ANCHOR[2:], *ANCHOR[:2], 12) != (v, u)
    denominator, cleared = coupled_cleared_derivatives(*ANCHOR, 12)
    assert denominator == 10
    for row, integers in zip((u, v), cleared, strict=True):
        assert all(isinstance(x, int) for x in integers)
        assert all(F(x) == row[n] * 10 ** (n * (n + 1)) for n, x in enumerate(integers))


def test_exact_integer_perron_anchors():
    assert coupled_integer_growth_anchor(*ANCHOR) == F(1, 2)
    assert coupled_integer_growth_anchor("3/5", "3/5", "4/5", "4/5", exponent=2) == F(9, 16)
    with pytest.raises(ValidationError, match="characteristic"):
        coupled_integer_growth_anchor(*ANCHOR, exponent=2)
    with pytest.raises(ValidationError, match="positive"):
        coupled_integer_growth_anchor(*ANCHOR, exponent=0)


def jet(scales):
    u, v = coupled_odd_derivatives(*scales, 3)
    return tuple(
        value for n in range(1, 4) for value in (u[n] / factorial(2 * n + 1), v[n] / factorial(2 * n + 1))
    )


@pytest.mark.parametrize("scales", list(product((F(1, 5), F(2, 5), F(3, 5), F(4, 5)), repeat=4)))
def test_exact_inverse_grid_and_reduction_locus(scales):
    q, r, a, b = scales
    values = jet(scales)
    p, k = q * r, a * b
    D = k * k * values[2] - p * p * values[3]
    assert coupled_reduction_discriminant(*scales) == D
    assert (D == 0) == (q == b or q * b == a * r)
    if D:
        assert recover_coupled_rational_scales(*values) == scales
        assert recover_coupled_squared_scales(*values) == tuple(x * x for x in scales)
    else:
        with pytest.raises(ValidationError, match="reduction"):
            recover_coupled_rational_scales(*values)


def test_inverse_rejects_noise_and_requires_rational_input():
    values = list(jet(ANCHOR))
    values[-1] += F(1, 10**20)
    with pytest.raises(ValidationError, match="recovery"):
        recover_coupled_rational_scales(*values)
    values = list(jet(ANCHOR))
    values[0] = float(values[0])
    with pytest.raises(ValidationError, match="rational"):
        recover_coupled_rational_scales(*values)


def test_squared_scale_inverse_has_a_broader_rational_domain():
    # q=sqrt(2)/3, r=sqrt(2)/4, a=sqrt(3)/4, b=sqrt(3)/5.
    # Their products and squared scales are rational, so the ordinary jet is exact.
    qs, rs, av, bs = F(2, 9), F(1, 8), F(3, 16), F(3, 25)
    p, k = F(1, 6), F(3, 20)
    u, v = [F(1)], [F(1)]
    for n in range(1, 4):
        u.append(p * sum(qs**i * rs ** (n - 1 - i) * u[i] * v[n - 1 - i] for i in range(n)) / (2 * n + 1))
        v.append(k * sum(av**i * bs ** (n - 1 - i) * u[i] * v[n - 1 - i] for i in range(n)) / (2 * n + 1))
    values = tuple(c for n in range(1, 4) for c in (u[n], v[n]))
    assert recover_coupled_squared_scales(*values) == (qs, rs, av, bs)
    with pytest.raises(ValidationError, match="not rational"):
        recover_coupled_rational_scales(*values)


@pytest.mark.parametrize("x", (F(0), F(1, 4), F(1, 2), F(1), F(3, 2)))
def test_exact_tail_contains_longer_partial(x):
    cert = coupled_certified_evaluation(*ANCHOR, x, last=12, outer_radius=max(1, 2 * x))
    rows = coupled_odd_derivatives(*ANCHOR, 24)
    for interval, row in zip((cert.u_interval, cert.v_interval), rows, strict=True):
        refined = sum(value * x ** (2 * n + 1) / factorial(2 * n + 1) for n, value in enumerate(row))
        assert interval[0] <= refined <= interval[1]
        assert interval[1] - interval[0] == cert.tail_bound


def test_known_sinh_enclosure_and_worked_tail():
    x = F(3, 2)
    cert = coupled_certified_evaluation(*([F(1, 2)] * 4), x, last=20, outer_radius=4)
    terms = [x ** (2 * n + 1) / F(2**n * factorial(2 * n + 1)) for n in range(41)]
    ratio = x * x / F(2 * 82 * 83)
    remaining = terms[-1] * ratio / (1 - ratio)
    assert cert.u_interval[0] <= sum(terms)
    assert sum(terms) + remaining <= cert.u_interval[1]
    assert coupled_disk_majorant(*ANCHOR, F(1, 2)) == F(2, 3)
    assert coupled_disk_majorant(*ANCHOR, 2) == F(3542485607810948, 2401490025)
    worked = coupled_certified_evaluation(*ANCHOR, 1, last=30, outer_radius=2)
    assert worked.tail_bound < F(16, 10**14)


def evaluate(poly, x):
    value = F(0)
    for c in reversed(poly):
        value = value * x + c
    return value


def test_interpolation_whole_space_and_positive_degree_four():
    cert = sieve_interpolation_certificate((1, 3, 5, 7, 11))
    assert cert.degree == 4
    for i, polynomial in enumerate(cert.basis_polynomials):
        assert tuple(evaluate(polynomial, x) for x in cert.nodes) == tuple(F(i == j) for j in range(5))
    # Reconstruct every basis monomial as an exact coefficient identity.
    for power in range(5):
        rebuilt = tuple(
            sum(cert.nodes[i] ** power * cert.basis_polynomials[i][k] for i in range(5)) for k in range(5)
        )
        assert rebuilt == tuple(F(k == power) for k in range(5))
    assert cert.defect_threshold > F(79790, 10**15)
    assert cert.positivity_margin(F(1, 10**12)) > 0
    assert cert.positivity_margin(cert.defect_threshold) == 0


def test_detector_exact_witness_and_finite_norm():
    cert = sieve_negative_detector(F(1, 10**12))
    assert cert.degree == 7 and cert.following_label == 11
    assert cert.tail_constant == F(756482048, 20777538825)
    assert cert.tail_ratio == F(3856, 102487)
    assert cert.detects_negativity
    assert cert.upper_bound < -F(58845708899, 10**24)
    assert evaluate(cert.polynomial, (1 - F(1, 4)) ** 2) == 1
    assert all(evaluate(cert.polynomial, (1 - F(1, n * n)) ** 2) == 0 for n in cert.killed)

    # Use an independent squarefree factorization predicate in the finite norm.
    def squarefree(n):
        trial = 2
        while trial * trial <= n:
            if n % (trial * trial) == 0:
                return False
            trial += 1
        return True

    norm = sum(
        F(1, n**4) * (1 if n % 2 else -F(1, 10**12)) * evaluate(cert.polynomial, (1 - F(1, n * n)) ** 2) ** 2
        for n in range(1, 101)
        if squarefree(n)
    )
    assert norm < 0 and norm <= cert.upper_bound


def test_inconclusive_detector_and_initial_segment_guard():
    assert not sieve_negative_detector(F(1, 10**20), killed=(1,), suppression_power=0).detects_negativity
    for killed in ((1, 3, 7), (3, 1), (1, 3, 5, 11)):
        with pytest.raises(ValidationError, match="initial segment"):
            sieve_negative_detector(F(1, 10**12), killed=killed)


@pytest.mark.parametrize(
    "prime, ceiling", [(101, F(1283, 10**11)), (1009, F(1287, 10**15)), (10007, F(1330, 10**19))]
)
def test_prime_exact_disk_bound(prime, ceiling):
    base = prime_exclusion_disk_bound(prime)
    assert 0 < base < ceiling
    for k in range(4):
        assert prime_exclusion_disk_bound(prime, derivative_order=k) == base * F(2, prime * prime - 1) ** k


def test_weak_tap_compact_certificate():
    assert weak_tap_disk_bound(F(1, 10**12)) < F(241890267, 10**21)
    assert weak_tap_disk_bound(0) == 0
    assert weak_tap_disk_bound(1, radius=0) == 0


@pytest.mark.parametrize(
    "operation",
    [
        lambda: coupled_odd_derivatives(True, 1, 1, 1),
        lambda: coupled_odd_derivatives(0.5, 1, 1, 1),
        lambda: coupled_odd_derivatives(1, 1, 1, 1, True),
        lambda: coupled_certified_evaluation(1, "1/2", "1/2", "1/2", 1),
        lambda: coupled_certified_evaluation(*ANCHOR, -1),
        lambda: coupled_certified_evaluation(*ANCHOR, 1, outer_radius=1),
        lambda: prime_exclusion_disk_bound(9),
        lambda: prime_exclusion_disk_bound(101, radius="5"),
        lambda: sieve_interpolation_certificate(()),
        lambda: sieve_interpolation_certificate((1, 1)),
        lambda: sieve_interpolation_certificate((1, 9)),
        lambda: sieve_negative_detector(-1),
        lambda: weak_tap_disk_bound(F(3, 2)),
        lambda: PaperMachineryBudget(max_exact_bits=True),
    ],
)
def test_invalid_domains(operation):
    with pytest.raises(ValidationError):
        operation()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: coupled_odd_derivatives(*ANCHOR, 2, budget=PaperMachineryBudget(max_odd_index=1)),
        lambda: coupled_odd_derivatives(*ANCHOR, 4, budget=PaperMachineryBudget(max_exact_bits=8)),
        lambda: coupled_disk_majorant(*ANCHOR, 2, budget=PaperMachineryBudget(max_majorant_steps=0)),
        lambda: coupled_disk_majorant(*([F(99, 100)] * 4), 2),
        lambda: feedback_odd_derivatives(((1, 1, 1),) * 3, budget=PaperMachineryBudget(max_channels=2)),
        lambda: sieve_interpolation_certificate((1, 3, 5), budget=PaperMachineryBudget(max_nodes=2)),
        lambda: prime_exclusion_disk_bound(10007, budget=PaperMachineryBudget(max_label=1000)),
        lambda: recover_coupled_squared_scales(*jet(ANCHOR), budget=PaperMachineryBudget(max_exact_bits=8)),
        lambda: sieve_interpolation_certificate((1,), budget=PaperMachineryBudget(max_exact_bits=1)),
        lambda: sieve_interpolation_certificate(
            (1,), budget=PaperMachineryBudget(max_exact_bits=8)
        ).positivity_margin(10**4),
    ],
)
def test_resource_caps_raise_instead_of_truncating(operation):
    with pytest.raises(ResourceLimitError):
        operation()


def test_installed_paper_stress_factory_and_exact_check_evidence():
    import random

    from komorebi_math.audit.runner import StressContext
    from komorebi_math.audit.special_function_checks import special_function_checks
    from komorebi_math.certificates import Exactness

    checks = special_function_checks(samples=3)
    assert len(checks) == 4 and len({x.identifier for x in checks}) == 4
    for check in checks:
        assert check.exactness is Exactness.EXACT
        result = check.function(StressContext(20261009, random.Random(20261009), check.parameters))
        assert result.passed
        assert result.to_dict()["evidence"]
    for value in (True, 0, 129, F(2)):
        with pytest.raises(ValueError):
            special_function_checks(value)
