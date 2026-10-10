"""Exact finite machinery from Special Function Discovery 1.1.

Formal feedback coefficients and analytic contraction certificates have separate
domains. All scalar inputs are explicit rationals, and all returned bounds are
rational. No floating fit is used to approve an inverse or a tail certificate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, factorial, isqrt, lcm
from typing import Any

from .._number_theory import is_prime
from ..exceptions import ResourceLimitError, ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import integer, rational


@dataclass(frozen=True, slots=True)
class PaperMachineryBudget:
    """Explicit finite resource caps; exceeding a cap raises, never truncates."""

    max_odd_index: int = 256
    max_exact_bits: int = 2_000_000
    max_majorant_steps: int = 20
    max_channels: int = 32
    max_nodes: int = 64
    max_label: int = 1_000_000

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            integer(
                getattr(self, name), name, minimum=0 if name in {"max_odd_index", "max_majorant_steps"} else 1
            )


_DEFAULT_BUDGET = PaperMachineryBudget()


def _budget(value):
    if value is None:
        return _DEFAULT_BUDGET
    if not isinstance(value, PaperMachineryBudget):
        raise ValidationError("budget must be a PaperMachineryBudget")
    return value


def _check(value: Fraction, budget: PaperMachineryBudget) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > budget.max_exact_bits:
        raise ResourceLimitError("exact rational exceeds max_exact_bits")
    return value


def _rat(value, name, budget):
    return _check(rational(value, name), budget)


def _index(value, budget):
    result = integer(value, "last")
    if result > budget.max_odd_index:
        raise ResourceLimitError("last exceeds max_odd_index")
    return result


def _scales(q, r, a, b, budget, *, contraction=False):
    values = tuple(_rat(x, name, budget) for x, name in zip((q, r, a, b), ("q", "r", "a", "b"), strict=True))
    if contraction and not all(0 < x < 1 for x in values):
        raise ValidationError("analytic certificates require 0 < q,r,a,b < 1")
    return values


def _power(value, exponent, budget):
    # This is a conservative resource check, not a mathematical approximation.
    if max(value.numerator.bit_length(), value.denominator.bit_length()) * exponent > budget.max_exact_bits:
        raise ResourceLimitError("exact power exceeds max_exact_bits")
    return _check(value**exponent, budget)


def _sum(terms, budget):
    result = Fraction(0)
    for term in terms:
        result = _check(result + _check(term, budget), budget)
    return result


@documented(
    ProvenanceRecord(
        identifier="special-functions.feedback-derivatives",
        title="Exact scalar feedback derivative arrays",
        status=ClaimStatus.DEFINITION,
        source="Special Function Discovery 1.1, family_lab.py; Appendix E",
        assumptions=("rational channel weights and scales", "finite last odd index"),
        limitations=("Arbitrary rational parameters define formal coefficients only.",),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def feedback_odd_derivatives(
    channels: Iterable[tuple[Any, Any, Any]], last: int = 12, *, budget: PaperMachineryBudget | None = None
) -> tuple[Fraction, ...]:
    """Return F^(2n+1)(0), n=0..last, for F'=1+sum h F(qz)F(rz).

    Empty channels give F=z. Negative, zero, and noncontracting rational scales
    are accepted as formal data. Floats and booleans are rejected.
    """
    cap = _budget(budget)
    end = _index(last, cap)
    values = []
    for item in channels:
        if len(values) >= cap.max_channels:
            raise ResourceLimitError("channel count exceeds max_channels")
        try:
            h, q, r = item
        except (TypeError, ValueError) as exc:
            raise ValidationError("each channel must contain weight and two scales") from exc
        values.append(
            tuple(_rat(x, name, cap) for x, name in zip((h, q, r), ("weight", "q", "r"), strict=True))
        )
    out = [Fraction(1)]
    for n in range(1, end + 1):
        out.append(
            _sum(
                (
                    h
                    * comb(2 * n, 2 * i + 1)
                    * _power(q, 2 * i + 1, cap)
                    * _power(r, 2 * (n - 1 - i) + 1, cap)
                    * out[i]
                    * out[n - 1 - i]
                    for h, q, r in values
                    for i in range(n)
                ),
                cap,
            )
        )
    return tuple(out)


def asymmetric_odd_derivatives(q: Any, r: Any, last: int = 12, *, budget=None):
    """Scalar one-channel specialization; derivative normalization is retained."""
    return feedback_odd_derivatives(((1, q, r),), last, budget=budget)


@documented(
    ProvenanceRecord(
        identifier="special-functions.coupled-derivatives",
        title="Exact coupled pantograph derivative arrays",
        status=ClaimStatus.DEFINITION,
        source="Special Function Discovery 1.1, coupled_lab.py; Appendix J",
        assumptions=("rational ordered scales", "U(0)=V(0)=0 and U'(0)=V'(0)=1"),
        limitations=(
            "Formal coefficients outside positive strict contractions carry no entire-growth claim.",
        ),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def coupled_odd_derivatives(
    q: Any, r: Any, a: Any, b: Any, last: int = 12, *, budget: PaperMachineryBudget | None = None
):
    """Exact derivatives for U'=1+U(qz)V(rz), V'=1+U(az)V(bz).

    Both arrays contain derivatives of orders 1,3,...,2*last+1. Divide entry n
    by (2n+1)! for the ordinary Taylor coefficient. Formal rational scales are
    allowed; analytic certificates below require positive strict contractions.
    """
    cap = _budget(budget)
    end = _index(last, cap)
    q, r, a, b = _scales(q, r, a, b, cap)
    u, v = [Fraction(1)], [Fraction(1)]
    for n in range(1, end + 1):
        terms = tuple(
            (comb(2 * n, 2 * i + 1) * u[i] * v[n - 1 - i], 2 * i + 1, 2 * (n - 1 - i) + 1) for i in range(n)
        )
        u.append(_sum((c * _power(q, i, cap) * _power(r, j, cap) for c, i, j in terms), cap))
        v.append(_sum((c * _power(a, i, cap) * _power(b, j, cap) for c, i, j in terms), cap))
    return tuple(u), tuple(v)


def coupled_cleared_derivatives(q, r, a, b, last=12, *, budget=None):
    """Return D and integer arrays D**(n*(n+1))*A_n, D=lcm of scale denominators."""
    cap = _budget(budget)
    p = _scales(q, r, a, b, cap)
    rows = coupled_odd_derivatives(*p, last, budget=cap)
    denominator = lcm(*(x.denominator for x in p))
    cleared = []
    for row in rows:
        values = []
        for n, value in enumerate(row):
            multiplier = _power(Fraction(denominator), n * (n + 1), cap)
            scaled = _check(value * multiplier, cap)
            if scaled.denominator != 1:
                raise AssertionError("coupled denominator-clearing identity failed")
            values.append(scaled.numerator)
        cleared.append(tuple(values))
    return denominator, tuple(cleared)


@documented(
    ProvenanceRecord(
        identifier="special-functions.coupled-majorant",
        title="Exact coupled disk majorant",
        status=ClaimStatus.CONDITIONAL,
        source="Special Function Discovery 1.1, coupled_lab.py; Appendix J.3",
        assumptions=("all four scales are positive strict contractions", "nonnegative rational radius"),
        limitations=(
            "Bound can be conservative.",
            "Resource exhaustion raises instead of weakening the bound.",
        ),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def coupled_disk_majorant(q, r, a, b, radius, *, budget=None):
    """Bound both components on |z|<=radius with an exact contraction ladder."""
    cap = _budget(budget)
    p = _scales(q, r, a, b, cap, contraction=True)
    x = _rat(radius, "radius", cap)
    if x < 0:
        raise ValidationError("radius must be nonnegative")
    ladder = []
    while x >= 1:
        if len(ladder) >= cap.max_majorant_steps:
            raise ResourceLimitError("contraction ladder exceeds max_majorant_steps")
        ladder.append(x)
        x = _check(x * max(p), cap)
    bound = _check(x / (1 - x * x), cap)
    for x in reversed(ladder):
        bound = _check(x * (1 + bound * bound), cap)
    return bound


@dataclass(frozen=True, slots=True)
class CoupledEvaluation:
    """Rational real enclosures and a uniform complex-disk truncation bound."""

    x: Fraction
    last_index: int
    outer_radius: Fraction
    majorant: Fraction
    tail_bound: Fraction
    u_interval: tuple[Fraction, Fraction]
    v_interval: tuple[Fraction, Fraction]


@documented(
    ProvenanceRecord(
        identifier="special-functions.coupled-tail-enclosure",
        title="Exact coupled Taylor-tail enclosure",
        status=ClaimStatus.CONDITIONAL,
        source="Special Function Discovery 1.1, coupled_lab.py; Appendix J.3",
        assumptions=("positive strict contraction scales", "rational x>=0 and outer_radius>x"),
        limitations=("No approximate parameter or coefficient input is accepted.",),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def coupled_certified_evaluation(q, r, a, b, x, last=24, outer_radius=None, *, budget=None):
    """Enclose U(x),V(x) at rational x>=0; the tail bound holds on |z|<=x."""
    cap = _budget(budget)
    end = _index(last, cap)
    p = _scales(q, r, a, b, cap, contraction=True)
    point = _rat(x, "x", cap)
    if point < 0:
        raise ValidationError("certified real evaluation requires x>=0")
    outer = (
        _rat(outer_radius, "outer_radius", cap) if outer_radius is not None else max(Fraction(1), 2 * point)
    )
    if outer <= point:
        raise ValidationError("outer_radius must be strictly greater than x")
    majorant = coupled_disk_majorant(*p, outer, budget=cap)
    tail = _check(majorant * _power(point / outer, 2 * end + 3, cap), cap)
    partials = tuple(
        _sum(
            (value * _power(point, 2 * n + 1, cap) / factorial(2 * n + 1) for n, value in enumerate(row)), cap
        )
        for row in coupled_odd_derivatives(*p, end, budget=cap)
    )
    return CoupledEvaluation(
        point,
        end,
        outer,
        majorant,
        tail,
        (partials[0], _check(partials[0] + tail, cap)),
        (partials[1], _check(partials[1] + tail, cap)),
    )


def coupled_reduction_discriminant(q, r, a, b, *, budget=None):
    """Exact D; within (0,1)^4, D=0 iff q=b or qb=ar."""
    cap = _budget(budget)
    q, r, a, b = _scales(q, r, a, b, cap, contraction=True)
    p, k = q * r, a * b
    return _check(p * k * a * r * (q * q - b * b) * (q * b - a * r) / 15, cap)


@documented(
    ProvenanceRecord(
        identifier="special-functions.coupled-growth-anchor",
        title="Exact integer coupled characteristic anchor",
        status=ClaimStatus.CONDITIONAL,
        source="Special Function Discovery 1.1, research_v11/coupled_growth_proof.md, Theorems 2-3",
        assumptions=(
            "positive strict contraction scales",
            "positive integer exponent satisfies the characteristic identity",
        ),
        limitations=("No general real-exponent root solver or ordinary type-limit assertion.",),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def coupled_integer_growth_anchor(q, r, a, b, exponent=1, *, budget=None):
    """Verify an exact integer-order Perron anchor and return w_U/w_V.

    The characteristic identity (1-q**rho)*(1-b**rho)=(a*r)**rho
    must hold exactly. The source growth theorem then identifies this exponent
    as the common order of the normalized pair. This function does not solve
    the characteristic equation at arbitrary real exponents.
    """
    cap = _budget(budget)
    p = _scales(q, r, a, b, cap, contraction=True)
    degree = _index(exponent, cap)
    if degree == 0:
        raise ValidationError("exponent must be a positive integer")
    qv, rv, av, bv = tuple(_power(x, degree, cap) for x in p)
    if _check((1 - qv) * (1 - bv) - av * rv, cap) != 0:
        raise ValidationError("exponent fails the exact Perron characteristic identity")
    return _check(rv / (1 - qv), cap)


def _rational_sqrt(value):
    if value < 0:
        return None
    numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
    if numerator * numerator != value.numerator or denominator * denominator != value.denominator:
        return None
    return Fraction(numerator, denominator)


def _roots(total, product, cap):
    total, product = _check(total, cap), _check(product, cap)
    root = _rational_sqrt(_check(total * total - 4 * product, cap))
    if root is None:
        raise ValidationError("quadratic squared-scale candidates must be real rationals")
    return tuple(sorted({(total - root) / 2, (total + root) / 2}))


@documented(
    ProvenanceRecord(
        identifier="special-functions.coupled-exact-inverse",
        title="Degree-seven coupled exact inverse",
        status=ClaimStatus.CONDITIONAL,
        source="Special Function Discovery 1.1, coupled_inverse.py; Appendix I",
        assumptions=(
            "six exact ordinary Taylor coefficients",
            "positive strict contractions",
            "D!=0",
            "rational squared-scale candidates",
        ),
        limitations=(
            "No finite-precision stability guarantee.",
            "Reduction surfaces and inconsistent data are rejected.",
        ),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def recover_coupled_squared_scales(u1, v1, u2, v2, u3, v3, *, budget=None):
    """Recover (q²,r²,a²,b²) from degrees 3,5,7 of the labeled pair.

    This exact-data domain requires rational quadratic candidates. It does not
    implement arbitrary algebraic-number recovery or certify noisy coefficients.
    """
    cap = _budget(budget)
    u1, v1, u2, v2, u3, v3 = tuple(
        _rat(x, name, cap)
        for x, name in zip((u1, v1, u2, v2, u3, v3), ("u1", "v1", "u2", "v2", "u3", "v3"), strict=True)
    )
    p, k = _check(3 * u1, cap), _check(3 * v1, cap)
    if min(p, k, u2, v2, u3, v3) <= 0:
        raise ValidationError("positive coupled Taylor coefficients are required")
    if _check(k * k * u2 - p * p * v2, cap) == 0:
        raise ValidationError("D=0 lies on the scalar reduction surfaces")
    q_selected = tuple(
        x
        for x in _roots(15 * u2 / (p * p), k * p, cap)
        if 0 < x < 1
        and 0 < p * p / x < 1
        and _check(p * (x * x * u2 + p * p * u1 * v1 + (p * p / x) ** 2 * v2) / 7, cap) == u3
    )
    a_selected = tuple(
        y
        for y in _roots(15 * v2 / (p * k), k**3 / p, cap)
        if 0 < y < 1
        and 0 < k * k / y < 1
        and _check(k * (y * y * u2 + k * k * u1 * v1 + (k * k / y) ** 2 * v2) / 7, cap) == v3
    )
    if len(q_selected) != 1 or len(a_selected) != 1:
        raise ValidationError("coefficients fail unique contraction-domain recovery")
    x, y = q_selected[0], a_selected[0]
    return tuple(_check(value, cap) for value in (x, p * p / x, y, k * k / y))


def recover_coupled_rational_scales(u1, v1, u2, v2, u3, v3, *, budget=None):
    """Exact rational-scale subset of recover_coupled_squared_scales."""
    result = tuple(
        _rational_sqrt(x) for x in recover_coupled_squared_scales(u1, v1, u2, v2, u3, v3, budget=budget)
    )
    if any(x is None for x in result):
        raise ValidationError("recovered scales are not rational; use squared scales")
    return result


def _label(value, name, cap):
    result = integer(value, name, minimum=1)
    if result > cap.max_label:
        raise ResourceLimitError(f"{name} exceeds max_label")
    return result


def _prime(value, cap):
    result = _label(value, "prime", cap)
    if not is_prime(result):
        raise ValidationError("prime must be a positive prime integer")
    return result


def _squarefree(value):
    return all(value % (k * k) for k in range(2, isqrt(value) + 1))


def _node(value):
    return (1 - Fraction(1, value * value)) ** 2


@documented(
    ProvenanceRecord(
        identifier="special-functions.prime-resolution-bound",
        title="Rational prime-exclusion disk envelope",
        status=ClaimStatus.CONDITIONAL,
        source="Special Function Discovery 1.1, research_v11/resolution_atlas.md, RA-4-RA-5",
        assumptions=(
            "validated prime",
            "a=beta=2 arithmetic family",
            "nonnegative integer radius and derivative order",
        ),
        limitations=("Sufficient envelope; no uniform prime-recovery assertion.",),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def prime_exclusion_disk_bound(prime, radius=5, derivative_order=0, *, budget=None):
    """Rational bound for sup |D_p^(k)(s)|, |s|<=R, integer R,k>=0.

    Here D_p=A_1-A_p for A_Q(s)=sum squarefree n>=2, (n,Q)=1,
    n^-4*(1-n^-2)^(2s). Uses D(4)<4/3 and lambda_p<=2/(p²-1).
    """
    cap = _budget(budget)
    p = _prime(prime, cap)
    radius = _index(radius, cap)
    degree = _index(derivative_order, cap)
    return _check(
        Fraction(4, 3 * (p**4 + 1))
        * _power(Fraction(p * p, p * p - 1), 2 * radius, cap)
        * _power(Fraction(2, p * p - 1), degree, cap),
        cap,
    )


def _multiply(left, right, cap):
    out = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            out[i + j] = _check(out[i + j] + x * y, cap)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class SieveInterpolation:
    """Exact Lagrange envelope for the full polynomial space through degree d."""

    prime: int
    labels: tuple[int, ...]
    nodes: tuple[Fraction, ...]
    basis_polynomials: tuple[tuple[Fraction, ...], ...]
    sup_bounds: tuple[Fraction, ...]
    constant: Fraction
    _budget: PaperMachineryBudget = field(default=_DEFAULT_BUDGET, repr=False, compare=False)

    @property
    def degree(self):
        return len(self.labels) - 1

    @property
    def defect_threshold(self):
        return _check(Fraction(3 * self.prime**4, 4 * self.constant), self._budget)

    def positivity_margin(self, delta):
        defect = _rat(delta, "delta", self._budget)
        if defect < 0:
            raise ValidationError("delta must be nonnegative")
        return _check(1 - defect / self.defect_threshold, self._budget)


@documented(
    ProvenanceRecord(
        identifier="special-functions.sieve-interpolation",
        title="Exact polynomial-space positivity envelope",
        status=ClaimStatus.CONDITIONAL,
        source="Special Function Discovery 1.1, research_v11/resolution_atlas.md, low-degree interpolation bound",
        assumptions=(
            "distinct positive prime-free squarefree labels",
            "one negative prime shelf with strength -delta",
        ),
        limitations=(
            "Only a strictly positive margin is sufficient for positivity.",
            "No optimal resolution claim.",
        ),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def sieve_interpolation_certificate(labels, prime=2, *, budget=None):
    """Bound Q_delta(f) below on polynomials of degree <=len(labels)-1.

    Labels are distinct positive squarefree integers not divisible by prime.
    A positive positivity_margin(delta) certifies strict positivity on every
    nonzero polynomial in that space. Equality or a negative margin is inconclusive.
    """
    cap = _budget(budget)
    p = _prime(prime, cap)
    selected = []
    for item in labels:
        if len(selected) >= cap.max_nodes:
            raise ResourceLimitError("label count exceeds max_nodes")
        value = _label(item, "label", cap)
        if value % p == 0 or not _squarefree(value) or value in selected:
            raise ValidationError("labels must be distinct prime-free squarefree integers")
        selected.append(value)
    if not selected:
        raise ValidationError("at least one label is required")
    nodes = tuple(_node(n) for n in selected)
    bounds, polynomials = [], []
    for i, x in enumerate(nodes):
        bound, poly = Fraction(1), (Fraction(1),)
        for j, y in enumerate(nodes):
            if i != j:
                bound = _check(bound * max(y, 1 - y) / abs(x - y), cap)
                poly = _multiply(poly, (-y / (x - y), 1 / (x - y)), cap)
        bounds.append(bound)
        polynomials.append(poly)
    constant = _sum((n**4 * c * c for n, c in zip(selected, bounds, strict=True)), cap)
    result = SieveInterpolation(p, tuple(selected), nodes, tuple(polynomials), tuple(bounds), constant, cap)
    _ = result.defect_threshold  # Validate the derived output against the same budget.
    return result


@dataclass(frozen=True, slots=True)
class SieveDetector:
    """A normalized exact polynomial and a rigorous upper bound on its signed norm."""

    prime: int
    delta: Fraction
    killed: tuple[int, ...]
    suppression_power: int
    polynomial: tuple[Fraction, ...]
    following_label: int
    tail_constant: Fraction
    tail_ratio: Fraction
    upper_bound: Fraction

    @property
    def degree(self):
        return len(self.polynomial) - 1

    @property
    def detects_negativity(self):
        return self.upper_bound < 0


@documented(
    ProvenanceRecord(
        identifier="special-functions.sieve-detector",
        title="Exact signed-sieve negative detector",
        status=ClaimStatus.CONDITIONAL,
        source="Special Function Discovery 1.1, research_v11/resolution_atlas.md, tailored detector bound",
        assumptions=(
            "positive rational delta",
            "killed nodes form the complete ordered positive initial segment",
        ),
        limitations=("Nonnegative upper bound is inconclusive.", "Detection degree need not be minimal."),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def sieve_negative_detector(delta, killed=(1, 3, 5, 7), suppression_power=3, prime=2, *, budget=None):
    """Certify a negative norm when upper_bound<0, otherwise return inconclusive.

    Q_delta(f)=sum_{squarefree p-free n} n^-4*f(x_n)^2
              -delta*sum_{squarefree p|n} n^-4*f(x_n)^2.
    The killed list MUST be a complete initial segment of the positive labels.
    The polynomial equals one at x_prime and vanishes at every killed node.
    """
    cap = _budget(budget)
    p = _prime(prime, cap)
    defect = _rat(delta, "delta", cap)
    if defect <= 0:
        raise ValidationError("delta must be positive")
    power = _index(suppression_power, cap)
    labels = sieve_interpolation_certificate(killed, p, budget=cap).labels
    expected = []
    candidate = 1
    while len(expected) <= len(labels):
        if candidate > cap.max_label:
            raise ResourceLimitError("positive-label search exceeds max_label")
        if candidate % p and _squarefree(candidate):
            expected.append(candidate)
        candidate += 1
    if labels != tuple(expected[:-1]):
        raise ValidationError("killed labels must be the complete ordered initial segment")
    following = expected[-1]
    target = _node(p)
    poly, constant = (Fraction(1),), Fraction(1)
    for n in labels:
        y = _node(n)
        poly = _multiply(poly, (-y / (target - y), 1 / (target - y)), cap)
        constant = _check(constant * (1 - y) / abs(target - y), cap)
    for _ in range(power):
        poly = _multiply(poly, (1 / (1 - target), -1 / (1 - target)), cap)
    ratio = (1 - _node(following)) / (1 - target)
    upper = _check(
        -defect / Fraction(p**4)
        + constant * constant * _power(ratio, 2 * power, cap) / (3 * labels[-1] ** 3),
        cap,
    )
    return SieveDetector(p, defect, labels, power, poly, following, constant, ratio, upper)


@documented(
    ProvenanceRecord(
        identifier="special-functions.weak-tap-resolution",
        title="Exact weak-tap compact bound",
        status=ClaimStatus.CONDITIONAL,
        source="Special Function Discovery 1.1, research_v11/resolution_atlas.md, RA-1-RA-2",
        assumptions=("0<=epsilon<=1", "0<=rational radius<=1", "finite tail multiplier is nonincreasing"),
        limitations=("Compact closeness does not preserve global entire order.",),
        verification_suite="tests/test_special_functions_paper_machinery.py",
    )
)
def weak_tap_disk_bound(epsilon, radius=1, last=40, *, budget=None):
    """Exact compact bound for F_epsilon-F_0 on |z|<=R<=1.

    F_epsilon'=1+(1-epsilon)F_epsilon(z/4)^2+epsilon*F_epsilon(3z/4)^2.
    Requires 0<=epsilon<=1 and a last index satisfying the monotone-tail guard.
    """
    cap = _budget(budget)
    end = _index(last, cap)
    weight = _rat(epsilon, "epsilon", cap)
    radius = _rat(radius, "radius", cap)
    if not 0 <= weight <= 1 or not 0 <= radius <= 1 or end < 1:
        raise ValidationError("require 0<=epsilon<=1, 0<=radius<=1 and last>=1")
    t = Fraction(3, 4) * radius
    if (end + 2) * t * t > end + 1:
        raise ValidationError("last index fails the monotone-tail condition")
    row = asymmetric_odd_derivatives(Fraction(3, 4), Fraction(3, 4), end, budget=cap)
    finite = _sum(
        (
            n * value * _power(radius, 2 * n + 1, cap) / factorial(2 * n + 1)
            for n, value in enumerate(row)
            if n
        ),
        cap,
    )
    tail = _check(12 * (end + 1) * _power(t, 2 * end + 3, cap), cap)
    return _check(weight * (finite + tail), cap)


__all__ = [
    "CoupledEvaluation",
    "PaperMachineryBudget",
    "SieveDetector",
    "SieveInterpolation",
    "asymmetric_odd_derivatives",
    "coupled_certified_evaluation",
    "coupled_cleared_derivatives",
    "coupled_disk_majorant",
    "coupled_integer_growth_anchor",
    "coupled_odd_derivatives",
    "coupled_reduction_discriminant",
    "feedback_odd_derivatives",
    "prime_exclusion_disk_bound",
    "recover_coupled_rational_scales",
    "recover_coupled_squared_scales",
    "sieve_interpolation_certificate",
    "sieve_negative_detector",
    "weak_tap_disk_bound",
]
