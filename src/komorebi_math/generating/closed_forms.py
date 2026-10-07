"""Exact finite-prefix closed-form and recurrence recognition tools.

Every result distinguishes a proved finite-prefix identity from an infinite
claim.  A guessed model becomes an infinite certificate only after an
independent recurrence, functional equation, or symbolic proof is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from importlib import import_module
from typing import Any, Iterable, Sequence

from ..exceptions import BackendUnavailableError


def _sympy() -> Any:
    """Return the optional SymPy backend without making it a base dependency."""

    try:
        return import_module("sympy")
    except ModuleNotFoundError as exc:
        raise BackendUnavailableError(
            "symbolic generating-function recognition requires the optional SymPy dependency"
        ) from exc


def _fractions(values: Iterable[int | Fraction]) -> tuple[Fraction, ...]:
    return tuple(Fraction(value) for value in values)


def _primitive_integer_vector(values: Sequence[Any]) -> tuple[int, ...]:
    denominators = [int(value.q) for value in values]
    lcm = 1
    for denominator in denominators:
        lcm = lcm * denominator // gcd(lcm, denominator)
    integers = [int(value * lcm) for value in values]
    common = 0
    for value in integers:
        common = gcd(common, abs(value))
    if common:
        integers = [value // common for value in integers]
    first = next((value for value in integers if value), 1)
    if first < 0:
        integers = [-value for value in integers]
    return tuple(integers)


def _fraction_vector(integers: Sequence[int]) -> tuple[Fraction, ...]:
    return tuple(Fraction(value) for value in integers)


@dataclass(frozen=True, slots=True)
class LinearRecurrenceGuess:
    """Bounded exact candidate for a constant-coefficient recurrence."""

    order: int
    coefficients: tuple[Fraction, ...]
    fit_terms: int
    holdout_terms: int
    exact_on_prefix: bool

    def equation(self) -> str:
        terms = " + ".join(f"({coefficient}) a[n+{j}]" for j, coefficient in enumerate(self.coefficients))
        return f"a[n+{self.order}] = {terms}"


@dataclass(frozen=True, slots=True)
class HypergeometricGuess:
    """Bounded rational candidate for a consecutive-term ratio."""

    numerator: tuple[int, ...]
    denominator: tuple[int, ...]
    fit_relations: int
    holdout_relations: int
    exact_on_prefix: bool

    def ratio_expression(self) -> Any:
        sp = _sympy()
        n = sp.Symbol("n")
        p = sum(coefficient * n**degree for degree, coefficient in enumerate(self.numerator))
        q = sum(coefficient * n**degree for degree, coefficient in enumerate(self.denominator))
        return sp.cancel(p / q)


@dataclass(frozen=True, slots=True)
class PRecursiveGuess:
    """Bounded polynomial-coefficient recurrence candidate."""

    order: int
    degree: int
    polynomials: tuple[tuple[int, ...], ...]
    fit_relations: int
    holdout_relations: int
    exact_on_prefix: bool

    def expression(self) -> Any:
        sp = _sympy()
        n = sp.Symbol("n")
        symbols = sp.Function("a")
        return sp.expand(
            sum(
                sum(coefficient * n**degree for degree, coefficient in enumerate(polynomial))
                * symbols(n + shift)
                for shift, polynomial in enumerate(self.polynomials)
            )
        )


@dataclass(frozen=True, slots=True)
class AlgebraicGFGuess:
    """Bounded algebraic relation candidate for an ordinary series."""

    x_degree: int
    y_degree: int
    monomials: tuple[tuple[int, int, int], ...]
    fit_coefficients: int
    holdout_coefficients: int
    exact_on_prefix: bool

    def expression(self) -> Any:
        sp = _sympy()
        x, y = sp.symbols("x y")
        return sp.expand(sum(coefficient * x**i * y**j for i, j, coefficient in self.monomials))


@dataclass(frozen=True, slots=True)
class QuasiPolynomialGuess:
    """Bounded residue-class polynomial candidate."""

    period: int
    degree: int
    residue_values: tuple[tuple[Fraction, ...], ...]
    fit_terms: int
    holdout_terms: int
    exact_on_prefix: bool


def guess_linear_recurrence(
    values: Iterable[int | Fraction],
    *,
    max_order: int = 12,
    holdout: int | None = None,
) -> LinearRecurrenceGuess | None:
    """Guess the smallest exact constant-coefficient recurrence."""

    sp = _sympy()
    data = _fractions(values)
    if len(data) < 4:
        return None
    hold = max(1, len(data) // 4) if holdout is None else holdout
    fit_end = len(data) - hold
    for order in range(1, min(max_order, fit_end // 2) + 1):
        rows = []
        rhs = []
        for n in range(fit_end - order):
            rows.append([sp.Rational(data[n + j].numerator, data[n + j].denominator) for j in range(order)])
            rhs.append(sp.Rational(data[n + order].numerator, data[n + order].denominator))
        matrix = sp.Matrix(rows)
        target = sp.Matrix(rhs)
        try:
            solution_set = sp.linsolve((matrix, target))
        except ValueError:
            continue
        if solution_set is sp.EmptySet:
            continue
        candidates = list(solution_set)
        if not candidates:
            continue
        candidate = candidates[0]
        if any(value.free_symbols for value in candidate):
            candidate = tuple(value.subs({symbol: 0 for symbol in value.free_symbols}) for value in candidate)
        coefficients = tuple(Fraction(int(value.p), int(value.q)) for value in candidate)
        valid = all(
            data[n + order] == sum(coefficients[j] * data[n + j] for j in range(order))
            for n in range(len(data) - order)
        )
        if valid:
            return LinearRecurrenceGuess(order, coefficients, fit_end, hold, True)
    return None


def rational_gf_from_recurrence(
    recurrence: LinearRecurrenceGuess,
    initial: Sequence[int | Fraction],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Return numerator and denominator of the ordinary rational GF."""

    order = recurrence.order
    if len(initial) < order:
        raise ValueError("insufficient initial values")
    data = _fractions(initial[:order])
    denominator = [Fraction(1)]
    denominator.extend(-recurrence.coefficients[order - k] for k in range(1, order + 1))
    numerator: list[Fraction] = []
    for n in range(order):
        value = data[n]
        for k in range(1, n + 1):
            value += denominator[k] * data[n - k]
        numerator.append(value)
    while len(numerator) > 1 and numerator[-1] == 0:
        numerator.pop()
    return tuple(numerator), tuple(denominator)


def guess_hypergeometric_ratio(
    values: Iterable[int | Fraction],
    *,
    max_numerator_degree: int = 4,
    max_denominator_degree: int = 4,
    holdout: int | None = None,
) -> HypergeometricGuess | None:
    """Guess ``a[n+1]/a[n]=P(n)/Q(n)`` by exact nullspace search."""

    sp = _sympy()
    data = _fractions(values)
    if len(data) < 6 or any(value == 0 for value in data[:-1]):
        return None
    relation_count = len(data) - 1
    hold = max(2, relation_count // 4) if holdout is None else holdout
    fit_relations = relation_count - hold
    for total_degree in range(max_numerator_degree + max_denominator_degree + 1):
        for p_degree in range(max_numerator_degree + 1):
            q_degree = total_degree - p_degree
            if q_degree < 0 or q_degree > max_denominator_degree:
                continue
            unknowns = (p_degree + 1) + (q_degree + 1)
            if fit_relations < unknowns - 1:
                continue
            rows = []
            for n in range(fit_relations):
                a_n = sp.Rational(data[n].numerator, data[n].denominator)
                a_next = sp.Rational(data[n + 1].numerator, data[n + 1].denominator)
                rows.append(
                    [-a_n * n**degree for degree in range(p_degree + 1)]
                    + [a_next * n**degree for degree in range(q_degree + 1)]
                )
            nullspace = sp.Matrix(rows).nullspace()
            for vector in nullspace:
                primitive = _primitive_integer_vector(tuple(vector))
                numerator = primitive[: p_degree + 1]
                denominator = primitive[p_degree + 1 :]
                if not any(denominator):
                    continue
                valid = True
                for n in range(relation_count):
                    p_value = sum(numerator[d] * n**d for d in range(len(numerator)))
                    q_value = sum(denominator[d] * n**d for d in range(len(denominator)))
                    if q_value == 0 or data[n + 1] * q_value != data[n] * p_value:
                        valid = False
                        break
                if valid:
                    return HypergeometricGuess(
                        tuple(numerator),
                        tuple(denominator),
                        fit_relations,
                        hold,
                        True,
                    )
    return None


def guess_p_recursive(
    values: Iterable[int | Fraction],
    *,
    max_order: int = 4,
    max_degree: int = 5,
    holdout: int | None = None,
) -> PRecursiveGuess | None:
    r"""Guess ``sum_(j=0)^r P_j(n)a[n+j]=0`` exactly on a prefix."""

    sp = _sympy()
    data = _fractions(values)
    if len(data) < 10:
        return None
    for complexity in range(1, max_order + max_degree + 1):
        for order in range(1, max_order + 1):
            degree = complexity - order
            if degree < 0 or degree > max_degree:
                continue
            relation_count = len(data) - order
            hold = max(3, relation_count // 4) if holdout is None else holdout
            fit_relations = relation_count - hold
            unknowns = (order + 1) * (degree + 1)
            if fit_relations < unknowns - 1:
                continue
            rows = []
            for n in range(fit_relations):
                row = []
                for shift in range(order + 1):
                    value = sp.Rational(data[n + shift].numerator, data[n + shift].denominator)
                    row.extend(value * n**d for d in range(degree + 1))
                rows.append(row)
            nullspace = sp.Matrix(rows).nullspace()
            for vector in nullspace:
                primitive = _primitive_integer_vector(tuple(vector))
                polynomials = tuple(
                    tuple(primitive[(degree + 1) * shift + d] for d in range(degree + 1))
                    for shift in range(order + 1)
                )
                if not any(polynomials[-1]):
                    continue
                valid = True
                for n in range(relation_count):
                    total = Fraction(0)
                    for shift, polynomial in enumerate(polynomials):
                        p_value = sum(coefficient * n**d for d, coefficient in enumerate(polynomial))
                        total += p_value * data[n + shift]
                    if total != 0:
                        valid = False
                        break
                if valid:
                    return PRecursiveGuess(order, degree, polynomials, fit_relations, hold, True)
    return None


def _series_multiply(left: Sequence[Fraction], right: Sequence[Fraction], degree: int) -> tuple[Fraction, ...]:
    out = [Fraction(0) for _ in range(degree + 1)]
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if i + j > degree:
                break
            out[i + j] += a * b
    return tuple(out)


def guess_algebraic_gf(
    values: Iterable[int | Fraction],
    *,
    max_x_degree: int = 4,
    max_y_degree: int = 4,
    holdout: int | None = None,
) -> AlgebraicGFGuess | None:
    r"""Guess an exact relation ``P(x,F(x))=0`` from ordinary coefficients."""

    sp = _sympy()
    data = _fractions(values)
    if len(data) < 10:
        return None
    hold = max(3, len(data) // 4) if holdout is None else holdout
    fit_coefficients = len(data) - hold
    degree_bound = len(data) - 1
    powers: list[tuple[Fraction, ...]] = []
    identity = (Fraction(1),) + (Fraction(0),) * degree_bound
    powers.append(identity)
    for _ in range(max_y_degree):
        powers.append(_series_multiply(powers[-1], data, degree_bound))

    for complexity in range(2, max_x_degree + max_y_degree + 1):
        for x_degree in range(max_x_degree + 1):
            y_degree = complexity - x_degree
            if y_degree < 1 or y_degree > max_y_degree:
                continue
            monomial_indices = [(i, j) for j in range(y_degree + 1) for i in range(x_degree + 1)]
            if fit_coefficients < len(monomial_indices) - 1:
                continue
            rows = []
            for n in range(fit_coefficients):
                row = []
                for i, j in monomial_indices:
                    value = powers[j][n - i] if n >= i else Fraction(0)
                    row.append(sp.Rational(value.numerator, value.denominator))
                rows.append(row)
            nullspace = sp.Matrix(rows).nullspace()
            candidates: list[tuple[int, ...]] = []
            for vector in nullspace:
                primitive = _primitive_integer_vector(tuple(vector))
                if any(primitive[index] for index, (_i, j) in enumerate(monomial_indices) if j > 0):
                    candidates.append(primitive)
            candidates.sort(key=lambda vector: (sum(value != 0 for value in vector), sum(abs(value) for value in vector)))
            for primitive in candidates:
                residual = [Fraction(0) for _ in range(len(data))]
                for coefficient, (i, j) in zip(primitive, monomial_indices, strict=True):
                    if coefficient == 0:
                        continue
                    for n in range(i, len(data)):
                        residual[n] += coefficient * powers[j][n - i]
                if all(value == 0 for value in residual):
                    terms = tuple(
                        (i, j, coefficient)
                        for coefficient, (i, j) in zip(primitive, monomial_indices, strict=True)
                        if coefficient
                    )
                    return AlgebraicGFGuess(
                        x_degree,
                        y_degree,
                        terms,
                        fit_coefficients,
                        hold,
                        True,
                    )
    return None


def guess_quasipolynomial(
    values: Iterable[int | Fraction],
    *,
    max_period: int = 12,
    max_degree: int = 6,
    holdout: int | None = None,
) -> QuasiPolynomialGuess | None:
    """Guess a quasipolynomial by constant finite differences on each lane."""

    data = _fractions(values)
    if len(data) < 8:
        return None
    hold = max(2, len(data) // 4) if holdout is None else holdout
    fit_end = len(data) - hold
    for period in range(1, max_period + 1):
        for degree in range(max_degree + 1):
            lane_coefficients: list[tuple[Fraction, ...]] = []
            possible = True
            for residue in range(period):
                lane = [data[n] for n in range(residue, fit_end, period)]
                if len(lane) < degree + 1:
                    possible = False
                    break
                current = lane[:]
                newton = []
                for _ in range(degree + 1):
                    newton.append(current[0])
                    current = [current[index + 1] - current[index] for index in range(len(current) - 1)]
                if current and any(value != 0 for value in current):
                    possible = False
                    break
                lane_coefficients.append(tuple(newton))
            if not possible:
                continue

            def evaluate(index: int) -> Fraction:
                residue = index % period
                q = index // period
                coefficients = lane_coefficients[residue]
                total = Fraction(0)
                binomial = Fraction(1)
                for k, coefficient in enumerate(coefficients):
                    if k:
                        binomial = binomial * (q - k + 1) / k
                    total += coefficient * binomial
                return total

            if all(evaluate(n) == data[n] for n in range(len(data))):
                return QuasiPolynomialGuess(period, degree, tuple(lane_coefficients), fit_end, hold, True)
    return None


def characteristic_closed_form(
    recurrence: LinearRecurrenceGuess,
    initial: Sequence[int | Fraction],
) -> Any | None:
    """Ask SymPy for a symbolic closed form of a constant recurrence."""

    sp = _sympy()
    n = sp.Symbol("n", integer=True, nonnegative=True)
    a = sp.Function("a")
    equation = sp.Eq(
        a(n + recurrence.order),
        sum(sp.Rational(c.numerator, c.denominator) * a(n + j) for j, c in enumerate(recurrence.coefficients)),
    )
    conditions = {a(j): sp.Rational(Fraction(initial[j]).numerator, Fraction(initial[j]).denominator) for j in range(recurrence.order)}
    try:
        return sp.simplify(sp.rsolve(equation, a(n), conditions))
    except (ValueError, NotImplementedError):
        return None


__all__ = [
    "AlgebraicGFGuess",
    "HypergeometricGuess",
    "LinearRecurrenceGuess",
    "PRecursiveGuess",
    "QuasiPolynomialGuess",
    "characteristic_closed_form",
    "guess_algebraic_gf",
    "guess_hypergeometric_ratio",
    "guess_linear_recurrence",
    "guess_p_recursive",
    "guess_quasipolynomial",
    "rational_gf_from_recurrence",
]
