"""Declarative custom generating-function specifications for exact finite computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Mapping

from .basis import WeightedMonomialBasis, ordinary_basis
from .sequences import generalized_bell_numbers, weighted_generalized_bell_numbers
from .series import FormalSeries


CoefficientBuilder = Callable[[int], tuple[Fraction, ...]]


@dataclass(frozen=True, slots=True)
class CustomFunctionSpec:
    """A named exact coefficient source with an explicit basis and metadata."""

    name: str
    basis: WeightedMonomialBasis
    builder: CoefficientBuilder = field(repr=False)
    source_kind: str = "custom"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def series(self, degree: int) -> FormalSeries:
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        values = self.builder(degree)
        if len(values) != degree + 1:
            raise ValueError("custom coefficient builder returned the wrong number of coefficients")
        return FormalSeries(self.basis, values)


class FunctionRegistry:
    """An explicit custom-function registry; names cannot silently overwrite."""

    def __init__(self) -> None:
        self._functions: dict[str, CustomFunctionSpec] = {}

    def register(self, function: CustomFunctionSpec) -> None:
        key = function.name.strip().lower()
        if not key:
            raise ValueError("function name must be nonempty")
        if key in self._functions:
            raise KeyError(f"custom function already registered: {function.name}")
        self._functions[key] = function

    def get(self, name: str) -> CustomFunctionSpec:
        key = name.strip().lower()
        if key not in self._functions:
            raise KeyError(f"unknown custom function: {name}")
        return self._functions[key]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._functions))


def explicit_function(
    name: str,
    basis: WeightedMonomialBasis,
    coefficients: tuple[int | Fraction, ...],
) -> CustomFunctionSpec:
    """Create a finite exact coefficient source in a declared normalized basis."""

    values = tuple(Fraction(value) for value in coefficients)

    def build(degree: int) -> tuple[Fraction, ...]:
        if degree >= len(values):
            raise ValueError("explicit function has insufficient declared coefficients")
        return values[: degree + 1]

    return CustomFunctionSpec(name, basis, build, source_kind="explicit")


def linear_recurrence_function(
    name: str,
    coefficients: tuple[int | Fraction, ...],
    initial: tuple[int | Fraction, ...],
    *,
    basis: WeightedMonomialBasis | None = None,
) -> CustomFunctionSpec:
    r"""Create ``a[n+d]=sum_j c_j a[n+j]``."""

    recurrence = tuple(Fraction(value) for value in coefficients)
    seeds = tuple(Fraction(value) for value in initial)
    if not recurrence or len(seeds) != len(recurrence):
        raise ValueError("initial value count must equal positive recurrence order")

    def build(degree: int) -> tuple[Fraction, ...]:
        values = list(seeds[: degree + 1])
        while len(values) <= degree:
            values.append(sum(recurrence[j] * values[len(values) - len(recurrence) + j] for j in range(len(recurrence))))
        return tuple(values)

    return CustomFunctionSpec(
        name,
        basis or ordinary_basis(),
        build,
        source_kind="linear-recurrence",
        metadata={"order": len(recurrence), "coefficients": recurrence, "initial": seeds},
    )


def hypergeometric_function(
    name: str,
    initial: int | Fraction,
    ratio: Callable[[int], int | Fraction],
    *,
    basis: WeightedMonomialBasis | None = None,
) -> CustomFunctionSpec:
    r"""Create a term sequence with ``a[n+1]/a[n]=ratio(n)``."""

    seed = Fraction(initial)

    def build(degree: int) -> tuple[Fraction, ...]:
        values = [seed]
        for n in range(degree):
            values.append(values[-1] * Fraction(ratio(n)))
        return tuple(values)

    return CustomFunctionSpec(name, basis or ordinary_basis(), build, source_kind="hypergeometric")


def rational_function(
    name: str,
    numerator: tuple[int | Fraction, ...],
    denominator: tuple[int | Fraction, ...],
    *,
    basis: WeightedMonomialBasis | None = None,
) -> CustomFunctionSpec:
    """Create the coefficient sequence of one exact rational function."""

    p = tuple(Fraction(value) for value in numerator)
    q = tuple(Fraction(value) for value in denominator)
    if not q or q[0] == 0:
        raise ValueError("rational denominator must have a nonzero constant coefficient")

    def build(degree: int) -> tuple[Fraction, ...]:
        ordinary: list[Fraction] = []
        for n in range(degree + 1):
            rhs = p[n] if n < len(p) else Fraction(0)
            rhs -= sum(q[k] * ordinary[n - k] for k in range(1, min(n, len(q) - 1) + 1))
            ordinary.append(rhs / q[0])
        target = basis or ordinary_basis()
        return tuple(ordinary[n] * target.gamma(n) for n in range(degree + 1))

    return CustomFunctionSpec(
        name,
        basis or ordinary_basis(),
        build,
        source_kind="rational-function",
        metadata={"numerator": p, "denominator": q},
    )



def algebraic_function(
    name: str,
    polynomial_terms: Mapping[tuple[int, int], int | Fraction],
    initial_value: int | Fraction,
    *,
    basis: WeightedMonomialBasis | None = None,
) -> CustomFunctionSpec:
    r"""Create the unique formal branch of ``P(z,y)=0`` at ``y(0)=y0``.

    Terms are supplied as ``(z_degree, y_degree) -> coefficient``.  The
    formal implicit-function condition ``P_y(0,y0) != 0`` is checked exactly.
    Coefficients are then solved one degree at a time over the rationals.
    """

    terms = {
        (int(z_degree), int(y_degree)): Fraction(coefficient)
        for (z_degree, y_degree), coefficient in polynomial_terms.items()
        if Fraction(coefficient) != 0
    }
    if not terms:
        raise ValueError("algebraic polynomial must contain at least one nonzero term")
    if any(z_degree < 0 or y_degree < 0 for z_degree, y_degree in terms):
        raise ValueError("algebraic polynomial degrees must be nonnegative")
    y0 = Fraction(initial_value)
    constant = sum(
        coefficient * y0**y_degree
        for (z_degree, y_degree), coefficient in terms.items()
        if z_degree == 0
    )
    if constant != 0:
        raise ValueError("initial value does not satisfy P(0,y0)=0")
    derivative = sum(
        coefficient * y_degree * y0 ** (y_degree - 1)
        for (z_degree, y_degree), coefficient in terms.items()
        if z_degree == 0 and y_degree > 0
    )
    if derivative == 0:
        raise ValueError("formal implicit branch requires P_y(0,y0) to be nonzero")
    target = basis or ordinary_basis()

    def multiply(left: list[Fraction], right: list[Fraction], degree: int) -> list[Fraction]:
        output = [Fraction(0) for _ in range(degree + 1)]
        for i, left_value in enumerate(left):
            if i > degree:
                break
            for j, right_value in enumerate(right):
                if i + j > degree:
                    break
                output[i + j] += left_value * right_value
        return output

    def coefficient_of_polynomial(values: list[Fraction], degree: int) -> Fraction:
        max_y_degree = max(y_degree for _z_degree, y_degree in terms)
        powers: list[list[Fraction]] = [[Fraction(1)] + [Fraction(0)] * degree]
        padded = values + [Fraction(0)] * (degree + 1 - len(values))
        for _ in range(max_y_degree):
            powers.append(multiply(powers[-1], padded, degree))
        total = Fraction(0)
        for (z_degree, y_degree), coefficient in terms.items():
            if z_degree <= degree:
                total += coefficient * powers[y_degree][degree - z_degree]
        return total

    def build(degree: int) -> tuple[Fraction, ...]:
        ordinary = [y0]
        for n in range(1, degree + 1):
            residual = coefficient_of_polynomial(ordinary + [Fraction(0)], n)
            ordinary.append(-residual / derivative)
        return tuple(ordinary[n] * target.gamma(n) for n in range(degree + 1))

    return CustomFunctionSpec(
        name,
        target,
        build,
        source_kind="algebraic-implicit",
        metadata={
            "terms": tuple(sorted((z_degree, y_degree, coefficient) for (z_degree, y_degree), coefficient in terms.items())),
            "initial_value": y0,
        },
    )

def euler_product_function(
    name: str,
    exponents: Mapping[int, int | Fraction] | Callable[[int], int | Fraction],
    *,
    basis: WeightedMonomialBasis | None = None,
) -> CustomFunctionSpec:
    r"""Create ``prod_(m>=1) (1-z^m)^(-c_m)`` by log-derivative recurrence."""

    def exponent(index: int) -> Fraction:
        if callable(exponents):
            return Fraction(exponents(index))
        return Fraction(exponents.get(index, 0))

    target = basis or ordinary_basis()

    def build(degree: int) -> tuple[Fraction, ...]:
        ordinary = [Fraction(1)]
        logarithmic = [Fraction(0)] * (degree + 1)
        for k in range(1, degree + 1):
            logarithmic[k] = sum(Fraction(d) * exponent(d) for d in range(1, k + 1) if k % d == 0)
        for n in range(1, degree + 1):
            ordinary.append(sum(logarithmic[k] * ordinary[n - k] for k in range(1, n + 1)) / n)
        return tuple(ordinary[n] * target.gamma(n) for n in range(degree + 1))

    return CustomFunctionSpec(name, target, build, source_kind="euler-product")


def factorial_power_bell_function(power: int, *, name: str | None = None) -> CustomFunctionSpec:
    r"""Create the unweighted factorial-power Bell family of a positive order."""

    from .basis import factorial_power_basis

    basis = factorial_power_basis(power)

    def build(degree: int) -> tuple[Fraction, ...]:
        return tuple(Fraction(value) for value in generalized_bell_numbers(power, degree + 1))

    return CustomFunctionSpec(
        name or f"factorial-power Bell {power}",
        basis,
        build,
        source_kind="exponential-of-hypergeometric",
        metadata={
            "power": power,
            "ordinary_closed_form": f"exp(0F{power - 1}(;1,...,1;z)-1)",
        },
    )



def weighted_factorial_power_bell_function(
    power: int,
    weights: tuple[int | Fraction, ...] | Callable[[int], int | Fraction],
    *,
    name: str | None = None,
) -> CustomFunctionSpec:
    """Create a live custom block-weighted factorial-power Bell family."""

    from .basis import factorial_power_basis

    basis = factorial_power_basis(power)

    def build(degree: int) -> tuple[Fraction, ...]:
        return weighted_generalized_bell_numbers(power, degree + 1, weights)

    return CustomFunctionSpec(
        name or f"weighted factorial-power Bell {power}",
        basis,
        build,
        source_kind="weighted-exponential-of-hypergeometric",
        metadata={"power": power, "live_weight_rule": callable(weights)},
    )

def compose_functions(
    name: str,
    outer: CustomFunctionSpec,
    inner: CustomFunctionSpec,
) -> CustomFunctionSpec:
    """Create an exact ordinary-composition specification."""

    def build(degree: int) -> tuple[Fraction, ...]:
        outer_series = outer.series(degree)
        inner_series = inner.series(degree).convert(outer.basis)
        return outer_series.compose(inner_series, degree=degree).coefficients

    return CustomFunctionSpec(
        name,
        outer.basis,
        build,
        source_kind="composition",
        metadata={"outer": outer.name, "inner": inner.name},
    )


__all__ = [
    "CustomFunctionSpec",
    "FunctionRegistry",
    "algebraic_function",
    "compose_functions",
    "euler_product_function",
    "explicit_function",
    "factorial_power_bell_function",
    "hypergeometric_function",
    "linear_recurrence_function",
    "rational_function",
    "weighted_factorial_power_bell_function",
]
