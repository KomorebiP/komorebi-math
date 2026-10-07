"""Generic integration dispatch and honest quadrature error semantics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import product
from math import fsum, isfinite
from numbers import Complex, Real
from sys import float_info
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .measures import (
    AtomicMeasure,
    ExactAffineMap,
    IntervalMeasure,
    Measure,
    ProductMeasure,
    PushforwardMeasure,
)
from .polynomial import ExactPolynomial


_EXACTNESS_STRENGTH = {
    Exactness.EXACT: 0,
    Exactness.NUMERICAL: 1,
    Exactness.CONDITIONAL: 2,
    Exactness.HEURISTIC: 3,
    Exactness.PLANNED: 4,
}


def _weakest_exactness(*values: Exactness) -> Exactness:
    return max(values, key=_EXACTNESS_STRENGTH.__getitem__)


class ErrorGuarantee(StrEnum):
    """Strength of an integration error statement."""

    EXACT = "exact"
    RIGOROUS = "rigorous"
    CONDITIONAL = "conditional"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IntegrationError:
    """One absolute error statement attached to an integral result."""

    absolute_bound: Fraction | float | None
    guarantee: ErrorGuarantee
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.absolute_bound is not None:
            raw = self.absolute_bound
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise ValidationError("integration error bounds must be finite nonnegative reals")
            if isinstance(raw, (int, Fraction)):
                value: Fraction | float = Fraction(raw)
            else:
                value = float(raw)
            if value < 0 or (isinstance(value, float) and not isfinite(value)):
                raise ValidationError("integration error bounds must be finite and nonnegative")
            object.__setattr__(self, "absolute_bound", value)
        if self.guarantee in (ErrorGuarantee.EXACT, ErrorGuarantee.RIGOROUS) and self.absolute_bound is None:
            raise ValidationError("exact and rigorous error statements require an absolute bound")

    @property
    def rigorous(self) -> bool:
        return self.guarantee in (ErrorGuarantee.EXACT, ErrorGuarantee.RIGOROUS)

    @classmethod
    def exact(cls) -> "IntegrationError":
        return cls(Fraction(0), ErrorGuarantee.EXACT, ("exact arithmetic",))


@dataclass(frozen=True, slots=True)
class IntegralResult:
    """An integral value with method, evaluation count, certificate, and error."""

    value: Any
    certificate: Certificate
    error: IntegrationError
    method: str
    evaluations: int

    @property
    def valid(self) -> bool:
        return self.certificate.valid

    @property
    def exact(self) -> bool:
        return self.error.guarantee is ErrorGuarantee.EXACT

    def require(self) -> Any:
        self.certificate.require()
        return self.value


@dataclass(frozen=True, slots=True)
class DerivativeBound:
    r"""A declared bound ``sup |f^(order)| <= bound`` with its own proof status."""

    order: int
    bound: Fraction | float
    certificate: Certificate
    description: str = "derivative supremum bound"
    polynomial: ExactPolynomial | None = None
    interval: tuple[Fraction, Fraction] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            raise ValidationError("derivative-bound order must be a positive integer")
        raw_bound = self.bound
        if isinstance(raw_bound, bool) or not isinstance(raw_bound, Real):
            raise ValidationError("derivative bound must be a finite nonnegative real")
        if isinstance(raw_bound, (int, Fraction)):
            value: Fraction | float = Fraction(raw_bound)
        else:
            value = float(raw_bound)
        if value < 0 or (isinstance(value, float) and not isfinite(value)):
            raise ValidationError("derivative bound must be finite and nonnegative")
        self.certificate.require()
        object.__setattr__(self, "bound", value)
        label = str(self.description).strip()
        if not label:
            raise ValidationError("derivative-bound description must be nonempty")
        object.__setattr__(self, "description", label)
        if (self.polynomial is None) != (self.interval is None):
            raise ValidationError("polynomial derivative-bound provenance requires polynomial and interval")
        if self.polynomial is not None:
            if self.polynomial.dimension != 1:
                raise DimensionError("polynomial derivative-bound provenance must be univariate")
            if not isinstance(self.interval, tuple) or len(self.interval) != 2:
                raise ValidationError("derivative-bound interval provenance is malformed")
            lower, upper = self.interval
            if any(not isinstance(endpoint, Fraction) for endpoint in (lower, upper)):
                raise ValidationError("derivative-bound interval provenance must be rational")
            if upper < lower:
                raise ValidationError("derivative-bound interval provenance must be ordered")
            expected = self.polynomial.triangle_derivative_bound(
                lower,
                upper,
                order=self.order,
            )
            if not isinstance(value, Fraction) or value < expected:
                raise ValidationError(
                    "internally rigorous polynomial bounds must preserve the exact rational upper bound"
                )

    @property
    def guarantee(self) -> ErrorGuarantee:
        if (
            self.polynomial is not None
            and self.interval is not None
            and self.certificate.exactness is Exactness.EXACT
            and not self.certificate.assumptions_unchecked
        ):
            return ErrorGuarantee.RIGOROUS
        if (
            self.certificate.exactness in (Exactness.EXACT, Exactness.CONDITIONAL)
            or self.certificate.assumptions_unchecked
        ):
            return ErrorGuarantee.CONDITIONAL
        return ErrorGuarantee.ESTIMATED

    def applies_to(self, polynomial: ExactPolynomial, measure: IntervalMeasure) -> bool:
        """Return whether internal exact provenance matches an integrand and domain."""

        if self.polynomial is None or self.interval is None:
            return False
        if measure.exactness is not Exactness.EXACT:
            return False
        structure = measure.certificate()
        if (
            not structure.valid
            or structure.exactness is not Exactness.EXACT
            or structure.assumptions_unchecked
        ):
            return False
        return self.polynomial == polynomial and self.interval == (
            measure.lower,
            measure.upper,
        )

    @classmethod
    def assumed(
        cls,
        order: int,
        bound: Fraction | float,
        *,
        description: str = "assumed derivative supremum bound",
    ) -> "DerivativeBound":
        return cls(
            order,
            bound,
            Certificate(
                valid=True,
                exactness=Exactness.CONDITIONAL,
                assumptions_unchecked=(description,),
                theorem_id="INT.QUAD.BOUND.ASSUMED",
                message="derivative bound accepted as an explicit assumption",
            ),
            description,
        )


@dataclass(frozen=True, slots=True)
class CompositeSimpson:
    """Composite Simpson quadrature for bounded interval factors."""

    subintervals: int = 64
    derivative_bound: DerivativeBound | None = None
    max_evaluations: int = 1_000_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.subintervals, bool)
            or not isinstance(self.subintervals, int)
            or self.subintervals < 4
            or self.subintervals % 4
        ):
            raise ValidationError("Simpson subintervals must be a positive multiple of four")
        if (
            isinstance(self.max_evaluations, bool)
            or not isinstance(self.max_evaluations, int)
            or self.max_evaluations < 1
        ):
            raise ValidationError("max_evaluations must be a positive integer")
        if self.derivative_bound is not None and self.derivative_bound.order != 4:
            raise ValidationError("composite Simpson certification requires a fourth-derivative bound")

    def without_derivative_bound(self) -> "CompositeSimpson":
        return CompositeSimpson(self.subintervals, None, self.max_evaluations)


def polynomial_derivative_bound(
    polynomial: ExactPolynomial,
    measure: IntervalMeasure,
    *,
    order: int = 4,
) -> DerivativeBound:
    """Create a rigorous coefficient-triangle derivative bound."""

    if not isinstance(polynomial, ExactPolynomial):
        raise TypeError("polynomial derivative bounds require an ExactPolynomial")
    if not isinstance(measure, IntervalMeasure):
        raise TypeError("polynomial derivative bounds require an IntervalMeasure")
    structure = measure.certificate()
    structure.require()
    if (
        measure.exactness is not Exactness.EXACT
        or structure.exactness is not Exactness.EXACT
        or structure.assumptions_unchecked
    ):
        raise ValidationError("rigorous polynomial derivative bounds require rational interval data")
    bound = polynomial.triangle_derivative_bound(measure.lower, measure.upper, order=order)
    return DerivativeBound(
        order,
        bound,
        Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "exact polynomial differentiation",
                "coefficient triangle inequality",
                "rational bounded interval",
            )
            + tuple(structure.assumptions_checked),
            diagnostics={"exact_bound": str(bound), "order": order},
            theorem_id="INT.QUAD.POLY.BOUND",
            message="rigorous polynomial derivative bound derived",
        ),
        "coefficient-triangle polynomial derivative bound",
        polynomial,
        (measure.lower, measure.upper),
    )


def _is_exact_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, Fraction)):
        return True
    if isinstance(value, np.ndarray):
        return all(_is_exact_value(entry) for entry in value.reshape(-1))
    if isinstance(value, (tuple, list)):
        return all(_is_exact_value(entry) for entry in value)
    return False


def _finite_value(value: Any) -> None:
    """Validate numeric finiteness without coercing arbitrary-size exact values."""

    if isinstance(value, bool):
        raise ValidationError("integrand output must be numeric")
    if isinstance(value, (int, Fraction)):
        return
    if isinstance(value, np.ndarray):
        for entry in value.reshape(-1):
            _finite_value(entry)
        return
    if isinstance(value, (tuple, list)):
        for entry in value:
            _finite_value(entry)
        return
    if isinstance(value, Complex):
        scalar = complex(value)
        if isfinite(scalar.real) and isfinite(scalar.imag):
            return
    raise ValidationError("integrand output must be finite numeric data")


def _finite_float_scalar(value: Any, *, name: str) -> float:
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError(f"{name} cannot be represented as a finite float") from exc
    if not isfinite(converted):
        raise ValidationError(f"{name} cannot be represented as a finite float")
    return converted


def _scale(value: Any, weight: Any) -> Any:
    try:
        if isinstance(value, (tuple, list)):
            return np.asarray(value, dtype=object) * weight
        return value * weight
    except OverflowError as exc:
        raise ValidationError("integration scaling overflowed floating-point range") from exc


def _norm_difference(left: Any, right: Any) -> float:
    try:
        value = float(
            np.linalg.norm(
                np.asarray(left, dtype=complex) - np.asarray(right, dtype=complex)
            )
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError("quadrature values cannot be compared numerically") from exc
    if not isfinite(value):
        raise ValidationError("quadrature comparison overflowed or became nonfinite")
    return value


def _diagnostic_float(value: Any) -> float:
    """Return a finite display residual without using it for exact theorem logic."""

    try:
        converted = float(abs(value))
    except (OverflowError, TypeError, ValueError):
        return float_info.max
    if not isfinite(converted):
        return float_info.max
    return converted


def _call(function: Callable[[Any], Any], point: tuple[Any, ...]) -> Any:
    value = function(point[0] if len(point) == 1 else point)
    _finite_value(value)
    return value


def _weighted_sum(
    function: Callable[[Any], Any],
    nodes: Sequence[tuple[Any, ...]],
    weights: Sequence[Any],
) -> tuple[Any, bool]:
    terms: list[Any] = []
    exact = True
    output_shape: tuple[int, ...] | None = None
    for point, weight in zip(nodes, weights, strict=True):
        value = _call(function, point)
        shape = tuple(np.asarray(value, dtype=object).shape)
        if output_shape is None:
            output_shape = shape
        elif shape != output_shape:
            raise DimensionError("integrand output shape changed between quadrature nodes")
        exact = exact and _is_exact_value(value) and _is_exact_value(weight)
        term = _scale(value, weight)
        _finite_value(term)
        terms.append(term)
    if not terms:
        raise ValidationError("quadrature rule contains no nodes")
    if output_shape == () and exact:
        total = sum(terms, Fraction(0))
    elif output_shape == ():
        try:
            if any(complex(term).imag != 0 for term in terms):
                total = complex(
                    fsum(float(complex(term).real) for term in terms),
                    fsum(float(complex(term).imag) for term in terms),
                )
            else:
                total = fsum(float(term) for term in terms)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValidationError("quadrature scalar summation overflowed") from exc
    else:
        total = terms[0]
        for term in terms[1:]:
            try:
                total = total + term
            except OverflowError as exc:
                raise ValidationError(
                    "integration array summation overflowed floating-point range"
                ) from exc
    _finite_value(total)
    return total, exact


def _atomic_integral(function: Callable[[Any], Any], measure: AtomicMeasure) -> IntegralResult:
    value, exact_values = _weighted_sum(function, measure.points, measure.weights)
    exact = exact_values and measure.exactness is Exactness.EXACT
    return IntegralResult(
        value,
        Certificate(
            valid=True,
            exactness=Exactness.EXACT if exact else Exactness.NUMERICAL,
            assumptions_checked=("finite weighted sum over every atom",),
            assumptions_unchecked=()
            if exact
            else ("floating-point roundoff is not enclosed",),
            diagnostics={"atom_count": measure.atom_count},
            theorem_id="INT.EVAL.ATOM",
            message="atomic integral evaluated",
        ),
        IntegrationError.exact()
        if exact
        else IntegrationError(None, ErrorGuarantee.UNKNOWN, ("floating-point atomic evaluation",)),
        "finite atomic sum",
        measure.atom_count,
    )


def _inherit_measure_assumptions(
    result: IntegralResult,
    measure: Measure,
    *,
    theorem_id: str,
) -> IntegralResult:
    """Attach structural measure assumptions without upgrading the result."""

    structure = measure.certificate()
    structure.require()
    inherited_nonexact = (
        structure.exactness is not Exactness.EXACT
        or bool(structure.assumptions_unchecked)
    )
    structural_unchecked = tuple(structure.assumptions_unchecked)
    if structure.exactness is not Exactness.EXACT and not structural_unchecked:
        structural_unchecked = (
            f"measure structure has {structure.exactness.value} rather than exact status",
        )
    unchecked = tuple(
        dict.fromkeys(
            result.certificate.assumptions_unchecked
            + structural_unchecked
        )
    )
    exactness = _weakest_exactness(
        result.certificate.exactness,
        structure.exactness,
    )
    error = result.error
    if inherited_nonexact and error.guarantee in (
        ErrorGuarantee.EXACT,
        ErrorGuarantee.RIGOROUS,
    ):
        error = IntegrationError(
            None,
            ErrorGuarantee.UNKNOWN,
            ("measure semantics depend on unchecked assumptions",),
        )
    return IntegralResult(
        result.value,
        Certificate(
            valid=result.valid and structure.valid,
            exactness=exactness,
            residuals=result.certificate.residuals,
            tolerance=result.certificate.tolerance,
            assumptions_checked=tuple(
                dict.fromkeys(
                    result.certificate.assumptions_checked
                    + structure.assumptions_checked
                )
            ),
            assumptions_unchecked=unchecked,
            diagnostics={
                **result.certificate.diagnostics,
                "measure_exactness": structure.exactness.value,
            },
            theorem_id=theorem_id,
            message=f"{result.certificate.message}; measure assumptions propagated",
        ),
        error,
        result.method,
        result.evaluations,
    )


def _simpson_nodes(measure: IntervalMeasure, subintervals: int) -> tuple[tuple[tuple[float], ...], tuple[float, ...]]:
    a = _finite_float_scalar(measure.lower, name="interval lower endpoint")
    b = _finite_float_scalar(measure.upper, name="interval upper endpoint")
    density = _finite_float_scalar(measure.density, name="interval density")
    if measure.lower != measure.upper and a == b:
        raise ValidationError("floating-point endpoints collapse a nondegenerate interval")
    if measure.density != 0 and density == 0:
        raise ValidationError("floating-point density underflowed to zero")
    step = (b - a) / subintervals
    if measure.lower != measure.upper and step == 0:
        raise ValidationError("floating-point Simpson step underflowed to zero")
    points = tuple((a + index * step,) for index in range(subintervals + 1))
    coefficients = tuple(
        1 if index in (0, subintervals) else 4 if index % 2 else 2
        for index in range(subintervals + 1)
    )
    weights = tuple(density * step * coefficient / 3.0 for coefficient in coefficients)
    if measure.density != 0 and measure.lower != measure.upper and not any(weights):
        raise ValidationError("floating-point Simpson weights underflowed to zero")
    if not all(isfinite(value) for value in weights):
        raise ValidationError("floating-point Simpson weights became nonfinite")
    return points, weights


def _exact_simpson_nodes(
    measure: IntervalMeasure,
    subintervals: int,
) -> tuple[tuple[tuple[Fraction], ...], tuple[Fraction, ...]]:
    """Construct a Simpson rule without any floating-point conversion."""

    if measure.exactness is not Exactness.EXACT:
        raise ValidationError("exact Simpson nodes require rational interval data")
    a = Fraction(measure.lower)
    b = Fraction(measure.upper)
    density = Fraction(measure.density)
    step = (b - a) / subintervals
    points = tuple((a + index * step,) for index in range(subintervals + 1))
    coefficients = tuple(
        1 if index in (0, subintervals) else 4 if index % 2 else 2
        for index in range(subintervals + 1)
    )
    weights = tuple(density * step * coefficient / 3 for coefficient in coefficients)
    return points, weights


def _zero_interval_result() -> IntegralResult:
    return IntegralResult(
        Fraction(0),
        Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=("structurally zero finite measure",),
            diagnostics={"evaluations": 0},
            theorem_id="INT.EVAL.ZERO.MEASURE",
            message="integral over the zero measure evaluated exactly",
        ),
        IntegrationError.exact(),
        "zero-measure integral",
        0,
    )


def _simpson_integral(
    function: Callable[[Any], Any],
    measure: IntervalMeasure,
    provider: CompositeSimpson,
) -> IntegralResult:
    if measure.lower == measure.upper or measure.density == 0:
        return _zero_interval_result()
    # This validates overflow/underflow in numerical measure arithmetic, but
    # structural zero detection above never relies on a computed float mass.
    measure.mass().require()
    evaluations = provider.subintervals + 1 + provider.subintervals // 2 + 1
    if evaluations > provider.max_evaluations:
        raise ValidationError("quadrature evaluation count exceeds max_evaluations")
    fine_nodes, fine_weights = _simpson_nodes(measure, provider.subintervals)
    coarse_nodes, coarse_weights = _simpson_nodes(measure, provider.subintervals // 2)
    fine, _ = _weighted_sum(function, fine_nodes, fine_weights)
    coarse, _ = _weighted_sum(function, coarse_nodes, coarse_weights)
    estimate = _norm_difference(fine, coarse) / 15.0
    residuals = {"richardson_difference": _norm_difference(fine, coarse)}
    unchecked: tuple[str, ...] = ()
    checked = ["bounded interval", "composite Simpson rule", "finite integrand samples"]
    if provider.derivative_bound is None:
        error = IntegrationError(
            estimate,
            ErrorGuarantee.ESTIMATED,
            ("Richardson fine/coarse difference",),
        )
        exactness = Exactness.HEURISTIC
        unchecked = ("Richardson estimate bounds the true quadrature error",)
    else:
        if provider.derivative_bound.guarantee is ErrorGuarantee.RIGOROUS:
            raise ValidationError(
                "an internally rigorous polynomial bound requires the exact polynomial integrand"
            )
        if np.asarray(fine, dtype=object).shape != ():
            raise ValidationError(
                "derivative-bound Simpson currently requires a scalar-valued integrand"
            )
        length = _finite_float_scalar(
            measure.upper - measure.lower,
            name="interval length",
        )
        step = length / provider.subintervals
        try:
            bound = (
                abs(_finite_float_scalar(measure.density, name="interval density"))
                * length
                * step**4
                * _finite_float_scalar(
                    provider.derivative_bound.bound,
                    name="derivative bound",
                )
                / 180.0
            )
        except OverflowError as exc:
            raise ValidationError("conditional Simpson error bound overflowed") from exc
        if not isfinite(bound):
            raise ValidationError("conditional Simpson error bound became nonfinite")
        guarantee = provider.derivative_bound.guarantee
        error = IntegrationError(bound, guarantee, (provider.derivative_bound.description,))
        if guarantee is ErrorGuarantee.CONDITIONAL:
            exactness = Exactness.CONDITIONAL
            unchecked = tuple(provider.derivative_bound.certificate.assumptions_unchecked) + (
                "derivative bound applies to this callable and interval",
                "floating evaluation, node, weight, and summation roundoff lie within the stated bound",
            )
        else:
            exactness = Exactness.HEURISTIC
            unchecked = (
                "derivative bound is not independently rigorous",
                "floating-point roundoff is not enclosed",
            )
    return IntegralResult(
        fine,
        Certificate(
            valid=True,
            exactness=exactness,
            residuals=residuals,
            assumptions_checked=tuple(checked),
            assumptions_unchecked=unchecked,
            diagnostics={
                "subintervals": provider.subintervals,
                "evaluations": evaluations,
                "error_guarantee": error.guarantee.value,
                "absolute_error_bound": error.absolute_bound,
            },
            theorem_id="INT.QUAD.SIMPSON",
            message="composite Simpson integral evaluated with explicit error status",
        ),
        error,
        "composite Simpson",
        evaluations,
    )


def _simpson_polynomial_integral(
    polynomial: ExactPolynomial,
    measure: IntervalMeasure,
    provider: CompositeSimpson,
) -> IntegralResult:
    """Evaluate a polynomial Simpson rule and its bound with exact rationals."""

    if polynomial.dimension != 1:
        raise DimensionError("Simpson polynomial integration requires a univariate polynomial")
    if measure.exactness is not Exactness.EXACT:
        raise ValidationError("exact polynomial Simpson evaluation requires rational interval data")
    if measure.lower == measure.upper or measure.density == 0:
        return _zero_interval_result()
    evaluations = provider.subintervals + 1 + provider.subintervals // 2 + 1
    if evaluations > provider.max_evaluations:
        raise ValidationError("quadrature evaluation count exceeds max_evaluations")
    fine_nodes, fine_weights = _exact_simpson_nodes(measure, provider.subintervals)
    coarse_nodes, coarse_weights = _exact_simpson_nodes(
        measure,
        provider.subintervals // 2,
    )
    fine, fine_exact = _weighted_sum(polynomial, fine_nodes, fine_weights)
    coarse, coarse_exact = _weighted_sum(polynomial, coarse_nodes, coarse_weights)
    if not fine_exact or not coarse_exact or not isinstance(fine, Fraction) or not isinstance(coarse, Fraction):
        raise ValidationError("exact polynomial Simpson evaluation lost rational closure")
    difference = abs(fine - coarse)
    if provider.derivative_bound is None:
        error = IntegrationError(
            difference / 15,
            ErrorGuarantee.ESTIMATED,
            ("exact fine/coarse Simpson difference",),
        )
        exactness = Exactness.HEURISTIC
        unchecked = ("Richardson estimate bounds the true quadrature error",)
        checked = (
            "exact rational Simpson nodes and weights",
            "exact polynomial evaluation",
        )
    else:
        derivative_bound = provider.derivative_bound
        if not derivative_bound.applies_to(polynomial, measure):
            raise ValidationError(
                "polynomial derivative bound does not match the exact integrand and interval"
            )
        if derivative_bound.guarantee is not ErrorGuarantee.RIGOROUS:
            raise ValidationError("polynomial provenance did not establish a rigorous bound")
        bound_value = Fraction(derivative_bound.bound)
        length = Fraction(measure.upper) - Fraction(measure.lower)
        step = length / provider.subintervals
        rigorous_bound = (
            abs(Fraction(measure.density))
            * length
            * step**4
            * bound_value
            / 180
        )
        error = IntegrationError(
            rigorous_bound,
            ErrorGuarantee.RIGOROUS,
            (derivative_bound.description, "exact rational Simpson evaluation"),
        )
        exactness = Exactness.EXACT
        unchecked = ()
        checked = (
            "provenance-matched exact polynomial and interval",
            "exact rational Simpson nodes, weights, evaluation, and summation",
            "exact fourth-derivative supremum bound",
            "composite Simpson truncation theorem",
        )
    return IntegralResult(
        fine,
        Certificate(
            valid=True,
            exactness=exactness,
            residuals={"richardson_difference": _diagnostic_float(difference)},
            assumptions_checked=checked,
            assumptions_unchecked=unchecked,
            diagnostics={
                "subintervals": provider.subintervals,
                "evaluations": evaluations,
                "error_guarantee": error.guarantee.value,
                "absolute_error_bound_exact": str(error.absolute_bound),
                "richardson_difference_exact": str(difference),
            },
            theorem_id="INT.QUAD.SIMPSON.EXACT.POLY",
            message="exact polynomial Simpson integral evaluated with explicit error status",
        ),
        error,
        "exact-rational composite Simpson",
        evaluations,
    )


def _node_count(measure: Measure, subintervals: int) -> int:
    if isinstance(measure, AtomicMeasure):
        return measure.atom_count
    if isinstance(measure, IntervalMeasure):
        return subintervals + 1
    if isinstance(measure, ProductMeasure):
        count = 1
        for factor in measure.factors:
            count *= _node_count(factor, subintervals)
        return count
    if isinstance(measure, PushforwardMeasure):
        return _node_count(measure.base, subintervals)
    raise ValidationError("measure does not expose quadrature nodes")


def _finite_atomic_support(measure: Measure) -> bool:
    if isinstance(measure, AtomicMeasure):
        return True
    if isinstance(measure, PushforwardMeasure):
        return _finite_atomic_support(measure.base)
    if isinstance(measure, ProductMeasure):
        return all(_finite_atomic_support(factor) for factor in measure.factors)
    return False


def _rule(measure: Measure, subintervals: int) -> tuple[tuple[tuple[Any, ...], ...], tuple[Any, ...], bool]:
    if isinstance(measure, AtomicMeasure):
        return measure.points, measure.weights, measure.exactness is Exactness.EXACT
    if isinstance(measure, IntervalMeasure):
        nodes, weights = _simpson_nodes(measure, subintervals)
        return nodes, weights, False
    if isinstance(measure, ProductMeasure):
        rules = tuple(_rule(factor, subintervals) for factor in measure.factors)
        nodes: list[tuple[Any, ...]] = []
        weights: list[Any] = []
        for choices in product(*(range(len(rule[0])) for rule in rules)):
            point: tuple[Any, ...] = ()
            weight: Any = Fraction(1)
            for choice, rule in zip(choices, rules, strict=True):
                point += rule[0][choice]
                weight *= rule[1][choice]
            nodes.append(point)
            weights.append(weight)
        exact = all(rule[2] for rule in rules)
        return tuple(nodes), tuple(weights), exact
    if isinstance(measure, PushforwardMeasure):
        nodes, weights, exact = _rule(measure.base, subintervals)
        mapped = [measure.map_point(point) for point in nodes]
        return tuple(mapped), weights, exact and isinstance(measure.mapping, ExactAffineMap)
    raise ValidationError("measure does not expose quadrature nodes")


def _product_integral(
    function: Callable[[Any], Any],
    measure: ProductMeasure,
    provider: CompositeSimpson,
) -> IntegralResult:
    fine_count = _node_count(measure, provider.subintervals)
    if _finite_atomic_support(measure):
        if fine_count > provider.max_evaluations:
            raise ValidationError("finite atomic product evaluation exceeds max_evaluations")
        fine_nodes, fine_weights, fine_exact = _rule(measure, provider.subintervals)
        value, exact_values = _weighted_sum(function, fine_nodes, fine_weights)
        measure_certificate = measure.certificate()
        measure_certificate.require()
        unchecked = tuple(measure_certificate.assumptions_unchecked)
        exact = (
            fine_exact
            and exact_values
            and measure_certificate.exactness is Exactness.EXACT
            and not unchecked
        )
        if exact:
            result_exactness = Exactness.EXACT
            error = IntegrationError.exact()
        elif measure_certificate.exactness is not Exactness.EXACT:
            result_exactness = measure_certificate.exactness
            if not unchecked:
                unchecked = (
                    "finite atomic product structure is not exact",
                )
            error = IntegrationError(
                None,
                ErrorGuarantee.UNKNOWN,
                ("finite atomic product structure is not exact",),
            )
        elif unchecked:
            result_exactness = Exactness.CONDITIONAL
            error = IntegrationError(
                None,
                ErrorGuarantee.UNKNOWN,
                ("finite atomic measure assumptions are unchecked",),
            )
        else:
            result_exactness = Exactness.NUMERICAL
            unchecked = ("floating-point roundoff is not enclosed",)
            error = IntegrationError(
                None,
                ErrorGuarantee.UNKNOWN,
                ("floating-point product sum",),
            )
        return IntegralResult(
            value,
            Certificate(
                valid=True,
                exactness=result_exactness,
                assumptions_checked=("one finite weighted sum over product support",)
                + tuple(measure_certificate.assumptions_checked),
                assumptions_unchecked=unchecked,
                diagnostics={"evaluations": fine_count},
                theorem_id="INT.EVAL.ATOMIC.PRODUCT",
                message="finite atomic product integral evaluated",
            ),
            error,
            "finite atomic product sum",
            fine_count,
        )
    coarse_count = _node_count(measure, provider.subintervals // 2)
    if fine_count + coarse_count > provider.max_evaluations:
        raise ValidationError("tensor quadrature evaluation count exceeds max_evaluations")
    fine_nodes, fine_weights, _ = _rule(measure, provider.subintervals)
    if provider.derivative_bound is not None:
        raise ValidationError(
            "a one-dimensional derivative bound cannot certify tensor-product quadrature"
        )
    coarse_nodes, coarse_weights, _ = _rule(measure, provider.subintervals // 2)
    fine, _ = _weighted_sum(function, fine_nodes, fine_weights)
    coarse, _ = _weighted_sum(function, coarse_nodes, coarse_weights)
    difference = _norm_difference(fine, coarse)
    estimate = difference / 15.0
    evaluations = fine_count + coarse_count
    result = IntegralResult(
        fine,
        Certificate(
            valid=True,
            exactness=Exactness.HEURISTIC,
            residuals={"tensor_refinement_difference": difference},
            assumptions_checked=("finite tensor-product Simpson rule", "finite integrand samples"),
            assumptions_unchecked=(
                "tensor refinement estimate bounds the true multidimensional quadrature error",
            ),
            diagnostics={
                "subintervals_per_interval_factor": provider.subintervals,
                "evaluations": evaluations,
                "error_guarantee": ErrorGuarantee.ESTIMATED.value,
            },
            theorem_id="INT.QUAD.TENSOR.SIMPSON",
            message="tensor-product quadrature evaluated with a non-rigorous refinement estimate",
        ),
        IntegrationError(
            estimate,
            ErrorGuarantee.ESTIMATED,
            ("tensor fine/coarse difference",),
        ),
        "tensor-product composite Simpson",
        evaluations,
    )
    return _inherit_measure_assumptions(
        result,
        measure,
        theorem_id="INT.QUAD.TENSOR.SIMPSON.MEASURE",
    )


def _exact_polynomial_integral(polynomial: ExactPolynomial, measure: Measure) -> IntegralResult:
    if polynomial.dimension != measure.dimension:
        raise DimensionError("polynomial and measure dimensions do not agree")
    if (
        isinstance(measure, PushforwardMeasure)
        and isinstance(measure.base, AtomicMeasure)
        and not isinstance(measure.mapping, ExactAffineMap)
    ):
        snapshot = measure.materialize_atomic()
        result = _exact_polynomial_integral(polynomial, snapshot)
        measure_certificate = measure.certificate()
        measure_certificate.require()
        unchecked = tuple(measure_certificate.assumptions_unchecked)
        certificate = Certificate(
            valid=result.valid,
            exactness=Exactness.CONDITIONAL if unchecked else result.certificate.exactness,
            assumptions_checked=result.certificate.assumptions_checked
            + ("callable pushforward support materialized once",),
            assumptions_unchecked=result.certificate.assumptions_unchecked + unchecked,
            diagnostics={
                **result.certificate.diagnostics,
                "materialized_atom_count": snapshot.atom_count,
            },
            theorem_id="INT.PUSH.ATOMIC.SNAPSHOT.POLY",
            message="polynomial integrated over one callable-pushforward support snapshot",
        )
        return _inherit_measure_assumptions(
            IntegralResult(
                result.value,
                certificate,
                result.error
                if not unchecked
                else IntegrationError(
                    None,
                    ErrorGuarantee.UNKNOWN,
                    ("callable pushforward determinism is unchecked",),
                ),
                "atomic callable-pushforward snapshot and moments",
                snapshot.atom_count,
            ),
            measure,
            theorem_id="INT.PUSH.ATOMIC.SNAPSHOT.POLY.MEASURE",
        )
    if isinstance(measure, PushforwardMeasure) and isinstance(measure.mapping, ExactAffineMap):
        pulled = measure.mapping.pullback(polynomial)
        base = _exact_polynomial_integral(pulled, measure.base)
        certificate = Certificate(
            valid=base.valid,
            exactness=base.certificate.exactness,
            assumptions_checked=base.certificate.assumptions_checked
            + ("pushforward integral equals pullback integral by definition",),
            assumptions_unchecked=base.certificate.assumptions_unchecked,
            diagnostics={
                **base.certificate.diagnostics,
                "source_dimension": measure.base.dimension,
                "target_dimension": measure.dimension,
                "pulled_term_count": pulled.term_count,
            },
            theorem_id="INT.PUSH.POLY",
            message="polynomial integrated through exact affine pushforward",
        )
        return _inherit_measure_assumptions(
            IntegralResult(
                base.value,
                certificate,
                base.error,
                "exact affine pullback and moments",
                base.evaluations,
            ),
            measure,
            theorem_id="INT.PUSH.POLY.MEASURE",
        )
    measure_certificate = measure.certificate()
    measure_certificate.require()
    value: Any = Fraction(0)
    certificates = [measure_certificate]
    moment_values: list[Any] = []
    for exponent, coefficient in polynomial.terms:
        moment = measure.moment(exponent)
        moment.certificate.require()
        moment_values.append(moment.value)
        try:
            value += coefficient * moment.value
        except OverflowError as exc:
            raise ValidationError(
                "polynomial moment contraction overflowed floating-point range"
            ) from exc
        _finite_value(value)
        certificates.append(moment.certificate)
    checked = tuple(
        dict.fromkeys(
            assumption
            for certificate in certificates
            for assumption in certificate.assumptions_checked
        )
    )
    inherited_unchecked = tuple(
        dict.fromkeys(
            assumption
            for certificate in certificates
            for assumption in certificate.assumptions_unchecked
        )
    )
    exact_value_closure = (
        all(_is_exact_value(moment) for moment in moment_values)
        and _is_exact_value(value)
    )
    exact = all(
        certificate.exactness is Exactness.EXACT
        and not certificate.assumptions_unchecked
        for certificate in certificates
    ) and exact_value_closure
    inherited_exactness = _weakest_exactness(
        *(certificate.exactness for certificate in certificates)
    )
    if exact:
        error = IntegrationError.exact()
        exactness = Exactness.EXACT
        unchecked: tuple[str, ...] = ()
    else:
        error = IntegrationError(
            None,
            ErrorGuarantee.UNKNOWN,
            ("measure or moment computation is not unconditionally exact",),
        )
        exactness = (
            Exactness.CONDITIONAL
            if inherited_exactness is Exactness.EXACT and inherited_unchecked
            else Exactness.NUMERICAL
            if inherited_exactness is Exactness.EXACT and not exact_value_closure
            else inherited_exactness
        )
        if inherited_unchecked:
            unchecked = inherited_unchecked
        elif exactness is Exactness.NUMERICAL:
            unchecked = ("floating-point moment roundoff is not enclosed",)
        else:
            unchecked = (
                f"measure or moment status is {exactness.value} rather than exact",
            )
    return IntegralResult(
        value,
        Certificate(
            valid=True,
            exactness=exactness,
            assumptions_checked=("finite sparse polynomial", "measure monomial moments")
            + checked,
            assumptions_unchecked=unchecked,
            diagnostics={
                "dimension": polynomial.dimension,
                "term_count": polynomial.term_count,
                "measure": measure.name,
            },
            theorem_id="INT.EVAL.POLY.MOMENTS",
            message="polynomial integral evaluated by measure moments",
        ),
        error,
        "polynomial moment contraction",
        polynomial.term_count,
    )


def integrate(
    integrand: ExactPolynomial | Callable[[Any], Any],
    measure: Measure,
    *,
    provider: CompositeSimpson | None = None,
) -> IntegralResult:
    """Integrate an exact polynomial or callable against a finite measure.

    Exact polynomials use measure moments.  Arbitrary callables use finite sums
    on atomic measures and composite Simpson quadrature on interval factors.
    The returned error object distinguishes exact, rigorous, conditional,
    estimated, and unknown claims.
    """

    if not isinstance(measure, Measure):
        raise TypeError("measure must implement the Measure protocol")
    measure.certificate().require()
    if provider is not None and not isinstance(provider, CompositeSimpson):
        raise TypeError("provider must be a CompositeSimpson object")
    if isinstance(integrand, ExactPolynomial):
        if provider is not None:
            if not isinstance(measure, IntervalMeasure):
                raise ValidationError(
                    "explicit Simpson evaluation of an exact polynomial requires an IntervalMeasure"
                )
            result = _simpson_polynomial_integral(integrand, measure, provider)
            return _inherit_measure_assumptions(
                result,
                measure,
                theorem_id="INT.QUAD.SIMPSON.EXACT.POLY.MEASURE",
            )
        return _exact_polynomial_integral(integrand, measure)
    if not callable(integrand):
        raise TypeError("integrand must be callable or ExactPolynomial")
    rule = CompositeSimpson() if provider is None else provider
    if isinstance(measure, AtomicMeasure):
        return _inherit_measure_assumptions(
            _atomic_integral(integrand, measure),
            measure,
            theorem_id="INT.EVAL.ATOM.MEASURE",
        )
    if isinstance(measure, IntervalMeasure):
        return _inherit_measure_assumptions(
            _simpson_integral(integrand, measure, rule),
            measure,
            theorem_id="INT.QUAD.SIMPSON.MEASURE",
        )
    if isinstance(measure, ProductMeasure):
        return _product_integral(integrand, measure, rule)
    if isinstance(measure, PushforwardMeasure):
        if rule.derivative_bound is not None:
            raise ValidationError(
                "pushforward quadrature requires a derivative bound for the composed integrand"
            )
        if isinstance(measure.base, AtomicMeasure):
            result = _atomic_integral(integrand, measure.materialize_atomic())
            return _inherit_measure_assumptions(
                result,
                measure,
                theorem_id="INT.EVAL.PUSHFORWARD.ATOMIC.CALLABLE",
            )

        def composed(point: Any) -> Any:
            image = measure.map_point(point)
            target = image[0] if measure.dimension == 1 else image
            return integrand(target)

        result = integrate(
            composed,
            measure.base,
            provider=rule.without_derivative_bound(),
        )
        return _inherit_measure_assumptions(
            result,
            measure,
            theorem_id="INT.EVAL.PUSHFORWARD.CALLABLE",
        )
    raise ValidationError("no callable integration provider is available for this measure")


def integrate_polynomial_oriented(
    polynomial: ExactPolynomial,
    lower: int | Fraction,
    upper: int | Fraction,
    *,
    density: int | Fraction = 1,
) -> IntegralResult:
    r"""Integrate over oriented bounds without mislabeling orientation as a measure."""

    if not isinstance(polynomial, ExactPolynomial) or polynomial.dimension != 1:
        raise DimensionError("oriented polynomial integration requires a univariate ExactPolynomial")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, Fraction))
        for value in (lower, upper, density)
    ):
        raise ValidationError("oriented exact integration requires integers or Fractions")
    a = Fraction(lower)
    b = Fraction(upper)
    rho = Fraction(density)
    if rho < 0:
        raise ValidationError("oriented integration density must be nonnegative")
    if a == b or rho == 0:
        value = Fraction(0)
    else:
        sign = Fraction(1) if a < b else Fraction(-1)
        measure = IntervalMeasure(min(a, b), max(a, b), density=rho)
        value = sign * _exact_polynomial_integral(polynomial, measure).require()
    return IntegralResult(
        value,
        Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "exact rational polynomial",
                "oriented endpoint convention",
                "nonnegative constant density",
            ),
            diagnostics={"lower": str(a), "upper": str(b), "density": str(rho)},
            theorem_id="INT.EVAL.ORIENTED.POLY",
            message="oriented polynomial integral evaluated exactly",
        ),
        IntegrationError.exact(),
        "exact oriented polynomial antiderivative",
        polynomial.term_count,
    )


def integrate_derivative_provider(
    provider: Any,
    measure: Measure,
    *,
    quadrature: CompositeSimpson | None = None,
) -> IntegralResult:
    """Integrate the value map of an existing finite derivative provider."""

    try:
        source_dimension = provider.source_dimension
        target_dimension = provider.target_dimension
        value = provider.value
    except AttributeError as exc:
        raise TypeError(
            "provider must expose source_dimension, target_dimension, and value(point)"
        ) from exc
    if (
        isinstance(source_dimension, bool)
        or not isinstance(source_dimension, int)
        or source_dimension < 1
        or isinstance(target_dimension, bool)
        or not isinstance(target_dimension, int)
        or target_dimension < 1
    ):
        raise DimensionError("derivative-provider dimensions must be positive integers")
    if source_dimension != measure.dimension:
        raise DimensionError("derivative provider source dimension does not match the measure")

    def function(point: Any) -> Any:
        vector = np.asarray([point] if source_dimension == 1 and np.asarray(point).ndim == 0 else point)
        output = np.asarray(value(vector), dtype=object)
        if output.shape != (target_dimension,):
            raise DimensionError("derivative provider value has the wrong target shape")
        return output

    return integrate(function, measure, provider=quadrature)


__all__ = [
    "CompositeSimpson",
    "DerivativeBound",
    "ErrorGuarantee",
    "IntegralResult",
    "IntegrationError",
    "integrate",
    "integrate_derivative_provider",
    "integrate_polynomial_oriented",
    "polynomial_derivative_bound",
]
