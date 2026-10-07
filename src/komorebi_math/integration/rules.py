"""Signed quadrature functionals kept distinct from positive measures."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import fsum, isfinite
from numbers import Complex, Real
from sys import float_info
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .measures import AtomicMeasure, Measure
from .polynomial import ExactPolynomial
from .quadrature import ErrorGuarantee, IntegralResult, IntegrationError

Number = Fraction | float


_EXACTNESS_STRENGTH = {
    Exactness.EXACT: 0,
    Exactness.NUMERICAL: 1,
    Exactness.CONDITIONAL: 2,
    Exactness.HEURISTIC: 3,
    Exactness.PLANNED: 4,
}


def _weakest_exactness(*values: Exactness) -> Exactness:
    return max(values, key=_EXACTNESS_STRENGTH.__getitem__)


def _number(value: Any, *, exact: bool, name: str) -> Number:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a finite real scalar")
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    if exact:
        raise ValidationError(f"{name} must be an integer or Fraction")
    result = float(value)
    if not isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _canonical_points(values: Iterable[Any], *, exact: bool) -> tuple[tuple[tuple[Number, ...], ...], int]:
    raw = tuple(values)
    if not raw:
        raise ValidationError("quadrature rules require at least one node")
    first = raw[0]
    if isinstance(first, Real) and not isinstance(first, bool):
        dimension = 1
    else:
        try:
            dimension = len(tuple(first))
        except TypeError as exc:
            raise DimensionError("quadrature node is not vector-like") from exc
        if dimension < 1:
            raise DimensionError("quadrature node dimension must be positive")
    points: list[tuple[Number, ...]] = []
    for raw_point in raw:
        if dimension == 1 and isinstance(raw_point, Real) and not isinstance(raw_point, bool):
            point = (_number(raw_point, exact=exact, name="quadrature node"),)
        else:
            try:
                entries = tuple(raw_point)
            except TypeError as exc:
                raise DimensionError("quadrature node is not vector-like") from exc
            if len(entries) != dimension:
                raise DimensionError("quadrature nodes must share a common dimension")
            point = tuple(
                _number(entry, exact=exact, name="quadrature node coordinate")
                for entry in entries
            )
        points.append(point)
    return tuple(points), dimension


def _exact_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, Fraction)):
        return True
    if isinstance(value, np.ndarray):
        return all(_exact_value(entry) for entry in value.reshape(-1))
    if isinstance(value, (tuple, list)):
        return all(_exact_value(entry) for entry in value)
    return False


def _finite(value: Any) -> None:
    if isinstance(value, bool):
        raise ValidationError("quadrature integrand output must be numeric")
    if isinstance(value, (int, Fraction)):
        return
    if isinstance(value, np.ndarray):
        for entry in value.reshape(-1):
            _finite(entry)
        return
    if isinstance(value, (tuple, list)):
        for entry in value:
            _finite(entry)
        return
    if isinstance(value, Complex):
        scalar = complex(value)
        if isfinite(scalar.real) and isfinite(scalar.imag):
            return
    raise ValidationError("quadrature integrand output must be finite numeric data")


def _display_residual(value: Any) -> float:
    try:
        converted = float(abs(value))
    except (OverflowError, TypeError, ValueError):
        return float_info.max
    if not isfinite(converted):
        return float_info.max
    return converted


def _scale(value: Any, weight: Number) -> Any:
    try:
        if isinstance(value, (tuple, list)):
            return np.asarray(value, dtype=object) * weight
        return value * weight
    except OverflowError as exc:
        raise ValidationError("quadrature scaling overflowed floating-point range") from exc


@dataclass(frozen=True, slots=True)
class QuadratureExactness:
    """Moment comparison between a quadrature functional and one measure."""

    degree: int
    max_tested_degree: int
    exact: bool
    certificate: Certificate


@dataclass(frozen=True, slots=True, init=False)
class QuadratureRule:
    r"""A finite linear functional ``Q[f]=sum_i w_i f(x_i)``.

    Weights may be negative.  Therefore this object is deliberately not a
    measure, even when its nodes and weights are represented exactly.
    """

    points: tuple[tuple[Number, ...], ...]
    weights: tuple[Number, ...]
    dimension: int
    name: str

    def __init__(
        self,
        points: Iterable[Any],
        weights: Iterable[Any],
        *,
        name: str = "quadrature rule",
        _require_exact: bool = False,
    ) -> None:
        nodes, dimension = _canonical_points(points, exact=_require_exact)
        coefficients = tuple(
            _number(value, exact=_require_exact, name="quadrature weight")
            for value in weights
        )
        if len(nodes) != len(coefficients):
            raise DimensionError("quadrature nodes and weights must have equal length")
        label = str(name).strip()
        if not label:
            raise ValidationError("quadrature rule name must be nonempty")
        object.__setattr__(self, "points", nodes)
        object.__setattr__(self, "weights", coefficients)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "name", label)

    @property
    def node_count(self) -> int:
        return len(self.points)

    @property
    def exact_data(self) -> bool:
        return all(
            isinstance(entry, Fraction)
            for point in self.points
            for entry in point
        ) and all(isinstance(weight, Fraction) for weight in self.weights)

    @property
    def positive(self) -> bool:
        return all(weight >= 0 for weight in self.weights)

    def apply(self, function: Callable[[Any], Any]) -> IntegralResult:
        if not callable(function):
            raise TypeError("quadrature application requires a callable")
        terms: list[Any] = []
        exact_outputs = True
        output_shape: tuple[int, ...] | None = None
        for point, weight in zip(self.points, self.weights, strict=True):
            value = function(point[0] if self.dimension == 1 else point)
            _finite(value)
            shape = tuple(np.asarray(value, dtype=object).shape)
            if output_shape is None:
                output_shape = shape
            elif shape != output_shape:
                raise DimensionError("quadrature output shape changed between nodes")
            exact_outputs = exact_outputs and _exact_value(value)
            term = _scale(value, weight)
            _finite(term)
            terms.append(term)
        if not terms:
            raise ValidationError("quadrature rule has no nodes")
        exact_evaluation = self.exact_data and exact_outputs
        if output_shape == () and exact_evaluation:
            total: Any = sum(terms, Fraction(0))
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
                        "quadrature array summation overflowed floating-point range"
                    ) from exc
        _finite(total)
        return IntegralResult(
            total,
            Certificate(
                valid=True,
                exactness=Exactness.EXACT if exact_evaluation else Exactness.NUMERICAL,
                assumptions_checked=(
                    "finite quadrature functional evaluation",
                    "no target-integral equality inferred",
                ),
                diagnostics={
                    "node_count": self.node_count,
                    "positive_weights": self.positive,
                    "exact_functional_evaluation": exact_evaluation,
                },
                theorem_id="INT.QUAD.RULE.APPLY",
                message="quadrature functional evaluated; no integral exactness inferred",
            ),
            IntegrationError(
                None,
                ErrorGuarantee.UNKNOWN,
                ("no target-measure error theorem supplied",),
            ),
            self.name,
            self.node_count,
        )

    def moment(self, exponents: Sequence[int]) -> Any:
        powers = tuple(exponents)
        if len(powers) != self.dimension:
            raise DimensionError("quadrature moment exponent dimension is incorrect")
        if any(isinstance(power, bool) or not isinstance(power, int) or power < 0 for power in powers):
            raise ValidationError("quadrature moment exponents must be nonnegative integers")
        terms: list[Any] = []
        for point, weight in zip(self.points, self.weights, strict=True):
            term: Any = weight
            for coordinate, power in zip(point, powers, strict=True):
                try:
                    term *= coordinate**power
                except OverflowError as exc:
                    raise ValidationError(
                        "quadrature moment overflowed floating-point range"
                    ) from exc
            _finite(term)
            terms.append(term)
        if self.exact_data:
            return sum(terms, Fraction(0))
        try:
            value = fsum(float(term) for term in terms)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValidationError("quadrature moment summation overflowed") from exc
        _finite(value)
        return value

    def integrate_polynomial(self, polynomial: ExactPolynomial) -> IntegralResult:
        if not isinstance(polynomial, ExactPolynomial):
            raise TypeError("quadrature polynomial evaluation requires an ExactPolynomial")
        if polynomial.dimension != self.dimension:
            raise DimensionError("polynomial dimension does not match the quadrature rule")
        value: Any = Fraction(0)
        for exponent, coefficient in polynomial.terms:
            try:
                value += coefficient * self.moment(exponent)
            except OverflowError as exc:
                raise ValidationError(
                    "polynomial quadrature contraction overflowed floating-point range"
                ) from exc
            _finite(value)
        exact_evaluation = self.exact_data
        return IntegralResult(
            value,
            Certificate(
                valid=True,
                exactness=Exactness.EXACT if exact_evaluation else Exactness.NUMERICAL,
                assumptions_checked=(
                    "finite polynomial",
                    "finite quadrature moments",
                    "no target-integral equality inferred",
                ),
                diagnostics={
                    "polynomial_degree": polynomial.degree,
                    "node_count": self.node_count,
                },
                theorem_id="INT.QUAD.RULE.POLY",
                message="polynomial quadrature functional evaluated without claiming integral equality",
            ),
            IntegrationError(None, ErrorGuarantee.UNKNOWN, ("quadrature degree not certified",)),
            f"{self.name} polynomial evaluation",
            polynomial.term_count,
        )

    def tensor_product(self, other: "QuadratureRule") -> "QuadratureRule":
        if not isinstance(other, QuadratureRule):
            raise TypeError("quadrature tensor product requires another QuadratureRule")
        points: list[tuple[Number, ...]] = []
        weights: list[Number] = []
        for left, right in product(range(self.node_count), range(other.node_count)):
            points.append(self.points[left] + other.points[right])
            try:
                weights.append(self.weights[left] * other.weights[right])
            except OverflowError as exc:
                raise ValidationError(
                    "quadrature tensor-product weight overflowed floating-point range"
                ) from exc
        cls = ExactQuadratureRule if self.exact_data and other.exact_data else QuadratureRule
        return cls(points, weights, name=f"{self.name} tensor {other.name}")

    def as_atomic_measure(self) -> AtomicMeasure:
        if not self.positive:
            raise ValidationError("quadrature rules with negative weights are not positive measures")
        return AtomicMeasure(self.points, self.weights, name=f"positive rule {self.name}")

    def degree_of_exactness(
        self,
        measure: Measure,
        *,
        max_degree: int,
        tolerance: float = 1e-12,
    ) -> QuadratureExactness:
        if self.dimension != 1 or measure.dimension != 1:
            raise DimensionError("degree of exactness currently requires one-dimensional objects")
        if isinstance(max_degree, bool) or not isinstance(max_degree, int) or max_degree < 0:
            raise ValidationError("max_degree must be nonnegative")
        try:
            threshold = float(tolerance)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValidationError(
                "quadrature comparison tolerance must be finite and nonnegative"
            ) from exc
        if threshold < 0 or not isfinite(threshold):
            raise ValidationError("quadrature comparison tolerance must be finite and nonnegative")
        measure_certificate = measure.certificate()
        measure_certificate.require()
        exact_comparison = (
            self.exact_data
            and measure_certificate.exactness is Exactness.EXACT
            and not measure_certificate.assumptions_unchecked
        )
        matched = -1
        residuals: dict[str, float] = {}
        exact_differences: dict[str, str] = {}
        inherited_unchecked = list(measure_certificate.assumptions_unchecked)
        inherited_exactness = measure_certificate.exactness
        if not self.exact_data:
            inherited_exactness = _weakest_exactness(
                inherited_exactness,
                Exactness.NUMERICAL,
            )
        for degree in range(max_degree + 1):
            observed = self.moment((degree,))
            expected_result = measure.moment((degree,))
            expected_result.certificate.require()
            expected = expected_result.value
            if not _exact_value(observed) or not _exact_value(expected):
                exact_comparison = False
                inherited_exactness = _weakest_exactness(
                    inherited_exactness,
                    Exactness.NUMERICAL,
                )
            inherited_exactness = _weakest_exactness(
                inherited_exactness,
                expected_result.certificate.exactness,
            )
            if (
                expected_result.certificate.exactness is not Exactness.EXACT
                or expected_result.certificate.assumptions_unchecked
            ):
                exact_comparison = False
                inherited_unchecked.extend(expected_result.certificate.assumptions_unchecked)
            try:
                difference = abs(observed - expected)
            except OverflowError as exc:
                raise ValidationError(
                    "quadrature moment comparison overflowed floating-point range"
                ) from exc
            if exact_comparison:
                equal = difference == 0
                residuals[f"moment_{degree}"] = 0.0 if equal else 1.0
                exact_differences[f"moment_{degree}"] = str(difference)
            else:
                residual = _display_residual(difference)
                if residual == float_info.max and difference != 0:
                    raise ValidationError("numerical quadrature moment residual is not finite")
                residuals[f"moment_{degree}"] = residual
                equal = residual <= threshold
            if not equal:
                break
            matched = degree
        certificate_exactness = Exactness.EXACT if exact_comparison else inherited_exactness
        if (
            not exact_comparison
            and certificate_exactness is Exactness.EXACT
            and inherited_unchecked
        ):
            certificate_exactness = Exactness.CONDITIONAL
        certificate = Certificate(
            valid=True,
            exactness=certificate_exactness,
            residuals=residuals,
            tolerance=0.0 if exact_comparison else threshold,
            assumptions_checked=("monomial moments compared consecutively from degree zero",)
            + tuple(measure_certificate.assumptions_checked),
            assumptions_unchecked=()
            if exact_comparison
            else tuple(dict.fromkeys(inherited_unchecked))
            + ("floating moment agreement implies exact algebraic equality",),
            diagnostics={
                "certified_degree": matched,
                "max_tested_degree": max_degree,
                "exact_comparison": exact_comparison,
                "exact_moment_differences": exact_differences,
            },
            theorem_id="INT.QUAD.RULE.DEGREE",
            message="quadrature degree of exactness evaluated",
        )
        return QuadratureExactness(matched, max_degree, exact_comparison, certificate)

    def certify_polynomial_integral(
        self,
        polynomial: ExactPolynomial,
        measure: Measure,
    ) -> IntegralResult:
        if polynomial.dimension != 1 or measure.dimension != 1 or self.dimension != 1:
            raise DimensionError("polynomial quadrature certification currently requires dimension one")
        exactness = self.degree_of_exactness(measure, max_degree=polynomial.degree, tolerance=0.0)
        if not exactness.exact or exactness.degree < polynomial.degree:
            raise ValidationError("quadrature rule is not exactly certified through the polynomial degree")
        application = self.integrate_polynomial(polynomial)
        return IntegralResult(
            application.value,
            Certificate(
                valid=True,
                exactness=Exactness.EXACT,
                residuals=exactness.certificate.residuals,
                tolerance=0.0,
                assumptions_checked=(
                    "exact monomial moments through polynomial degree",
                    "linearity of finite quadrature functional",
                ),
                diagnostics={
                    "polynomial_degree": polynomial.degree,
                    "certified_degree": exactness.degree,
                },
                theorem_id="INT.QUAD.RULE.CERTIFIED.POLY",
                message="quadrature equals the target integral for this polynomial",
            ),
            IntegrationError.exact(),
            f"certified {self.name}",
            application.evaluations,
        )

    def certificate(self) -> Certificate:
        return Certificate(
            valid=True,
            exactness=Exactness.EXACT if self.exact_data else Exactness.NUMERICAL,
            assumptions_checked=(
                "finite nodes",
                "finite signed weights",
                "common node dimension",
                "signed weights are not treated as a positive measure",
            ),
            diagnostics={
                "dimension": self.dimension,
                "node_count": self.node_count,
                "positive_weights": self.positive,
            },
            theorem_id="INT.QUAD.RULE.1",
            message="quadrature functional validated",
        )


class ExactQuadratureRule(QuadratureRule):
    """A quadrature functional with rational nodes and signed rational weights."""

    __slots__ = ()

    def __init__(
        self,
        points: Iterable[Any],
        weights: Iterable[Any],
        *,
        name: str = "exact quadrature rule",
    ) -> None:
        super().__init__(points, weights, name=name, _require_exact=True)


__all__ = ["ExactQuadratureRule", "QuadratureExactness", "QuadratureRule"]
