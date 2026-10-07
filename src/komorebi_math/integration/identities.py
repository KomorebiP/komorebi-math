"""Exact finite integral identities with replayable witness data."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Any

from ..certificates import Certificate, Exactness
from ..exceptions import DimensionError, ValidationError
from .measures import AtomicMeasure, ExactAffineMap, IntervalMeasure, Measure, ProductMeasure, PushforwardMeasure
from .polynomial import ExactPolynomial
from .quadrature import integrate


def _require_exact_measure_structure(measure: Measure, *, context: str) -> Certificate:
    certificate = measure.certificate()
    certificate.require()
    if (
        certificate.exactness is not Exactness.EXACT
        or certificate.assumptions_unchecked
    ):
        raise ValidationError(f"{context} requires an unconditionally exact measure certificate")
    return certificate


@dataclass(frozen=True, slots=True)
class IntegralIdentity:
    """Two computed sides of an integral identity and their certificate."""

    name: str
    left: Any
    right: Any
    certificate: Certificate

    @property
    def valid(self) -> bool:
        return self.certificate.valid

    def require(self) -> "IntegralIdentity":
        self.certificate.require()
        return self


def _identity_certificate(
    *,
    name: str,
    left: Any,
    right: Any,
    theorem_id: str,
    checked: tuple[str, ...],
    diagnostics: dict[str, Any] | None = None,
) -> Certificate:
    valid = left == right
    exact_difference = left - right
    details = {} if diagnostics is None else dict(diagnostics)
    details["exact_difference"] = str(exact_difference)
    return Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={"identity": 0.0 if valid else 1.0},
        tolerance=0.0,
        assumptions_checked=checked,
        diagnostics=details,
        theorem_id=theorem_id,
        message=f"{name} verified exactly" if valid else f"{name} failed exact verification",
    )


def fundamental_theorem(
    polynomial: ExactPolynomial,
    measure: IntervalMeasure,
) -> IntegralIdentity:
    r"""Certify ``integral p d(rho x) = rho(F(b)-F(a))`` exactly."""

    if not isinstance(polynomial, ExactPolynomial) or polynomial.dimension != 1:
        raise DimensionError("fundamental-theorem certification requires a univariate polynomial")
    if not isinstance(measure, IntervalMeasure) or measure.exactness is not Exactness.EXACT:
        raise ValidationError("fundamental-theorem certification requires an exact interval measure")
    _require_exact_measure_structure(measure, context="fundamental-theorem certification")
    integral = integrate(polynomial, measure)
    antiderivative = polynomial.antiderivative()
    right = measure.density * (
        antiderivative(measure.upper) - antiderivative(measure.lower)
    )
    certificate = _identity_certificate(
        name="fundamental theorem of calculus",
        left=integral.value,
        right=right,
        theorem_id="INT.ID.FTC",
        checked=(
            "exact rational polynomial antiderivative",
            "constant-density bounded interval",
            "endpoint evaluation",
        ),
        diagnostics={
            "integrand_degree": polynomial.degree,
            "antiderivative_degree": antiderivative.degree,
        },
    )
    return IntegralIdentity("fundamental theorem of calculus", integral.value, right, certificate)


def integration_by_parts(
    left_factor: ExactPolynomial,
    right_factor: ExactPolynomial,
    measure: IntervalMeasure,
) -> IntegralIdentity:
    r"""Certify ``int u v' = rho[uv]_a^b - int u' v`` exactly."""

    if not isinstance(left_factor, ExactPolynomial) or not isinstance(right_factor, ExactPolynomial):
        raise TypeError("integration by parts requires ExactPolynomial factors")
    if left_factor.dimension != 1 or right_factor.dimension != 1:
        raise DimensionError("integration by parts currently requires univariate polynomials")
    if not isinstance(measure, IntervalMeasure) or measure.exactness is not Exactness.EXACT:
        raise ValidationError("integration-by-parts certification requires an exact interval measure")
    _require_exact_measure_structure(measure, context="integration-by-parts certification")
    product_uv = left_factor.multiply(right_factor)
    left_integrand = left_factor.multiply(right_factor.derivative())
    remainder_integrand = left_factor.derivative().multiply(right_factor)
    left = integrate(left_integrand, measure).value
    boundary = measure.density * (
        product_uv(measure.upper) - product_uv(measure.lower)
    )
    right = boundary - integrate(remainder_integrand, measure).value
    certificate = _identity_certificate(
        name="integration by parts",
        left=left,
        right=right,
        theorem_id="INT.ID.IBP",
        checked=("exact product rule", "exact endpoint term", "exact polynomial moments"),
        diagnostics={
            "left_factor_degree": left_factor.degree,
            "right_factor_degree": right_factor.degree,
            "boundary_term": str(boundary),
        },
    )
    return IntegralIdentity("integration by parts", left, right, certificate)


def fubini_identity(
    polynomial: ExactPolynomial,
    measure: ProductMeasure,
    *,
    order: tuple[int, ...] | None = None,
) -> IntegralIdentity:
    """Certify direct versus iterated integration on an exact interval box."""

    if not isinstance(polynomial, ExactPolynomial):
        raise TypeError("Fubini certification requires an ExactPolynomial")
    if not isinstance(measure, ProductMeasure):
        raise TypeError("Fubini certification requires a ProductMeasure")
    if polynomial.dimension != measure.dimension:
        raise DimensionError("polynomial and product-measure dimensions do not agree")
    interval_factors = all(
        isinstance(factor, IntervalMeasure)
        and factor.dimension == 1
        and factor.exactness is Exactness.EXACT
        for factor in measure.factors
    )
    atomic_factors = all(
        isinstance(factor, AtomicMeasure)
        and factor.dimension == 1
        and factor.exactness is Exactness.EXACT
        for factor in measure.factors
    )
    if not interval_factors and not atomic_factors:
        raise ValidationError(
            "exact Fubini certification requires all-rational interval or one-dimensional atomic factors"
        )
    _require_exact_measure_structure(measure, context="exact Fubini certification")
    for factor in measure.factors:
        _require_exact_measure_structure(factor, context="exact Fubini certification")
    variables = tuple(range(polynomial.dimension)) if order is None else tuple(order)
    if sorted(variables) != list(range(polynomial.dimension)):
        raise ValidationError("Fubini order must be a permutation of every variable")
    direct = integrate(polynomial, measure).value
    if interval_factors:
        current: Fraction | ExactPolynomial = polynomial
        labels = list(range(polynomial.dimension))
        for original_variable in variables:
            if not isinstance(current, ExactPolynomial):
                raise ValidationError("Fubini iteration exhausted variables too early")
            position = labels.index(original_variable)
            factor = measure.factors[original_variable]
            current = current.integrate_out(
                position,
                factor.lower,
                factor.upper,
                density=factor.density,
            )
            labels.pop(position)
        if not isinstance(current, Fraction):
            raise ValidationError("Fubini iteration did not reduce to a scalar")
        iterated = current
        factor_kind = "interval"
    else:
        assigned = [Fraction(0) for _ in range(polynomial.dimension)]

        def nested(level: int) -> Fraction:
            if level == len(variables):
                return polynomial(tuple(assigned))
            variable = variables[level]
            factor = measure.factors[variable]
            total = Fraction(0)
            for point, weight in zip(factor.points, factor.weights, strict=True):
                assigned[variable] = point[0]
                total += weight * nested(level + 1)
            return total

        iterated = nested(0)
        factor_kind = "atomic"
    certificate = _identity_certificate(
        name="Fubini identity",
        left=direct,
        right=iterated,
        theorem_id="INT.ID.FUBINI",
        checked=(
            "finite sparse polynomial",
            "finite exact product measure",
            "exact iterated finite sums or antiderivatives",
        ),
        diagnostics={
            "order": variables,
            "factor_count": len(measure.factors),
            "factor_kind": factor_kind,
        },
    )
    return IntegralIdentity("Fubini identity", direct, iterated, certificate)


def all_fubini_orders(
    polynomial: ExactPolynomial,
    measure: ProductMeasure,
    *,
    max_dimensions: int = 7,
) -> tuple[IntegralIdentity, ...]:
    """Certify every variable order within an explicit factorial budget."""

    if polynomial.dimension > max_dimensions:
        raise ValidationError("all-order Fubini verification exceeds max_dimensions")
    return tuple(
        fubini_identity(polynomial, measure, order=order)
        for order in permutations(range(polynomial.dimension))
    )


def pushforward_identity(
    polynomial: ExactPolynomial,
    base: Measure,
    mapping: ExactAffineMap,
) -> IntegralIdentity:
    r"""Certify ``int f d(mapping_# mu) = int (f o mapping) dmu``."""

    if not isinstance(polynomial, ExactPolynomial):
        raise TypeError("pushforward certification requires an ExactPolynomial")
    if not isinstance(mapping, ExactAffineMap):
        raise TypeError("pushforward certification requires an ExactAffineMap")
    if base.dimension != mapping.source_dimension:
        raise DimensionError("base dimension does not match the affine source")
    if polynomial.dimension != mapping.target_dimension:
        raise DimensionError("polynomial dimension does not match the affine target")
    _require_exact_measure_structure(base, context="exact pushforward certification")
    pushed = PushforwardMeasure(base, mapping, name=f"{mapping.name} pushforward")
    left_result = integrate(polynomial, pushed)
    pulled = mapping.pullback(polynomial)
    right_result = integrate(pulled, base)
    exact = left_result.exact and right_result.exact
    if not exact:
        raise ValidationError("exact pushforward identity requires exact base moments")
    certificate = _identity_certificate(
        name="pushforward integration identity",
        left=left_result.value,
        right=right_result.value,
        theorem_id="INT.ID.PUSHFORWARD",
        checked=("exact affine pullback", "pushforward definition", "exact base moments"),
        diagnostics={
            "source_dimension": mapping.source_dimension,
            "target_dimension": mapping.target_dimension,
            "pulled_term_count": pulled.term_count,
        },
    )
    return IntegralIdentity(
        "pushforward integration identity",
        left_result.value,
        right_result.value,
        certificate,
    )


def affine_interval_change_of_variables(
    polynomial: ExactPolynomial,
    base: IntervalMeasure,
    mapping: ExactAffineMap,
) -> IntegralIdentity:
    r"""Certify the one-dimensional affine Jacobian density law exactly."""

    if polynomial.dimension != 1:
        raise DimensionError("interval change of variables requires a univariate polynomial")
    if base.exactness is not Exactness.EXACT:
        raise ValidationError("exact interval change of variables requires rational interval data")
    _require_exact_measure_structure(base, context="exact interval change of variables")
    if mapping.source_dimension != 1 or mapping.target_dimension != 1:
        raise DimensionError("interval change of variables requires a one-dimensional affine map")
    scale = mapping.matrix[0][0]
    if scale == 0:
        raise ValidationError("Jacobian change of variables requires a nonzero affine scale")
    pushed = base.pushforward_affine(mapping)
    if not isinstance(pushed, IntervalMeasure):
        raise ValidationError("nonzero affine scale must produce an interval pushforward")
    left = integrate(polynomial, pushed).value
    right = integrate(mapping.pullback(polynomial), base).value
    certificate = _identity_certificate(
        name="affine interval change of variables",
        left=left,
        right=right,
        theorem_id="INT.ID.COV.AFFINE",
        checked=(
            "nonzero exact affine scale",
            "pushforward density divided by absolute Jacobian",
            "orientation-normalized transformed interval",
        ),
        diagnostics={
            "scale": str(scale),
            "absolute_jacobian": str(abs(scale)),
            "pushed_density": str(pushed.density),
        },
    )
    return IntegralIdentity("affine interval change of variables", left, right, certificate)


__all__ = [
    "IntegralIdentity",
    "affine_interval_change_of_variables",
    "all_fubini_orders",
    "fubini_identity",
    "fundamental_theorem",
    "integration_by_parts",
    "pushforward_identity",
]
