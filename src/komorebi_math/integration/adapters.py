"""Adapters from existing coordinate, series, derivative, and frame calculi."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite
from numbers import Complex, Integral
from typing import Any

import numpy as np

from ..certificates import Certificate, Exactness
from ..coordinates.analysis_synthesis import AnalysisSynthesisSystem
from ..coordinates.polynomials import (
    PolynomialFamily,
    polynomial_add,
    polynomial_scale,
)
from ..exceptions import DimensionError, ValidationError
from ..generating.series import FormalSeries
from .measures import AtomicMeasure, Measure
from .polynomial import ExactPolynomial, Rational
from .quadrature import IntegralResult, integrate


_EXACTNESS_STRENGTH = {
    Exactness.EXACT: 0,
    Exactness.NUMERICAL: 1,
    Exactness.CONDITIONAL: 2,
    Exactness.HEURISTIC: 3,
    Exactness.PLANNED: 4,
}


def _weakest_exactness(*values: Exactness) -> Exactness:
    return max(values, key=_EXACTNESS_STRENGTH.__getitem__)


def _fraction(value: Any, *, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValidationError(f"{name} must be an integer or Fraction")
    return Fraction(value)


def _provider_dimension(provider: Any, attribute: str, index: int) -> int:
    try:
        value = getattr(provider, attribute)
    except AttributeError as exc:
        raise TypeError(
            f"provider {index} must expose {attribute}"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise DimensionError(
            f"provider {index} {attribute} must be a positive integer"
        )
    return int(value)


def _merged_strings(*groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def _finite_provider_vector(value: np.ndarray, *, name: str) -> None:
    """Validate provider data without float-coercing arbitrary-size exact values."""

    for entry in value.reshape(-1):
        if isinstance(entry, bool):
            raise ValidationError(f"{name} must contain finite numeric data")
        if isinstance(entry, (Integral, Fraction)):
            continue
        if isinstance(entry, Complex):
            try:
                scalar = complex(entry)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValidationError(f"{name} must contain finite numeric data") from exc
            if isfinite(scalar.real) and isfinite(scalar.imag):
                continue
        raise ValidationError(f"{name} must contain finite numeric data")


def polynomial_from_family_coordinates(
    family: PolynomialFamily,
    coefficients: Iterable[Rational],
) -> ExactPolynomial:
    """Convert finite coordinates in any exact polynomial family to a polynomial."""

    if not isinstance(family, PolynomialFamily):
        raise TypeError("family must be a PolynomialFamily")
    values = tuple(_fraction(value, name="family coordinate") for value in coefficients)
    if not values:
        raise ValidationError("family coordinate vector must be nonempty")
    polynomial = (Fraction(0),)
    for degree, coefficient in enumerate(values):
        polynomial = polynomial_add(
            polynomial,
            polynomial_scale(family.polynomial(degree), coefficient),
        )
    return ExactPolynomial.univariate(polynomial)


def polynomial_from_formal_series(
    series: FormalSeries,
    *,
    degree: int | None = None,
) -> ExactPolynomial:
    """Convert one finite normalized formal series to ordinary polynomial form."""

    if not isinstance(series, FormalSeries):
        raise TypeError("series must be a FormalSeries")
    return ExactPolynomial.univariate(series.ordinary_coefficients(degree))


def integrate_polynomial_coordinates(
    family: PolynomialFamily,
    coefficients: Iterable[Rational],
    measure: Measure,
) -> IntegralResult:
    return integrate(polynomial_from_family_coordinates(family, coefficients), measure)


def integrate_formal_series(
    series: FormalSeries,
    measure: Measure,
    *,
    degree: int | None = None,
) -> IntegralResult:
    return integrate(polynomial_from_formal_series(series, degree=degree), measure)


@dataclass(slots=True)
class AtomicIntegratedDerivativeProvider:
    r"""A fixed finite weighted sum of compatible derivative providers.

    The weights are constant.  The class intentionally does not implement the
    formula for a parameter-dependent measure, which would require an
    additional derivative-of-measure term.
    """

    providers: Sequence[Any]
    weights: Sequence[int | Fraction | float]
    name: str = "atomic integrated derivative provider"
    source_dimension: int = field(init=False)
    target_dimension: int = field(init=False)
    exactness: Exactness = field(init=False)

    def __post_init__(self) -> None:
        self.providers = tuple(self.providers)
        if not self.providers:
            raise ValidationError("integrated derivative providers require at least one provider")
        raw_weights = tuple(self.weights)
        if len(raw_weights) != len(self.providers):
            raise DimensionError("provider and weight counts must agree")
        normalized: list[Fraction | float] = []
        for value in raw_weights:
            if isinstance(value, bool) or not isinstance(value, (int, Fraction, float)):
                raise ValidationError("provider weights must be finite real scalars")
            coefficient: Fraction | float = Fraction(value) if isinstance(value, (int, Fraction)) else float(value)
            if isinstance(coefficient, float) and not isfinite(coefficient):
                raise ValidationError("provider weights must be finite")
            if coefficient < 0:
                raise ValidationError("positive atomic integration requires nonnegative weights")
            normalized.append(coefficient)
        self.weights = tuple(normalized)
        dimensions = tuple(
            (
                _provider_dimension(provider, "source_dimension", index),
                _provider_dimension(provider, "target_dimension", index),
            )
            for index, provider in enumerate(self.providers)
        )
        self.source_dimension, self.target_dimension = dimensions[0]
        if any(dimension != dimensions[0] for dimension in dimensions[1:]):
            raise DimensionError("integrated derivative providers must share source and target dimensions")
        try:
            exactnesses = tuple(provider.exactness for provider in self.providers)
        except AttributeError as exc:
            raise TypeError("every provider must expose exactness") from exc
        if any(not isinstance(value, Exactness) for value in exactnesses):
            raise TypeError("every provider exactness must be an Exactness value")
        exact_weights = all(isinstance(value, Fraction) for value in self.weights)
        self.exactness = _weakest_exactness(
            *exactnesses,
            Exactness.EXACT if exact_weights else Exactness.NUMERICAL,
        )
        self.name = str(self.name).strip()
        if not self.name:
            raise ValidationError("integrated derivative provider name must be nonempty")

    def supports_order(self, order: int) -> bool:
        return all(provider.supports_order(order) for provider in self.providers)

    def _weighted(self, values: Sequence[Any], *, operation: str) -> np.ndarray:
        if len(values) != len(self.providers):
            raise DimensionError("provider output count does not match provider count")
        total = np.zeros(self.target_dimension, dtype=object)
        for index, (weight, value) in enumerate(
            zip(self.weights, values, strict=True)
        ):
            try:
                output = np.asarray(value, dtype=object)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"provider {index} {operation} output must be vector-like"
                ) from exc
            if output.shape != (self.target_dimension,):
                raise DimensionError(
                    f"provider {index} {operation} output must have shape "
                    f"({self.target_dimension},)"
                )
            _finite_provider_vector(
                output,
                name=f"provider {index} {operation} output",
            )
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    total = total + output * weight
            except OverflowError as exc:
                raise ValidationError(
                    f"provider {index} {operation} weighting overflowed floating-point range"
                ) from exc
            _finite_provider_vector(
                total,
                name=f"provider {index} {operation} weighted sum",
            )
        return total

    def value(self, point: Any) -> Any:
        return self._weighted(
            tuple(provider.value(point) for provider in self.providers),
            operation="value",
        )

    def derivative(self, order: int, point: Any, directions: Sequence[Any]) -> Any:
        if not self.supports_order(order):
            raise ValidationError(f"derivative order {order} is not supported by every provider")
        return self._weighted(
            tuple(
                provider.derivative(order, point, directions)
                for provider in self.providers
            ),
            operation="derivative",
        )

    def certificate(self) -> Certificate:
        return Certificate(
            valid=True,
            exactness=self.exactness,
            assumptions_checked=(
                "finite fixed weights",
                "common provider source and target dimensions",
                "linearity of finite differentiation",
            ),
            assumptions_unchecked=(
                "providers represent globally correct derivatives",
            ),
            diagnostics={
                "provider_count": len(self.providers),
                "source_dimension": self.source_dimension,
                "target_dimension": self.target_dimension,
            },
            theorem_id="INT.ADAPTER.DERIVATIVE",
            message="finite weighted derivative provider validated",
        )


def analysis_synthesis_integration_certificate(
    system: AnalysisSynthesisSystem,
    measure: AtomicMeasure,
    physical_values: Sequence[Any],
    *,
    tolerance: float = 1e-10,
) -> Certificate:
    """Check that a fixed analysis/synthesis system commutes with finite integration."""

    if not isinstance(system, AnalysisSynthesisSystem):
        raise TypeError("system must be an AnalysisSynthesisSystem")
    if not isinstance(measure, AtomicMeasure):
        raise TypeError("frame integration certification requires an AtomicMeasure")
    if isinstance(tolerance, bool):
        raise ValidationError("tolerance must be finite and nonnegative")
    try:
        threshold = float(tolerance)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError("tolerance must be finite and nonnegative") from exc
    if not isfinite(threshold) or threshold < 0:
        raise ValidationError("tolerance must be finite and nonnegative")
    system_certificate = system.certificate()
    measure_certificate = measure.certificate()
    values = tuple(np.asarray(value) for value in physical_values)
    if len(values) != measure.atom_count:
        raise DimensionError("physical value count must match the atomic support")
    if any(value.ndim != 1 or value.shape[0] != system.source_dimension for value in values):
        raise DimensionError("physical values do not match the analysis-synthesis source dimension")
    try:
        weights = np.asarray(measure.weights, dtype=float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError(
            "atomic weights cannot be represented as finite floating-point values"
        ) from exc
    if not np.all(np.isfinite(weights)):
        raise ValidationError(
            "atomic weights cannot be represented as finite floating-point values"
        )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            physical = sum(
                (
                    weight * value
                    for weight, value in zip(weights, values, strict=True)
                ),
                np.zeros(system.source_dimension),
            )
            analyzed_samples = tuple(system.analyze(value) for value in values)
            coefficients = sum(
                (
                    weight * value
                    for weight, value in zip(weights, analyzed_samples, strict=True)
                ),
                np.zeros(system.coordinate_dimension),
            )
    except (OverflowError, TypeError) as exc:
        raise ValidationError(
            "analysis-synthesis integration arithmetic overflowed"
        ) from exc
    analysis_residual = float(np.linalg.norm(system.analyze(physical) - coefficients))
    synthesis_residual = float(np.linalg.norm(system.synthesize(coefficients) - physical))
    projection = system.analysis @ system.synthesis
    gauge_residual = float(np.linalg.norm(projection @ coefficients - coefficients))
    commutation_valid = (
        max(analysis_residual, synthesis_residual, gauge_residual) <= threshold
    )
    valid = (
        system_certificate.valid
        and measure_certificate.valid
        and commutation_valid
    )
    residuals = dict(system_certificate.residuals)
    residuals.update(
        {
            "analysis_commutation": analysis_residual,
            "synthesis_reconstruction": synthesis_residual,
            "coefficient_gauge": gauge_residual,
        }
    )
    diagnostics = dict(system_certificate.diagnostics)
    diagnostics.update(
        {
            "atom_count": measure.atom_count,
            "source_dimension": system.source_dimension,
            "coefficient_dimension": system.coordinate_dimension,
            "system_certificate_valid": system_certificate.valid,
            "system_certificate_exactness": system_certificate.exactness.value,
            "system_certificate_tolerance": system_certificate.tolerance,
            "system_certificate_theorem_id": system_certificate.theorem_id,
            "system_certificate_message": system_certificate.message,
            "measure_certificate_valid": measure_certificate.valid,
            "measure_certificate_exactness": measure_certificate.exactness.value,
            "measure_certificate_theorem_id": measure_certificate.theorem_id,
            "measure_certificate_message": measure_certificate.message,
        }
    )
    if not system_certificate.valid:
        message = "analysis-synthesis integration commutation rejected an invalid system certificate"
    elif not measure_certificate.valid:
        message = "analysis-synthesis integration commutation rejected an invalid measure certificate"
    elif valid:
        message = "analysis-synthesis integration commutation checked"
    else:
        message = "analysis-synthesis integration commutation residual exceeded tolerance"
    return Certificate(
        valid=valid,
        exactness=_weakest_exactness(
            Exactness.NUMERICAL,
            system_certificate.exactness,
            measure_certificate.exactness,
        ),
        residuals=residuals,
        tolerance=threshold,
        assumptions_checked=_merged_strings(
            system_certificate.assumptions_checked,
            measure_certificate.assumptions_checked,
            (
                "analysis-synthesis system certificate incorporated",
                "atomic-measure certificate incorporated",
                "finite atomic integration",
                "constant integration weights",
            ),
        ),
        assumptions_unchecked=_merged_strings(
            system_certificate.assumptions_unchecked,
            measure_certificate.assumptions_unchecked,
        ),
        diagnostics=diagnostics,
        theorem_id="INT.ADAPTER.FRAME",
        message=message,
    )


__all__ = [
    "AtomicIntegratedDerivativeProvider",
    "analysis_synthesis_integration_certificate",
    "integrate_formal_series",
    "integrate_polynomial_coordinates",
    "polynomial_from_family_coordinates",
    "polynomial_from_formal_series",
]
