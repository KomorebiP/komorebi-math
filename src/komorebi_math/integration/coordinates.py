"""Integration covectors in arbitrary exact polynomial coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

from ..certificates import Certificate, Exactness
from ..coordinates.calculus import derivative_coordinate_operator
from ..coordinates.polynomials import (
    PolynomialFamily,
    connection_transform,
    polynomial_evaluate,
)
from ..exceptions import DimensionError, ValidationError
from .identities import IntegralIdentity
from .measures import IntervalMeasure
from .polynomial import ExactPolynomial, Rational
from .quadrature import integrate


_EXACTNESS_STRENGTH = {
    Exactness.EXACT: 0,
    Exactness.NUMERICAL: 1,
    Exactness.CONDITIONAL: 2,
    Exactness.HEURISTIC: 3,
    Exactness.PLANNED: 4,
}


def _weakest_exactness(*values: Exactness) -> Exactness:
    return max(values, key=_EXACTNESS_STRENGTH.__getitem__)


def _require_exact_interval_structure(
    measure: IntervalMeasure,
    *,
    context: str,
) -> Certificate:
    certificate = measure.certificate()
    certificate.require()
    if (
        certificate.exactness is not Exactness.EXACT
        or certificate.assumptions_unchecked
    ):
        raise ValidationError(f"{context} requires an unconditionally exact interval certificate")
    return certificate


def _degree(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError("coordinate degree must be a nonnegative integer")
    return value


def _coordinate(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValidationError("coordinate coefficients must be integers or Fractions")
    return Fraction(value)


def _basis_witness(
    value: Sequence[Sequence[Rational]] | None,
    degree: int,
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Normalize an exact graded-basis witness through ``degree``."""

    if value is None:
        return None
    try:
        polynomials = tuple(value)
    except TypeError as exc:
        raise ValidationError("basis witness must be a finite polynomial sequence") from exc
    if len(polynomials) != degree + 1:
        raise DimensionError("basis witness length does not match degree")
    normalized: list[tuple[Fraction, ...]] = []
    for index, polynomial in enumerate(polynomials):
        try:
            coefficients = tuple(
                _coordinate(coefficient) for coefficient in polynomial
            )
        except TypeError as exc:
            raise ValidationError(
                "basis witness polynomials must be finite coefficient sequences"
            ) from exc
        if len(coefficients) != index + 1 or coefficients[-1] == 0:
            raise ValidationError(
                "basis witness must contain one exact polynomial of each graded degree"
            )
        normalized.append(coefficients)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class PolynomialIntegrationCovector:
    """The integral functional in one finite polynomial basis.

    A covector constructed directly from weights is usable as a finite linear
    functional, but its claimed relationship to ``family`` is conditional.
    Supplying ``basis_witness`` makes that provenance replayable: construction
    validates the graded basis and recomputes every exact interval moment.
    :func:`polynomial_integration_covector` supplies this witness automatically.
    """

    family: str
    degree: int
    measure: IntervalMeasure
    weights: tuple[Fraction, ...]
    basis_witness: tuple[tuple[Fraction, ...], ...] | None = None

    def __post_init__(self) -> None:
        _degree(self.degree)
        if (
            not isinstance(self.measure, IntervalMeasure)
            or self.measure.exactness is not Exactness.EXACT
        ):
            raise ValidationError("polynomial integration covectors require exact interval data")
        weights = tuple(self.weights)
        if len(weights) != self.degree + 1:
            raise DimensionError("integration covector length does not match degree")
        if any(not isinstance(value, Fraction) for value in weights):
            raise ValidationError("integration covector entries must be Fractions")
        if not str(self.family).strip():
            raise ValidationError("polynomial family name must be nonempty")
        witness = _basis_witness(self.basis_witness, self.degree)
        if witness is not None:
            _require_exact_interval_structure(
                self.measure,
                context="verified polynomial integration covectors",
            )
            recomputed = tuple(
                integrate(ExactPolynomial.univariate(polynomial), self.measure).require()
                for polynomial in witness
            )
            if weights != recomputed:
                raise ValidationError(
                    "integration covector weights do not match the exact basis witness"
                )
        object.__setattr__(self, "family", str(self.family).strip())
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "basis_witness", witness)

    @property
    def basis_verified(self) -> bool:
        """Whether exact basis polynomials replay and justify the weights."""

        return self.basis_witness is not None

    def apply(self, coefficients: Sequence[Rational]) -> Fraction:
        values = tuple(coefficients)
        if len(values) != self.degree + 1:
            raise DimensionError("coefficient vector length does not match the integration covector")
        exact = tuple(_coordinate(value) for value in values)
        return sum(
            (weight * coefficient for weight, coefficient in zip(self.weights, exact, strict=True)),
            Fraction(0),
        )

    def certificate(self) -> Certificate:
        structure = self.measure.certificate()
        structure.require()
        structurally_exact = (
            structure.exactness is Exactness.EXACT
            and not structure.assumptions_unchecked
        )
        if not self.basis_verified or not structurally_exact:
            unchecked = list(structure.assumptions_unchecked)
            if not self.basis_verified:
                unchecked.append(
                    "declared weights are the exact moments of the named polynomial family"
                )
            elif structure.exactness is not Exactness.EXACT and not unchecked:
                unchecked.append(
                    f"interval structure has {structure.exactness.value} rather than exact status"
                )
            status = _weakest_exactness(
                Exactness.CONDITIONAL if not self.basis_verified else Exactness.EXACT,
                structure.exactness,
            )
            if structure.exactness is Exactness.EXACT and structure.assumptions_unchecked:
                status = Exactness.CONDITIONAL
            return Certificate(
                valid=structure.valid,
                exactness=status,
                assumptions_checked=(
                    "exact rational interval data",
                    "finite rational covector",
                    "coordinate dimension",
                )
                + tuple(structure.assumptions_checked),
                assumptions_unchecked=tuple(dict.fromkeys(unchecked)),
                diagnostics={
                    "family": self.family,
                    "degree": self.degree,
                    "dimension": len(self.weights),
                    "basis_witness_verified": False,
                },
                theorem_id="INT.COORD.COVECTOR",
                message="polynomial integration covector is structurally valid but its basis provenance is unchecked",
            )
        return Certificate(
            valid=True,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "replayed exact graded polynomial basis witness",
                "exact rational interval moments",
                "finite coordinate truncation",
            ),
            diagnostics={
                "family": self.family,
                "degree": self.degree,
                "dimension": len(self.weights),
                "basis_witness_verified": True,
            },
            theorem_id="INT.COORD.COVECTOR",
            message="polynomial integration covector validated from its exact basis witness",
        )


def _require_verified_covector(
    covector: PolynomialIntegrationCovector,
) -> PolynomialIntegrationCovector:
    certificate = covector.certificate()
    if (
        not covector.basis_verified
        or not certificate.valid
        or certificate.exactness is not Exactness.EXACT
        or certificate.assumptions_unchecked
    ):
        raise ValidationError(
            "exact coordinate identities require a factory-verified integration covector"
        )
    return covector


def polynomial_integration_covector(
    family: PolynomialFamily,
    degree: int,
    measure: IntervalMeasure,
) -> PolynomialIntegrationCovector:
    if not isinstance(family, PolynomialFamily):
        raise TypeError("family must be a PolynomialFamily")
    bound = _degree(degree)
    if not isinstance(measure, IntervalMeasure) or measure.exactness is not Exactness.EXACT:
        raise ValidationError("integration covectors require an exact IntervalMeasure")
    _require_exact_interval_structure(
        measure,
        context="exact integration covectors",
    )
    witness = tuple(family.polynomial(index) for index in range(bound + 1))
    weights = tuple(
        integrate(ExactPolynomial.univariate(polynomial), measure).require()
        for polynomial in witness
    )
    return _require_verified_covector(
        PolynomialIntegrationCovector(
            family.name,
            bound,
            measure,
            weights,
            basis_witness=witness,
        )
    )


def coordinate_fundamental_theorem(
    family: PolynomialFamily,
    degree: int,
    measure: IntervalMeasure,
) -> IntegralIdentity:
    r"""Certify ``ell D = rho(ev_b - ev_a)`` in the declared basis."""

    covector = _require_verified_covector(
        polynomial_integration_covector(family, degree, measure)
    )
    derivative = derivative_coordinate_operator(family, degree)
    composed = tuple(
        sum(
            (
                covector.weights[row] * derivative.matrix[row][column]
                for row in range(degree + 1)
            ),
            Fraction(0),
        )
        for column in range(degree + 1)
    )
    boundary = tuple(
        measure.density
        * (
            polynomial_evaluate(family.polynomial(index), measure.upper)
            - polynomial_evaluate(family.polynomial(index), measure.lower)
        )
        for index in range(degree + 1)
    )
    valid = composed == boundary
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={"coordinate_ftc": 0.0 if valid else 1.0},
        tolerance=0.0,
        assumptions_checked=(
            "exact derivative coordinate operator",
            "exact integration covector",
            "exact endpoint evaluation covectors",
        ),
        diagnostics={"family": family.name, "degree": degree},
        theorem_id="INT.COORD.FTC",
        message="coordinate FTC verified exactly" if valid else "coordinate FTC failed",
    )
    return IntegralIdentity("coordinate fundamental theorem", composed, boundary, certificate)


def coordinate_transport_identity(
    source: PolynomialFamily,
    target: PolynomialFamily,
    degree: int,
    measure: IntervalMeasure,
) -> IntegralIdentity:
    r"""Certify contragredient transport ``ell_target C = ell_source``."""

    source_covector = _require_verified_covector(
        polynomial_integration_covector(source, degree, measure)
    )
    target_covector = _require_verified_covector(
        polynomial_integration_covector(target, degree, measure)
    )
    transform = connection_transform(source, target, degree)
    transported = tuple(
        sum(
            (
                target_covector.weights[row] * transform.matrix[row][column]
                for row in range(degree + 1)
            ),
            Fraction(0),
        )
        for column in range(degree + 1)
    )
    valid = transported == source_covector.weights
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={"covector_transport": 0.0 if valid else 1.0},
        tolerance=0.0,
        assumptions_checked=(
            "exact polynomial connection transform",
            "contragredient covector action",
            "common physical interval measure",
        ),
        diagnostics={
            "source_family": source.name,
            "target_family": target.name,
            "degree": degree,
        },
        theorem_id="INT.COORD.TRANSPORT",
        message="integration covector transport verified exactly" if valid else "integration covector transport failed",
    )
    return IntegralIdentity(
        "integration covector coordinate transport",
        transported,
        source_covector.weights,
        certificate,
    )


__all__ = [
    "PolynomialIntegrationCovector",
    "coordinate_fundamental_theorem",
    "coordinate_transport_identity",
    "polynomial_integration_covector",
]
