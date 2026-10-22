"""Installed verification for finite measure integration."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..certificates import Exactness
from ..coordinates import (
    AnalysisSynthesisSystem,
    euler_polynomial_family,
    fibonacci_polynomial_family,
    legendre_polynomial_family,
    monomial_polynomial_family,
)
from ..exceptions import ValidationError
from ..integration import (
    AtomicIntegratedDerivativeProvider,
    AtomicMeasure,
    CompositeSimpson,
    ErrorGuarantee,
    ExactAffineMap,
    ExactPolynomial,
    ExactQuadratureRule,
    IntervalMeasure,
    ProductMeasure,
    PushforwardMeasure,
    affine_interval_change_of_variables,
    all_fubini_orders,
    analysis_synthesis_integration_certificate,
    coordinate_fundamental_theorem,
    coordinate_transport_identity,
    fundamental_theorem,
    integrate,
    integrate_polynomial_oriented,
    integration_by_parts,
    polynomial_derivative_bound,
)
from .core import VerificationCheck, VerificationReport, _capture


def run_integration_verification() -> VerificationReport:
    """Verify exact measure laws, quadrature boundaries, and adapters."""

    checks: list[VerificationCheck] = []

    def interval_moments() -> None:
        measure = IntervalMeasure(-1, 2, density=Fraction(3, 2))
        expected = (
            Fraction(9, 2),
            Fraction(9, 4),
            Fraction(9, 2),
            Fraction(45, 8),
            Fraction(99, 10),
        )
        assert tuple(measure.moment((degree,)).require() for degree in range(5)) == expected
        polynomial = ExactPolynomial.univariate((4, -2, 1))
        assert integrate(polynomial, measure).require() == 18

    checks.append(_capture("exact interval moments", interval_moments))

    def atomic_outputs() -> None:
        measure = AtomicMeasure(
            (-1, 0, 2),
            (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
        )
        vector = integrate(lambda x: (1, x, x * x), measure)
        assert vector.exact
        assert tuple(vector.value) == (1, Fraction(-1, 6), Fraction(7, 6))
        matrix = integrate(lambda x: ((1, x), (x * x, x + 1)), measure)
        assert matrix.exact
        assert tuple(tuple(row) for row in matrix.value) == (
            (1, Fraction(-1, 6)),
            (Fraction(7, 6), Fraction(5, 6)),
        )

    checks.append(_capture("exact atomic vector and matrix integration", atomic_outputs))

    def noninjective_pushforward() -> None:
        base = AtomicMeasure(
            (-2, -1, 1, 2),
            (Fraction(1, 6), Fraction(1, 3), Fraction(1, 4), Fraction(1, 4)),
        )
        pushed = PushforwardMeasure(base, lambda x: x * x, target_dimension=1)
        materialized = pushed.materialize_atomic()
        assert dict(zip(materialized.points, materialized.weights, strict=True)) == {
            (Fraction(1),): Fraction(7, 12),
            (Fraction(4),): Fraction(5, 12),
        }
        polynomial = ExactPolynomial.univariate((1, 0, 1))
        assert integrate(polynomial, materialized).require() == Fraction(33, 4)

    checks.append(_capture("noninjective atomic pushforward", noninjective_pushforward))

    def fubini_laws() -> None:
        box = ProductMeasure((IntervalMeasure(0, 1), IntervalMeasure(1, 2)))
        polynomial = ExactPolynomial(2, {(2, 1): 1, (1, 2): 3})
        interval_identities = all_fubini_orders(polynomial, box)
        assert all(identity.require().left == 4 for identity in interval_identities)
        left = AtomicMeasure((0, 2), (Fraction(1, 3), Fraction(2, 3)))
        right = AtomicMeasure((-1, 3), (Fraction(3, 4), Fraction(1, 4)))
        atomic = ProductMeasure((left, right))
        affine = ExactPolynomial(2, {(1, 0): 2, (0, 1): -3, (1, 1): 1})
        atomic_identities = all_fubini_orders(affine, atomic)
        assert all(identity.require().left == Fraction(8, 3) for identity in atomic_identities)

    checks.append(_capture("exact interval and atomic Fubini", fubini_laws))

    def quadrature_boundary() -> None:
        rule = ExactQuadratureRule(
            (-1, Fraction(1, 2), 2),
            (Fraction(1, 2), 2, Fraction(1, 2)),
            name="exact Simpson",
        )
        measure = IntervalMeasure(-1, 2)
        assert rule.degree_of_exactness(measure, max_degree=6).degree == 3
        fourth = ExactPolynomial.univariate((0, 0, 0, 0, 1))
        assert rule.integrate_polynomial(fourth).require() == Fraction(69, 8)
        assert integrate(fourth, measure).require() == Fraction(33, 5)
        try:
            rule.certify_polynomial_integral(fourth, measure)
        except ValidationError:
            pass
        else:
            raise AssertionError("degree-four Simpson exactness was falsely certified")
        adversary = ExactPolynomial.univariate(
            (0, 0, Fraction(1, 4), Fraction(-3, 2), Fraction(13, 4), -3, 1)
        )
        unit_rule = ExactQuadratureRule(
            (0, Fraction(1, 2), 1),
            (Fraction(1, 6), Fraction(2, 3), Fraction(1, 6)),
        )
        assert unit_rule.integrate_polynomial(adversary).require() == 0
        assert integrate(adversary, IntervalMeasure(0, 1)).require() == Fraction(1, 840)

    checks.append(_capture("quadrature exactness boundary and node adversary", quadrature_boundary))

    def calculus_identities() -> None:
        polynomial = ExactPolynomial.univariate(
            (7, -5, Fraction(3, 2), Fraction(-2, 3), Fraction(1, 4))
        )
        measure = IntervalMeasure(-1, 2)
        fundamental_theorem(polynomial.derivative(), measure).require()
        u = ExactPolynomial.univariate((1, 0, 1))
        v = ExactPolynomial.univariate((0, -2, 0, 1))
        identity = integration_by_parts(u, v, measure).require()
        assert identity.left == Fraction(84, 5)
        assert identity.right == Fraction(84, 5)

    checks.append(_capture("FTC and integration by parts", calculus_identities))

    def affine_orientation() -> None:
        polynomial = ExactPolynomial.univariate((2,))
        mapping = ExactAffineMap(((-2,),), (1,))
        identity = affine_interval_change_of_variables(polynomial, IntervalMeasure(-1, 1), mapping)
        identity.require()
        assert identity.left == identity.right
        forward = integrate_polynomial_oriented(polynomial, -1, 1).require()
        reverse = integrate_polynomial_oriented(polynomial, 1, -1).require()
        assert reverse == -forward

    checks.append(_capture("affine Jacobian and oriented bounds", affine_orientation))

    def coordinate_covariance() -> None:
        measure = IntervalMeasure(-1, 2)
        for family in (
            monomial_polynomial_family(),
            euler_polynomial_family(),
            fibonacci_polynomial_family(),
            legendre_polynomial_family(),
        ):
            coordinate_fundamental_theorem(family, 6, measure).require()
        coordinate_transport_identity(
            euler_polynomial_family(),
            fibonacci_polynomial_family(),
            6,
            measure,
        ).require()

    checks.append(_capture("coordinate integration covector covariance", coordinate_covariance))

    def quadrature_error_status() -> None:
        polynomial = ExactPolynomial.univariate((0, 0, 0, 0, 1))
        measure = IntervalMeasure(0, 1)
        bound = polynomial_derivative_bound(polynomial, measure)
        result = integrate(
            polynomial,
            measure,
            provider=CompositeSimpson(8, bound),
        )
        assert result.error.guarantee is ErrorGuarantee.RIGOROUS
        assert abs(float(result.value) - 0.2) <= result.error.absolute_bound
        estimated = integrate(lambda x: np.sin(x), measure, provider=CompositeSimpson(32))
        assert estimated.error.guarantee is ErrorGuarantee.ESTIMATED

    checks.append(_capture("rigorous versus estimated quadrature error", quadrature_error_status))

    def derivative_commutation() -> None:
        class Provider:
            source_dimension = 1
            target_dimension = 1
            exactness = Exactness.EXACT

            def __init__(self, scale: int) -> None:
                self.scale = scale

            def supports_order(self, order: int) -> bool:
                return order in (1, 2)

            def value(self, point: np.ndarray) -> np.ndarray:
                return np.asarray([self.scale * point[0] ** 2], dtype=object)

            def derivative(self, order: int, point: np.ndarray, directions: tuple[np.ndarray, ...]) -> np.ndarray:
                if order == 1:
                    return np.asarray([2 * self.scale * point[0] * directions[0][0]], dtype=object)
                return np.asarray([2 * self.scale * directions[0][0] * directions[1][0]], dtype=object)

        combined = AtomicIntegratedDerivativeProvider(
            (Provider(1), Provider(3)),
            (Fraction(1, 4), Fraction(3, 4)),
        )
        combined.certificate().require()
        point = np.asarray([2], dtype=object)
        direction = np.asarray([5], dtype=object)
        assert tuple(combined.derivative(1, point, (direction,))) == (50,)

    checks.append(_capture("differentiation through fixed finite measure", derivative_commutation))

    def frame_commutation() -> None:
        system = AnalysisSynthesisSystem(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            np.array([[0.5, -0.5, 0.5], [-0.25, 0.75, 0.25]]),
        )
        measure = AtomicMeasure((0, 1, 2), (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)))
        values = (
            np.array([1.0, -2.0]),
            np.array([3.0, 4.0]),
            np.array([-1.0, 5.0]),
        )
        analysis_synthesis_integration_certificate(system, measure, values).require()

    checks.append(_capture("analysis-synthesis integration commutation", frame_commutation))

    def zero_measure() -> None:
        measure = IntervalMeasure(2, 2, density=7)
        result = integrate(lambda _: (_ for _ in ()).throw(RuntimeError("must not evaluate")), measure)
        assert result.exact and result.require() == 0 and result.evaluations == 0

    checks.append(_capture("zero measure avoids integrand evaluation", zero_measure))

    return VerificationReport(tuple(checks))


__all__ = ["run_integration_verification"]
