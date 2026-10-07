from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import komorebi_math
from komorebi_math import beta, stable
from komorebi_math.certificates import Certificate, ComputationResult, Exactness
from komorebi_math.coordinates import (
    AnalysisSynthesisSystem,
    euler_polynomial_family,
    fibonacci_polynomial_family,
    legendre_polynomial_family,
    monomial_polynomial_family,
)
from komorebi_math.exceptions import DimensionError, ValidationError
from komorebi_math.export import portable_codecs
from komorebi_math.generating import FormalSeries, ordinary_basis
from komorebi_math.persistence_contract import (
    check_portable_codec_contract,
    load_portable_codec_contract,
)
from komorebi_math.integration import (
    AtomicIntegratedDerivativeProvider,
    AtomicMeasure,
    CompositeSimpson,
    DerivativeBound,
    ErrorGuarantee,
    ExactAffineMap,
    ExactPolynomial,
    ExactQuadratureRule,
    IntervalMeasure,
    PolynomialIntegrationCovector,
    ProductMeasure,
    PushforwardMeasure,
    QuadratureRule,
    affine_interval_change_of_variables,
    all_fubini_orders,
    analysis_synthesis_integration_certificate,
    coordinate_fundamental_theorem,
    coordinate_transport_identity,
    fubini_identity,
    fundamental_theorem,
    integrate,
    integrate_formal_series,
    integrate_polynomial_coordinates,
    integrate_polynomial_oriented,
    integration_by_parts,
    polynomial_derivative_bound,
    polynomial_from_family_coordinates,
    polynomial_integration_covector,
    pushforward_identity,
)

F = Fraction


def test_exact_polynomial_arithmetic_and_float_rejection() -> None:
    polynomial = ExactPolynomial(2, {((2, 0)): 1, ((0, 1)): 3})
    assert polynomial((2, 5)) == 19
    assert polynomial.derivative(0)((2, 5)) == 4
    assert polynomial.antiderivative(1).derivative(1) == polynomial
    assert polynomial.multiply(polynomial).degree == 4
    with pytest.raises(ValidationError, match="integer or Fraction"):
        ExactPolynomial.univariate((0.1, 1))


def test_exact_affine_pullback_and_composition() -> None:
    polynomial = ExactPolynomial(2, {(2, 0): 1, (0, 1): 3})
    first = ExactAffineMap(((1, 2), (0, 1)), (1, -1))
    second = ExactAffineMap(((2, 0), (1, 1)), (0, 3))
    point = (F(2), F(-1))
    composed = first.then(second)
    assert composed(point) == second(first(point))
    assert polynomial.pullback_affine(first.matrix, first.shift)(point) == polynomial(first(point))


def test_interval_measure_exact_moments_and_polynomial_integral() -> None:
    measure = IntervalMeasure(-1, 2, density=F(3, 2))
    assert tuple(measure.moment((degree,)).require() for degree in range(5)) == (
        F(9, 2),
        F(9, 4),
        F(9, 2),
        F(45, 8),
        F(99, 10),
    )
    assert integrate(ExactPolynomial.univariate((4, -2, 1)), measure).require() == 18


def test_interval_measure_zero_and_validation_edges() -> None:
    zero_interval = IntervalMeasure(2, 2, density=7)
    assert zero_interval.mass().require() == 0
    assert integrate(lambda _: (_ for _ in ()).throw(RuntimeError()), zero_interval).require() == 0
    zero_density = IntervalMeasure(-10, 10, density=0)
    assert integrate(ExactPolynomial.univariate((1, 2, 3)), zero_density).require() == 0
    with pytest.raises(ValidationError, match="lower <= upper"):
        IntervalMeasure(2, -1)
    with pytest.raises(ValidationError, match="density"):
        IntervalMeasure(0, 1, density=-1)
    with pytest.raises(ValidationError, match="zero-mass"):
        zero_interval.normalized()


def test_atomic_measure_validation_and_exact_outputs() -> None:
    measure = AtomicMeasure((-1, 0, 2), (F(1, 2), F(1, 3), F(1, 6)))
    vector = integrate(lambda x: (1, x, x * x), measure)
    assert vector.exact
    assert tuple(vector.value) == (1, F(-1, 6), F(7, 6))
    matrix = integrate(lambda x: ((1, x), (x * x, 2 * x + 1)), measure)
    assert matrix.exact
    assert tuple(tuple(row) for row in matrix.value) == (
        (1, F(-1, 6)),
        (F(7, 6), F(2, 3)),
    )
    with pytest.raises(DimensionError):
        AtomicMeasure((0, 1), (1,))
    with pytest.raises(ValidationError, match="nonnegative"):
        AtomicMeasure((0, 1), (1, -1))
    with pytest.raises(ValidationError, match="finite"):
        AtomicMeasure((0, 1), (1, float("nan")))


def test_atomic_inconsistent_integrand_shape_is_rejected() -> None:
    measure = AtomicMeasure((0, 1), (F(1, 2), F(1, 2)))
    with pytest.raises(DimensionError, match="shape changed"):
        integrate(lambda x: (x,) if x == 0 else (x, x), measure)


def test_atomic_numerical_scalar_uses_accurate_summation() -> None:
    measure = AtomicMeasure((0, 1, 2), (1.0, 1.0, 1.0))
    values = (1e16, 1.0, -1e16)
    result = integrate(lambda x: values[int(x)], measure)
    assert result.require() == 1.0
    assert not result.exact


def test_noninjective_and_constant_atomic_pushforwards() -> None:
    base = AtomicMeasure(
        (-2, -1, 1, 2),
        (F(1, 6), F(1, 3), F(1, 4), F(1, 4)),
    )
    square = PushforwardMeasure(base, lambda x: x * x, target_dimension=1).materialize_atomic()
    assert dict(zip(square.points, square.weights, strict=True)) == {
        (F(1),): F(7, 12),
        (F(4),): F(5, 12),
    }
    assert integrate(ExactPolynomial.univariate((1, 0, 1)), square).require() == F(33, 4)
    constant = PushforwardMeasure(base.normalized(), lambda _: 7, target_dimension=1).materialize_atomic()
    assert constant.points == ((F(7),),)
    assert constant.weights == (F(1),)
    assert integrate(ExactPolynomial.univariate((0, 1)), constant).require() == 7


def test_exact_affine_pushforward_identity_has_no_jacobian_factor() -> None:
    base = IntervalMeasure(0, 1)
    mapping = ExactAffineMap(((2,),), (1,))
    polynomial = ExactPolynomial.univariate((1, -2, 0, 0, 1))
    identity = pushforward_identity(polynomial, base, mapping).require()
    assert identity.left == identity.right == F(106, 5)


def test_affine_interval_density_uses_absolute_jacobian() -> None:
    base = IntervalMeasure(0, 1)
    mapping = ExactAffineMap(((-2,),), (1,))
    pushed = base.pushforward_affine(mapping)
    assert isinstance(pushed, IntervalMeasure)
    assert (pushed.lower, pushed.upper, pushed.density) == (-1, 1, F(1, 2))
    identity = affine_interval_change_of_variables(
        ExactPolynomial.univariate((2,)),
        base,
        mapping,
    ).require()
    assert identity.left == identity.right == 2


def test_oriented_bounds_are_not_positive_measures() -> None:
    polynomial = ExactPolynomial.univariate((2,))
    assert integrate_polynomial_oriented(polynomial, -1, 1).require() == 4
    assert integrate_polynomial_oriented(polynomial, 1, -1).require() == -4
    assert integrate_polynomial_oriented(polynomial, 1, 1).require() == 0


def test_exact_interval_and_atomic_fubini_all_orders() -> None:
    box = ProductMeasure((IntervalMeasure(0, 1), IntervalMeasure(1, 2)))
    polynomial = ExactPolynomial(2, {(2, 1): 1, (1, 2): 3})
    assert {identity.require().left for identity in all_fubini_orders(polynomial, box)} == {F(4)}
    left = AtomicMeasure((0, 2), (F(1, 3), F(2, 3)))
    right = AtomicMeasure((-1, 3), (F(3, 4), F(1, 4)))
    product_measure = ProductMeasure((left, right))
    affine = ExactPolynomial(2, {(1, 0): 2, (0, 1): -3, (1, 1): 1})
    square = ExactPolynomial(2, {(2, 0): 1, (1, 1): -2, (0, 2): 1})
    assert {identity.require().left for identity in all_fubini_orders(affine, product_measure)} == {F(8, 3)}
    assert {identity.require().left for identity in all_fubini_orders(square, product_measure)} == {F(17, 3)}


def test_joint_measure_is_not_product_of_marginals() -> None:
    joint = AtomicMeasure(((0, 0), (1, 1)), (F(1, 2), F(1, 2)))
    marginal = AtomicMeasure((0, 1), (F(1, 2), F(1, 2)))
    product_measure = ProductMeasure((marginal, marginal))
    xy = ExactPolynomial.monomial(1, (1, 1))
    assert integrate(xy, joint).require() == F(1, 2)
    assert integrate(xy, product_measure).require() == F(1, 4)


def test_exact_simpson_degree_boundary_and_false_claim_blocked() -> None:
    rule = ExactQuadratureRule(
        (-1, F(1, 2), 2),
        (F(1, 2), 2, F(1, 2)),
        name="exact Simpson",
    )
    measure = IntervalMeasure(-1, 2)
    assert rule.degree_of_exactness(measure, max_degree=7).degree == 3
    cubic = ExactPolynomial.univariate((-5, 3, -2, 1))
    assert rule.certify_polynomial_integral(cubic, measure).require() == F(-51, 4)
    fourth = ExactPolynomial.univariate((0, 0, 0, 0, 1))
    assert rule.integrate_polynomial(fourth).require() == F(69, 8)
    assert integrate(fourth, measure).require() == F(33, 5)
    with pytest.raises(ValidationError, match="not exactly certified"):
        rule.certify_polynomial_integral(fourth, measure)


def test_signed_quadrature_cannot_become_positive_measure() -> None:
    rule = ExactQuadratureRule(
        (0, F(1, 2), 1, 2),
        (F(-1, 12), F(4, 3), F(-1, 3), F(1, 12)),
    )
    assert not rule.positive
    with pytest.raises(ValidationError, match="negative weights"):
        rule.as_atomic_measure()
    assert rule.certificate().valid


def test_quadrature_node_adversary_has_nonzero_true_integral() -> None:
    adversary = ExactPolynomial.univariate(
        (0, 0, F(1, 4), F(-3, 2), F(13, 4), -3, 1)
    )
    rule = ExactQuadratureRule(
        (0, F(1, 2), 1),
        (F(1, 6), F(2, 3), F(1, 6)),
    )
    quadrature = rule.integrate_polynomial(adversary)
    assert quadrature.require() == 0
    assert quadrature.error.guarantee is ErrorGuarantee.UNKNOWN
    assert integrate(adversary, IntervalMeasure(0, 1)).require() == F(1, 840)


def test_tensor_quadrature_exactness_and_boundary() -> None:
    x_rule = ExactQuadratureRule((0, F(1, 2), 1), (F(1, 6), F(2, 3), F(1, 6)))
    y_rule = ExactQuadratureRule((-1, 1), (1, 1))
    tensor = x_rule.tensor_product(y_rule)
    exact_polynomial = ExactPolynomial(2, {(3, 0): 2, (3, 1): 3, (0, 1): 1})
    boundary = ExactPolynomial(2, {(4, 0): 1, (4, 1): 1})
    box = ProductMeasure((IntervalMeasure(0, 1), IntervalMeasure(-1, 1)))
    assert tensor.integrate_polynomial(exact_polynomial).require() == integrate(exact_polynomial, box).require() == 1
    assert tensor.integrate_polynomial(boundary).require() == F(5, 12)
    assert integrate(boundary, box).require() == F(2, 5)


def test_fundamental_theorem_and_one_sided_inverse() -> None:
    polynomial = ExactPolynomial.univariate((7, -5, F(3, 2), F(-2, 3), F(1, 4)))
    derivative = polynomial.derivative()
    identity = fundamental_theorem(derivative, IntervalMeasure(-1, 2)).require()
    assert identity.left == identity.right == F(-51, 4)
    zero_constant = derivative.antiderivative()
    assert zero_constant.derivative() == derivative
    assert zero_constant == polynomial.subtract(ExactPolynomial.constant(7))


def test_integration_by_parts_keeps_boundary_term() -> None:
    u = ExactPolynomial.univariate((1, 0, 1))
    v = ExactPolynomial.univariate((0, -2, 0, 1))
    identity = integration_by_parts(u, v, IntervalMeasure(-1, 2)).require()
    assert identity.left == F(84, 5)
    assert identity.right == F(84, 5)


def test_coordinate_ftc_and_contragredient_transport() -> None:
    measure = IntervalMeasure(-1, 2)
    for family in (
        monomial_polynomial_family(),
        euler_polynomial_family(),
        fibonacci_polynomial_family(),
        legendre_polynomial_family(),
    ):
        coordinate_fundamental_theorem(family, 7, measure).require()
    coordinate_transport_identity(
        euler_polynomial_family(),
        fibonacci_polynomial_family(),
        7,
        measure,
    ).require()


def test_basis_invariant_integration_outside_reference_interval() -> None:
    monomial = (3, -2, 0, 0, 1)
    legendre = (F(16, 5), -2, F(4, 7), 0, F(8, 35))
    measure = IntervalMeasure(-1, 2)
    assert integrate_polynomial_coordinates(monomial_polynomial_family(), monomial, measure).require() == F(63, 5)
    assert integrate_polynomial_coordinates(legendre_polynomial_family(), legendre, measure).require() == F(63, 5)


def test_formal_series_adapter_preserves_exact_coefficients() -> None:
    series = FormalSeries.from_ordinary_coefficients(ordinary_basis(), (4, -2, 1))
    assert integrate_formal_series(series, IntervalMeasure(-1, 2, density=F(3, 2))).require() == 18


def test_rigorous_and_estimated_simpson_are_distinguished() -> None:
    polynomial = ExactPolynomial.univariate((0, 0, 0, 0, 1))
    measure = IntervalMeasure(0, 1)
    bound = polynomial_derivative_bound(polynomial, measure)
    rigorous = integrate(polynomial, measure, provider=CompositeSimpson(8, bound))
    assert rigorous.error.guarantee is ErrorGuarantee.RIGOROUS
    assert isinstance(rigorous.value, Fraction)
    assert isinstance(rigorous.error.absolute_bound, Fraction)
    assert abs(rigorous.value - F(1, 5)) <= rigorous.error.absolute_bound
    with pytest.raises(ValidationError, match="exact polynomial integrand"):
        integrate(lambda x: x**4, measure, provider=CompositeSimpson(8, bound))
    estimated = integrate(lambda x: np.exp(-x * x), measure, provider=CompositeSimpson(32))
    assert estimated.error.guarantee is ErrorGuarantee.ESTIMATED
    assert estimated.certificate.exactness is Exactness.HEURISTIC


def test_one_dimensional_bound_cannot_certify_tensor_quadrature() -> None:
    polynomial = ExactPolynomial.univariate((0, 0, 0, 0, 1))
    bound = polynomial_derivative_bound(polynomial, IntervalMeasure(0, 1))
    box = ProductMeasure((IntervalMeasure(0, 1), IntervalMeasure(0, 1)))
    with pytest.raises(ValidationError, match="one-dimensional derivative bound"):
        integrate(lambda point: point[0] ** 4 + point[1], box, provider=CompositeSimpson(8, bound))


def test_pushforward_does_not_reuse_untransformed_derivative_bound() -> None:
    polynomial = ExactPolynomial.univariate((0, 0, 0, 0, 1))
    bound = polynomial_derivative_bound(polynomial, IntervalMeasure(0, 1))
    pushed = PushforwardMeasure(IntervalMeasure(0, 1), lambda x: x * x, target_dimension=1)
    with pytest.raises(ValidationError, match="composed integrand"):
        integrate(lambda y: y**2, pushed, provider=CompositeSimpson(8, bound))


def test_atomic_integrated_derivative_provider() -> None:
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

    provider = AtomicIntegratedDerivativeProvider(
        (Provider(1), Provider(3)),
        (F(1, 4), F(3, 4)),
    )
    provider.certificate().require()
    point = np.asarray([2], dtype=object)
    direction = np.asarray([5], dtype=object)
    assert tuple(provider.derivative(1, point, (direction,))) == (50,)


def test_analysis_synthesis_commutes_with_atomic_integration() -> None:
    system = AnalysisSynthesisSystem(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([[0.5, -0.5, 0.5], [-0.25, 0.75, 0.25]]),
    )
    measure = AtomicMeasure((0, 1, 2), (F(1, 4), F(1, 2), F(1, 4)))
    values = (
        np.array([1.0, -2.0]),
        np.array([3.0, 4.0]),
        np.array([-1.0, 5.0]),
    )
    analysis_synthesis_integration_certificate(system, measure, values).require()


def test_expansion_and_tensor_resource_guards() -> None:
    polynomial = ExactPolynomial(2, {(5, 5): 1})
    mapping = ExactAffineMap(((1, 1), (1, -1)), (0, 0))
    with pytest.raises(ValidationError, match="max_terms"):
        mapping.pullback(polynomial, max_terms=2)
    box = ProductMeasure((IntervalMeasure(0, 1),) * 4)
    with pytest.raises(ValidationError, match="max_evaluations"):
        integrate(lambda point: sum(point), box, provider=CompositeSimpson(16, max_evaluations=100))


def test_block19_is_research_only_and_frozen_contracts_remain_unchanged() -> None:
    assert komorebi_math.__version__ == "1.0.0b4"
    assert len(komorebi_math.__all__) == 516
    assert len(stable.__all__) == 131
    assert len(beta.__all__) == 83
    reference = load_portable_codec_contract()
    assert len(reference.entries) == 51
    check_portable_codec_contract().certificate.require()
    block19 = {
        "AtomicMeasure",
        "ExactPolynomial",
        "IntervalMeasure",
        "QuadratureRule",
        "integrate",
    }
    assert block19.isdisjoint(stable.__all__)
    assert block19.isdisjoint(beta.__all__)


def test_rigorous_simpson_preserves_tiny_exact_bounds_and_density() -> None:
    tiny = F(1, 10**325)
    scale = 10**60
    polynomial = ExactPolynomial.univariate((0, 0, 0, 0, tiny))
    measure = IntervalMeasure(0, scale)
    bound = polynomial_derivative_bound(polynomial, measure)
    assert isinstance(bound.bound, Fraction) and bound.bound > 0
    result = integrate(polynomial, measure, provider=CompositeSimpson(4, bound))
    exact = integrate(polynomial, measure).require()
    assert result.error.guarantee is ErrorGuarantee.RIGOROUS
    assert abs(result.value - exact) <= result.error.absolute_bound

    density_measure = IntervalMeasure(0, 1, density=F(1, 10**400))
    constant = ExactPolynomial.constant(1)
    zero_bound = polynomial_derivative_bound(constant, density_measure)
    density_result = integrate(
        constant,
        density_measure,
        provider=CompositeSimpson(4, zero_bound),
    )
    assert density_result.value == F(1, 10**400)
    assert density_result.error.absolute_bound == 0


def test_numerical_mass_underflow_and_overflow_are_rejected() -> None:
    underflow = IntervalMeasure(0.0, 1e-300, density=1e-300)
    with pytest.raises(ValidationError, match="underflowed"):
        underflow.mass()
    with pytest.raises(ValidationError, match="underflowed"):
        integrate(lambda _: 1.0, underflow)
    with pytest.raises(ValidationError, match="overflowed"):
        AtomicMeasure((0.0, 1.0), (1e308, 1e308)).mass()
    with pytest.raises(ValidationError, match="finite|overflowed"):
        IntervalMeasure(0.0, 1e308, density=1e308).mass()
    with pytest.raises(ValidationError, match="overflowed"):
        AtomicMeasure((1e308,), (1.0,)).moment((2,))


def test_complex_quadrature_and_huge_exact_atomic_outputs() -> None:
    numerical = QuadratureRule((0.0, 1.0), (0.5, 0.5))
    assert numerical.apply(lambda x: 1j * x).require() == 0.5j
    exact = AtomicMeasure((0,), (1,))
    assert integrate(lambda _: 10**400, exact).require() == 10**400


def test_numerical_atomic_product_is_not_simpson_or_zero_error() -> None:
    product = ProductMeasure((AtomicMeasure((0.0, 1.0), (0.5, 0.5)),))
    result = integrate(lambda _: 0.1, product)
    assert result.method == "finite atomic product sum"
    assert result.evaluations == 2
    assert result.error.guarantee is ErrorGuarantee.UNKNOWN


def test_unverified_covector_and_invalid_frame_do_not_certify() -> None:
    forged = PolynomialIntegrationCovector(
        "monomials",
        1,
        IntervalMeasure(0, 1),
        (F(0), F(0)),
    )
    assert forged.certificate().exactness is Exactness.CONDITIONAL
    assert forged.certificate().assumptions_unchecked

    invalid = AnalysisSynthesisSystem(
        np.eye(2),
        np.zeros((2, 2)),
    )
    assert not invalid.certificate().valid
    measure = AtomicMeasure((0,), (1,))
    certificate = analysis_synthesis_integration_certificate(
        invalid,
        measure,
        (np.zeros(2),),
    )
    assert not certificate.valid


def test_derivative_provider_dimensions_and_shapes_are_strict() -> None:
    class BadDimension:
        source_dimension = 1.9
        target_dimension = 1
        exactness = Exactness.EXACT

    with pytest.raises(DimensionError, match="positive integer"):
        AtomicIntegratedDerivativeProvider((BadDimension(),), (1,))

    class BadShape:
        source_dimension = 1
        target_dimension = 2
        exactness = Exactness.PLANNED

        def supports_order(self, order: int) -> bool:
            return True

        def value(self, point: np.ndarray) -> int:
            return 1

        def derivative(self, order: int, point: np.ndarray, directions: tuple[np.ndarray, ...]) -> int:
            return 1

    provider = AtomicIntegratedDerivativeProvider((BadShape(),), (1,))
    assert provider.exactness is Exactness.PLANNED
    with pytest.raises(DimensionError, match="output must have shape"):
        provider.value(np.asarray([0]))


def test_simpson_resource_guard_runs_before_node_allocation() -> None:
    with pytest.raises(ValidationError, match="max_evaluations"):
        integrate(
            lambda x: x,
            IntervalMeasure(0, 1),
            provider=CompositeSimpson(10**9, max_evaluations=1),
        )


def test_callable_pushforward_typing_and_assumptions_are_enforced() -> None:
    pushed = PushforwardMeasure(
        IntervalMeasure(0, 1),
        lambda x: (x, x),
        target_dimension=1,
    )
    with pytest.raises(DimensionError, match="dimension"):
        integrate(lambda y: y, pushed)
    structural = pushed.certificate()
    assert structural.exactness is Exactness.CONDITIONAL
    assert any("deterministic" in assumption for assumption in structural.assumptions_unchecked)


def test_custom_measure_assumptions_are_not_laundered() -> None:
    class ConditionalMeasure:
        dimension = 1
        name = "conditional exact moments"
        exactness = Exactness.EXACT

        def mass(self) -> ComputationResult[Fraction]:
            return ComputationResult(F(1), self.certificate())

        def moment(self, exponents: tuple[int, ...]) -> ComputationResult[Fraction]:
            return ComputationResult(
                F(1),
                Certificate(
                    valid=True,
                    exactness=Exactness.EXACT,
                    assumptions_unchecked=("moment theorem supplied externally",),
                ),
            )

        def certificate(self) -> Certificate:
            return Certificate(valid=True, exactness=Exactness.EXACT)

    result = integrate(ExactPolynomial.constant(1), ConditionalMeasure())
    assert result.certificate.exactness is Exactness.CONDITIONAL
    assert result.error.guarantee is ErrorGuarantee.UNKNOWN

    class InconsistentExactMeasure:
        dimension = 1
        name = "float moments mislabeled exact"
        exactness = Exactness.EXACT

        def certificate(self) -> Certificate:
            return Certificate(valid=True, exactness=Exactness.EXACT)

        def mass(self) -> ComputationResult[float]:
            return ComputationResult(1.0, self.certificate())

        def moment(self, exponents: tuple[int, ...]) -> ComputationResult[float]:
            return ComputationResult(0.1, self.certificate())

    inconsistent = InconsistentExactMeasure()
    downgraded = integrate(ExactPolynomial.univariate((0, 1)), inconsistent)
    assert downgraded.certificate.exactness is Exactness.NUMERICAL
    assert downgraded.error.guarantee is ErrorGuarantee.UNKNOWN
    degree = ExactQuadratureRule((0,), (1,)).degree_of_exactness(
        inconsistent,
        max_degree=1,
    )
    assert not degree.exact
    assert degree.certificate.exactness is Exactness.NUMERICAL


def test_exact_quadrature_residuals_do_not_float_overflow() -> None:
    rule = ExactQuadratureRule((10**400,), (1,))
    exactness = rule.degree_of_exactness(IntervalMeasure(0, 1), max_degree=1)
    assert exactness.degree == 0
    assert exactness.exact
    assert exactness.certificate.residuals["moment_1"] == 1.0


def test_numerical_affine_pushforward_cannot_collapse_mass() -> None:
    translated = IntervalMeasure(0.0, 1.0)
    large_shift = ExactAffineMap(((1,),), (10**20,))
    with pytest.raises(ValidationError, match="collapsed|lost numerical interval mass"):
        translated.pushforward_affine(large_shift)

    dilute = IntervalMeasure(0.0, 1.0, density=1e-300)
    huge_scale = ExactAffineMap(((10**100,),), (0,))
    with pytest.raises(ValidationError, match="lost numerical interval mass"):
        dilute.pushforward_affine(huge_scale)


def test_measure_structure_status_cannot_be_laundered() -> None:
    class ConditionalAtomic(AtomicMeasure):
        def certificate(self) -> Certificate:
            return Certificate(
                valid=True,
                exactness=Exactness.CONDITIONAL,
                theorem_id="TEST.CONDITIONAL.ATOM",
                message="conditional atomic semantics",
            )

    conditional = ConditionalAtomic((0,), (1,))
    direct = integrate(lambda _: 1, conditional)
    assert direct.certificate.exactness is Exactness.CONDITIONAL
    assert direct.error.guarantee is ErrorGuarantee.UNKNOWN
    assert direct.certificate.assumptions_unchecked

    pushed = PushforwardMeasure(conditional, ExactAffineMap.identity(1))
    transported = integrate(lambda _: 1, pushed)
    assert transported.certificate.exactness is Exactness.CONDITIONAL
    assert transported.error.guarantee is ErrorGuarantee.UNKNOWN
    assert transported.certificate.assumptions_unchecked

    product = integrate(lambda _: 1, ProductMeasure((conditional,)))
    assert product.certificate.exactness is Exactness.CONDITIONAL
    assert product.error.guarantee is ErrorGuarantee.UNKNOWN

    class PlannedAtomic(AtomicMeasure):
        def certificate(self) -> Certificate:
            return Certificate(
                valid=True,
                exactness=Exactness.PLANNED,
                theorem_id="TEST.PLANNED.ATOM",
                message="planned atomic semantics",
            )

    planned_measure = PlannedAtomic((0,), (1,))
    planned = integrate(ExactPolynomial.constant(1), planned_measure)
    assert planned.certificate.exactness is Exactness.PLANNED
    assert planned.error.guarantee is ErrorGuarantee.UNKNOWN
    assert planned.certificate.assumptions_unchecked
    degree = ExactQuadratureRule((0,), (1,)).degree_of_exactness(
        planned_measure,
        max_degree=0,
    )
    assert degree.certificate.exactness is Exactness.PLANNED


def test_rigorous_bound_requires_exact_measure_structure() -> None:
    class ConditionalInterval(IntervalMeasure):
        def certificate(self) -> Certificate:
            return Certificate(
                valid=True,
                exactness=Exactness.CONDITIONAL,
                theorem_id="TEST.CONDITIONAL.INTERVAL",
                message="conditional interval semantics",
            )

    polynomial = ExactPolynomial.univariate((0, 0, 0, 0, 1))
    conditional = ConditionalInterval(0, 1)
    with pytest.raises(ValidationError, match="rational interval data"):
        polynomial_derivative_bound(polynomial, conditional)

    exact = IntervalMeasure(0, 1)
    bound = polynomial_derivative_bound(polynomial, exact)
    with pytest.raises(ValidationError, match="does not match"):
        integrate(polynomial, conditional, provider=CompositeSimpson(8, bound))


def test_callable_simpson_wraps_unrepresentable_exact_interval_data() -> None:
    with pytest.raises(ValidationError, match="finite float"):
        integrate(lambda _: 1, IntervalMeasure(0, F(10**400)))
    with pytest.raises(ValidationError, match="finite float"):
        integrate(
            lambda _: 1,
            IntervalMeasure(0, 1, density=F(10**400)),
        )
    with pytest.raises(ValidationError, match="error bound overflowed"):
        integrate(
            lambda _: 1.0,
            IntervalMeasure(0.0, 1e100, density=1e-300),
            provider=CompositeSimpson(4, DerivativeBound.assumed(4, 1.0)),
        )


def test_exact_identities_require_exact_measure_certificates() -> None:
    class ConditionalInterval(IntervalMeasure):
        def certificate(self) -> Certificate:
            return Certificate(
                valid=True,
                exactness=Exactness.CONDITIONAL,
                theorem_id="TEST.CONDITIONAL.IDENTITY.INTERVAL",
                message="conditional interval semantics",
            )

    measure = ConditionalInterval(0, 1)
    polynomial = ExactPolynomial.univariate((1, 2, 1))
    mapping = ExactAffineMap(((2,),), (1,))
    actions = (
        lambda: fundamental_theorem(polynomial, measure),
        lambda: integration_by_parts(polynomial, polynomial, measure),
        lambda: pushforward_identity(polynomial, measure, mapping),
        lambda: affine_interval_change_of_variables(polynomial, measure, mapping),
        lambda: fubini_identity(polynomial, ProductMeasure((measure,))),
        lambda: polynomial_integration_covector(
            monomial_polynomial_family(),
            2,
            measure,
        ),
    )
    for action in actions:
        with pytest.raises(ValidationError, match="unconditionally exact"):
            action()


def test_polynomial_pushforward_propagates_its_structure() -> None:
    class ConditionalPushforward(PushforwardMeasure):
        def certificate(self) -> Certificate:
            return Certificate(
                valid=True,
                exactness=Exactness.CONDITIONAL,
                theorem_id="TEST.CONDITIONAL.PUSHFORWARD",
                message="conditional pushforward semantics",
            )

    mappings = (
        (ExactAffineMap.identity(1), {}),
        (lambda x: x, {"target_dimension": 1}),
    )
    for mapping, options in mappings:
        measure = ConditionalPushforward(
            AtomicMeasure((0,), (1,)),
            mapping,
            **options,
        )
        result = integrate(ExactPolynomial.constant(1), measure)
        assert result.certificate.exactness is Exactness.CONDITIONAL
        assert result.error.guarantee is ErrorGuarantee.UNKNOWN
        assert result.certificate.assumptions_unchecked


def test_frame_integration_propagates_atomic_measure_certificate() -> None:
    class InvalidAtomic(AtomicMeasure):
        def certificate(self) -> Certificate:
            return Certificate(
                valid=False,
                exactness=Exactness.CONDITIONAL,
                assumptions_unchecked=("atomic semantics are invalid",),
                theorem_id="TEST.INVALID.FRAME.MEASURE",
                message="invalid atomic semantics",
            )

    class ConditionalAtomic(AtomicMeasure):
        def certificate(self) -> Certificate:
            return Certificate(
                valid=True,
                exactness=Exactness.CONDITIONAL,
                assumptions_unchecked=("atomic semantics are conditional",),
                theorem_id="TEST.CONDITIONAL.FRAME.MEASURE",
                message="conditional atomic semantics",
            )

    system = AnalysisSynthesisSystem(np.eye(1), np.eye(1))
    invalid = analysis_synthesis_integration_certificate(
        system,
        InvalidAtomic((0,), (1,)),
        ([2.0],),
    )
    assert not invalid.valid
    assert invalid.diagnostics["measure_certificate_valid"] is False

    conditional = analysis_synthesis_integration_certificate(
        system,
        ConditionalAtomic((0,), (1,)),
        ([2.0],),
    )
    assert conditional.valid
    assert conditional.exactness is Exactness.CONDITIONAL
    assert "atomic semantics are conditional" in conditional.assumptions_unchecked


def test_public_numerical_integration_apis_wrap_overflow() -> None:
    with pytest.raises(ValidationError, match="polynomial evaluation overflowed"):
        ExactPolynomial.constant(10**400)(1.0)
    with pytest.raises(ValidationError, match="affine point evaluation overflowed"):
        ExactAffineMap(((10**400,),), (0,))(1.0)
    with pytest.raises(ValidationError, match="quadrature moment overflowed"):
        QuadratureRule((1e308,), (1.0,)).moment((2,))
    with pytest.raises(ValidationError, match="quadrature scaling overflowed"):
        QuadratureRule((0,), (1.0,)).apply(lambda _: 10**400)
    with pytest.raises(ValidationError, match="integration scaling overflowed"):
        integrate(lambda _: 10**400, AtomicMeasure((0,), (1.0,)))
    mixed_vector = lambda x: [10**400] if x == 0 else [1.0]
    with pytest.raises(ValidationError, match="integration array summation overflowed"):
        integrate(mixed_vector, AtomicMeasure((0, 1), (1, 1)))
    with pytest.raises(ValidationError, match="quadrature array summation overflowed"):
        ExactQuadratureRule((0, 1), (1, 1)).apply(mixed_vector)
    huge_polynomial = ExactPolynomial.constant(10**400)
    with pytest.raises(ValidationError, match="moment contraction overflowed"):
        integrate(huge_polynomial, IntervalMeasure(0.0, 1.0))
    with pytest.raises(ValidationError, match="quadrature contraction overflowed"):
        QuadratureRule((0.0,), (1.0,)).integrate_polynomial(huge_polynomial)
    with pytest.raises(ValidationError, match="moment comparison overflowed"):
        QuadratureRule((0.0,), (1.0,)).degree_of_exactness(
            AtomicMeasure((10**400,), (1,)),
            max_degree=1,
        )
    with pytest.raises(ValidationError, match="tensor-product weight overflowed"):
        QuadratureRule((0,), (1.0,)).tensor_product(
            ExactQuadratureRule((0,), (10**400,))
        )

    class HugeProvider:
        source_dimension = 1
        target_dimension = 1
        exactness = Exactness.EXACT

        def supports_order(self, order: int) -> bool:
            return True

        def value(self, point: np.ndarray) -> list[int]:
            return [10**400]

        def derivative(
            self,
            order: int,
            point: np.ndarray,
            directions: tuple[np.ndarray, ...],
        ) -> list[int]:
            return [10**400]

    with pytest.raises(ValidationError, match="weighting overflowed"):
        AtomicIntegratedDerivativeProvider((HugeProvider(),), (1.0,)).value(
            np.asarray([0.0])
        )

    class LargeFloatProvider(HugeProvider):
        exactness = Exactness.NUMERICAL

        def value(self, point: np.ndarray) -> list[float]:
            return [1e308]

    with pytest.raises(ValidationError, match="finite numeric data"):
        AtomicIntegratedDerivativeProvider(
            (LargeFloatProvider(),),
            (2.0,),
        ).value(np.asarray([0.0]))

    with pytest.raises(ValidationError, match="tolerance"):
        ExactQuadratureRule((0,), (1,)).degree_of_exactness(
            IntervalMeasure(0, 1),
            max_degree=0,
            tolerance=F(10**400),
        )

    system = AnalysisSynthesisSystem(np.eye(1), np.eye(1))
    with pytest.raises(ValidationError, match="tolerance"):
        analysis_synthesis_integration_certificate(
            system,
            AtomicMeasure((0,), (1,)),
            ([1.0],),
            tolerance=F(10**400),
        )
    with pytest.raises(ValidationError, match="atomic weights"):
        analysis_synthesis_integration_certificate(
            system,
            AtomicMeasure((0,), (10**400,)),
            ([1.0],),
        )
    with pytest.raises(ValidationError, match="integration arithmetic overflowed"):
        analysis_synthesis_integration_certificate(
            system,
            AtomicMeasure((0,), (1,)),
            ([10**400],),
        )
