"""Installed verification for the finite typed expression calculus."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from fractions import Fraction
from typing import Any

from ..certificates import Certificate, Exactness
from ..coordinates import (
    PolynomialFamily,
    euler_polynomial_family,
    fibonacci_polynomial_family,
    monomial_polynomial_family,
)
from ..coordinates.exact import (
    identity_matrix,
    matrix_multiply,
    matrix_vector_multiply,
)
from ..exceptions import DimensionError, ResourceLimitError, ValidationError
from ..expression import (
    RATIONAL_SPACE,
    Expression,
    ExpressionBinding,
    ExpressionBudget,
    LinearMapType,
    PolynomialCoordinateSpace,
    PolynomialPrimitive,
    PrimitiveKind,
    RationalSpace,
    add,
    affine_substitution,
    antiderivative,
    apply,
    boundary_functional,
    check_expression,
    compile_expression,
    compose,
    constant_injection,
    coordinate_change,
    derivative,
    equivalent,
    evaluate_expression,
    evaluation,
    exact_map,
    exact_value,
    execute_plan,
    expression_digest,
    expression_from_dict,
    expression_from_json,
    expression_to_dict,
    expression_to_json,
    finite_difference,
    identity,
    interval_integration,
    let,
    matrix_chain_multiply,
    matrix_chain_parenthesization,
    matrix_chain_plan,
    normalize_expression,
    primitive_certificate,
    replay,
    scale,
    simplify_expression,
    symbol,
    then,
)
from .core import VerificationCheck, VerificationReport, _capture


def _spaces() -> tuple[
    PolynomialCoordinateSpace,
    PolynomialCoordinateSpace,
    PolynomialCoordinateSpace,
    PolynomialCoordinateSpace,
    PolynomialCoordinateSpace,
    PolynomialCoordinateSpace,
]:
    fibonacci = fibonacci_polynomial_family()
    euler = euler_polynomial_family()
    monomial = monomial_polynomial_family()
    return (
        PolynomialCoordinateSpace.from_family(fibonacci, 3),
        PolynomialCoordinateSpace.from_family(fibonacci, 4),
        PolynomialCoordinateSpace.from_family(euler, 3),
        PolynomialCoordinateSpace.from_family(euler, 4),
        PolynomialCoordinateSpace.from_family(monomial, 3),
        PolynomialCoordinateSpace.from_family(monomial, 4),
    )


def _expect_error(
    exceptions: type[BaseException] | tuple[type[BaseException], ...],
    function: Callable[[], Any],
) -> None:
    try:
        function()
    except exceptions:
        return
    raise AssertionError("expected validation failure was not raised")


def _fibonacci_pipeline() -> tuple[Expression, PolynomialCoordinateSpace]:
    fibonacci3, fibonacci4, euler3, euler4, _, _ = _spaces()
    pipeline = then(
        antiderivative(fibonacci3, fibonacci4, 0),
        coordinate_change(fibonacci4, euler4),
        derivative(euler4),
        coordinate_change(euler3, fibonacci3),
    )
    return pipeline, fibonacci3


def run_expression_verification() -> VerificationReport:
    """Verify typed exact semantics, rewriting, planning, and transport."""

    checks: list[VerificationCheck] = []

    def semantic_spaces() -> None:
        fibonacci3, fibonacci4, _, _, _, _ = _spaces()
        assert RationalSpace() == RATIONAL_SPACE
        assert RATIONAL_SPACE.dimension == 1
        assert fibonacci3.dimension == 4
        assert fibonacci4.truncate(3) == fibonacci3
        renamed = PolynomialCoordinateSpace(
            fibonacci3.variable,
            fibonacci3.degree,
            fibonacci3.basis,
            "a display-only replacement",
        )
        assert renamed == fibonacci3
        assert renamed.semantic_key == fibonacci3.semantic_key

    checks.append(_capture("semantic coordinate-space typing", semantic_spaces))

    def strict_rationals() -> None:
        fibonacci3, _, _, _, _, _ = _spaces()
        _expect_error(
            ValidationError,
            lambda: exact_value(fibonacci3, (0, 0, 0, 0.5)),
        )
        _expect_error(
            ValidationError,
            lambda: PolynomialCoordinateSpace("x", True, (((1,),))),
        )
        _expect_error(
            ValidationError,
            lambda: scale(True, exact_value(fibonacci3, (0, 0, 0, 1))),
        )
        decomposed = symbol("e\u0301", fibonacci3)
        assert decomposed.name == "é"

    checks.append(_capture("strict rational and NFC constructors", strict_rationals))

    def primitive_replay() -> None:
        fibonacci3, _, euler3, _, _, _ = _spaces()
        change = coordinate_change(fibonacci3, euler3)
        matrix = replay(change)
        certificate = primitive_certificate(change, claimed_matrix=matrix)
        certificate.require()
        assert certificate.exactness is Exactness.EXACT
        assert certificate.theorem_id == "EXPR.POLY.COORDINATE"
        assert replay(evaluation(fibonacci3, 2)) == (
            (Fraction(1), Fraction(2), Fraction(5), Fraction(12)),
        )
        assert replay(interval_integration(fibonacci3, 0, 1)) == (
            (
                Fraction(1),
                Fraction(1, 2),
                Fraction(4, 3),
                Fraction(5, 4),
            ),
        )
        assert replay(constant_injection(fibonacci3)) == (
            (Fraction(1),),
            (Fraction(0),),
            (Fraction(0),),
            (Fraction(0),),
        )

    checks.append(_capture("replayable polynomial primitives", primitive_replay))

    def coordinate_cocycle() -> None:
        fibonacci3, _, euler3, _, monomial3, _ = _spaces()
        fibonacci_to_euler = replay(coordinate_change(fibonacci3, euler3))
        euler_to_monomial = replay(coordinate_change(euler3, monomial3))
        fibonacci_to_monomial = replay(coordinate_change(fibonacci3, monomial3))
        assert matrix_multiply(euler_to_monomial, fibonacci_to_euler) == fibonacci_to_monomial

    checks.append(_capture("exact coordinate cocycle", coordinate_cocycle))

    def derivative_antiderivative() -> None:
        fibonacci3, fibonacci4, _, _, _, _ = _spaces()
        right_inverse = matrix_multiply(
            replay(derivative(fibonacci4)),
            replay(antiderivative(fibonacci3, fibonacci4, 0)),
        )
        assert right_inverse == identity_matrix(4)
        constant_space = fibonacci3.truncate(0)
        assert replay(derivative(constant_space)) == ((Fraction(0),),)

    checks.append(_capture("tight derivative-antiderivative right inverse", derivative_antiderivative))

    def antiderivative_derivative_defect() -> None:
        fibonacci3, _, _, _, _, _ = _spaces()
        product = matrix_multiply(
            replay(antiderivative(fibonacci3.truncate(2), fibonacci3, 0)),
            replay(derivative(fibonacci3)),
        )
        expected = (
            (Fraction(0), Fraction(0), Fraction(-1), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        )
        assert product == expected
        assert product != identity_matrix(4)
        assert matrix_vector_multiply(product, (1, 0, 0, 0)) == (0, 0, 0, 0)

    checks.append(_capture("basepoint JD defect", antiderivative_derivative_defect))

    def density_weighted_ftc() -> None:
        _, fibonacci4, _, _, _, _ = _spaces()
        density = Fraction(3, 2)
        left = matrix_multiply(
            replay(interval_integration(fibonacci4.truncate(3), -1, 2, density)),
            replay(derivative(fibonacci4)),
        )
        right = replay(boundary_functional(fibonacci4, -1, 2, density))
        expected = (
            (
                Fraction(0),
                Fraction(9, 2),
                Fraction(9, 2),
                Fraction(45, 2),
                Fraction(36),
            ),
        )
        assert left == right == expected

    checks.append(_capture("density-weighted coordinate FTC", density_weighted_ftc))

    def fibonacci_worked_example() -> None:
        fibonacci3, fibonacci4, euler3, euler4, _, _ = _spaces()
        value = (Fraction(0), Fraction(0), Fraction(0), Fraction(1))
        operators = (
            antiderivative(fibonacci3, fibonacci4, 0),
            coordinate_change(fibonacci4, euler4),
            derivative(euler4),
            coordinate_change(euler3, fibonacci3),
        )
        expected = (
            (Fraction(-1, 2), 0, Fraction(1, 4), 0, Fraction(1, 4)),
            (Fraction(5, 8), Fraction(3, 2), Fraction(7, 4), Fraction(1, 2), Fraction(1, 4)),
            (Fraction(3, 2), Fraction(7, 2), Fraction(3, 2), 1),
            (0, 0, 0, 1),
        )
        current = value
        observed: list[tuple[Fraction, ...]] = []
        for operator in operators:
            current = matrix_vector_multiply(replay(operator), current)
            observed.append(current)
        assert tuple(observed) == expected
        composite = normalize_expression(then(*operators))
        assert composite.data == identity_matrix(4)
        integral = matrix_vector_multiply(
            replay(interval_integration(fibonacci3, 0, 1)),
            current,
        )
        assert integral == (Fraction(5, 4),)

    checks.append(_capture("Fibonacci-Euler I4 and five-fourths example", fibonacci_worked_example))

    def affine_worked_example() -> None:
        _, _, _, _, _, monomial4 = _spaces()
        polynomial = exact_value(monomial4, (1, -2, 0, 0, 1))
        expression = apply(
            then(
                affine_substitution(monomial4, 2, 1),
                interval_integration(monomial4, 0, 1),
            ),
            polynomial,
        )
        assert normalize_expression(expression).data == (Fraction(106, 5),)

    checks.append(_capture("affine transported integral 106/5", affine_worked_example))

    def normal_form_equality() -> None:
        fibonacci3, _, _, _, _, _ = _spaces()
        constructed = add(
            scale(2, identity(fibonacci3)),
            scale(-1, identity(fibonacci3)),
        )
        decision = equivalent(constructed, identity(fibonacci3))
        decision.require_equivalent()
        assert decision.certificate.exactness is Exactness.EXACT
        assert decision.left is not None and decision.left.data == identity_matrix(4)

    checks.append(_capture("decidable exact normal-form equality", normal_form_equality))

    def simplifier_preservation() -> None:
        pipeline, _ = _fibonacci_pipeline()
        simplified = simplify_expression(pipeline)
        simplified.certificate.require()
        assert simplified.before.data == simplified.after.data == identity_matrix(4)
        assert len(simplified.steps) >= 3
        assert any(step.theorem_id == "EXPR.REWRITE.DJ" for step in simplified.steps)
        assert all(step.before_digest and step.after_digest for step in simplified.steps)

    checks.append(_capture("certified simplifier preserves denotation", simplifier_preservation))

    def matrix_chain_optimality() -> None:
        plan = matrix_chain_plan((30, 35, 15, 5, 10, 20, 25))
        plan.certificate.require()
        assert plan.certificate.theorem_id == "EXPR.PLAN.MATRIX_CHAIN_DP"
        assert plan.scalar_operation_cost == 15125
        assert plan.left_associative_cost == 40500
        assert plan.right_associative_cost == 47500
        assert matrix_chain_parenthesization(plan) == (
            "((A0 @ (A1 @ A2)) @ ((A3 @ A4) @ A5))"
        )
        tie = matrix_chain_plan((2, 2, 2, 2))
        assert tie.tree.split == 1
        assert matrix_chain_parenthesization(tie) == "(A0 @ (A1 @ A2))"

    checks.append(_capture("exact matrix-chain dynamic program", matrix_chain_optimality))

    def matrix_chain_execution() -> None:
        left = ((1, 2, 3), (4, 5, 6))
        middle = ((1, 0), (0, 1), (1, 1))
        right = ((2,), (3,))
        plan = matrix_chain_plan((2, 3, 2, 1))
        result = matrix_chain_multiply((left, middle, right), plan)
        assert result == ((Fraction(23),), (Fraction(53),))
        _expect_error(
            DimensionError,
            lambda: matrix_chain_multiply((left, middle), plan),
        )
        _expect_error(
            ValidationError,
            lambda: matrix_chain_plan((2, True, 1)),
        )

    checks.append(_capture("exact matrix-chain execution and guards", matrix_chain_execution))

    def dual_path_evaluation() -> None:
        _, _, _, _, _, monomial4 = _spaces()
        expression = apply(
            then(
                affine_substitution(monomial4, 2, 1),
                interval_integration(monomial4, 0, 1),
            ),
            exact_value(monomial4, (1, -2, 0, 0, 1)),
        )
        result = evaluate_expression(expression)
        result.certificate.require()
        assert result.certificate.theorem_id == "EXPR.EVALUATE.DUAL_PATH"
        assert result.scalar == Fraction(106, 5)
        assert result.normal_form.data == result.direct_normal_form.data
        compiled = compile_expression(expression)
        assert execute_plan(compiled).data == (Fraction(106, 5),)

    checks.append(_capture("direct and planned evaluation agree", dual_path_evaluation))

    def portable_roundtrip() -> None:
        pipeline, fibonacci3 = _fibonacci_pipeline()
        expression = apply(
            then(pipeline, interval_integration(fibonacci3, 0, 1)),
            exact_value(fibonacci3, (0, 0, 0, 1)),
        )
        encoded = expression_to_json(expression)
        restored = expression_from_json(encoded)
        assert expression_digest(restored) == expression_digest(expression)
        assert normalize_expression(restored).data == (Fraction(5, 4),)
        assert expression_to_json(restored) == encoded

    checks.append(_capture("portable expression roundtrip", portable_roundtrip))

    def alpha_equivalence() -> None:
        _, _, _, _, monomial3, _ = _spaces()
        left_symbol = symbol("left", monomial3)
        right_symbol = symbol("right", monomial3)
        value = exact_value(monomial3, (1, 2, 3, 4))
        left = let(left_symbol, value, apply(identity(monomial3), left_symbol))
        right = let(right_symbol, value, apply(identity(monomial3), right_symbol))
        assert expression_digest(left) == expression_digest(right)
        assert expression_to_json(left) != expression_to_json(right)
        assert normalize_expression(left).data == normalize_expression(right).data
        same_name_free = symbol("left", monomial3)
        assert let(left_symbol, value, left_symbol) != let(
            left_symbol,
            value,
            same_name_free,
        )

    checks.append(_capture("alpha-equivalent lexical transport", alpha_equivalence))

    def portable_tamper_rejection() -> None:
        _, _, _, _, monomial3, _ = _spaces()
        value = exact_value(monomial3, (1, 2, 3, 4))
        record = deepcopy(expression_to_dict(value))
        record["expression"]["entries"][0]["numerator"] = 9
        _expect_error(ValidationError, lambda: expression_from_dict(record))
        encoded = expression_to_json(value)
        _expect_error(ValidationError, lambda: expression_from_json(encoded + " "))

    checks.append(_capture("portable digest and canonical-text tamper rejection", portable_tamper_rejection))

    def binding_certificate_propagation() -> None:
        _, _, _, _, monomial3, _ = _spaces()
        free = symbol("external", monomial3)
        assumption = "external coordinate value is trusted"
        binding = ExpressionBinding(
            (1, 2, 3, 4),
            monomial3,
            Certificate(
                valid=True,
                exactness=Exactness.CONDITIONAL,
                assumptions_unchecked=(assumption,),
                theorem_id="EXPR.TEST.CONDITIONAL_BINDING",
            ),
        )
        normalized = normalize_expression(
            apply(identity(monomial3), free),
            {free: binding},
        )
        assert normalized.data == (1, 2, 3, 4)
        assert normalized.certificate.exactness is Exactness.CONDITIONAL
        assert assumption in normalized.certificate.assumptions_unchecked

    checks.append(_capture("binding certificate propagation", binding_certificate_propagation))

    def resource_rejection() -> None:
        fibonacci3, _, _, _, _, _ = _spaces()
        expression = apply(identity(fibonacci3), exact_value(fibonacci3, (0, 0, 0, 1)))
        _expect_error(
            ResourceLimitError,
            lambda: ExpressionBudget(max_nodes=1).preflight(expression),
        )
        _expect_error(
            ResourceLimitError,
            lambda: ExpressionBudget(max_degree=2).preflight(expression),
        )
        shared = exact_value(RATIONAL_SPACE, (1,))
        for _ in range(24):
            shared = add(shared, shared)
        assert normalize_expression(shared).data == (Fraction(1 << 24),)
        growing = exact_value(RATIONAL_SPACE, (1,))
        for _ in range(20):
            growing = scale(2, growing)
        _expect_error(
            ResourceLimitError,
            lambda: normalize_expression(
                growing,
                budget=ExpressionBudget(max_coefficient_bits=2),
            ),
        )
        _expect_error(
            ResourceLimitError,
            lambda: compile_expression(
                compose(identity(fibonacci3), identity(fibonacci3)),
                budget=ExpressionBudget(max_serialized_bytes=400),
            ),
        )

    checks.append(_capture("expression resource limits", resource_rejection))

    def cycle_rejection() -> None:
        fibonacci3, _, _, _, _, _ = _spaces()
        cyclic = scale(1, exact_value(fibonacci3, (0, 0, 0, 1)))
        object.__setattr__(cyclic, "value", cyclic)
        _expect_error(
            ValidationError,
            lambda: ExpressionBudget().preflight(cyclic),
        )
        shifted = PolynomialCoordinateSpace(
            "x",
            3,
            (
                (Fraction(1),),
                (Fraction(1), Fraction(1)),
                (Fraction(0), Fraction(0), Fraction(1)),
                (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
            ),
            "shifted",
        )
        retyped = compose(identity(fibonacci3), identity(fibonacci3))
        object.__setattr__(
            retyped,
            "factors",
            (identity(fibonacci3), identity(shifted)),
        )
        _expect_error(ValidationError, lambda: check_expression(retyped))

    checks.append(_capture("iterative expression cycle rejection", cycle_rejection))

    def same_label_spoof() -> None:
        monomial = PolynomialFamily(
            "same label",
            lambda degree: tuple(
                Fraction(int(index == degree)) for index in range(degree + 1)
            ),
        )
        scaled = PolynomialFamily(
            "same label",
            lambda degree: tuple(
                Fraction(2 * int(index == degree)) for index in range(degree + 1)
            ),
        )
        left = PolynomialCoordinateSpace.from_family(monomial, 3)
        right = PolynomialCoordinateSpace.from_family(scaled, 3)
        assert left.label == right.label and left != right
        _expect_error(
            ValidationError,
            lambda: apply(identity(left), exact_value(right, (1, 0, 0, 0))),
        )

    checks.append(_capture("same-label basis spoof rejection", same_label_spoof))

    def singular_and_forged_rejection() -> None:
        _, _, _, _, monomial3, _ = _spaces()
        monomial1 = monomial3.truncate(1)
        singular = exact_map(
            LinearMapType(monomial1, monomial1),
            ((1, 0), (0, 0)),
            label="singular general map",
        )
        decision = equivalent(singular, identity(monomial1))
        assert not decision.equivalent
        _expect_error(
            ValidationError,
            lambda: PolynomialCoordinateSpace(
                "x",
                1,
                ((Fraction(1),), (Fraction(0), Fraction(0))),
            ),
        )
        _expect_error(
            ValidationError,
            lambda: PolynomialPrimitive(
                PrimitiveKind.DERIVATIVE,
                LinearMapType(monomial1, monomial1),
            ),
        )
        derivative_primitive = derivative(monomial1)
        assert not primitive_certificate(
            derivative_primitive,
            claimed_matrix=((0, 0),),
        ).valid
        assert replay(finite_difference(monomial1, 1)) != replay(derivative_primitive)

    checks.append(_capture("singular and forged primitive rejection", singular_and_forged_rejection))

    return VerificationReport(tuple(checks))


__all__ = ["run_expression_verification"]
