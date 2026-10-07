"""Adversarial production tests for the Block 20 finite expression calculus.

The independent mathematical oracle lives in ``verification/block20_math_verifier.py``.
This module instead drives the public production expression API end to end.
All witnesses are declared locally so the worked regression does not inherit a
polynomial-family implementation's answer.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from functools import cache

import pytest

from komorebi_math.certificates import Certificate, Exactness
from komorebi_math.exceptions import DimensionError, ResourceLimitError, ValidationError
from komorebi_math.expression import (
    RATIONAL_SPACE,
    Apply,
    Compose,
    ExactMapLiteral,
    ExactValueLiteral,
    ExpressionBinding,
    ExpressionBudget,
    IdentityMap,
    Let,
    LinearMapType,
    PolynomialCoordinateSpace,
    PolynomialPrimitive,
    PrimitiveKind,
    Scale,
    Sum,
    Symbol,
    ZeroMap,
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
    exact_binding,
    exact_internal_certificate,
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
    replay,
    scale,
    simplify_expression,
    symbol,
    type_from_dict,
    type_to_dict,
    zero_map,
)

Q = Fraction

FIBONACCI_WITNESS = (
    (Q(1),),
    (Q(0), Q(1)),
    (Q(1), Q(0), Q(1)),
    (Q(0), Q(2), Q(0), Q(1)),
    (Q(1), Q(0), Q(3), Q(0), Q(1)),
)

EULER_WITNESS = (
    (Q(1),),
    (Q(-1, 2), Q(1)),
    (Q(0), Q(-1), Q(1)),
    (Q(1, 4), Q(0), Q(-3, 2), Q(1)),
    (Q(0), Q(1), Q(0), Q(-2), Q(1)),
)

MONOMIAL_WITNESS = tuple(
    tuple(Q(1) if column == degree else Q(0) for column in range(degree + 1))
    for degree in range(5)
)


def _space(
    witness: tuple[tuple[Fraction, ...], ...],
    degree: int,
    label: str,
    *,
    variable: str = "x",
) -> PolynomialCoordinateSpace:
    return PolynomialCoordinateSpace(variable, degree, witness[: degree + 1], label)


F3 = _space(FIBONACCI_WITNESS, 3, "Fibonacci")
F4 = _space(FIBONACCI_WITNESS, 4, "Fibonacci")
E3 = _space(EULER_WITNESS, 3, "Euler")
E4 = _space(EULER_WITNESS, 4, "Euler")
M0 = _space(MONOMIAL_WITNESS, 0, "monomial")
M1 = _space(MONOMIAL_WITNESS, 1, "monomial")
M3 = _space(MONOMIAL_WITNESS, 3, "monomial")
M4 = _space(MONOMIAL_WITNESS, 4, "monomial")


def _worked_nodes() -> dict[str, object]:
    value = exact_value(F3, (0, 0, 0, 1))
    antiderivative_f = antiderivative(F3, F4, 0)
    fibonacci_to_euler = coordinate_change(F4, E4)
    derivative_e = derivative(E4)
    euler_to_fibonacci = coordinate_change(E3, F3)
    pipeline = compose(
        euler_to_fibonacci,
        derivative_e,
        fibonacci_to_euler,
        antiderivative_f,
    )
    return {
        "value": value,
        "antiderivative": antiderivative_f,
        "fibonacci_to_euler": fibonacci_to_euler,
        "derivative": derivative_e,
        "euler_to_fibonacci": euler_to_fibonacci,
        "pipeline": pipeline,
    }


def _identity_matrix(size: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Q(1) if row == column else Q(0) for column in range(size))
        for row in range(size)
    )


def _direct_matrix_multiply(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            sum((left[row][inner] * right[inner][column] for inner in range(len(right))), Q(0))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _exhaustive_chain_costs(dimensions: tuple[int, ...]) -> tuple[int, ...]:
    @cache
    def costs(start: int, end: int) -> tuple[int, ...]:
        if end == start + 1:
            return (0,)
        values: list[int] = []
        for split in range(start + 1, end):
            for left in costs(start, split):
                for right in costs(split, end):
                    values.append(
                        left
                        + right
                        + dimensions[start] * dimensions[split] * dimensions[end]
                    )
        return tuple(values)

    return costs(0, len(dimensions) - 1)


def _scalar_envelope() -> dict[str, object]:
    return expression_to_dict(exact_value(RATIONAL_SPACE, (Q(1, 2),)))


def test_fibonacci_worked_expression_has_every_documented_intermediate() -> None:
    nodes = _worked_nodes()
    value = nodes["value"]
    antiderivative_f = nodes["antiderivative"]
    fibonacci_to_euler = nodes["fibonacci_to_euler"]
    derivative_e = nodes["derivative"]
    euler_to_fibonacci = nodes["euler_to_fibonacci"]
    assert isinstance(value, ExactValueLiteral)
    assert isinstance(antiderivative_f, PolynomialPrimitive)
    assert isinstance(fibonacci_to_euler, PolynomialPrimitive)
    assert isinstance(derivative_e, PolynomialPrimitive)
    assert isinstance(euler_to_fibonacci, PolynomialPrimitive)

    stage_1 = apply(antiderivative_f, value)
    stage_2 = apply(fibonacci_to_euler, stage_1)
    stage_3 = apply(derivative_e, stage_2)
    stage_4 = apply(euler_to_fibonacci, stage_3)

    assert normalize_expression(value).data == (Q(0), Q(0), Q(0), Q(1))
    assert normalize_expression(stage_1).data == (Q(-1, 2), Q(0), Q(1, 4), Q(0), Q(1, 4))
    assert normalize_expression(stage_2).data == (Q(5, 8), Q(3, 2), Q(7, 4), Q(1, 2), Q(1, 4))
    assert normalize_expression(stage_3).data == (Q(3, 2), Q(7, 2), Q(3, 2), Q(1))
    assert normalize_expression(stage_4).data == (Q(0), Q(0), Q(0), Q(1))

    integrated = apply(interval_integration(F3, 0, 1), stage_4)
    assert evaluate_expression(integrated).scalar == Q(5, 4)


def test_fibonacci_pipeline_normal_form_is_i4() -> None:
    pipeline = _worked_nodes()["pipeline"]
    assert isinstance(pipeline, Compose)
    normal = normalize_expression(pipeline)
    assert normal.data == _identity_matrix(4)
    assert normal.certificate.valid
    assert normal.certificate.exactness is Exactness.EXACT


def test_simplifier_uses_naturality_cocycle_and_dj_to_reach_identity() -> None:
    pipeline = _worked_nodes()["pipeline"]
    assert isinstance(pipeline, Compose)
    simplified = simplify_expression(pipeline)
    assert isinstance(simplified.expression, IdentityMap)
    rules = {step.rule for step in simplified.steps}
    assert {
        "derivative coordinate naturality",
        "coordinate cocycle",
        "derivative-antiderivative cancellation",
    } <= rules
    assert simplified.before.data == simplified.after.data == _identity_matrix(4)
    assert all(step.certificate.valid for step in simplified.steps)


def test_affine_pullback_and_weighted_pushforward_are_both_106_over_5() -> None:
    polynomial = exact_value(M4, (1, -2, 0, 0, 1))
    pulled_back = apply(affine_substitution(M4, 2, 1), polynomial)
    assert normalize_expression(pulled_back).data == (Q(0), Q(4), Q(24), Q(32), Q(16))

    pullback_integral = apply(interval_integration(M4, 0, 1), pulled_back)
    pushforward_integral = apply(interval_integration(M4, 1, 3, Q(1, 2)), polynomial)
    assert evaluate_expression(pullback_integral).scalar == Q(106, 5)
    assert evaluate_expression(pushforward_integral).scalar == Q(106, 5)


def test_density_omission_is_a_distinct_212_over_5_computation() -> None:
    polynomial = exact_value(M4, (1, -2, 0, 0, 1))
    omitted = apply(interval_integration(M4, 1, 3), polynomial)
    assert evaluate_expression(omitted).scalar == Q(212, 5)
    assert evaluate_expression(omitted).scalar != Q(106, 5)


def test_ftc_rewrite_preserves_density_and_endpoints() -> None:
    source = M4
    density = Q(3, 5)
    expression = compose(interval_integration(M3, -2, 3, density), derivative(source))
    expected = boundary_functional(source, -2, 3, density)
    simplified = simplify_expression(expression)
    assert simplified.expression == expected
    assert simplified.before.data == simplified.after.data == replay(expected)
    assert any(step.theorem_id == "EXPR.REWRITE.FTC" for step in simplified.steps)


def test_jd_rewrites_to_basepoint_defect_and_not_identity() -> None:
    jd = compose(antiderivative(M3, M4, 0), derivative(M4))
    simplified = simplify_expression(jd)
    assert isinstance(simplified.expression, Sum)
    assert any(step.theorem_id == "EXPR.REWRITE.JD_DEFECT" for step in simplified.steps)

    expected = add(
        identity(M4),
        scale(-1, compose(constant_injection(M4), evaluation(M4, 0))),
    )
    assert normalize_expression(simplified.expression).data == normalize_expression(expected).data
    constant = exact_value(M4, (1, 0, 0, 0, 0))
    assert normalize_expression(apply(jd, constant)).data == (Q(0), Q(0), Q(0), Q(0), Q(0))
    assert normalize_expression(apply(identity(M4), constant)).data == (Q(1), Q(0), Q(0), Q(0), Q(0))


def test_same_label_and_dimension_do_not_make_basis_types_equal() -> None:
    basis = PolynomialCoordinateSpace(
        "x",
        2,
        ((Q(1),), (Q(0), Q(1)), (Q(0), Q(0), Q(1))),
        "custom basis",
    )
    changed = PolynomialCoordinateSpace(
        "x",
        2,
        ((Q(1),), (Q(0), Q(1)), (Q(0), Q(1), Q(1))),
        "custom basis",
    )
    assert basis.label == changed.label
    assert basis.dimension == changed.dimension
    assert basis != changed
    assert basis.semantic_key != changed.semantic_key

    with pytest.raises(ValidationError, match="argument"):
        apply(identity(basis), exact_value(changed, (0, 0, 1)))
    with pytest.raises(ValidationError, match="endpoints"):
        compose(identity(basis), identity(changed))


def test_exact_normal_form_equality_has_true_false_and_type_mismatch_cases() -> None:
    pipeline = _worked_nodes()["pipeline"]
    assert isinstance(pipeline, Compose)
    same = equivalent(pipeline, identity(F3))
    assert same.equivalent
    assert same.left is not None and same.right is not None
    same.require_equivalent()

    different = equivalent(exact_value(F3, (0, 0, 0, 1)), exact_value(F3, (0, 0, 1, 0)))
    assert not different.equivalent
    with pytest.raises(ValidationError, match="different exact normal forms"):
        different.require_equivalent()

    other_type = equivalent(exact_value(M0, (1,)), exact_value(RATIONAL_SPACE, (1,)))
    assert not other_type.equivalent
    assert other_type.left is None and other_type.right is None


def test_matrix_chain_classic_cost_and_parenthesization() -> None:
    plan = matrix_chain_plan((30, 35, 15, 5, 10, 20, 25))
    assert plan.scalar_operation_cost == 15125
    assert plan.left_associative_cost == 40500
    assert plan.right_associative_cost == 47500
    assert matrix_chain_parenthesization(plan) == "((A0 @ (A1 @ A2)) @ ((A3 @ A4) @ A5))"
    assert plan.certificate.valid
    assert plan.certificate.exactness is Exactness.EXACT


def test_matrix_chain_dp_matches_exhaustive_binary_parenthesizations() -> None:
    chains = (
        (5, 4, 6),
        (5, 10, 3, 12),
        (2, 7, 3, 6, 4),
        (3, 3, 3, 3, 3, 3),
    )
    for dimensions in chains:
        plan = matrix_chain_plan(dimensions)
        exhaustive = _exhaustive_chain_costs(dimensions)
        assert plan.scalar_operation_cost == min(exhaustive)


def test_matrix_chain_execution_matches_direct_exact_multiplication() -> None:
    matrices = (
        ((Q(1), Q(2), Q(0)), (Q(-1), Q(3), Q(4))),
        ((Q(2), Q(1)), (Q(0), Q(-1)), (Q(3), Q(2))),
        ((Q(5),), (Q(-2),)),
    )
    plan = matrix_chain_plan((2, 3, 2, 1))
    expected = _direct_matrix_multiply(_direct_matrix_multiply(matrices[0], matrices[1]), matrices[2])
    assert matrix_chain_multiply(matrices, plan) == expected


def test_direct_normalization_and_planned_execution_agree() -> None:
    nodes = _worked_nodes()
    expression = apply(
        interval_integration(F3, 0, 1),
        apply(nodes["pipeline"], nodes["value"]),
    )
    result = evaluate_expression(expression)
    assert result.scalar == Q(5, 4)
    assert result.normal_form.data == result.direct_normal_form.data
    assert execute_plan(result.plan).data == result.direct_normal_form.data
    assert result.certificate.valid


def _fresh_plan():
    nodes = _worked_nodes()
    return compile_expression(apply(nodes["pipeline"], nodes["value"]))


def test_plan_rejects_tampered_plan_and_expression_digests() -> None:
    for attribute in ("plan_digest", "expression_digest"):
        plan = _fresh_plan()
        replacement = "0" * 64 if getattr(plan, attribute) != "0" * 64 else "1" * 64
        object.__setattr__(plan, attribute, replacement)
        with pytest.raises(ValidationError, match="digest"):
            execute_plan(plan)


def test_plan_rejects_tampered_opcode_and_dependencies() -> None:
    plan = _fresh_plan()
    object.__setattr__(plan.instructions[-1], "opcode", "untrusted_eval")
    with pytest.raises(ValidationError):
        execute_plan(plan)

    plan = _fresh_plan()
    object.__setattr__(plan.instructions[-1], "inputs", (len(plan.instructions),))
    with pytest.raises(ValidationError, match="earlier outputs"):
        execute_plan(plan)


def test_plan_rejects_tampered_instruction_and_aggregate_costs() -> None:
    plan = _fresh_plan()
    instruction = plan.instructions[-1]
    object.__setattr__(instruction, "scalar_operation_cost", instruction.scalar_operation_cost + 1)
    with pytest.raises(ValidationError, match="cost"):
        execute_plan(plan)

    plan = _fresh_plan()
    object.__setattr__(plan, "scalar_operation_cost", plan.scalar_operation_cost + 1)
    with pytest.raises(ValidationError, match="cost"):
        execute_plan(plan)


def test_every_expression_type_round_trips_through_portable_data() -> None:
    expression_types = (
        RATIONAL_SPACE,
        F4,
        LinearMapType(F3, E3),
        LinearMapType(RATIONAL_SPACE, M4),
    )
    for expression_type in expression_types:
        encoded = type_to_dict(expression_type)
        assert type_from_dict(encoded) == expression_type
        assert encoded == type_to_dict(type_from_dict(encoded))


def test_every_model_node_and_polynomial_primitive_round_trips() -> None:
    binder = symbol("bound", RATIONAL_SPACE)
    value = exact_value(RATIONAL_SPACE, (Q(2, 3),))
    map_type = LinearMapType(RATIONAL_SPACE, RATIONAL_SPACE)
    model_nodes = (
        value,
        exact_map(map_type, ((Q(5, 7),),), label="literal label"),
        symbol("free", RATIONAL_SPACE),
        let(binder, value, binder),
        identity(M1),
        zero_map(map_type),
        compose(identity(M1), identity(M1)),
        add(value, scale(2, value)),
        scale(Q(-3, 5), value),
        apply(exact_map(map_type, ((Q(2),),)), value),
    )
    primitives = (
        coordinate_change(F3, E3),
        derivative(F4),
        antiderivative(F3, F4, Q(1, 3)),
        evaluation(F3, Q(2, 3)),
        interval_integration(F3, -1, 2, Q(3, 4)),
        affine_substitution(F3, Q(2), Q(-1)),
        finite_difference(F3, Q(1, 2)),
        boundary_functional(F3, -1, 2, Q(3, 4)),
        constant_injection(F3),
    )
    expected_model_classes = {
        ExactValueLiteral,
        ExactMapLiteral,
        Symbol,
        Let,
        IdentityMap,
        ZeroMap,
        Compose,
        Sum,
        Scale,
        Apply,
    }
    assert {type(node) for node in model_nodes} == expected_model_classes
    assert {primitive.kind for primitive in primitives} == set(PrimitiveKind)

    for expression in (*model_nodes, *primitives):
        envelope = expression_to_dict(expression)
        decoded_dict = expression_from_dict(envelope)
        text = expression_to_json(expression)
        decoded_json = expression_from_json(text)
        assert expression_to_json(decoded_dict) == text
        assert expression_to_json(decoded_json) == text
        assert expression_digest(decoded_dict) == expression_digest(expression)
        assert expression_to_json(decoded_json) == text


def test_canonical_serialization_is_deterministic_and_labels_are_nonsemantic() -> None:
    map_type = LinearMapType(M1, M1)
    left = exact_map(map_type, _identity_matrix(2), label="first display label")
    right = exact_map(map_type, _identity_matrix(2), label="second display label")
    assert expression_digest(left) == expression_digest(right)
    assert expression_to_json(left) != expression_to_json(right)

    text = expression_to_json(_worked_nodes()["pipeline"])
    assert text == expression_to_json(_worked_nodes()["pipeline"])
    assert text == json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with pytest.raises(ValidationError, match="canonical form"):
        expression_from_json(json.dumps(json.loads(text), indent=2, sort_keys=True))


def test_alpha_equivalence_shadowing_and_free_vs_bound_identity() -> None:
    x = symbol("x", RATIONAL_SPACE)
    renamed = symbol("renamed", RATIONAL_SPACE)
    value = exact_value(RATIONAL_SPACE, (7,))
    alpha_left = let(x, value, x)
    alpha_right = let(renamed, value, renamed)
    assert expression_digest(alpha_left) == expression_digest(alpha_right)

    outer = symbol("same", RATIONAL_SPACE)
    inner = symbol("same", RATIONAL_SPACE)
    shadow_inner = let(
        outer,
        exact_value(RATIONAL_SPACE, (1,)),
        let(inner, exact_value(RATIONAL_SPACE, (2,)), inner),
    )
    shadow_outer = let(
        outer,
        exact_value(RATIONAL_SPACE, (1,)),
        let(inner, exact_value(RATIONAL_SPACE, (2,)), outer),
    )
    assert normalize_expression(expression_from_json(expression_to_json(shadow_inner))).data == (Q(2),)
    assert normalize_expression(expression_from_json(expression_to_json(shadow_outer))).data == (Q(1),)
    assert expression_digest(shadow_inner) != expression_digest(shadow_outer)

    binder = symbol("collision", RATIONAL_SPACE)
    same_name_free = symbol("collision", RATIONAL_SPACE)
    bound_body = let(binder, exact_value(RATIONAL_SPACE, (3,)), binder)
    free_body = let(binder, exact_value(RATIONAL_SPACE, (3,)), same_name_free)
    assert bound_body != free_body
    assert expression_digest(bound_body) != expression_digest(free_body)
    assert normalize_expression(free_body, {"collision": exact_binding((9,), RATIONAL_SPACE)}).data == (Q(9),)


def test_portable_rejects_duplicate_keys_floats_bools_and_noncanonical_fractions() -> None:
    with pytest.raises(ValidationError, match="duplicate key"):
        expression_from_json('{"schema":"a","schema":"b"}')
    with pytest.raises(ValidationError, match="floating-point"):
        expression_from_json('{"schema_version":1.0}')

    boolean_payload = _scalar_envelope()
    boolean_payload["schema_version"] = True
    with pytest.raises(ValidationError, match="nonexact primitive"):
        expression_from_dict(boolean_payload)

    fraction_payload = _scalar_envelope()
    expression = fraction_payload["expression"]
    assert isinstance(expression, dict)
    expression["entries"][0] = {"numerator": 2, "denominator": 4}
    with pytest.raises(ValidationError, match="canonical reduced fraction"):
        expression_from_dict(fraction_payload)

    class RetypedScale(Scale):
        @property
        def expression_type(self):
            return M0

    spoof = RetypedScale(Q(1), exact_value(RATIONAL_SPACE, (1,)))
    with pytest.raises(TypeError, match="exact built-in"):
        check_expression(spoof)
    with pytest.raises(TypeError, match="exact built-in"):
        expression_digest(spoof)


def test_portable_rejects_unknown_fields_versions_schema_and_digest() -> None:
    unknown_field = _scalar_envelope()
    unknown_field["unexpected"] = 1
    with pytest.raises(ValidationError, match="fields do not match"):
        expression_from_dict(unknown_field)

    node_version = _scalar_envelope()
    node = node_version["expression"]
    assert isinstance(node, dict)
    node["version"] = 2
    with pytest.raises(ValidationError, match="unknown expression node version"):
        expression_from_dict(node_version)

    schema = _scalar_envelope()
    schema["schema"] = "komorebi_math.expression.future"
    with pytest.raises(ValidationError, match="unknown expression schema"):
        expression_from_dict(schema)

    schema_version = _scalar_envelope()
    schema_version["schema_version"] = 2
    with pytest.raises(ValidationError, match="unknown expression schema version"):
        expression_from_dict(schema_version)

    malformed_digest = _scalar_envelope()
    malformed_digest["semantic_sha256"] = "ABC"
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        expression_from_dict(malformed_digest)

    wrong_digest = _scalar_envelope()
    wrong_digest["semantic_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="mismatch"):
        expression_from_dict(wrong_digest)


def test_portable_rejects_dangling_binders_and_unknown_type_fields() -> None:
    dangling = _scalar_envelope()
    dangling["expression"] = {"index": 0, "kind": "bound_symbol", "version": 1}
    with pytest.raises(ValidationError, match="dangling"):
        expression_from_dict(dangling)

    expression_type = type_to_dict(F3)
    expression_type["unknown"] = 1
    with pytest.raises(ValidationError, match="fields do not match"):
        type_from_dict(expression_type)


def test_binding_evidence_can_weaken_but_never_launder_exactness() -> None:
    free = symbol("input", RATIONAL_SPACE)
    numerical = Certificate(True, Exactness.NUMERICAL, theorem_id="EXTERNAL.NUMERICAL")
    numerical_form = normalize_expression(
        free,
        {"input": ExpressionBinding((Q(2),), RATIONAL_SPACE, numerical)},
    )
    assert numerical_form.certificate.valid
    assert numerical_form.certificate.exactness is Exactness.NUMERICAL

    conditional_source = Certificate(
        True,
        Exactness.EXACT,
        assumptions_unchecked=("external premise",),
    )
    conditional_form = normalize_expression(
        free,
        {"input": ExpressionBinding((Q(2),), RATIONAL_SPACE, conditional_source)},
    )
    assert conditional_form.certificate.exactness is Exactness.CONDITIONAL
    assert "external premise" in conditional_form.certificate.assumptions_unchecked

    invalid = Certificate(False, Exactness.EXACT, message="invalid external evidence")
    invalid_form = normalize_expression(
        free,
        {"input": ExpressionBinding((Q(2),), RATIONAL_SPACE, invalid)},
    )
    assert not invalid_form.certificate.valid
    with pytest.raises(ValidationError, match="failed"):
        invalid_form.certificate.require()


def test_binding_shape_and_declared_type_are_rechecked() -> None:
    free = symbol("input", RATIONAL_SPACE)
    certificate = exact_internal_certificate("test binding")
    malformed = ExpressionBinding((Q(1), Q(2)), RATIONAL_SPACE, certificate)
    with pytest.raises(DimensionError, match="1 entries"):
        normalize_expression(free, {"input": malformed})

    other = symbol("input", M0)
    with pytest.raises(ValidationError, match="binding type"):
        check_expression(other, {"input": exact_binding((1,), RATIONAL_SPACE)})


def test_resource_guards_cover_nodes_depth_entries_bits_degree_and_size() -> None:
    first = exact_value(RATIONAL_SPACE, (1,))
    second = exact_value(RATIONAL_SPACE, (2,))
    with pytest.raises(ResourceLimitError, match="node count"):
        check_expression(add(first, second), budget=ExpressionBudget(max_nodes=2))

    nested = scale(2, scale(3, first))
    with pytest.raises(ResourceLimitError, match="depth"):
        check_expression(nested, budget=ExpressionBudget(max_depth=2))

    with pytest.raises(ResourceLimitError, match="entry count"):
        check_expression(exact_value(M0, (1,)), budget=ExpressionBudget(max_entries=1))

    with pytest.raises(ResourceLimitError, match="coefficient size"):
        check_expression(
            exact_value(RATIONAL_SPACE, (1 << 20,)),
            budget=ExpressionBudget(max_coefficient_bits=8),
        )

    with pytest.raises(ResourceLimitError, match="degree"):
        check_expression(identity(M4), budget=ExpressionBudget(max_degree=3))

    with pytest.raises(ResourceLimitError, match="structural size"):
        expression_to_json(
            symbol("serialized-boundary", RATIONAL_SPACE),
            budget=ExpressionBudget(max_serialized_bytes=16),
        )

    shared = exact_value(RATIONAL_SPACE, (1,))
    for _ in range(24):
        shared = add(shared, shared)
    assert normalize_expression(shared).data == (Q(1 << 24),)

    growing = exact_value(RATIONAL_SPACE, (1,))
    for _ in range(20):
        growing = scale(2, growing)
    with pytest.raises(ResourceLimitError, match="max_coefficient_bits"):
        normalize_expression(
            growing,
            budget=ExpressionBudget(max_coefficient_bits=2),
        )

    tight_serialization = ExpressionBudget(max_serialized_bytes=400)
    with pytest.raises(ResourceLimitError, match="max_serialized_bytes"):
        compile_expression(
            compose(identity(M4), identity(M4)),
            budget=tight_serialization,
        )

    # JSON transports expand shared DAG nodes.  The encoder must enforce the
    # decoder's occurrence budget instead of emitting a non-round-trippable
    # record.
    repeated_leaf = exact_value(RATIONAL_SPACE, (1,))
    repeated = Sum((repeated_leaf,) * 11)
    with pytest.raises(ResourceLimitError, match="node count"):
        expression_to_dict(
            repeated,
            budget=ExpressionBudget(max_nodes=10, max_steps=100),
        )

    long_label = exact_map(
        LinearMapType(RATIONAL_SPACE, RATIONAL_SPACE),
        ((1,),),
        label="x" * 9,
    )
    with pytest.raises(ResourceLimitError, match="label"):
        expression_to_dict(
            long_label,
            budget=ExpressionBudget(max_symbol_length=8),
        )


def test_expression_budget_rejects_an_object_level_cycle() -> None:
    cyclic = Compose((identity(M0), identity(M0)))
    object.__setattr__(cyclic, "factors", (cyclic, identity(M0)))
    with pytest.raises(ValidationError, match="cycle"):
        ExpressionBudget().preflight(cyclic)

    shifted = PolynomialCoordinateSpace(
        "x",
        1,
        ((Q(1),), (Q(1), Q(1))),
        "shifted",
    )
    retyped = Compose((identity(M1), identity(M1)))
    object.__setattr__(retyped, "factors", (identity(M1), identity(shifted)))
    with pytest.raises(ValidationError, match="endpoints"):
        check_expression(retyped)
    with pytest.raises(ValidationError, match="endpoints"):
        normalize_expression(retyped)


def test_invalid_and_singular_basis_adversaries_are_rejected() -> None:
    with pytest.raises(DimensionError, match="degree plus one"):
        PolynomialCoordinateSpace("x", 2, ((Q(1),), (Q(0), Q(1))), "short")
    with pytest.raises(DimensionError, match="graded degree"):
        PolynomialCoordinateSpace("x", 1, ((Q(1),), (Q(1),)), "wrong grade")
    with pytest.raises(ValidationError, match="nonzero leading"):
        PolynomialCoordinateSpace("x", 1, ((Q(1),), (Q(1), Q(0))), "singular")
    with pytest.raises(ValidationError, match="same physical variable"):
        coordinate_change(M1, _space(MONOMIAL_WITNESS, 1, "monomial", variable="y"))

    # Constructor bypass simulates hostile in-memory mutation after validation.
    singular = object.__new__(PolynomialCoordinateSpace)
    object.__setattr__(singular, "variable", "x")
    object.__setattr__(singular, "degree", 1)
    object.__setattr__(singular, "basis", ((Q(1),), (Q(1), Q(0))))
    object.__setattr__(singular, "label", "tampered singular witness")
    with pytest.raises(ValidationError, match="singular"):
        replay(coordinate_change(M1, singular))


def test_branded_looking_exact_map_labels_never_trigger_primitive_rewrites() -> None:
    map_type = LinearMapType(M1, M1)
    outer = exact_map(map_type, _identity_matrix(2), label="derivative coordinate naturality")
    inner = exact_map(map_type, _identity_matrix(2), label="antiderivative EXPR.REWRITE.DJ")
    expression = compose(outer, inner)
    simplified = simplify_expression(expression)
    assert isinstance(simplified.expression, Compose)
    assert simplified.steps == ()
    assert simplified.after.data == _identity_matrix(2)


def test_degree_zero_derivative_is_a_tight_zero_endomorphism() -> None:
    derivative_zero = derivative(M0)
    assert derivative_zero.map_type == LinearMapType(M0, M0)
    assert replay(derivative_zero) == ((Q(0),),)
    assert normalize_expression(apply(derivative_zero, exact_value(M0, (7,)))).data == (Q(0),)

    dj = compose(derivative(M1), antiderivative(M0, M1, 0))
    assert isinstance(simplify_expression(dj).expression, IdentityMap)


def test_empty_ill_shaped_and_ill_typed_inputs_fail_closed() -> None:
    with pytest.raises(ValidationError, match="at least one map"):
        compose()
    with pytest.raises(ValidationError, match="at least one term"):
        add()
    with pytest.raises(ValidationError, match="at least two dimensions"):
        matrix_chain_plan((2,))
    with pytest.raises(ResourceLimitError, match="planning-step guard"):
        matrix_chain_plan((2,) * 300)
    with pytest.raises(DimensionError, match="shape"):
        exact_map(LinearMapType(M1, M1), ((1, 0),), label="ill-shaped")
    with pytest.raises(ValidationError, match="argument"):
        apply(evaluation(M1, 0), exact_value(M0, (1,)))
    with pytest.raises(DimensionError, match="matrix count"):
        matrix_chain_multiply((), matrix_chain_plan((1, 1)))
    with pytest.raises(ValidationError, match="integer or Fraction"):
        exact_value(RATIONAL_SPACE, (0.5,))
    with pytest.raises(ValidationError, match="integer or Fraction"):
        scale(True, exact_value(RATIONAL_SPACE, (1,)))


def test_transport_envelope_is_copy_safe_for_adversarial_mutation() -> None:
    original = _scalar_envelope()
    hostile = copy.deepcopy(original)
    hostile["semantic_sha256"] = "f" * 64
    assert original != hostile
    assert expression_from_dict(original) == exact_value(RATIONAL_SPACE, (Q(1, 2),))
    with pytest.raises(ValidationError, match="mismatch"):
        expression_from_dict(hostile)
