"""Fixture-free adversarial runner for the production Block 20 expression API.

The runner depends only on the Python standard library and ``komorebi_math``.
It deliberately keeps its case table local instead of importing pytest tests,
and it records every independent failure before choosing its exit status.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from komorebi_math.certificates import Certificate, Exactness  # noqa: E402
from komorebi_math.exceptions import (  # noqa: E402
    DimensionError,
    ResourceLimitError,
    ValidationError,
)
from komorebi_math.expression import (  # noqa: E402
    RATIONAL_SPACE,
    Compose,
    ExpressionBinding,
    ExpressionBudget,
    IdentityMap,
    LinearMapType,
    PolynomialCoordinateSpace,
    PrimitiveKind,
    Scale,
    Sum,
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
Case = tuple[str, Callable[[], None]]
CASES: list[Case] = []

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


def case(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def register(function: Callable[[], None]) -> Callable[[], None]:
        CASES.append((name, function))
        return function

    return register


def _expect(
    expected: type[BaseException] | tuple[type[BaseException], ...],
    operation: Callable[[], Any],
    *,
    contains: str | None = None,
) -> BaseException:
    try:
        operation()
    except expected as exc:
        if contains is not None and contains not in str(exc):
            raise AssertionError(f"{exc!s} does not contain {contains!r}") from exc
        return exc
    except Exception as exc:
        expected_names = (
            ", ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise AssertionError(
            f"expected {expected_names}, received {type(exc).__name__}: {exc}"
        ) from exc
    expected_names = (
        ", ".join(item.__name__ for item in expected)
        if isinstance(expected, tuple)
        else expected.__name__
    )
    raise AssertionError(f"operation did not raise {expected_names}")


def _identity(size: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Q(1) if row == column else Q(0) for column in range(size))
        for row in range(size)
    )


def _worked() -> tuple[Any, Any, Any, Any, Any, Any]:
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
    return (
        value,
        antiderivative_f,
        fibonacci_to_euler,
        derivative_e,
        euler_to_fibonacci,
        pipeline,
    )


def _fresh_plan():
    value, _, _, _, _, pipeline = _worked()
    return compile_expression(apply(pipeline, value))


def _scalar_envelope() -> dict[str, Any]:
    return expression_to_dict(exact_value(RATIONAL_SPACE, (Q(1, 2),)))


def _direct_multiply(
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


def _exhaustive_costs(dimensions: tuple[int, ...]) -> tuple[int, ...]:
    @cache
    def costs(start: int, end: int) -> tuple[int, ...]:
        if end == start + 1:
            return (0,)
        output: list[int] = []
        for split in range(start + 1, end):
            for left in costs(start, split):
                for right in costs(split, end):
                    output.append(
                        left
                        + right
                        + dimensions[start] * dimensions[split] * dimensions[end]
                    )
        return tuple(output)

    return costs(0, len(dimensions) - 1)


@case("fibonacci_documented_intermediates")
def _case_fibonacci_intermediates() -> None:
    value, antiderivative_f, fibonacci_to_euler, derivative_e, euler_to_fibonacci, _ = _worked()
    stage_1 = apply(antiderivative_f, value)
    stage_2 = apply(fibonacci_to_euler, stage_1)
    stage_3 = apply(derivative_e, stage_2)
    stage_4 = apply(euler_to_fibonacci, stage_3)
    assert normalize_expression(value).data == (Q(0), Q(0), Q(0), Q(1))
    assert normalize_expression(stage_1).data == (Q(-1, 2), Q(0), Q(1, 4), Q(0), Q(1, 4))
    assert normalize_expression(stage_2).data == (Q(5, 8), Q(3, 2), Q(7, 4), Q(1, 2), Q(1, 4))
    assert normalize_expression(stage_3).data == (Q(3, 2), Q(7, 2), Q(3, 2), Q(1))
    assert normalize_expression(stage_4).data == (Q(0), Q(0), Q(0), Q(1))
    assert evaluate_expression(apply(interval_integration(F3, 0, 1), stage_4)).scalar == Q(5, 4)


@case("pipeline_i4_and_certified_rewrites")
def _case_pipeline_identity() -> None:
    *_, pipeline = _worked()
    assert normalize_expression(pipeline).data == _identity(4)
    simplified = simplify_expression(pipeline)
    assert isinstance(simplified.expression, IdentityMap)
    rules = {step.rule for step in simplified.steps}
    assert "derivative coordinate naturality" in rules
    assert "coordinate cocycle" in rules
    assert "derivative-antiderivative cancellation" in rules
    assert all(step.certificate.valid for step in simplified.steps)


@case("affine_pullback_and_weighted_pushforward_106_over_5")
def _case_affine_integration() -> None:
    polynomial = exact_value(M4, (1, -2, 0, 0, 1))
    pulled = apply(affine_substitution(M4, 2, 1), polynomial)
    assert normalize_expression(pulled).data == (Q(0), Q(4), Q(24), Q(32), Q(16))
    assert evaluate_expression(apply(interval_integration(M4, 0, 1), pulled)).scalar == Q(106, 5)
    weighted = apply(interval_integration(M4, 1, 3, Q(1, 2)), polynomial)
    assert evaluate_expression(weighted).scalar == Q(106, 5)


@case("density_omission_212_over_5")
def _case_density_omission() -> None:
    polynomial = exact_value(M4, (1, -2, 0, 0, 1))
    omitted = evaluate_expression(apply(interval_integration(M4, 1, 3), polynomial)).scalar
    assert omitted == Q(212, 5)
    assert omitted != Q(106, 5)


@case("density_weighted_ftc_rewrite")
def _case_ftc() -> None:
    density = Q(3, 5)
    expression = compose(interval_integration(M3, -2, 3, density), derivative(M4))
    expected = boundary_functional(M4, -2, 3, density)
    simplified = simplify_expression(expression)
    assert simplified.expression == expected
    assert simplified.before.data == simplified.after.data == replay(expected)
    assert any(step.theorem_id == "EXPR.REWRITE.FTC" for step in simplified.steps)


@case("jd_defect_and_constant_counterexample")
def _case_jd_defect() -> None:
    jd = compose(antiderivative(M3, M4, 0), derivative(M4))
    simplified = simplify_expression(jd)
    assert isinstance(simplified.expression, Sum)
    assert any(step.theorem_id == "EXPR.REWRITE.JD_DEFECT" for step in simplified.steps)
    expected = add(identity(M4), scale(-1, compose(constant_injection(M4), evaluation(M4, 0))))
    assert normalize_expression(simplified.expression).data == normalize_expression(expected).data
    constant = exact_value(M4, (1, 0, 0, 0, 0))
    assert normalize_expression(apply(jd, constant)).data == (Q(0),) * 5
    assert normalize_expression(apply(identity(M4), constant)).data == (Q(1), Q(0), Q(0), Q(0), Q(0))


@case("same_label_basis_type_mismatch")
def _case_basis_identity() -> None:
    basis = PolynomialCoordinateSpace(
        "x", 2, ((Q(1),), (Q(0), Q(1)), (Q(0), Q(0), Q(1))), "custom basis"
    )
    changed = PolynomialCoordinateSpace(
        "x", 2, ((Q(1),), (Q(0), Q(1)), (Q(0), Q(1), Q(1))), "custom basis"
    )
    assert basis.label == changed.label and basis.dimension == changed.dimension
    assert basis.semantic_key != changed.semantic_key
    _expect(
        ValidationError,
        lambda: apply(identity(basis), exact_value(changed, (0, 0, 1))),
    )
    _expect(ValidationError, lambda: compose(identity(basis), identity(changed)))


@case("exact_normal_form_equality_true_false_and_typed")
def _case_equality() -> None:
    *_, pipeline = _worked()
    assert equivalent(pipeline, identity(F3)).equivalent
    assert not equivalent(
        exact_value(F3, (0, 0, 0, 1)), exact_value(F3, (0, 0, 1, 0))
    ).equivalent
    typed = equivalent(exact_value(M0, (1,)), exact_value(RATIONAL_SPACE, (1,)))
    assert not typed.equivalent and typed.left is None and typed.right is None


@case("matrix_chain_classic_15125")
def _case_matrix_chain_classic() -> None:
    plan = matrix_chain_plan((30, 35, 15, 5, 10, 20, 25))
    assert plan.scalar_operation_cost == 15125
    assert matrix_chain_parenthesization(plan) == "((A0 @ (A1 @ A2)) @ ((A3 @ A4) @ A5))"
    assert plan.certificate.valid and plan.certificate.exactness is Exactness.EXACT


@case("matrix_chain_exhaustive_small_chains")
def _case_matrix_chain_exhaustive() -> None:
    for dimensions in ((5, 4, 6), (5, 10, 3, 12), (2, 7, 3, 6, 4), (3, 3, 3, 3, 3)):
        assert matrix_chain_plan(dimensions).scalar_operation_cost == min(_exhaustive_costs(dimensions))


@case("matrix_chain_exact_execution")
def _case_matrix_chain_execution() -> None:
    matrices = (
        ((Q(1), Q(2), Q(0)), (Q(-1), Q(3), Q(4))),
        ((Q(2), Q(1)), (Q(0), Q(-1)), (Q(3), Q(2))),
        ((Q(5),), (Q(-2),)),
    )
    expected = _direct_multiply(_direct_multiply(matrices[0], matrices[1]), matrices[2])
    assert matrix_chain_multiply(matrices, matrix_chain_plan((2, 3, 2, 1))) == expected


@case("direct_and_planned_expression_execution")
def _case_dual_execution() -> None:
    value, *_, pipeline = _worked()
    expression = apply(interval_integration(F3, 0, 1), apply(pipeline, value))
    result = evaluate_expression(expression)
    assert result.scalar == Q(5, 4)
    assert result.direct_normal_form.data == execute_plan(result.plan).data


@case("tampered_plan_digests")
def _case_plan_digests() -> None:
    for attribute in ("plan_digest", "expression_digest"):
        plan = _fresh_plan()
        replacement = "0" * 64 if getattr(plan, attribute) != "0" * 64 else "1" * 64
        object.__setattr__(plan, attribute, replacement)
        _expect(ValidationError, lambda plan=plan: execute_plan(plan), contains="digest")


@case("tampered_plan_opcode_and_dependencies")
def _case_plan_structure() -> None:
    plan = _fresh_plan()
    object.__setattr__(plan.instructions[-1], "opcode", "untrusted_eval")
    _expect(ValidationError, lambda: execute_plan(plan))
    plan = _fresh_plan()
    object.__setattr__(plan.instructions[-1], "inputs", (len(plan.instructions),))
    _expect(ValidationError, lambda: execute_plan(plan), contains="earlier outputs")


@case("tampered_plan_costs")
def _case_plan_costs() -> None:
    plan = _fresh_plan()
    instruction = plan.instructions[-1]
    object.__setattr__(instruction, "scalar_operation_cost", instruction.scalar_operation_cost + 1)
    _expect(ValidationError, lambda: execute_plan(plan), contains="cost")
    plan = _fresh_plan()
    object.__setattr__(plan, "scalar_operation_cost", plan.scalar_operation_cost + 1)
    _expect(ValidationError, lambda: execute_plan(plan), contains="cost")


@case("portable_type_roundtrips")
def _case_type_roundtrips() -> None:
    for expression_type in (
        RATIONAL_SPACE,
        F4,
        LinearMapType(F3, E3),
        LinearMapType(RATIONAL_SPACE, M4),
    ):
        assert type_from_dict(type_to_dict(expression_type)) == expression_type


@case("portable_all_nodes_and_primitives")
def _case_expression_roundtrips() -> None:
    binder = symbol("bound", RATIONAL_SPACE)
    value = exact_value(RATIONAL_SPACE, (Q(2, 3),))
    map_type = LinearMapType(RATIONAL_SPACE, RATIONAL_SPACE)
    expressions = (
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
        coordinate_change(F3, E3),
        derivative(F4),
        antiderivative(F3, F4, Q(1, 3)),
        evaluation(F3, Q(2, 3)),
        interval_integration(F3, -1, 2, Q(3, 4)),
        affine_substitution(F3, 2, -1),
        finite_difference(F3, Q(1, 2)),
        boundary_functional(F3, -1, 2, Q(3, 4)),
        constant_injection(F3),
    )
    primitive_kinds = {
        expression.kind
        for expression in expressions
        if hasattr(expression, "kind") and isinstance(expression.kind, PrimitiveKind)
    }
    assert primitive_kinds == set(PrimitiveKind)
    for expression in expressions:
        decoded_dict = expression_from_dict(expression_to_dict(expression))
        text = expression_to_json(expression)
        decoded_json = expression_from_json(text)
        assert expression_to_json(decoded_dict) == text
        assert expression_to_json(decoded_json) == text
        assert expression_digest(decoded_json) == expression_digest(expression)
        assert expression_to_json(decoded_json) == text


@case("canonical_serialization_and_alpha_equivalence")
def _case_canonical_alpha() -> None:
    x = symbol("x", RATIONAL_SPACE)
    y = symbol("renamed", RATIONAL_SPACE)
    value = exact_value(RATIONAL_SPACE, (7,))
    assert expression_digest(let(x, value, x)) == expression_digest(let(y, value, y))
    text = expression_to_json(_worked()[-1])
    assert text == expression_to_json(_worked()[-1])
    assert text == json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    _expect(
        ValidationError,
        lambda: expression_from_json(json.dumps(json.loads(text), indent=2, sort_keys=True)),
        contains="canonical form",
    )


@case("lexical_shadowing_and_free_bound_identity")
def _case_scope_identity() -> None:
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
    free = symbol("collision", RATIONAL_SPACE)
    bound_body = let(binder, exact_value(RATIONAL_SPACE, (3,)), binder)
    free_body = let(binder, exact_value(RATIONAL_SPACE, (3,)), free)
    assert bound_body != free_body
    assert expression_digest(bound_body) != expression_digest(free_body)
    assert normalize_expression(free_body, {"collision": exact_binding((9,), RATIONAL_SPACE)}).data == (Q(9),)


@case("portable_duplicate_float_bool_fraction_rejections")
def _case_portable_primitives() -> None:
    _expect(ValidationError, lambda: expression_from_json('{"schema":"a","schema":"b"}'))
    _expect(ValidationError, lambda: expression_from_json('{"schema_version":1.0}'))
    boolean_payload = _scalar_envelope()
    boolean_payload["schema_version"] = True
    _expect(ValidationError, lambda: expression_from_dict(boolean_payload))
    fraction_payload = _scalar_envelope()
    expression = fraction_payload["expression"]
    assert isinstance(expression, dict)
    expression["entries"][0] = {"numerator": 2, "denominator": 4}
    _expect(ValidationError, lambda: expression_from_dict(fraction_payload))

    class RetypedScale(Scale):
        @property
        def expression_type(self):
            return M0

    spoof = RetypedScale(Q(1), exact_value(RATIONAL_SPACE, (1,)))
    _expect(TypeError, lambda: check_expression(spoof), contains="exact built-in")
    _expect(TypeError, lambda: expression_digest(spoof), contains="exact built-in")


@case("portable_schema_version_field_digest_rejections")
def _case_portable_envelope() -> None:
    for key, value in (
        ("schema", "komorebi_math.expression.future"),
        ("schema_version", 2),
        ("semantic_sha256", "ABC"),
        ("semantic_sha256", "0" * 64),
    ):
        payload = _scalar_envelope()
        payload[key] = value
        _expect(ValidationError, lambda payload=payload: expression_from_dict(payload))
    payload = _scalar_envelope()
    payload["unknown"] = 1
    _expect(ValidationError, lambda: expression_from_dict(payload))
    payload = _scalar_envelope()
    node = payload["expression"]
    assert isinstance(node, dict)
    node["version"] = 2
    _expect(ValidationError, lambda: expression_from_dict(payload))


@case("portable_dangling_binder_and_type_field_rejections")
def _case_portable_binders() -> None:
    payload = _scalar_envelope()
    payload["expression"] = {"index": 0, "kind": "bound_symbol", "version": 1}
    _expect(ValidationError, lambda: expression_from_dict(payload), contains="dangling")
    expression_type = type_to_dict(F3)
    expression_type["unknown"] = 1
    _expect(ValidationError, lambda: type_from_dict(expression_type))


@case("binding_exactness_and_invalidity_propagation")
def _case_binding_evidence() -> None:
    free = symbol("input", RATIONAL_SPACE)
    numerical = ExpressionBinding(
        (Q(2),), RATIONAL_SPACE, Certificate(True, Exactness.NUMERICAL)
    )
    numerical_form = normalize_expression(free, {"input": numerical})
    assert numerical_form.certificate.exactness is Exactness.NUMERICAL
    conditional = ExpressionBinding(
        (Q(2),),
        RATIONAL_SPACE,
        Certificate(True, Exactness.EXACT, assumptions_unchecked=("external premise",)),
    )
    conditional_form = normalize_expression(free, {"input": conditional})
    assert conditional_form.certificate.exactness is Exactness.CONDITIONAL
    invalid = ExpressionBinding(
        (Q(2),),
        RATIONAL_SPACE,
        Certificate(False, Exactness.EXACT, message="invalid evidence"),
    )
    invalid_form = normalize_expression(free, {"input": invalid})
    assert not invalid_form.certificate.valid
    _expect(ValidationError, invalid_form.certificate.require)
    malformed = ExpressionBinding(
        (Q(1), Q(2)), RATIONAL_SPACE, exact_internal_certificate("runner binding")
    )
    _expect(DimensionError, lambda: normalize_expression(free, {"input": malformed}))


@case("resource_node_depth_entry_bit_degree_size_guards")
def _case_resource_limits() -> None:
    first = exact_value(RATIONAL_SPACE, (1,))
    second = exact_value(RATIONAL_SPACE, (2,))
    _expect(
        ResourceLimitError,
        lambda: check_expression(add(first, second), budget=ExpressionBudget(max_nodes=2)),
    )
    _expect(
        ResourceLimitError,
        lambda: check_expression(scale(2, scale(3, first)), budget=ExpressionBudget(max_depth=2)),
    )
    _expect(
        ResourceLimitError,
        lambda: check_expression(exact_value(M0, (1,)), budget=ExpressionBudget(max_entries=1)),
    )
    _expect(
        ResourceLimitError,
        lambda: check_expression(
            exact_value(RATIONAL_SPACE, (1 << 20,)),
            budget=ExpressionBudget(max_coefficient_bits=8),
        ),
    )
    shared = exact_value(RATIONAL_SPACE, (1,))
    for _ in range(24):
        shared = add(shared, shared)
    assert normalize_expression(shared).data == (Q(1 << 24),)
    growing = exact_value(RATIONAL_SPACE, (1,))
    for _ in range(20):
        growing = scale(2, growing)
    _expect(
        ResourceLimitError,
        lambda: normalize_expression(
            growing,
            budget=ExpressionBudget(max_coefficient_bits=2),
        ),
        contains="max_coefficient_bits",
    )
    _expect(
        ResourceLimitError,
        lambda: compile_expression(
            compose(identity(M4), identity(M4)),
            budget=ExpressionBudget(max_serialized_bytes=400),
        ),
        contains="max_serialized_bytes",
    )

    repeated_leaf = exact_value(RATIONAL_SPACE, (1,))
    repeated = Sum((repeated_leaf,) * 11)
    _expect(
        ResourceLimitError,
        lambda: expression_to_dict(
            repeated,
            budget=ExpressionBudget(max_nodes=10, max_steps=100),
        ),
        contains="node count",
    )
    long_label = exact_map(
        LinearMapType(RATIONAL_SPACE, RATIONAL_SPACE),
        ((1,),),
        label="x" * 9,
    )
    _expect(
        ResourceLimitError,
        lambda: expression_to_dict(
            long_label,
            budget=ExpressionBudget(max_symbol_length=8),
        ),
        contains="label",
    )
    _expect(
        ResourceLimitError,
        lambda: check_expression(identity(M4), budget=ExpressionBudget(max_degree=3)),
    )
    _expect(
        ResourceLimitError,
        lambda: expression_to_json(
            symbol("serialized-boundary", RATIONAL_SPACE),
            budget=ExpressionBudget(max_serialized_bytes=16),
        ),
    )


@case("object_cycle_guard")
def _case_cycle() -> None:
    cyclic = Compose((identity(M0), identity(M0)))
    object.__setattr__(cyclic, "factors", (cyclic, identity(M0)))
    _expect(ValidationError, lambda: ExpressionBudget().preflight(cyclic), contains="cycle")
    shifted = PolynomialCoordinateSpace(
        "x",
        1,
        ((Q(1),), (Q(1), Q(1))),
        "shifted",
    )
    retyped = Compose((identity(M1), identity(M1)))
    object.__setattr__(retyped, "factors", (identity(M1), identity(shifted)))
    _expect(ValidationError, lambda: check_expression(retyped), contains="endpoints")
    _expect(ValidationError, lambda: normalize_expression(retyped), contains="endpoints")


@case("invalid_and_singular_basis_guards")
def _case_basis_guards() -> None:
    _expect(
        DimensionError,
        lambda: PolynomialCoordinateSpace("x", 2, ((Q(1),), (Q(0), Q(1))), "short"),
    )
    _expect(
        ValidationError,
        lambda: PolynomialCoordinateSpace("x", 1, ((Q(1),), (Q(1), Q(0))), "singular"),
    )
    _expect(
        ValidationError,
        lambda: coordinate_change(M1, _space(MONOMIAL_WITNESS, 1, "monomial", variable="y")),
    )
    singular = object.__new__(PolynomialCoordinateSpace)
    object.__setattr__(singular, "variable", "x")
    object.__setattr__(singular, "degree", 1)
    object.__setattr__(singular, "basis", ((Q(1),), (Q(1), Q(0))))
    object.__setattr__(singular, "label", "tampered")
    _expect(
        ValidationError,
        lambda: replay(coordinate_change(M1, singular)),
        contains="singular",
    )


@case("literal_labels_cannot_trigger_primitive_rewrites")
def _case_literal_labels() -> None:
    map_type = LinearMapType(M1, M1)
    outer = exact_map(map_type, _identity(2), label="derivative coordinate naturality")
    inner = exact_map(map_type, _identity(2), label="antiderivative EXPR.REWRITE.DJ")
    simplified = simplify_expression(compose(outer, inner))
    assert isinstance(simplified.expression, Compose)
    assert simplified.steps == ()


@case("degree_zero_derivative")
def _case_degree_zero() -> None:
    derivative_zero = derivative(M0)
    assert derivative_zero.map_type == LinearMapType(M0, M0)
    assert replay(derivative_zero) == ((Q(0),),)
    assert normalize_expression(apply(derivative_zero, exact_value(M0, (7,)))).data == (Q(0),)
    assert isinstance(
        simplify_expression(compose(derivative(M1), antiderivative(M0, M1))).expression,
        IdentityMap,
    )


@case("empty_ill_shaped_and_type_mismatch_rejections")
def _case_bad_shapes() -> None:
    _expect(ValidationError, compose)
    _expect(ValidationError, add)
    _expect(ValidationError, lambda: matrix_chain_plan((2,)))
    _expect(ResourceLimitError, lambda: matrix_chain_plan((2,) * 300))
    _expect(
        DimensionError,
        lambda: exact_map(LinearMapType(M1, M1), ((1, 0),), label="ill-shaped"),
    )
    _expect(
        ValidationError,
        lambda: apply(evaluation(M1, 0), exact_value(M0, (1,))),
    )
    _expect(DimensionError, lambda: matrix_chain_multiply((), matrix_chain_plan((1, 1))))
    _expect(ValidationError, lambda: exact_value(RATIONAL_SPACE, (0.5,)))
    _expect(ValidationError, lambda: scale(True, exact_value(RATIONAL_SPACE, (1,))))


@case("transport_copy_tamper_isolation")
def _case_copy_tamper() -> None:
    original = _scalar_envelope()
    hostile = copy.deepcopy(original)
    hostile["semantic_sha256"] = "f" * 64
    assert expression_from_dict(original) == exact_value(RATIONAL_SPACE, (Q(1, 2),))
    _expect(ValidationError, lambda: expression_from_dict(hostile), contains="mismatch")


def _report() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for name, operation in CASES:
        try:
            operation()
        except Exception as exc:
            results.append(
                {
                    "detail": f"{type(exc).__name__}: {exc}",
                    "name": name,
                    "passed": False,
                }
            )
        else:
            results.append({"detail": "", "name": name, "passed": True})
    passed_count = sum(bool(result["passed"]) for result in results)
    total_count = len(results)
    return {
        "failed_count": total_count - passed_count,
        "package_version": "1.0.0b4",
        "passed": passed_count == total_count,
        "passed_count": passed_count,
        "results": results,
        "schema": "komorebi-math.block20-adversarial-tests",
        "schema_version": 1,
        "total_count": total_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the deterministic JSON report")
    parser.add_argument("--quiet", action="store_true", help="suppress stdout")
    arguments = parser.parse_args(argv)
    report = _report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    if not arguments.quiet:
        print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
