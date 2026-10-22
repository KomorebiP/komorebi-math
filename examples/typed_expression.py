"""Typed finite expression and exact integration example."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from functools import cache
from math import comb
from typing import Any

from komorebi_math import __version__
from komorebi_math.expression import (
    IdentityMap,
    MatrixChainPlan,
    PolynomialCoordinateSpace,
    affine_substitution,
    antiderivative,
    apply,
    compile_expression,
    compose,
    coordinate_change,
    derivative,
    evaluate_expression,
    exact_value,
    expression_digest,
    expression_from_json,
    expression_to_json,
    interval_integration,
    matrix_chain_parenthesization,
    normalize_expression,
    replay,
    simplify_expression,
)

Q = Fraction


@cache
def _euler_at_zero(index: int) -> Fraction:
    if index == 0:
        return Q(1)
    return -sum(
        (Q(comb(index, lower)) * _euler_at_zero(lower) for lower in range(index)),
        Q(0),
    ) / 2


def _euler_polynomial(degree: int) -> tuple[Fraction, ...]:
    return tuple(Q(comb(degree, power)) * _euler_at_zero(degree - power) for power in range(degree + 1))


@cache
def _fibonacci_polynomial(index: int) -> tuple[Fraction, ...]:
    if index == 0:
        return (Q(0),)
    if index == 1:
        return (Q(1),)
    previous = _fibonacci_polynomial(index - 2)
    current = _fibonacci_polynomial(index - 1)
    output = [Q(0) for _ in range(max(len(previous), len(current) + 1))]
    for power, coefficient in enumerate(previous):
        output[power] += coefficient
    for power, coefficient in enumerate(current):
        output[power + 1] += coefficient
    while len(output) > 1 and output[-1] == 0:
        output.pop()
    return tuple(output)


def _fibonacci_basis_polynomial(degree: int) -> tuple[Fraction, ...]:
    return _fibonacci_polynomial(degree + 1)


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _json_value(value: Any) -> Any:
    if isinstance(value, Fraction):
        return _fraction_text(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def build_record() -> dict[str, Any]:
    fibonacci4 = PolynomialCoordinateSpace(
        "x",
        4,
        tuple(_fibonacci_basis_polynomial(degree) for degree in range(5)),
        "Fibonacci G_n = F_(n+1)",
    )
    euler4 = PolynomialCoordinateSpace(
        "x",
        4,
        tuple(_euler_polynomial(degree) for degree in range(5)),
        "Euler polynomials",
    )
    fibonacci3 = fibonacci4.truncate(3)
    euler3 = euler4.truncate(3)

    polynomial = exact_value(fibonacci3, (0, 0, 0, 1))
    j0 = antiderivative(fibonacci3, fibonacci4, 0)
    f4_to_e4 = coordinate_change(fibonacci4, euler4)
    d_e4 = derivative(euler4)
    e3_to_f3 = coordinate_change(euler3, fibonacci3)

    after_j = normalize_expression(apply(j0, polynomial)).data
    after_change = normalize_expression(apply(f4_to_e4, apply(j0, polynomial))).data
    after_derivative = normalize_expression(
        apply(d_e4, apply(f4_to_e4, apply(j0, polynomial)))
    ).data
    recovered = normalize_expression(
        apply(e3_to_f3, apply(d_e4, apply(f4_to_e4, apply(j0, polynomial))))
    ).data

    pipeline = compose(e3_to_f3, d_e4, f4_to_e4, j0)
    integration = interval_integration(fibonacci3, 0, 1)
    expression = apply(integration, apply(pipeline, polynomial))
    evaluated = evaluate_expression(expression)
    pipeline_form = normalize_expression(pipeline)
    simplified = simplify_expression(pipeline)
    plan = compile_expression(expression)
    chain = next(
        instruction.payload
        for instruction in plan.instructions
        if instruction.opcode == "matrix_chain"
    )
    assert isinstance(chain, MatrixChainPlan)

    expected_identity = tuple(
        tuple(Q(int(row == column)) for column in range(4)) for row in range(4)
    )
    expected_vectors = (
        (Q(-1, 2), Q(0), Q(1, 4), Q(0), Q(1, 4)),
        (Q(5, 8), Q(3, 2), Q(7, 4), Q(1, 2), Q(1, 4)),
        (Q(3, 2), Q(7, 2), Q(3, 2), Q(1)),
        (Q(0), Q(0), Q(0), Q(1)),
    )
    integration_covector = replay(integration)[0]
    assert (after_j, after_change, after_derivative, recovered) == expected_vectors
    assert integration_covector == (Q(1), Q(1, 2), Q(4, 3), Q(5, 4))
    assert pipeline_form.data == expected_identity
    assert evaluated.scalar == Q(5, 4)
    assert isinstance(simplified.expression, IdentityMap)
    assert simplified.before.data == simplified.after.data == expected_identity
    assert evaluated.direct_normal_form.digest == evaluated.normal_form.digest

    portable_json = expression_to_json(expression)
    portable_roundtrip = expression_from_json(portable_json)
    assert expression_digest(portable_roundtrip) == expression_digest(expression)
    assert normalize_expression(portable_roundtrip).data == (Q(5, 4),)

    monomial4 = PolynomialCoordinateSpace(
        "y",
        4,
        tuple(
            tuple(Q(int(power == degree)) for power in range(degree + 1))
            for degree in range(5)
        ),
        "monomials",
    )
    polynomial_y = exact_value(monomial4, (1, -2, 0, 0, 1))
    pulled_back = apply(affine_substitution(monomial4, 2, 1), polynomial_y)
    pullback_value = normalize_expression(pulled_back).data
    base_integral = evaluate_expression(
        apply(interval_integration(monomial4, 0, 1), pulled_back)
    ).scalar
    weighted_pushforward = evaluate_expression(
        apply(interval_integration(monomial4, 1, 3, Q(1, 2)), polynomial_y)
    ).scalar
    assert pullback_value == (Q(0), Q(4), Q(24), Q(32), Q(16))
    assert base_integral == weighted_pushforward == Q(106, 5)

    return {
        "schema": "komorebi-math.block20-worked-example",
        "schema_version": 1,
        "package_version": __version__,
        "input": {
            "basis_coordinate": [0, 0, 0, 1],
            "physical_polynomial": "G_3(x) = x^3 + 2x",
        },
        "intermediate_vectors": {
            "J_0_in_Fibonacci": after_j,
            "Fibonacci4_to_Euler4": after_change,
            "D_in_Euler": after_derivative,
            "Euler3_to_Fibonacci3": recovered,
        },
        "integration_covector_0_1": integration_covector,
        "pipeline_normal_form": pipeline_form.data,
        "pipeline_is_I4": pipeline_form.data == expected_identity,
        "result": evaluated.scalar,
        "simplifier": {
            "rules": [step.rule for step in simplified.steps],
            "result_node": type(simplified.expression).__name__,
        },
        "execution_plan": {
            "instruction_count": len(plan.instructions),
            "scalar_operation_cost": plan.scalar_operation_cost,
            "chain_dimensions": chain.dimensions,
            "chain_optimal_cost": chain.scalar_operation_cost,
            "chain_left_associative_cost": chain.left_associative_cost,
            "parenthesization": matrix_chain_parenthesization(
                chain,
                ("C_E3_to_F3", "D_E4", "C_F4_to_E4", "J_0_F3"),
            ),
            "direct_and_planned_match": evaluated.direct_normal_form.digest
            == evaluated.normal_form.digest,
        },
        "portable": {
            "expression_digest": expression_digest(expression),
            "roundtrip_equal_normal_form": True,
        },
        "affine_regression": {
            "pulled_back_coefficients": pullback_value,
            "base_integral": base_integral,
            "weighted_pushforward_integral": weighted_pushforward,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print the machine-readable record")
    args = parser.parse_args()
    record = build_record()
    if args.json:
        print(json.dumps(_json_value(record), indent=2, sort_keys=True))
        return 0

    print("Typed finite expression example")
    print(f"Input: {record['input']['physical_polynomial']}")
    for name, vector in record["intermediate_vectors"].items():
        print(f"  {name}: ({', '.join(_fraction_text(value) for value in vector)})")
    covector = record["integration_covector_0_1"]
    print(f"Integration covector: ({', '.join(_fraction_text(value) for value in covector)})")
    print("Compiled pipeline normal form: I4")
    print(f"Final integral: {_fraction_text(record['result'])}")
    print(f"Simplifier rules: {', '.join(record['simplifier']['rules'])}")
    print(f"Optimal chain: {record['execution_plan']['parenthesization']}")
    print(
        "Plan cost: "
        f"{record['execution_plan']['chain_optimal_cost']} versus "
        f"{record['execution_plan']['chain_left_associative_cost']} left-associated"
    )
    print(f"Portable digest: {record['portable']['expression_digest']}")
    print(
        "Affine/weighted regression: "
        f"{_fraction_text(record['affine_regression']['base_integral'])}"
    )
    print("All exact assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
