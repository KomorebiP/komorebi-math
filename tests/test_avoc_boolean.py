"""Independent finite oracles for AVOC's exact three-input tree grammar."""

from collections import Counter
from dataclasses import replace
from itertools import product, repeat

import pytest

from komorebi_math.algorithms.boolean_synthesis import (
    BooleanClosureBank,
    BooleanExpr,
    audit_boolean_closure_bank,
    boolean_equivalent,
    boolean_grammar_layer,
    boolean_node_count,
    boolean_signature,
    bounded_boolean_search,
    build_boolean_closure_bank,
    compact_boolean_entries,
    compacted_boolean_layer,
    evaluate_boolean,
    globally_minimal_boolean,
)
from komorebi_math.certificates import Exactness
from komorebi_math.exceptions import ValidationError


def _ordinary(expression, inputs):
    """Independent recursive semantics; uses neither bit-mask nor producer API."""
    if expression.operation == "literal":
        return expression.atom
    if expression.operation == "variable":
        return inputs[expression.atom - 1]
    if expression.operation == "not":
        return not _ordinary(expression.children[0], inputs)
    left = _ordinary(expression.children[0], inputs)
    right = _ordinary(expression.children[1], inputs)
    return {"and": left and right, "or": left or right, "xor": left != right}[expression.operation]


def _count(expression):
    return 1 + sum(_count(child) for child in expression.children)


def _truth(expression):
    return tuple(_ordinary(expression, inputs) for inputs in product((False, True), repeat=3))


def _demorgan():
    left, right = BooleanExpr.variable(1), BooleanExpr.variable(2)
    return BooleanExpr.negate(BooleanExpr.binary("or", BooleanExpr.negate(left), BooleanExpr.negate(right)))


@pytest.fixture(scope="module")
def bank():
    return build_boolean_closure_bank()


def test_lexicographic_signature_and_eager_visit_semantics():
    assert tuple(boolean_signature(BooleanExpr.variable(index)) for index in (1, 2, 3)) == (240, 204, 170)
    assert boolean_signature(BooleanExpr.literal(False)) == 0
    assert boolean_signature(BooleanExpr.literal(True)) == 255
    for expression in boolean_grammar_layer(1):
        assert boolean_node_count(expression) == _count(expression)
        for inputs in product((False, True), repeat=3):
            evaluated = evaluate_boolean(expression, inputs)
            assert evaluated.value == _ordinary(expression, inputs)
            assert evaluated.visits == _count(expression)
    # A short-circuit meter would visit only one child; the source meter visits
    # all five nodes even when the first child already determines the output.
    right = BooleanExpr.binary("xor", BooleanExpr.variable(1), BooleanExpr.variable(2))
    expression = BooleanExpr.binary("and", BooleanExpr.literal(False), right)
    assert evaluate_boolean(expression, (False, False, False)).visits == 5
    assert evaluate_boolean(expression, (True, True, True)).visits == 5
    # Python aliasing supplies no circuit-DAG discount.
    shared = BooleanExpr.binary("xor", right, right)
    assert boolean_node_count(shared) == 7 and boolean_signature(shared) == 0


def test_complete_equivalence_and_de_morgan_worked_search():
    reference = _demorgan()
    conjunction = BooleanExpr.binary("and", BooleanExpr.variable(1), BooleanExpr.variable(2))
    assert boolean_equivalent(reference, conjunction)
    assert not boolean_equivalent(reference, BooleanExpr.variable(1))
    result = bounded_boolean_search(reference, 1, budget=6890, trial_cap=85)
    assert result.certificate.valid and result.certificate.exactness is Exactness.EXACT
    assert (result.reference_nodes, result.selected_nodes, result.layer_entries) == (6, 3, 85)
    assert (result.generation_charge, result.comparison_tariff, result.modeled_charge) == (90, 80, 6890)
    assert result.complete_layer and result.examined == 85
    assert _truth(result.expression) == _truth(reference)


@pytest.mark.parametrize(
    "budget,trial_cap,examined,generation,charge",
    [
        (0, 85, 0, 0, 0),
        (89, 85, 0, 0, 0),
        (90, 85, 0, 90, 90),
        (169, 85, 0, 90, 90),
        (170, 85, 1, 90, 170),
        (6890, 3, 3, 90, 330),
        (6810, 85, 84, 90, 6810),
    ],
)
def test_prefix_budget_and_no_generation_fallback(budget, trial_cap, examined, generation, charge):
    reference = _demorgan()
    result = bounded_boolean_search(reference, 1, budget=budget, trial_cap=trial_cap)
    assert (result.examined, result.generation_charge, result.modeled_charge) == (
        examined,
        generation,
        charge,
    )
    assert result.certificate.valid and not result.complete_layer
    layer_prefix = boolean_grammar_layer(1)[:examined]
    oracle = min(
        _count(candidate)
        for candidate in (reference, *layer_prefix)
        if _truth(candidate) == _truth(reference)
    )
    assert result.selected_nodes == oracle
    assert result.modeled_charge <= budget


def test_full_grammar_cardinality_bounds_and_independent_compaction():
    for depth, count, node_bound in ((0, 5, 1), (1, 85, 3), (2, 21845, 7)):
        layer = boolean_grammar_layer(depth)
        assert len(layer) == count and max(_count(expression) for expression in layer) == node_bound
        oracle = {}
        for expression in layer:
            truth = _truth(expression)
            oracle[truth] = min(oracle.get(truth, 100), _count(expression))
        result = compacted_boolean_layer(depth)
        assert {_truth(entry.expression): entry.node_count for entry in result.entries} == oracle
    submitted = (
        _demorgan(),
        BooleanExpr.binary("and", BooleanExpr.variable(1), BooleanExpr.variable(2)),
        _demorgan(),
    )
    compacted = compact_boolean_entries(submitted)
    assert len(compacted.entries) == 1 and compacted.entries[0].node_count == 3
    assert (compacted.emitted_entries, compacted.total_node_mass, compacted.index_probes) == (3, 15, 3)


def test_compaction_instrumentation_is_declared_constructor_work():
    initial = compacted_boolean_layer(0)
    first = compacted_boolean_layer(1)
    full = boolean_grammar_layer(1)
    assert first.emitted_entries == 5 + len(full)
    assert first.total_node_mass == 5 + sum(_count(expression) for expression in full)
    assert first.index_probes == first.emitted_entries
    assert initial.depth == 0 and first.depth == 1


def test_complete_bank_independently_attains_and_closes_every_constructor(bank):
    # This checks truth tuples, not the producer's bitwise semantic algebra.
    costs = {_truth(entry.expression): _count(entry.expression) for entry in bank.entries}
    assert len(costs) == 256 and len(bank.entries) == 256
    for entry in bank.entries:
        assert entry.node_count == _count(entry.expression)
        assert entry.signature == sum(
            int(bit) * 2**index for index, bit in enumerate(_truth(entry.expression))
        )
    for literal in (False, True):
        assert costs[(literal,) * 8] <= 1
    for index in range(3):
        assert costs[tuple(inputs[index] for inputs in product((False, True), repeat=3))] <= 1
    for truth, cost in costs.items():
        assert costs[tuple(not bit for bit in truth)] <= cost + 1
    for left, left_cost in costs.items():
        for right, right_cost in costs.items():
            assert (
                costs[tuple(a and b for a, b in zip(left, right, strict=True))] <= left_cost + right_cost + 1
            )
            assert (
                costs[tuple(a or b for a, b in zip(left, right, strict=True))] <= left_cost + right_cost + 1
            )
            assert (
                costs[tuple(a != b for a, b in zip(left, right, strict=True))] <= left_cost + right_cost + 1
            )
    certificate = audit_boolean_closure_bank(bank)
    assert certificate.valid and certificate.exactness is Exactness.EXACT
    assert certificate.diagnostics["atom_checks"] == 5
    assert certificate.diagnostics["unary_checks"] == 256
    assert certificate.diagnostics["binary_checks"] == 196608
    assert certificate.diagnostics["max_attained_nodes"] == 10
    assert bank.construction_rounds == 3
    assert sorted(Counter(entry.node_count for entry in bank.entries).items()) == [
        (1, 5),
        (2, 3),
        (3, 9),
        (4, 21),
        (5, 21),
        (6, 81),
        (7, 43),
        (8, 43),
        (9, 18),
        (10, 12),
    ]
    result = globally_minimal_boolean(bank, _demorgan())
    assert _count(result) == 3 and _truth(result) == _truth(_demorgan())


def test_audit_rejects_incomplete_duplicate_and_forged_rows(bank):
    assert not audit_boolean_closure_bank(BooleanClosureBank(bank.entries[:-1])).valid
    duplicated = BooleanClosureBank((bank.entries[1], *bank.entries[1:]))
    assert not audit_boolean_closure_bank(duplicated).valid
    for field, value in (("signature", 1), ("signature", True), ("node_count", 2), ("node_count", True)):
        row = replace(bank.entries[0], **{field: value})
        assert not audit_boolean_closure_bank(BooleanClosureBank((row, *bank.entries[1:]))).valid
    # Exact semantics and exact cached node counts alone do not prove minimality.
    # Every witness below is valid, but double negation makes its cost nonminimal.
    inflated = tuple(
        replace(
            entry,
            expression=BooleanExpr.negate(BooleanExpr.negate(entry.expression)),
            node_count=entry.node_count + 2,
        )
        for entry in bank.entries
    )
    result = audit_boolean_closure_bank(BooleanClosureBank(inflated))
    assert not result.valid and "atom" in result.message
    row = bank.entries[42]
    inflated_row = replace(
        row, expression=BooleanExpr.negate(BooleanExpr.negate(row.expression)), node_count=row.node_count + 2
    )
    result = audit_boolean_closure_bank(
        BooleanClosureBank((*bank.entries[:42], inflated_row, *bank.entries[43:]))
    )
    assert not result.valid and "inequality" in result.message
    # Inflate a complement pair together: witnesses and every unary inequality
    # remain valid, so a binary closure inequality must reject the bank.
    inflated_pair = tuple(
        replace(
            entry,
            expression=BooleanExpr.negate(BooleanExpr.negate(entry.expression)),
            node_count=entry.node_count + 2,
        )
        if entry.signature in (42, 213)
        else entry
        for entry in bank.entries
    )
    result = audit_boolean_closure_bank(BooleanClosureBank(inflated_pair))
    assert not result.valid and "inequality" in result.message
    assert result.diagnostics["unary_checks"] == 256
    assert result.diagnostics["binary_checks"] > 0
    with pytest.raises(ValidationError):
        globally_minimal_boolean(BooleanClosureBank(inflated), _demorgan())


def test_resource_exhaustion_never_certifies_global_optimality(bank):
    with pytest.raises(ValidationError, match="256 rows"):
        BooleanClosureBank([*bank.entries, bank.entries[0]])
    with pytest.raises(ValidationError, match="max_entries"):
        boolean_grammar_layer(3)
    with pytest.raises(ValidationError, match="max_entries"):
        bounded_boolean_search(_demorgan(), 3, budget=10**12, trial_cap=1)
    # Reservation fails before any unmanageable grammar allocation is attempted.
    assert bounded_boolean_search(_demorgan(), 3, budget=0, trial_cap=1).modeled_charge == 0
    with pytest.raises(ValidationError, match="max_rounds"):
        build_boolean_closure_bank(max_rounds=2)
    with pytest.raises(ValidationError, match="max_emitted"):
        build_boolean_closure_bank(max_emitted=84)
    with pytest.raises(ValidationError, match="max_witness_nodes"):
        build_boolean_closure_bank(max_witness_nodes=1)
    assert not audit_boolean_closure_bank(bank, max_witness_nodes=9).valid
    with pytest.raises(ValidationError, match="resource limit"):
        evaluate_boolean(_demorgan(), (1, 1, 1), max_nodes=5)
    with pytest.raises(ValidationError, match="resource limit"):
        boolean_node_count(_demorgan(), max_depth=1)
    deep = BooleanExpr.literal(False)
    for _ in range(257):
        deep = BooleanExpr.negate(deep)
    with pytest.raises(ValidationError, match="resource limit"):
        evaluate_boolean(deep, (0, 0, 0))


@pytest.mark.parametrize(
    "expression",
    [
        lambda: BooleanExpr("nand", (BooleanExpr.literal(True), BooleanExpr.literal(True))),
        lambda: BooleanExpr.literal(1),
        lambda: BooleanExpr.variable(True),
        lambda: BooleanExpr.variable(0),
        lambda: BooleanExpr.variable(4),
        lambda: BooleanExpr("and", (BooleanExpr.literal(False),)),
        lambda: BooleanExpr("not", (BooleanExpr.literal(False),), atom=1),
        lambda: BooleanExpr("variable", atom=1.5),
        lambda: BooleanExpr("literal", children=repeat(BooleanExpr.literal(True)), atom=True),
    ],
)
def test_grammar_validation(expression):
    with pytest.raises(ValidationError):
        expression()


@pytest.mark.parametrize("inputs", [(0, 1), (0, 1, 2), (0, 1, 1, 0), (0, 1, 0.0), repeat(0)])
def test_input_validation_consumes_at_most_four_bits(inputs):
    with pytest.raises(ValidationError):
        evaluate_boolean(BooleanExpr.variable(1), inputs)


@pytest.mark.parametrize(
    "keyword,value", [("budget", -1), ("budget", 1.5), ("trial_cap", True), ("max_entries", -1)]
)
def test_search_does_not_coerce_invalid_limits(keyword, value):
    values = {"budget": 100, "trial_cap": 1, keyword: value}
    with pytest.raises(ValidationError):
        bounded_boolean_search(_demorgan(), 1, **values)
