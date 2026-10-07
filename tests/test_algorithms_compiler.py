from __future__ import annotations

import itertools

import pytest

from komorebi_math.algorithms import (
    TreeNode,
    combine_tree_objectives,
    compile_tree_budget,
    compile_tree_lagrangian,
    decision_margins,
    pareto_frontier,
    replace_tree_nodes,
)
from komorebi_math.exceptions import ValidationError


def sample_tree() -> TreeNode:
    return TreeNode(
        "root",
        keep_cost=9.0,
        keep_error=0.0,
        split_cost=0.5,
        children=(
            TreeNode("left", 2.0, 0.2),
            TreeNode(
                "right",
                5.0,
                0.1,
                split_cost=0.2,
                children=(
                    TreeNode("r0", 1.0, 0.4),
                    TreeNode("r1", 1.5, 0.3),
                ),
            ),
        ),
    )


def all_plans(node: TreeNode):
    yield (node.keep_cost, node.keep_error, (node.name,))
    if node.children:
        child_plans = [list(all_plans(child)) for child in node.children]
        for choices in itertools.product(*child_plans):
            yield (
                node.split_cost + sum(choice[0] for choice in choices),
                sum(choice[1] for choice in choices),
                tuple(leaf for choice in choices for leaf in choice[2]),
            )


def test_tree_rejects_duplicate_descendant_names_and_nonfinite_data() -> None:
    with pytest.raises(ValidationError):
        TreeNode(
            "root",
            1.0,
            0.0,
            children=(
                TreeNode("left", 1.0, 0.0, children=(TreeNode("same", 1.0, 0.0),)),
                TreeNode("right", 1.0, 0.0, children=(TreeNode("same", 1.0, 0.0),)),
            ),
        )
    with pytest.raises(ValidationError):
        TreeNode("root", float("nan"), 0.0)


def test_budget_rejects_nonintegral_budget() -> None:
    with pytest.raises(ValidationError):
        compile_tree_budget(sample_tree(), 1.5)


def test_lagrangian_compiler_matches_enumeration() -> None:
    root = sample_tree()
    penalty = 4.0
    plan = compile_tree_lagrangian(root, penalty)
    plan.certificate.require()
    brute = min(all_plans(root), key=lambda item: (item[0] + penalty * item[1], item[0], item[1], item[2]))
    assert (plan.cost, plan.error, plan.leaves) == brute


def test_budget_compiler_matches_enumeration() -> None:
    root = sample_tree()
    plan = compile_tree_budget(root, 6, error_unit=0.1)
    assert plan is not None
    plan.certificate.require()
    feasible = [
        value
        for value in all_plans(root)
        if int(__import__("numpy").ceil(value[1] / 0.1 - 1e-14)) <= 6
    ]
    brute = min(feasible, key=lambda item: (item[0], int(__import__("numpy").ceil(item[1] / 0.1 - 1e-14)), item[1], item[2]))
    assert (plan.cost, plan.error, plan.leaves) == brute


def test_budget_quantizes_each_selected_leaf_independently() -> None:
    root = TreeNode(
        "root",
        keep_cost=10.0,
        keep_error=0.0,
        children=(
            TreeNode("left", 1.0, 0.04),
            TreeNode("right", 1.0, 0.04),
        ),
    )
    one_unit = compile_tree_budget(root, 1, error_unit=0.1)
    assert one_unit is not None
    assert one_unit.leaves == ("root",)
    two_units = compile_tree_budget(root, 2, error_unit=0.1)
    assert two_units is not None
    assert two_units.leaves == ("left", "right")
    assert two_units.quantized_error == 2


def test_pareto_frontier_is_nondominated() -> None:
    frontier = pareto_frontier(sample_tree())
    assert frontier
    for left in frontier:
        for right in frontier:
            if left is right:
                continue
            assert not (left.cost <= right.cost and left.error <= right.error and (left.cost < right.cost or left.error < right.error))


def test_combine_tree_objectives_and_updates() -> None:
    first = sample_tree()
    second = replace_tree_nodes(first, {"root": {"keep_cost": 11.0}, "r0": {"keep_error": 0.2}})
    combined = combine_tree_objectives([first, second], [0.25, 0.75])
    assert combined.keep_cost == 0.25 * 9.0 + 0.75 * 11.0
    assert combined.children[1].children[0].keep_error == 0.25 * 0.4 + 0.75 * 0.2


def test_decision_margins() -> None:
    margins = {margin.node: margin for margin in decision_margins(sample_tree(), 2.0)}
    assert margins["root"].margin is not None
    assert margins["left"].chosen == "keep"

def test_replace_tree_nodes_rejects_unknown_nodes_and_fields() -> None:
    with pytest.raises(ValidationError):
        replace_tree_nodes(sample_tree(), {"missing": {"keep_cost": 1.0}})
    with pytest.raises(ValidationError):
        replace_tree_nodes(sample_tree(), {"root": {"unknown": 1.0}})
    with pytest.raises(ValidationError):
        pareto_frontier(sample_tree(), max_plans=2.5)

