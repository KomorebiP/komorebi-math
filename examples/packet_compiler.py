"""Compile one finite candidate representation tree exactly."""

from __future__ import annotations

from komorebi_math.algorithms import (
    TreeNode,
    compile_tree_budget,
    compile_tree_lagrangian,
    pareto_frontier,
)

candidate = TreeNode(
    "root",
    keep_cost=9.0,
    keep_error=0.0,
    split_cost=0.5,
    children=(
        TreeNode("left", keep_cost=2.0, keep_error=0.2),
        TreeNode(
            "right",
            keep_cost=5.0,
            keep_error=0.1,
            split_cost=0.2,
            children=(
                TreeNode("right_low", keep_cost=1.0, keep_error=0.4),
                TreeNode("right_high", keep_cost=1.5, keep_error=0.3),
            ),
        ),
    ),
)

lagrangian = compile_tree_lagrangian(candidate, penalty=4.0)
budget = compile_tree_budget(candidate, budget=6, error_unit=0.1)
frontier = pareto_frontier(candidate)

print("Lagrangian leaves", lagrangian.leaves)
print("budget leaves", None if budget is None else budget.leaves)
print("Pareto points", [(plan.cost, plan.error, plan.leaves) for plan in frontier])
