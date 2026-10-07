"""Exact finite compilers for additive packet representation trees."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from math import fsum
from typing import Mapping, Sequence

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from ..provenance import ClaimStatus, ProvenanceRecord, documented
from ._validation import finite_float, strict_integer


@dataclass(frozen=True, slots=True)
class TreeNode:
    """One candidate keep or split decision in a finite compiler tree."""

    name: str
    keep_cost: float
    keep_error: float
    split_cost: float = 0.0
    children: tuple["TreeNode", ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("a compiler node must have a nonempty string name")
        children = tuple(self.children)
        if any(not isinstance(child, TreeNode) for child in children):
            raise ValidationError("compiler children must be TreeNode objects")
        values = (
            finite_float(self.keep_cost, "keep_cost", minimum=0.0),
            finite_float(self.keep_error, "keep_error", minimum=0.0),
            finite_float(self.split_cost, "split_cost", minimum=0.0),
        )
        all_names = (self.name,) + tuple(
            descendant
            for child in children
            for descendant in child.names()
        )
        if len(set(all_names)) != len(all_names):
            raise ValidationError("compiler node names must be unique throughout the tree")
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "keep_cost", values[0])
        object.__setattr__(self, "keep_error", values[1])
        object.__setattr__(self, "split_cost", values[2])

    @property
    def terminal(self) -> bool:
        return not self.children

    def names(self) -> tuple[str, ...]:
        return (self.name,) + tuple(name for child in self.children for name in child.names())


@dataclass(frozen=True, slots=True)
class TreePlan:
    """One complete admissible pruning and its additive objective data."""

    cost: float
    error: float
    leaves: tuple[str, ...]
    decisions: tuple[tuple[str, str], ...]
    objective: float | None = None
    quantized_error: int | None = None
    certificate: Certificate | None = None

    def decision(self, node: str) -> str | None:
        return dict(self.decisions).get(node)


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T30",
        title="Exact additive tree compiler",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite rooted tree", "additive keep and split models"),
        verification_suite="algorithms.compiler",
    )
)
def compile_tree_lagrangian(root: TreeNode, penalty: float) -> TreePlan:
    """Return the exact minimizer of cost plus penalty times error."""

    penalty_value = finite_float(penalty, "penalty", minimum=0.0)
    penalty_exact = Fraction.from_float(penalty_value)

    def solve(node: TreeNode) -> tuple[TreePlan, Fraction, Fraction, Fraction]:
        keep_cost_exact = Fraction.from_float(float(node.keep_cost))
        keep_error_exact = Fraction.from_float(float(node.keep_error))
        keep_objective_exact = keep_cost_exact + penalty_exact * keep_error_exact
        keep = TreePlan(
            cost=float(keep_cost_exact),
            error=float(keep_error_exact),
            leaves=(node.name,),
            decisions=((node.name, "keep"),),
            objective=float(keep_objective_exact),
        )
        keep_record = (keep, keep_cost_exact, keep_error_exact, keep_objective_exact)
        if node.terminal:
            return keep_record
        child_records = tuple(solve(child) for child in node.children)
        split_cost_exact = Fraction.from_float(float(node.split_cost)) + sum(
            (record[1] for record in child_records), Fraction(0)
        )
        split_error_exact = sum((record[2] for record in child_records), Fraction(0))
        split_objective_exact = split_cost_exact + penalty_exact * split_error_exact
        split = TreePlan(
            cost=float(split_cost_exact),
            error=float(split_error_exact),
            leaves=tuple(leaf for record in child_records for leaf in record[0].leaves),
            decisions=((node.name, "split"),)
            + tuple(decision for record in child_records for decision in record[0].decisions),
            objective=float(split_objective_exact),
        )
        split_record = (split, split_cost_exact, split_error_exact, split_objective_exact)
        return min(
            (keep_record, split_record),
            key=lambda record: (record[3], record[1], record[2], record[0].leaves),
        )

    plan, exact_cost, exact_error, exact_objective = solve(root)
    certificate = Certificate(
        valid=True,
        exactness=Exactness.EXACT,
        assumptions_checked=(
            "finite tree",
            "additive Lagrangian objective",
            "exact dyadic comparison of the supplied finite floating data",
        ),
        diagnostics={
            "penalty": penalty_value,
            "leaf_count": len(plan.leaves),
            "exact_cost": str(exact_cost),
            "exact_error": str(exact_error),
            "exact_objective": str(exact_objective),
        },
        theorem_id="PSC-VIII-T30",
        message="exact additive tree compilation over the supplied binary floating data",
    )
    return replace(plan, certificate=certificate)


@documented(
    ProvenanceRecord(
        identifier="PSC-VIII-T33",
        title="Exact quantized budget compiler",
        status=ClaimStatus.PROVED,
        source="Paper VIII",
        assumptions=("finite rooted tree", "integer quantized additive error budget"),
        verification_suite="algorithms.compiler",
    )
)
def compile_tree_budget(
    root: TreeNode,
    budget: int,
    *,
    error_unit: float = 1.0,
) -> TreePlan | None:
    """Minimize cost under additive integer error weights.

    Each keep error is converted independently to
    ``ceil(keep_error / error_unit)``.  The selected leaf weights are then
    added exactly.  This matches the integer model in the budget theorem and
    is deliberately not the same as rounding the final real error sum once.
    """

    budget_value = strict_integer(budget, "budget", minimum=0)
    error_unit_value = finite_float(error_unit, "error_unit", minimum=0.0, strict_minimum=True)
    error_unit_exact = Fraction.from_float(error_unit_value)
    budget = budget_value

    def units(error: float) -> int:
        ratio = Fraction.from_float(float(error)) / error_unit_exact
        return -(-ratio.numerator // ratio.denominator)

    def solve(node: TreeNode) -> dict[int, tuple[TreePlan, Fraction, Fraction]]:
        table: dict[int, tuple[TreePlan, Fraction, Fraction]] = {}
        keep_units = units(node.keep_error)
        keep_cost_exact = Fraction.from_float(float(node.keep_cost))
        keep_error_exact = Fraction.from_float(float(node.keep_error))
        if keep_units <= budget:
            table[keep_units] = (
                TreePlan(
                    cost=float(keep_cost_exact),
                    error=float(keep_error_exact),
                    leaves=(node.name,),
                    decisions=((node.name, "keep"),),
                    quantized_error=keep_units,
                ),
                keep_cost_exact,
                keep_error_exact,
            )
        if node.terminal:
            return table

        split_cost_exact = Fraction.from_float(float(node.split_cost))
        merged: dict[int, tuple[TreePlan, Fraction, Fraction]] = {
            0: (
                TreePlan(
                    cost=float(split_cost_exact),
                    error=0.0,
                    leaves=(),
                    decisions=((node.name, "split"),),
                    quantized_error=0,
                ),
                split_cost_exact,
                Fraction(0),
            )
        }
        for child in node.children:
            child_table = solve(child)
            next_table: dict[int, tuple[TreePlan, Fraction, Fraction]] = {}
            for previous_units, previous_record in merged.items():
                previous, previous_cost, previous_error = previous_record
                for child_units, child_record in child_table.items():
                    child_plan, child_cost, child_error = child_record
                    total_units = previous_units + child_units
                    if total_units > budget:
                        continue
                    exact_cost = previous_cost + child_cost
                    exact_error = previous_error + child_error
                    candidate = TreePlan(
                        cost=float(exact_cost),
                        error=float(exact_error),
                        leaves=previous.leaves + child_plan.leaves,
                        decisions=previous.decisions + child_plan.decisions,
                        quantized_error=total_units,
                    )
                    record = (candidate, exact_cost, exact_error)
                    old = next_table.get(total_units)
                    if old is None or (exact_cost, exact_error, candidate.leaves) < (
                        old[1],
                        old[2],
                        old[0].leaves,
                    ):
                        next_table[total_units] = record
            merged = next_table
            if not merged:
                break
        for error_units, record in merged.items():
            plan, exact_cost, exact_error = record
            old = table.get(error_units)
            if old is None or (exact_cost, exact_error, plan.leaves) < (
                old[1],
                old[2],
                old[0].leaves,
            ):
                table[error_units] = record
        return table

    table = solve(root)
    if not table:
        return None
    plan, exact_cost, exact_error = min(
        table.values(),
        key=lambda record: (
            record[1],
            record[0].quantized_error,
            record[2],
            record[0].leaves,
        ),
    )
    certificate = Certificate(
        valid=int(plan.quantized_error) <= int(budget),
        exactness=Exactness.EXACT,
        assumptions_checked=(
            "finite tree",
            "quantized additive error model",
            "exact dyadic comparison and quantization of the supplied finite floating data",
        ),
        diagnostics={
            "budget": int(budget),
            "error_unit": error_unit_value,
            "used_units": int(plan.quantized_error),
            "exact_cost": str(exact_cost),
            "exact_error": str(exact_error),
        },
        theorem_id="PSC-VIII-T33",
        message="exact quantized budget compilation over the supplied binary floating data",
    )
    return replace(plan, certificate=certificate)


def _dominates(left: TreePlan, right: TreePlan, tolerance: float) -> bool:
    no_worse = left.cost <= right.cost + tolerance and left.error <= right.error + tolerance
    strictly_better = left.cost < right.cost - tolerance or left.error < right.error - tolerance
    return no_worse and strictly_better


def pareto_frontier(
    root: TreeNode,
    *,
    tolerance: float = 0.0,
    max_plans: int = 100000,
) -> tuple[TreePlan, ...]:
    """Enumerate nondominated cost and error plans for a finite tree.

    The default zero tolerance gives the exact frontier of the supplied
    floating values.  A positive tolerance treats smaller numerical
    differences as ties.
    """

    tolerance_value = finite_float(tolerance, "tolerance", minimum=0.0)
    maximum_plans = strict_integer(max_plans, "max_plans", minimum=1)
    max_plans = maximum_plans

    def prune(plans: Sequence[TreePlan]) -> tuple[TreePlan, ...]:
        ordered = sorted(plans, key=lambda plan: (plan.cost, plan.error, plan.leaves))
        frontier: list[TreePlan] = []
        for plan in ordered:
            if any(_dominates(existing, plan, tolerance_value) for existing in frontier):
                continue
            frontier = [
                existing
                for existing in frontier
                if not _dominates(plan, existing, tolerance_value)
            ]
            frontier.append(plan)
        return tuple(sorted(frontier, key=lambda plan: (plan.error, plan.cost, plan.leaves)))

    def solve(node: TreeNode) -> tuple[TreePlan, ...]:
        keep = TreePlan(
            cost=float(node.keep_cost),
            error=float(node.keep_error),
            leaves=(node.name,),
            decisions=((node.name, "keep"),),
        )
        if node.terminal:
            return (keep,)
        child_frontiers = tuple(solve(child) for child in node.children)
        combination_count = int(np.prod([len(frontier) for frontier in child_frontiers], dtype=object))
        if combination_count > max_plans:
            raise ValidationError(
                f"Pareto enumeration would create {combination_count} combinations, above max_plans"
            )
        split_plans: list[TreePlan] = []
        for combination in product(*child_frontiers):
            split_plans.append(
                TreePlan(
                    cost=float(node.split_cost) + fsum(plan.cost for plan in combination),
                    error=fsum(plan.error for plan in combination),
                    leaves=tuple(leaf for plan in combination for leaf in plan.leaves),
                    decisions=((node.name, "split"),)
                    + tuple(decision for plan in combination for decision in plan.decisions),
                )
            )
        return prune((keep, *split_plans))

    return solve(root)


def combine_tree_objectives(
    roots: Sequence[TreeNode],
    weights: Sequence[float],
) -> TreeNode:
    """Combine several operators that share one candidate tree topology."""

    if not roots or len(roots) != len(weights):
        raise ValidationError("roots and weights must have equal positive length")
    weight_values = tuple(finite_float(weight, "weight", minimum=0.0) for weight in weights)
    if fsum(weight_values) <= 0:
        raise ValidationError("weights must have positive total")

    def combine(nodes: Sequence[TreeNode]) -> TreeNode:
        first = nodes[0]
        if any(node.name != first.name or len(node.children) != len(first.children) for node in nodes[1:]):
            raise ValidationError("all objective trees must have the same named topology")
        children = tuple(
            combine(tuple(node.children[index] for node in nodes))
            for index in range(len(first.children))
        )
        return TreeNode(
            first.name,
            fsum(weight * node.keep_cost for weight, node in zip(weight_values, nodes, strict=True)),
            fsum(weight * node.keep_error for weight, node in zip(weight_values, nodes, strict=True)),
            fsum(weight * node.split_cost for weight, node in zip(weight_values, nodes, strict=True)),
            children,
        )

    return combine(tuple(roots))


@dataclass(frozen=True, slots=True)
class DecisionMargin:
    """Local keep and split score gap for one compiler node."""

    node: str
    keep_objective: float
    split_objective: float | None
    margin: float | None
    chosen: str


def decision_margins(root: TreeNode, penalty: float) -> tuple[DecisionMargin, ...]:
    """Return local strict decision margins under the exact recursive objective."""

    penalty_value = finite_float(penalty, "penalty", minimum=0.0)
    margins: list[DecisionMargin] = []

    def solve(node: TreeNode) -> float:
        keep = node.keep_cost + penalty_value * node.keep_error
        if node.terminal:
            margins.append(DecisionMargin(node.name, keep, None, None, "keep"))
            return keep
        split = node.split_cost + fsum(solve(child) for child in node.children)
        chosen = "keep" if keep <= split else "split"
        margins.append(DecisionMargin(node.name, keep, split, abs(keep - split), chosen))
        return min(keep, split)

    solve(root)
    return tuple(margins)


def replace_tree_nodes(
    root: TreeNode,
    updates: Mapping[str, Mapping[str, float]],
) -> TreeNode:
    """Return a tree with selected local model fields replaced."""

    unknown_nodes = set(updates) - set(root.names())
    if unknown_nodes:
        raise ValidationError(f"unknown tree update nodes: {sorted(unknown_nodes)}")

    def update(node: TreeNode) -> TreeNode:
        values = dict(updates.get(node.name, {}))
        allowed = {"keep_cost", "keep_error", "split_cost"}
        unknown = set(values) - allowed
        if unknown:
            raise ValidationError(f"unknown tree update fields for {node.name}: {sorted(unknown)}")
        return replace(
            node,
            keep_cost=float(values.get("keep_cost", node.keep_cost)),
            keep_error=float(values.get("keep_error", node.keep_error)),
            split_cost=float(values.get("split_cost", node.split_cost)),
            children=tuple(update(child) for child in node.children),
        )

    return update(root)


__all__ = [
    "DecisionMargin",
    "TreeNode",
    "TreePlan",
    "combine_tree_objectives",
    "compile_tree_budget",
    "compile_tree_lagrangian",
    "decision_margins",
    "pareto_frontier",
    "replace_tree_nodes",
]
