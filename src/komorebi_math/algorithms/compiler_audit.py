"""Bounded independent exact oracles for finite packet-tree compilers.

Paper VIII, Sections 9.2--9.3, defines an admissible pruning by one selected
node on every root-to-terminal path.  These helpers enumerate *all* such
prunings rather than using the optimized keep/split or budget recurrence.
They are intended for small stress cases and reject enumeration beyond the
declared limit.  Exactness concerns the supplied binary floating data and
their exact dyadic values, not a continuous cost model or hardware timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isfinite
from numbers import Integral, Real

from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from ._validation import finite_float, strict_integer
from .compiler import TreeNode, TreePlan


@dataclass(frozen=True, slots=True)
class ExactPruning:
    """Exact additive data of one admissible finite-tree pruning."""

    cost: Fraction
    error: Fraction
    leaves: tuple[str, ...]
    decisions: tuple[tuple[str, str], ...]
    leaf_errors: tuple[Fraction, ...]

    def error_units(self, error_unit: Fraction) -> int:
        """Sum independently rounded-up leaf errors in positive units."""
        if not isinstance(error_unit, Fraction) or error_unit <= 0:
            raise ValidationError("error_unit must be a strictly positive Fraction")
        ratios = tuple(error / error_unit for error in self.leaf_errors)
        return sum(-(-ratio.numerator // ratio.denominator) for ratio in ratios)


@dataclass(frozen=True, slots=True)
class TreeCompilerAudit:
    """An exhaustive finite optimum and a fresh audit of a supplied plan."""

    prunings_examined: int
    feasible_prunings: int
    optimum: ExactPruning | None
    objective: Fraction | None
    certificate: Certificate


def enumerate_tree_prunings(
    root: TreeNode, *, max_prunings: int = 10000, max_nodes: int = 1000
) -> tuple[ExactPruning, ...]:
    """Enumerate a complete small-tree pruning space using exact arithmetic.

    Each internal node contributes its keep option and every Cartesian
    product of child pruning spaces.  No objective-based pruning or budget
    dynamic program is used.  Limits are checked before a Cartesian product
    is formed.  An exhausted limit raises instead of returning a partial
    optimum.  A depth above 256 also raises to bound Python recursion.
    """
    if not isinstance(root, TreeNode):
        raise ValidationError("root must be a TreeNode")
    pruning_limit = strict_integer(max_prunings, "max_prunings", minimum=1)
    node_limit = strict_integer(max_nodes, "max_nodes", minimum=1)
    pending = [(root, 0)]
    nodes = 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if nodes > node_limit:
            raise ValidationError("tree exceeds max_nodes")
        if depth > 256:
            raise ValidationError("tree exceeds the bounded audit depth of 256")
        pending.extend((child, depth + 1) for child in node.children)

    def enumerate_subtree(node: TreeNode) -> tuple[ExactPruning, ...]:
        cost = Fraction.from_float(node.keep_cost)
        error = Fraction.from_float(node.keep_error)
        keep = ExactPruning(cost, error, (node.name,), ((node.name, "keep"),), (error,))
        if node.terminal:
            return (keep,)
        choices = tuple(enumerate_subtree(child) for child in node.children)
        combinations = 1
        for space in choices:
            combinations *= len(space)
            if combinations + 1 > pruning_limit:
                raise ValidationError("complete pruning space exceeds max_prunings")
        split_overhead = Fraction.from_float(node.split_cost)
        split = tuple(
            ExactPruning(
                split_overhead + sum((part.cost for part in parts), Fraction(0)),
                sum((part.error for part in parts), Fraction(0)),
                tuple(leaf for part in parts for leaf in part.leaves),
                ((node.name, "split"), *(decision for part in parts for decision in part.decisions)),
                tuple(error for part in parts for error in part.leaf_errors),
            )
            for parts in product(*choices)
        )
        return (keep, *split)

    return enumerate_subtree(root)


def _reported_float_matches(value: object, exact: Fraction) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        reported, rounded = float(value), float(exact)
    except (OverflowError, ValueError):
        return False
    return isfinite(reported) and reported == rounded


def _matches_plan(plan: TreePlan, pruning: ExactPruning) -> bool:
    if not isinstance(plan, TreePlan):
        return False
    if not isinstance(plan.leaves, tuple) or not all(isinstance(leaf, str) for leaf in plan.leaves):
        return False
    if not isinstance(plan.decisions, tuple) or not all(
        isinstance(decision, tuple)
        and len(decision) == 2
        and all(isinstance(value, str) for value in decision)
        for decision in plan.decisions
    ):
        return False
    return (
        plan.leaves == pruning.leaves
        and plan.decisions == pruning.decisions
        and _reported_float_matches(plan.cost, pruning.cost)
        and _reported_float_matches(plan.error, pruning.error)
    )


def audit_tree_lagrangian(
    root: TreeNode,
    plan: TreePlan,
    penalty: float,
    *,
    max_prunings: int = 10000,
    max_nodes: int = 1000,
) -> TreeCompilerAudit:
    """Check feasibility, reported totals and optimality by full enumeration.

    Any admissible optimum is accepted; the optimized compiler's tie-breaking
    rule is not assumed.  An existing ``plan.certificate`` is never trusted.
    """
    if isinstance(penalty, bool) or not isinstance(penalty, Real):
        raise ValidationError("penalty must be a finite nonnegative real number, not a boolean")
    penalty_value = finite_float(penalty, "penalty", minimum=0.0)
    exact_penalty = Fraction.from_float(penalty_value)
    prunings = enumerate_tree_prunings(root, max_prunings=max_prunings, max_nodes=max_nodes)

    def objective_key(item: ExactPruning) -> tuple[Fraction, Fraction, Fraction, tuple[str, ...]]:
        return item.cost + exact_penalty * item.error, item.cost, item.error, item.leaves

    optimum = min(prunings, key=objective_key)
    objective = optimum.cost + exact_penalty * optimum.error
    witness = next((item for item in prunings if _matches_plan(plan, item)), None)
    valid = (
        witness is not None
        and witness.cost + exact_penalty * witness.error == objective
        and _reported_float_matches(plan.objective, objective)
    )
    return TreeCompilerAudit(
        len(prunings),
        len(prunings),
        optimum,
        objective,
        Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "complete bounded finite pruning enumeration",
                "exact dyadic values of the supplied local data and penalty",
                "pruning feasibility, additive totals, and optimal objective",
            ),
            diagnostics={
                "prunings_examined": len(prunings),
                "max_prunings": max_prunings,
                "exact_optimum": str(objective),
                "reported_plan_admissible": witness is not None,
            },
            theorem_id="PSC-VIII-COMPILER-EXHAUSTIVE-AUDIT",
            message="bounded exhaustive compiler audit passed"
            if valid
            else "supplied plan failed the bounded exhaustive audit",
        ),
    )


def audit_tree_budget(
    root: TreeNode,
    plan: TreePlan | None,
    budget: int,
    *,
    error_unit: float = 1.0,
    max_prunings: int = 10000,
    max_nodes: int = 1000,
) -> TreeCompilerAudit:
    """Audit an integer-budget optimum or an infeasibility result exactly.

    Quantization is ``sum(ceil(leaf_error / error_unit))``.  No tolerance is
    subtracted at integer boundaries.  A ``None`` plan passes only when the
    complete bounded enumeration has no feasible pruning.
    """
    integer_budget = strict_integer(budget, "budget", minimum=0)
    if isinstance(error_unit, bool) or not isinstance(error_unit, Real):
        raise ValidationError("error_unit must be a finite positive real number, not a boolean")
    unit_value = finite_float(error_unit, "error_unit", minimum=0.0, strict_minimum=True)
    exact_unit = Fraction.from_float(unit_value)
    prunings = enumerate_tree_prunings(root, max_prunings=max_prunings, max_nodes=max_nodes)
    feasible = tuple(item for item in prunings if item.error_units(exact_unit) <= integer_budget)
    optimum = min(
        feasible,
        key=lambda item: (item.cost, item.error_units(exact_unit), item.error, item.leaves),
        default=None,
    )
    witness = next((item for item in feasible if plan is not None and _matches_plan(plan, item)), None)
    valid = (plan is None and optimum is None) or (
        witness is not None
        and optimum is not None
        and witness.cost == optimum.cost
        and isinstance(plan.quantized_error, Integral)
        and not isinstance(plan.quantized_error, bool)
        and plan.quantized_error == witness.error_units(exact_unit)
    )
    objective = None if optimum is None else optimum.cost
    return TreeCompilerAudit(
        len(prunings),
        len(feasible),
        optimum,
        objective,
        Certificate(
            valid=valid,
            exactness=Exactness.EXACT,
            assumptions_checked=(
                "complete bounded finite pruning enumeration",
                "exact dyadic local costs and independently quantized leaf errors",
                "budget feasibility, additive totals, and minimum cost",
            ),
            diagnostics={
                "prunings_examined": len(prunings),
                "feasible_prunings": len(feasible),
                "budget": integer_budget,
                "error_unit": str(exact_unit),
                "exact_minimum_cost": None if objective is None else str(objective),
                "reported_plan_admissible": witness is not None,
            },
            theorem_id="PSC-VIII-BUDGET-EXHAUSTIVE-AUDIT",
            message="bounded exhaustive budget audit passed"
            if valid
            else "supplied budget result failed the bounded exhaustive audit",
        ),
    )


__all__ = [
    "ExactPruning",
    "TreeCompilerAudit",
    "audit_tree_budget",
    "audit_tree_lagrangian",
    "enumerate_tree_prunings",
]
