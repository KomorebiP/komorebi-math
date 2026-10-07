from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import pathlib
import random
import time
from fractions import Fraction
from math import nextafter
from typing import Any

import numpy as np

from komorebi_math.algorithms.compiler import TreeNode, compile_tree_budget, compile_tree_lagrangian
from komorebi_math.algorithms.precision import DTypeChoice, assign_discrete_precisions
from komorebi_math.langlands.finite_image.groups import FiniteGroup
from komorebi_math.nonlinear.graded import FiniteAbelianGroup
from komorebi_math.core.packets import PacketSystem
from komorebi_math.dynamics.survival import PhaseTypeSurvival
from komorebi_math.signals.wavelets import best_basis, haar_packet_analyze


def frac(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def exact_certificate_inventory(root: pathlib.Path) -> dict[str, Any]:
    callsites: list[dict[str, Any]] = []
    for path in sorted((root / "src" / "komorebi_math").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name != "Certificate":
                continue
            text = ast.get_source_segment(source, node) or ""
            if "Exactness.EXACT" not in text:
                continue
            valid_expr = ""
            if node.args:
                valid_expr = ast.get_source_segment(source, node.args[0]) or ""
            for keyword in node.keywords:
                if keyword.arg == "valid":
                    valid_expr = ast.get_source_segment(source, keyword.value) or ""
            current: ast.AST = node
            while current in parents and not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current = parents[current]
            function = current.name if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)) else "<module>"
            callsites.append(
                {
                    "path": str(path.relative_to(root)),
                    "line": node.lineno,
                    "function": function,
                    "valid_expression": valid_expr,
                    "hardcoded_true": valid_expr == "True",
                }
            )
    return {
        "exact_certificate_callsites": len(callsites),
        "source_files": len({entry["path"] for entry in callsites}),
        "hardcoded_true_callsites": sum(entry["hardcoded_true"] for entry in callsites),
        "callsites": callsites,
    }


def group_oracle(elements: tuple[int, ...], table: dict[tuple[int, int], int], identity: int) -> bool:
    element_set = set(elements)
    for left in elements:
        if table[(identity, left)] != left or table[(left, identity)] != left:
            return False
        inverses = [
            right
            for right in elements
            if table[(left, right)] == identity and table[(right, left)] == identity
        ]
        if len(inverses) != 1:
            return False
        for right in elements:
            if table[(left, right)] not in element_set:
                return False
    for a in elements:
        for b in elements:
            for c in elements:
                if table[(table[(a, b)], c)] != table[(a, table[(b, c)])]:
                    return False
    return True



def validation_bypass_trials() -> dict[str, Any]:
    finite_group = FiniteGroup([0, 1], lambda left, right: 0, 0, validate=False).certificate()
    finite_abelian = FiniteAbelianGroup(
        ("0", "1"),
        {(left, right): "0" for left in ("0", "1") for right in ("0", "1")},
        zero="0",
        validate=False,
    ).certificate()
    packet = PacketSystem([np.array([[2]], dtype=object)], backend="sympy", validate=False, tolerance=0).certificate()
    survival = PhaseTypeSurvival(np.array([[1.0]]), np.array([1.0]), validate=False).certificate()
    records = {
        "finite_group": finite_group.valid,
        "finite_abelian_group": finite_abelian.valid,
        "packet_system": packet.valid,
        "phase_type_survival": survival.valid,
    }
    return {"records": records, "all_invalid_as_expected": not any(records.values())}


def group_trials(rng: random.Random, trials: int) -> dict[str, Any]:
    failures = 0
    valid_count = 0
    invalid_count = 0
    elements = (0, 1, 2)
    for index in range(trials):
        if index == 0:
            table = {(a, b): (a + b) % 3 for a in elements for b in elements}
        else:
            table = {(a, b): rng.choice(elements) for a in elements for b in elements}
        group = FiniteGroup(elements, lambda a, b, table=table: table[(a, b)], 0, validate=False, name=f"trial-{index}")
        expected = group_oracle(elements, table, 0)
        observed = group.certificate().valid
        valid_count += int(expected)
        invalid_count += int(not expected)
        failures += int(observed != expected)
    return {"trials": trials, "oracle_valid": valid_count, "oracle_invalid": invalid_count, "failures": failures}


def exact_best_basis(costs: dict[str, float], depth: int) -> tuple[Fraction, tuple[str, ...]]:
    def solve(path: str) -> tuple[Fraction, tuple[str, ...]]:
        keep = frac(costs[path])
        if len(path) == depth:
            return keep, (path,)
        left_cost, left = solve(path + "0")
        right_cost, right = solve(path + "1")
        split = left_cost + right_cost
        if keep <= split:
            return keep, (path,)
        return split, left + right
    return solve("")


def best_basis_trials(rng: random.Random, trials: int) -> dict[str, Any]:
    failures = 0
    pool = [0.0, 0.25, 0.5, nextafter(0.5, 0.0), nextafter(0.5, 1.0), 1.0, nextafter(1.0, 0.0), 1.5, 2.0]
    tree = haar_packet_analyze(np.arange(4, dtype=float), 2)
    paths = tuple(tree.nodes)
    for _ in range(trials):
        costs = {path: rng.choice(pool) for path in paths}
        expected_cost, expected_leaves = exact_best_basis(costs, 2)
        observed = best_basis(tree, costs)
        if observed.leaves != expected_leaves:
            failures += 1
            continue
        diag = observed.certificate.diagnostics
        got = Fraction(int(diag["exact_cost_numerator"]), int(diag["exact_cost_denominator"]))
        if got != expected_cost:
            failures += 1
    return {"trials": trials, "failures": failures}


def enumerate_tree_plans(node: TreeNode) -> list[tuple[Fraction, Fraction, tuple[str, ...]]]:
    plans = [(frac(node.keep_cost), frac(node.keep_error), (node.name,))]
    if node.children:
        child_plans = [enumerate_tree_plans(child) for child in node.children]
        for choices in itertools.product(*child_plans):
            plans.append(
                (
                    frac(node.split_cost) + sum((choice[0] for choice in choices), Fraction(0)),
                    sum((choice[1] for choice in choices), Fraction(0)),
                    tuple(leaf for choice in choices for leaf in choice[2]),
                )
            )
    return plans


def random_tree(rng: random.Random, pool: list[float], prefix: str = "r") -> TreeNode:
    return TreeNode(
        prefix,
        rng.choice(pool),
        rng.choice(pool),
        rng.choice(pool),
        (
            TreeNode(prefix + "0", rng.choice(pool), rng.choice(pool)),
            TreeNode(prefix + "1", rng.choice(pool), rng.choice(pool)),
        ),
    )


def compiler_trials(rng: random.Random, trials: int) -> dict[str, Any]:
    lag_failures = 0
    budget_failures = 0
    pool = [0.0, 0.1, 0.25, 0.5, nextafter(0.5, 0.0), nextafter(0.5, 1.0), 1.0, nextafter(1.0, 0.0), 1.5]
    for index in range(trials):
        root = random_tree(rng, pool, f"r{index}")
        penalty = rng.choice(pool)
        plans = enumerate_tree_plans(root)
        p = frac(penalty)
        expected = min(plans, key=lambda item: (item[0] + p * item[1], item[0], item[1], item[2]))
        observed = compile_tree_lagrangian(root, penalty)
        if observed.leaves != expected[2] or observed.certificate.diagnostics["exact_cost"] != str(expected[0]):
            lag_failures += 1

        unit = rng.choice([0.1, 0.25, 0.5, 1.0])
        budget = rng.randrange(0, 5)
        unit_exact = frac(unit)
        feasible = []
        for cost, error, leaves in plans:
            # compile_tree_budget quantizes each selected leaf independently, so recompute
            def leaf_units(name: str) -> int:
                stack = [root]
                while stack:
                    current = stack.pop()
                    if current.name == name:
                        return ceil_fraction(frac(current.keep_error) / unit_exact)
                    stack.extend(current.children)
                raise AssertionError(name)
            used = sum(leaf_units(name) for name in leaves)
            if used <= budget:
                feasible.append((cost, used, error, leaves))
        observed_budget = compile_tree_budget(root, budget, error_unit=unit)
        if not feasible:
            if observed_budget is not None:
                budget_failures += 1
        else:
            expected_budget = min(feasible, key=lambda item: (item[0], item[1], item[2], item[3]))
            if observed_budget is None or observed_budget.leaves != expected_budget[3]:
                budget_failures += 1
            elif observed_budget.certificate.diagnostics["exact_cost"] != str(expected_budget[0]):
                budget_failures += 1
    return {"trials": trials, "lagrangian_failures": lag_failures, "budget_failures": budget_failures}


def precision_trials(rng: random.Random, trials: int) -> dict[str, Any]:
    failures = 0
    pool = [0.0, 0.25, 0.5, nextafter(0.5, 0.0), nextafter(0.5, 1.0), 1.0, nextafter(1.0, 0.0), 2.0]
    for index in range(trials):
        options: list[list[DTypeChoice]] = []
        for packet in range(3):
            options.append(
                [
                    DTypeChoice(f"p{packet}o{choice}", rng.choice(pool), rng.randrange(0, 4))
                    for choice in range(3)
                ]
            )
        budget = rng.randrange(0, 8)
        feasible = []
        for choices in itertools.product(*options):
            used = sum(choice.error_units for choice in choices)
            if used <= budget:
                feasible.append((sum((frac(choice.cost) for choice in choices), Fraction(0)), used, tuple(choice.name for choice in choices)))
        observed = assign_discrete_precisions(options, budget)
        if not feasible:
            failures += int(observed is not None)
            continue
        expected = min(feasible)
        if observed is None or (observed.names, observed.error_units) != (expected[2], expected[1]):
            failures += 1
        elif observed.certificate.diagnostics["exact_total_cost"] != str(expected[0]):
            failures += 1
    return {"trials": trials, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--trials", type=int, default=600)
    args = parser.parse_args()
    start = time.perf_counter()
    rng = random.Random(args.seed)
    result = {
        "schema": "komorebi-math.episode018-certificate-consistency-probe",
        "seed": args.seed,
        "inventory": exact_certificate_inventory(args.root),
        "validation_bypass": validation_bypass_trials(),
        "finite_group": group_trials(rng, args.trials),
        "best_basis": best_basis_trials(rng, args.trials),
        "tree_compilers": compiler_trials(rng, args.trials),
        "precision_assignment": precision_trials(rng, args.trials),
    }
    result["passed"] = (
        result["validation_bypass"]["all_invalid_as_expected"]
        and result["finite_group"]["failures"] == 0
        and result["best_basis"]["failures"] == 0
        and result["tree_compilers"]["lagrangian_failures"] == 0
        and result["tree_compilers"]["budget_failures"] == 0
        and result["precision_assignment"]["failures"] == 0
    )
    result["runtime_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "inventory"}, indent=2, sort_keys=True))
    print(json.dumps({k: result["inventory"][k] for k in ("exact_certificate_callsites", "source_files", "hardcoded_true_callsites")}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
