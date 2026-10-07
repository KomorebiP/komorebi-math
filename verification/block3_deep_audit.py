from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from komorebi_math.algorithms import (  # noqa: E402
    BlockDiagonalLowRankOperator,
    BlockDiagonalOperator,
    DTypeChoice,
    KroneckerOperator,
    PacketIR,
    PacketLayout,
    Task,
    ToeplitzOperator,
    TreeNode,
    WalshHadamardOperator,
    assign_discrete_precisions,
    block_jacobi_certificate,
    clipped_precision_allocation,
    compile_tree_budget,
    compile_tree_lagrangian,
    conjugate_gradient,
    discover_block_diagonal_low_rank,
    discover_packet_support,
    optimal_precision_tolerances,
    pareto_frontier,
    recompress_factors,
    schedule_unlimited,
    schur_fill_support,
    support_product,
    transitive_closure,
    verify_schedule,
)


@dataclass
class GroupResult:
    name: str
    checks: int
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


class Group:
    def __init__(self, name: str) -> None:
        self.name = name
        self.checks = 0
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(message)

    def close(self) -> GroupResult:
        return GroupResult(self.name, self.checks, self.failures)


def all_tree_plans(node: TreeNode):
    yield node.keep_cost, node.keep_error, (node.name,)
    if node.children:
        child_plans = [tuple(all_tree_plans(child)) for child in node.children]
        for choices in itertools.product(*child_plans):
            yield (
                node.split_cost + sum(choice[0] for choice in choices),
                sum(choice[1] for choice in choices),
                tuple(leaf for choice in choices for leaf in choice[2]),
            )


def all_quantized_tree_plans(node: TreeNode, error_unit: float):
    keep_units = int(np.ceil(node.keep_error / error_unit - 1e-14))
    yield node.keep_cost, node.keep_error, keep_units, (node.name,)
    if node.children:
        child_plans = [
            tuple(all_quantized_tree_plans(child, error_unit))
            for child in node.children
        ]
        for choices in itertools.product(*child_plans):
            yield (
                node.split_cost + sum(choice[0] for choice in choices),
                sum(choice[1] for choice in choices),
                sum(choice[2] for choice in choices),
                tuple(leaf for choice in choices for leaf in choice[3]),
            )


def random_tree(rng: np.random.Generator, depth: int, name: str = "r") -> TreeNode:
    keep_cost = float(rng.integers(1, 20)) / 3
    keep_error = float(rng.integers(0, 10)) / 10
    if depth == 0 or rng.random() < 0.3:
        return TreeNode(name, keep_cost, keep_error)
    child_count = int(rng.integers(2, 4))
    children = tuple(random_tree(rng, depth - 1, f"{name}{index}") for index in range(child_count))
    return TreeNode(name, keep_cost, keep_error, float(rng.integers(0, 5)) / 5, children)


def graph_closure_bruteforce(n: int, support: set[tuple[int, int]]) -> set[tuple[int, int]]:
    result = {(index, index) for index in range(n)} | set(support)
    changed = True
    while changed:
        changed = False
        additions = support_product(result, result)
        for edge in additions:
            if edge not in result:
                result.add(edge)
                changed = True
    return result


def run() -> dict[str, object]:
    rng = np.random.default_rng(20260815)
    results: list[GroupResult] = []

    group = Group("packet_ir")
    for _ in range(250):
        sizes = tuple(int(value) for value in rng.integers(1, 5, size=int(rng.integers(1, 6))))
        layout = PacketLayout(sizes)
        matrix = rng.normal(size=(layout.dimension, layout.dimension))
        mask = rng.random((len(layout), len(layout))) < 0.45
        for target, target_slice in enumerate(layout.slices):
            for source, source_slice in enumerate(layout.slices):
                if not mask[target, source]:
                    matrix[target_slice, source_slice] = 0
        ir = PacketIR.from_dense(matrix, layout, keep_zeros=False)
        rhs = rng.normal(size=(layout.dimension, int(rng.integers(1, 5))))
        group.check(np.allclose(ir.apply(rhs), matrix @ rhs), "PacketIR apply mismatch")
        group.check(np.allclose(ir.materialize(), matrix), "PacketIR materialization mismatch")
        squared = ir.compose(ir).materialize()
        group.check(np.allclose(squared, matrix @ matrix), "PacketIR composition mismatch")
        group.check(ir.certificate().valid, "PacketIR certificate failed")
    results.append(group.close())

    group = Group("structured_operators")
    for n in range(2, 33):
        column = rng.normal(size=n)
        from komorebi_math.algorithms import CirculantOperator

        circulant = CirculantOperator(column)
        rhs = rng.normal(size=(n, 3))
        group.check(np.allclose(circulant.apply(rhs), circulant.materialize() @ rhs), f"circulant {n}")
    for rows in range(2, 11):
        for columns in range(2, 11):
            column = rng.normal(size=rows)
            row = rng.normal(size=columns)
            row[0] = column[0]
            toeplitz = ToeplitzOperator(column, row)
            rhs = rng.normal(size=(columns, 2))
            group.check(np.allclose(toeplitz.apply(rhs), toeplitz.materialize() @ rhs), f"Toeplitz {rows}x{columns}")
    for _ in range(150):
        a = rng.normal(size=(int(rng.integers(1, 6)), int(rng.integers(1, 6))))
        b = rng.normal(size=(int(rng.integers(1, 6)), int(rng.integers(1, 6))))
        operator = KroneckerOperator(a, b)
        rhs = rng.normal(size=operator.shape[1])
        group.check(np.allclose(operator.apply(rhs), np.kron(a, b) @ rhs), "Kronecker mismatch")
    for exponent in range(1, 9):
        n = 2**exponent
        operator = WalshHadamardOperator(n, normalized=True)
        rhs = rng.normal(size=n)
        group.check(np.allclose(operator.apply(operator.apply(rhs)), rhs), f"Walsh involution {n}")
    results.append(group.close())

    group = Group("low_rank")
    for _ in range(350):
        rows = int(rng.integers(2, 18))
        columns = int(rng.integers(2, 18))
        raw_rank = int(rng.integers(1, min(rows, columns) + 1))
        left = rng.normal(size=(rows, raw_rank))
        right = rng.normal(size=(columns, raw_rank))
        retained = int(rng.integers(0, raw_rank + 1))
        result = recompress_factors(left, right, retained)
        matrix = left @ right.T
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        expected = float(singular_values[retained]) if retained < singular_values.size else 0.0
        actual = float(np.linalg.norm(matrix - result.materialize(), 2))
        group.check(abs(actual - expected) <= 1e-8 * max(1.0, expected), "recompression optimality")
        group.check(result.certificate.valid, "recompression certificate")
    for _ in range(120):
        sizes = tuple(int(value) for value in rng.integers(1, 5, size=int(rng.integers(1, 5))))
        diagonal = BlockDiagonalOperator(tuple(np.eye(size) * rng.uniform(1.5, 4.0) for size in sizes))
        rank = int(rng.integers(0, min(4, diagonal.shape[0]) + 1))
        left = rng.normal(size=(diagonal.shape[0], rank)) * 0.05
        right = rng.normal(size=(diagonal.shape[0], rank)) * 0.05
        operator = BlockDiagonalLowRankOperator.from_factors(diagonal, left, right)
        x = rng.normal(size=diagonal.shape[0])
        solved = operator.solve(operator.materialize() @ x)
        group.check(np.allclose(solved, x, atol=1e-9), "Woodbury solve mismatch")
    results.append(group.close())

    group = Group("support")
    strict_edges = tuple((target, source) for source in range(4) for target in range(4) if target != source)
    for mask in range(1 << len(strict_edges)):
        support = {edge for index, edge in enumerate(strict_edges) if mask & (1 << index)}
        actual = transitive_closure(4, support)
        expected = graph_closure_bruteforce(4, support)
        group.check(actual == expected, f"closure mask {mask}")
    for _ in range(1500):
        support = {edge for edge in strict_edges if rng.random() < 0.25}
        retained = [0, 3]
        hidden = [1, 2]
        fill = schur_fill_support(retained, hidden, support)
        brute = {(target, source) for target, source in support if target in retained and source in retained}
        for source in retained:
            for target in retained:
                for length in range(2, 6):
                    for path in itertools.product(range(4), repeat=length - 1):
                        vertices = (source,) + path + (target,)
                        if any(vertex not in hidden for vertex in vertices[1:-1]):
                            continue
                        if all((vertices[index + 1], vertices[index]) in support for index in range(len(vertices) - 1)):
                            brute.add((target, source))
                            break
        group.check(fill == brute, "Schur fill mismatch")
    results.append(group.close())

    group = Group("krylov")
    for n in range(2, 35):
        for _ in range(8):
            raw = rng.normal(size=(n, n))
            matrix = raw.T @ raw + n * np.eye(n)
            right = rng.normal(size=n)
            result = conjugate_gradient(
                lambda value, matrix=matrix: matrix @ value,
                right,
                relative_tolerance=1e-11,
                max_iterations=2 * n,
            )
            group.check(result.converged, f"CG convergence n={n}")
            group.check(np.linalg.norm(matrix @ result.solution - right) <= 1e-8, f"CG residual n={n}")
            split = int(rng.integers(1, n))
            certificate = block_jacobi_certificate(matrix, PacketLayout((split, n - split)))
            group.check(certificate.certificate.valid, "block Jacobi certificate")
    results.append(group.close())

    group = Group("precision")
    for _ in range(2000):
        count = int(rng.integers(1, 9))
        sensitivities = rng.uniform(0.1, 5.0, size=count)
        costs = rng.uniform(0.1, 5.0, size=count)
        tolerance = float(rng.uniform(0.01, 2.0))
        allocation = optimal_precision_tolerances(sensitivities, costs, tolerance)
        group.check(abs(float(np.dot(sensitivities, allocation)) - tolerance) <= 1e-10, "continuous allocation budget")
        maxima = rng.uniform(0.02, 0.8, size=count)
        clipped = clipped_precision_allocation(sensitivities, costs, tolerance, maxima)
        group.check(float(np.dot(sensitivities, clipped)) <= tolerance + 1e-9 or np.allclose(clipped, maxima), "clipped budget")
        group.check(np.all(clipped <= maxima + 1e-12), "clipped maxima")
    for _ in range(1000):
        packet_count = int(rng.integers(1, 6))
        options = []
        for packet in range(packet_count):
            options.append(
                [
                    DTypeChoice(f"{packet}a", float(rng.integers(1, 8)), int(rng.integers(0, 6))),
                    DTypeChoice(f"{packet}b", float(rng.integers(1, 8)), int(rng.integers(0, 6))),
                    DTypeChoice(f"{packet}c", float(rng.integers(1, 8)), int(rng.integers(0, 6))),
                ]
            )
        budget = int(rng.integers(0, 15))
        result = assign_discrete_precisions(options, budget)
        feasible = []
        for choices in itertools.product(*options):
            error = sum(choice.error_units for choice in choices)
            if error <= budget:
                feasible.append((sum(choice.cost for choice in choices), error, tuple(choice.name for choice in choices)))
        group.check((result is None) == (not feasible), "discrete feasibility")
        if result is not None:
            group.check((result.total_cost, result.error_units, result.names) == min(feasible), "discrete optimum")
    results.append(group.close())

    group = Group("compiler")
    for _ in range(1200):
        tree = random_tree(rng, 2)
        penalty = float(rng.uniform(0, 8))
        plan = compile_tree_lagrangian(tree, penalty)
        brute = min(
            all_tree_plans(tree),
            key=lambda item: (item[0] + penalty * item[1], item[0], item[1], item[2]),
        )
        group.check((plan.cost, plan.error, plan.leaves) == brute, "Lagrangian compiler optimum")
        unit = 0.1
        budget = int(rng.integers(0, 20))
        budget_plan = compile_tree_budget(tree, budget, error_unit=unit)
        feasible = [
            item
            for item in all_quantized_tree_plans(tree, unit)
            if item[2] <= budget
        ]
        group.check((budget_plan is None) == (not feasible), "budget compiler feasibility")
        if budget_plan is not None:
            brute_budget = min(
                feasible,
                key=lambda item: (item[0], item[2], item[1], item[3]),
            )
            same_budget_plan = (
                np.isclose(budget_plan.cost, brute_budget[0])
                and np.isclose(budget_plan.error, brute_budget[1])
                and budget_plan.quantized_error == brute_budget[2]
                and budget_plan.leaves == brute_budget[3]
            )
            group.check(same_budget_plan, "budget compiler optimum")
        frontier = pareto_frontier(tree, max_plans=50000)
        for left, right in itertools.permutations(frontier, 2):
            group.check(not (left.cost <= right.cost and left.error <= right.error and (left.cost < right.cost or left.error < right.error)), "Pareto domination")
    results.append(group.close())

    group = Group("scheduling")
    for n in range(1, 20):
        for _ in range(120):
            tasks = []
            for target in range(n):
                dependencies = tuple(str(source) for source in range(target) if rng.random() < 0.18)
                tasks.append(Task(str(target), float(rng.integers(0, 10)), dependencies))
            schedule = schedule_unlimited(tasks)
            group.check(verify_schedule(tasks, schedule), "schedule feasibility")
            finish = {}
            for task in tasks:
                finish[task.name] = task.duration + max((finish[dependency] for dependency in task.dependencies), default=0.0)
            group.check(abs(schedule.makespan - max(finish.values())) <= 1e-12, "critical path makespan")
    results.append(group.close())

    group = Group("discovery")
    for _ in range(500):
        sizes = tuple(int(value) for value in rng.integers(1, 5, size=int(rng.integers(1, 5))))
        layout = PacketLayout(sizes)
        matrix = rng.normal(size=(layout.dimension, layout.dimension))
        support_result = discover_packet_support(matrix, layout, relative_tolerance=1e-10)
        group.check(np.linalg.norm(matrix - support_result.packet_ir.materialize(), 2) <= support_result.certificate.residuals["discarded_spectral_norm"] + 1e-12, "support discovery residual")
        rank = int(rng.integers(0, layout.dimension + 1))
        bdlr = discover_block_diagonal_low_rank(matrix, layout, rank)
        residual = matrix - bdlr.operator.materialize()
        group.check(abs(np.linalg.norm(residual, 2) - bdlr.spectral_error) <= 1e-8 * max(1.0, bdlr.spectral_error), "BDLR optimality")
        group.check(bdlr.certificate.valid, "BDLR certificate")
    results.append(group.close())

    total_checks = sum(result.checks for result in results)
    failures = [failure for result in results for failure in result.failures]
    payload = {
        "passed": not failures,
        "total_checks": total_checks,
        "total_failures": len(failures),
        "groups": [asdict(result) | {"passed": result.passed} for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    outcome = run()
    raise SystemExit(0 if outcome["passed"] else 1)
