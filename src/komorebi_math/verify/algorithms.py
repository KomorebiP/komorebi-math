"""Installed package verification for the structured algorithm block."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..algorithms import (
    BlockDiagonalLowRankOperator,
    BlockDiagonalOperator,
    DTypeChoice,
    PacketIR,
    PacketLayout,
    Task,
    TreeNode,
    assign_discrete_precisions,
    compile_tree_lagrangian,
    conjugate_gradient,
    discover_block_diagonal_low_rank,
    packet_executor,
    recompress_factors,
    schedule_unlimited,
    schur_fill_support,
)
from .core import VerificationCheck, VerificationReport


def _capture(name: str, function: Callable[[], None]) -> VerificationCheck:
    try:
        function()
        return VerificationCheck(name, True)
    except Exception as exc:
        return VerificationCheck(name, False, f"{type(exc).__name__}: {exc}")


def run_algorithms_verification() -> VerificationReport:
    """Run the installed structured algorithm verification suite."""

    checks: list[VerificationCheck] = []

    def ir_check() -> None:
        layout = PacketLayout((1, 2))
        ir = PacketIR.square(layout, {(0, 0): [[2.0]], (1, 1): np.eye(2)})
        x = np.array([1.0, 2.0, 3.0])
        assert np.allclose(ir.apply(x), ir.materialize() @ x)
        ir.certificate().require()

    checks.append(_capture("packet intermediate representation", ir_check))

    def low_rank_check() -> None:
        rng = np.random.default_rng(801)
        left = rng.normal(size=(8, 4))
        right = rng.normal(size=(7, 4))
        result = recompress_factors(left, right, 2)
        result.certificate.require()
        assert np.isclose(
            np.linalg.norm(left @ right.T - result.materialize(), 2),
            result.spectral_error,
        )

    checks.append(_capture("optimal low rank recompression", low_rank_check))

    def bdlr_check() -> None:
        diagonal = BlockDiagonalOperator((2 * np.eye(2), 3 * np.eye(1)))
        left = np.array([[0.1], [0.2], [0.3]])
        right = np.array([[0.2], [0.1], [0.4]])
        operator = BlockDiagonalLowRankOperator.from_factors(diagonal, left, right)
        x = np.array([1.0, 2.0, 3.0])
        assert np.allclose(operator.solve(operator.materialize() @ x), x)

    checks.append(_capture("Woodbury structured solve", bdlr_check))

    def support_check() -> None:
        fill = schur_fill_support(
            [0, 3],
            [1, 2],
            {(1, 0), (2, 1), (3, 2)},
        )
        assert (3, 0) in fill

    checks.append(_capture("Schur fill support", support_check))

    def krylov_check() -> None:
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        right = np.array([1.0, 2.0])
        result = conjugate_gradient(lambda value: matrix @ value, right, max_iterations=4)
        assert result.converged
        assert np.allclose(result.solution, np.linalg.solve(matrix, right))

    checks.append(_capture("conjugate gradient", krylov_check))

    def precision_check() -> None:
        result = assign_discrete_precisions(
            [
                [DTypeChoice("a", 1.0, 2), DTypeChoice("A", 2.0, 0)],
                [DTypeChoice("b", 1.0, 2), DTypeChoice("B", 2.0, 0)],
            ],
            2,
        )
        assert result is not None
        result.certificate.require()

    checks.append(_capture("discrete precision assignment", precision_check))

    def compiler_check() -> None:
        tree = TreeNode(
            "root",
            5.0,
            0.0,
            0.5,
            (TreeNode("left", 1.0, 0.2), TreeNode("right", 1.0, 0.2)),
        )
        plan = compile_tree_lagrangian(tree, 3.0)
        plan.certificate.require()
        assert plan.decision("root") == "split"

    checks.append(_capture("exact finite tree compiler", compiler_check))

    def scheduling_check() -> None:
        schedule = schedule_unlimited(
            [
                Task("a", 2.0),
                Task("b", 1.0),
                Task("c", 3.0, ("a", "b")),
            ]
        )
        assert schedule.makespan == 5.0

    checks.append(_capture("task graph scheduling", scheduling_check))

    def discovery_check() -> None:
        layout = PacketLayout((2, 1))
        matrix = np.diag([1.0, 2.0, 3.0])
        result = discover_block_diagonal_low_rank(matrix, layout, 0)
        result.certificate.require()
        assert np.allclose(result.operator.materialize(), matrix)

    checks.append(_capture("fixed layout structure discovery", discovery_check))

    def executor_check() -> None:
        layout = PacketLayout((1,))
        ir = PacketIR.square(layout, {(0, 0): [[3.0]]})
        result = packet_executor(ir).apply(np.array([2.0]))
        result.certificate.require()
        assert np.allclose(result.value, [6.0])

    checks.append(_capture("NumPy packet executor", executor_check))

    return VerificationReport(tuple(checks))
