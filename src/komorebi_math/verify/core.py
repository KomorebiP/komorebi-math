"""Lightweight installed package verification for the finite core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..certificates import Certificate, Exactness
from ..core import (
    OperatorGrammar,
    OperatorRealization,
    PacketOperator,
    PacketSystem,
    SeedSystem,
    TransferGraph,
    block_from_paths,
)


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    """One named verification outcome with an optional diagnostic detail."""


    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Aggregate report for a finite collection of verification checks."""


    checks: tuple[VerificationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(check.passed for check in self.checks)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    def certificate(self) -> Certificate:
        return Certificate(
            valid=self.passed,
            exactness=Exactness.NUMERICAL,
            assumptions_checked=tuple(check.name for check in self.checks if check.passed),
            diagnostics={
                "passed": self.passed_count,
                "total": self.total_count,
                "failures": tuple(check.name for check in self.checks if not check.passed),
            },
            message=(
                f"verification passed {self.passed_count}/{self.total_count} checks"
                if self.passed
                else f"verification failed {self.total_count - self.passed_count} checks"
            ),
        )


def _capture(name: str, function: Callable[[], None]) -> VerificationCheck:
    try:
        function()
        return VerificationCheck(name, True)
    except Exception as exc:
        return VerificationCheck(name, False, f"{type(exc).__name__}: {exc}")


def run_core_verification() -> VerificationReport:
    """Run the installed finite core verification suite."""

    checks: list[VerificationCheck] = []

    def grammar_check() -> None:
        grammar = OperatorGrammar(["X"], {"T": ("X", "X")})
        t = grammar.word("T")
        t3 = grammar.word("T", "T", "T")
        grammar.relation("cubic", [(1, t3), (-1, t)])
        matrix = np.diag([-1.0, 0.0, 1.0])
        realization = OperatorRealization(grammar, {"X": 3}, {"T": matrix})
        realization.relation_certificate().require()

    checks.append(_capture("typed grammar relations", grammar_check))

    def seed_check() -> None:
        shift = np.roll(np.eye(5), 1, axis=0)
        seed = np.array([1, 0, 0, 0, 0], dtype=float)
        system = SeedSystem([shift], [seed])
        assert system.hull_dimension() == 5
        system.generation_certificate().require()

    checks.append(_capture("finite seed hull", seed_check))

    def packet_check() -> None:
        p0 = np.diag([1, 1, 0, 0])
        p1 = np.diag([0, 0, 1, 1])
        packets = PacketSystem([p0, p1], names=["left", "right"])
        vector = np.array([1, 2, 3, 4], dtype=float)
        coordinates = packets.analyze(vector)
        assert np.allclose(coordinates.reconstruct(), vector)
        packets.certificate().require()

    checks.append(_capture("packet analysis and synthesis", packet_check))

    def block_check() -> None:
        p0 = np.diag([1, 1, 0, 0])
        p1 = np.diag([0, 0, 1, 1])
        packets = PacketSystem([p0, p1], names=["left", "right"])
        operator = np.arange(16, dtype=float).reshape(4, 4)
        blocked = PacketOperator.from_operator(operator, packets)
        assert np.allclose(blocked.reconstruct(), operator)
        blocked.certificate().require()

    checks.append(_capture("packet operator reconstruction", block_check))

    def path_check() -> None:
        projectors = [np.diag([1, 0, 0]), np.diag([0, 1, 0]), np.diag([0, 0, 1])]
        packets = PacketSystem(projectors)
        operator = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], dtype=float)
        blocked = PacketOperator.from_operator(operator, packets)
        result = block_from_paths(blocked, source=0, target=2, length=2)
        result.certificate.require()
        assert np.allclose(result.value, blocked.power(2).blocks[2][0])

    checks.append(_capture("packet path expansion", path_check))

    def graph_check() -> None:
        adjacency = [[False, True, False], [False, False, True], [False, False, False]]
        graph = TransferGraph(adjacency)
        assert graph.reachable([0]) == frozenset({0, 1, 2})
        assert graph.longest_path_vertices() == 3
        assert graph.source_components() == ((0,),)

    checks.append(_capture("transfer graph algorithms", graph_check))

    return VerificationReport(tuple(checks))
