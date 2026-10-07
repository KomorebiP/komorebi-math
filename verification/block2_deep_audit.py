"""Deterministic deep verification for the second komorebi-math wheel block."""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import sympy as sp

from komorebi_math import PacketSystem
from komorebi_math.nonlinear import (
    LinearLiftProblem,
    LinearObstructionProblem,
    NonlinearPacketMap,
    PolynomialLiftProblem,
    TransportEdge,
    TransportGraph,
)
from komorebi_math.refinement import (
    SmithCongruenceProblem,
    WeightedTransferEdge,
    WeightedTransferGraph,
    crt_many,
    nested_residue_analysis,
    residue_analyze,
    residue_synthesize,
)


@dataclass(frozen=True)
class AuditSummary:
    passed: bool
    checks: int
    groups: dict[str, int]
    seed: int
    failures: tuple[str, ...]


class Counter:
    def __init__(self) -> None:
        self.total = 0
        self.groups: dict[str, int] = {}
        self.failures: list[str] = []

    def check(self, group: str, condition: bool, message: str) -> None:
        self.total += 1
        self.groups[group] = self.groups.get(group, 0) + 1
        if not condition:
            self.failures.append(f"{group}: {message}")


def singleton_packets(size: int) -> PacketSystem:
    return PacketSystem(
        [sp.diag(*[1 if row == index else 0 for row in range(size)]) for index in range(size)],
        backend="sympy",
    )


def verify_residue_and_crt(counter: Counter) -> None:
    group = "residue_and_crt"
    for length in range(1, 45):
        values = np.arange(length, dtype=int) ** 3 - 2 * np.arange(length, dtype=int)
        for modulus in range(1, 9):
            coordinates = residue_analyze(values, modulus)
            reconstructed = residue_synthesize(coordinates.channels)
            counter.check(group, np.array_equal(reconstructed, values), f"round trip length={length} modulus={modulus}")
            for outer in range(1, 5):
                for inner in range(1, 5):
                    nested = nested_residue_analysis(values, outer, inner)
                    rebuilt = residue_synthesize(nested.channels)
                    counter.check(group, np.array_equal(rebuilt, values), f"nested round trip {outer}x{inner}")

    moduli = (2, 3, 5, 7)
    for residues in product(*[range(modulus) for modulus in moduli]):
        address = crt_many(residues, moduli)
        for residue, modulus in zip(residues, moduli, strict=True):
            counter.check(group, address.residue % modulus == residue, f"CRT address {residues} modulo {modulus}")


def brute_congruence_count(matrix: np.ndarray, rhs: np.ndarray, modulus: int) -> tuple[int, tuple[int, ...] | None]:
    count = 0
    first: tuple[int, ...] | None = None
    for vector in product(range(modulus), repeat=matrix.shape[1]):
        candidate = np.asarray(vector, dtype=int)
        if np.all((matrix @ candidate - rhs) % modulus == 0):
            count += 1
            if first is None:
                first = vector
    return count, first


def verify_smith(counter: Counter, rng: random.Random) -> None:
    group = "smith_congruences"
    for prime in (2, 3, 5):
        for depth in (1, 2, 3):
            modulus = prime**depth
            for coefficient in range(-6, 7):
                for rhs_value in range(-6, 7):
                    problem = SmithCongruenceProblem([[coefficient]], [rhs_value], prime=prime, depth=depth)
                    result = problem.solve()
                    brute_count, _ = brute_congruence_count(
                        np.array([[coefficient]], dtype=int), np.array([rhs_value], dtype=int), modulus
                    )
                    counter.check(group, result.solution_count == brute_count, f"1x1 count p={prime} d={depth} a={coefficient} b={rhs_value}")
                    counter.check(group, result.solvable == (brute_count > 0), "1x1 solvability")
                    if result.particular_solution is not None:
                        x = np.array(result.particular_solution, dtype=int)
                        counter.check(group, np.all((coefficient * x - rhs_value) % modulus == 0), "1x1 particular")

    for _ in range(260):
        prime = rng.choice((2, 3))
        depth = rng.choice((1, 2))
        modulus = prime**depth
        matrix = np.array([[rng.randint(-3, 3) for _ in range(2)] for _ in range(2)], dtype=int)
        rhs = np.array([rng.randint(-3, 3) for _ in range(2)], dtype=int)
        result = SmithCongruenceProblem(matrix.tolist(), rhs.tolist(), prime=prime, depth=depth).solve()
        brute_count, _ = brute_congruence_count(matrix, rhs, modulus)
        counter.check(group, result.solution_count == brute_count, f"2x2 count matrix={matrix.tolist()} rhs={rhs.tolist()}")
        counter.check(group, result.solvable == (brute_count > 0), "2x2 solvability")
        if result.particular_solution is not None:
            x = np.array(result.particular_solution, dtype=int)
            counter.check(group, np.all((matrix @ x - rhs) % modulus == 0), "2x2 particular")


def random_nonzero(rng: random.Random) -> int:
    value = 0
    while value == 0:
        value = rng.randint(-4, 4)
    return value


def verify_weighted_inverses(counter: Counter, rng: random.Random) -> None:
    group = "weighted_inverses"
    for size in range(2, 7):
        packets = singleton_packets(size)
        for _ in range(45):
            diagonal_entries = [random_nonzero(rng) for _ in range(size)]
            diagonal = sp.diag(*diagonal_entries)
            edges: list[WeightedTransferEdge] = []
            for source in range(size):
                for target in range(source + 1, size):
                    if rng.random() < 0.35:
                        operator = sp.zeros(size)
                        operator[target, source] = random_nonzero(rng)
                        edges.append(WeightedTransferEdge(source, target, rng.randint(1, 4), operator))
            graph = WeightedTransferGraph(packets, diagonal, edges, backend="sympy")
            result = graph.exact_inverse_if_acyclic()
            counter.check(group, result.certificate.valid, "acyclic inverse certificate")
            counter.check(group, result.value == graph.operator.inv(), "acyclic inverse equality")
            for source in range(size):
                for target in range(size):
                    weight = graph.minimum_path_weight(source, target)
                    if source == target:
                        counter.check(group, weight == 0, "diagonal minimum path")
                    elif math.isfinite(weight):
                        paths = graph.paths_below_weight(int(weight) + 1, source=source, target=target)
                        counter.check(group, any(path.weight == weight for path in paths), "minimum path realized")

    prime = 2
    packets = singleton_packets(4)
    for _ in range(80):
        edges = []
        for source in range(4):
            for target in range(source + 1, 4):
                if rng.random() < 0.45:
                    weight = rng.randint(1, 4)
                    operator = sp.zeros(4)
                    operator[target, source] = (prime**weight) * random_nonzero(rng)
                    edges.append(WeightedTransferEdge(source, target, weight, operator))
        graph = WeightedTransferGraph(packets, sp.eye(4), edges, backend="sympy", prime=prime)
        for precision in range(1, 7):
            expansion = graph.truncated_inverse(precision)
            counter.check(group, expansion.certificate.valid, f"adic truncation precision={precision}")


def verify_obstruction_and_lifting(counter: Counter, rng: random.Random) -> None:
    group = "obstruction_and_lifting"
    for rows in (1, 2, 3):
        for columns in (1, 2, 3):
            for _ in range(35):
                matrix = sp.Matrix([[rng.randint(-3, 3) for _ in range(columns)] for _ in range(rows)])
                defect = sp.Matrix([rng.randint(-3, 3) for _ in range(rows)])
                result = LinearObstructionProblem(matrix, defect, backend="sympy").solve()
                solvable = matrix.row_join(-defect).rank() == matrix.rank()
                counter.check(group, result.vanishes == solvable, "exact cokernel criterion")
                counter.check(group, result.cokernel_dimension == rows - matrix.rank(), "cokernel dimension")
                if result.correction is not None:
                    counter.check(group, matrix * result.correction == -defect, "exact correction")

    for prime in (2, 3, 5):
        for depth in (1, 2, 3):
            modulus = prime**depth
            next_modulus = modulus * prime
            for coefficient in range(-3, 4):
                for rhs_value in range(-3, 4):
                    for point in range(modulus):
                        if (coefficient * point - rhs_value) % modulus != 0:
                            continue
                        result = LinearLiftProblem(
                            [[coefficient]], [rhs_value], [point], prime=prime, depth=depth
                        ).step()
                        brute = tuple(
                            (point + modulus * correction) % next_modulus
                            for correction in range(prime)
                            if (coefficient * (point + modulus * correction) - rhs_value) % next_modulus == 0
                        )
                        actual = tuple(child[0] for child in result.children)
                        counter.check(group, actual == brute, f"linear lift p={prime} depth={depth}")

    for prime in (3, 5, 7):
        for constant in range(prime):
            function: Callable[[tuple[int, ...]], tuple[int, ...]] = lambda x, c=constant: (x[0] * x[0] - c,)
            jacobian = lambda x: [[2 * x[0]]]
            for root in range(prime):
                if function((root,))[0] % prime:
                    continue
                result = PolynomialLiftProblem(function, jacobian, [root], prime=prime, depth=1).step()
                brute = tuple(
                    (root + prime * correction,)
                    for correction in range(prime)
                    if function((root + prime * correction,))[0] % (prime**2) == 0
                )
                counter.check(group, result.children == brute, "polynomial lift children")


def invertible_integer_matrix(rng: random.Random, size: int = 2) -> np.ndarray:
    while True:
        matrix = np.array([[rng.randint(-3, 3) for _ in range(size)] for _ in range(size)], dtype=float)
        if abs(np.linalg.det(matrix)) > 0.5:
            return matrix


def verify_holonomy(counter: Counter, rng: random.Random) -> None:
    group = "holonomy"
    for _ in range(220):
        ab = invertible_integer_matrix(rng)
        bc = invertible_integer_matrix(rng)
        cycle = invertible_integer_matrix(rng)
        ac = bc @ ab @ cycle
        graph = TransportGraph(
            ("a", "b", "c"),
            (
                TransportEdge("a", "b", ab, "ab"),
                TransportEdge("b", "c", bc, "bc"),
                TransportEdge("a", "c", ac, "ac"),
            ),
        )
        holonomies = graph.fundamental_holonomies("a")
        counter.check(group, len(holonomies) == 1, "triangle Betti number")
        gauges = {vertex: invertible_integer_matrix(rng) for vertex in ("a", "b", "c")}
        transformed = graph.gauge_transform(gauges)
        left = transformed.fundamental_holonomies("a")[0].matrix
        right = gauges["a"] @ holonomies[0].matrix @ np.linalg.inv(gauges["a"])
        counter.check(group, np.allclose(left, right, atol=1e-8, rtol=1e-8), "gauge conjugacy")
        identical = graph.intertwine(graph, np.eye(2), root="a")
        counter.check(group, identical.certificate.valid, "identity intertwiner")

        exact_ac = bc @ ab
        flat = TransportGraph(
            ("a", "b", "c"),
            (
                TransportEdge("a", "b", ab),
                TransportEdge("b", "c", bc),
                TransportEdge("a", "c", exact_ac),
            ),
        )
        counter.check(group, flat.gluing_certificate("a").valid, "flat triangle glues")


def verify_nonlinear_coordinates(counter: Counter, rng: random.Random) -> None:
    group = "nonlinear_coordinates"
    projectors = [np.diag([1.0, 0.0, 1.0, 0.0]), np.diag([0.0, 1.0, 0.0, 1.0])]
    packets = PacketSystem(projectors)

    def function(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + x[1], x[1] * x[2], x[2] + x[3] ** 3, x[0] - x[3]])

    packet_map = NonlinearPacketMap(function, packets, packets)
    for _ in range(500):
        vector = np.array([rng.uniform(-2.0, 2.0) for _ in range(4)])
        coordinates = packets.analyze(vector)
        output = packet_map(coordinates).reconstruct()
        counter.check(group, np.allclose(output, function(vector)), "nonlinear packet reconstruction")
        counter.check(group, packet_map.certificate(vector).valid, "nonlinear coordinate certificate")


def main() -> int:
    seed = 20260815
    rng = random.Random(seed)
    counter = Counter()
    verify_residue_and_crt(counter)
    verify_smith(counter, rng)
    verify_weighted_inverses(counter, rng)
    verify_obstruction_and_lifting(counter, rng)
    verify_holonomy(counter, rng)
    verify_nonlinear_coordinates(counter, rng)
    summary = AuditSummary(
        passed=not counter.failures,
        checks=counter.total,
        groups=dict(sorted(counter.groups.items())),
        seed=seed,
        failures=tuple(counter.failures[:100]),
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
