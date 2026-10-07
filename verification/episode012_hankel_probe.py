"""Deterministic Episode 012 Hankel/route audit.

The finite computation checks exact integer/rational instances supporting the
research decision.  It does not prove the infinite Hankel-rank theorem.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from komorebi_math.arithmetic.recurrence import LinearRecurrence
from komorebi_math.signals.route_packets import PolyphaseRoutePacket

SEED = 20260923


def frac(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    return Fraction(value)  # type: ignore[arg-type]


def evaluate(packet: PolyphaseRoutePacket, z: int) -> list[list[Fraction]]:
    point = Fraction(z)
    return [[sum((frac(c) * point**q for q, c in enumerate(packet.kernels[r][s])), Fraction(0))
             for s in range(packet.order)] for r in range(packet.order)]


def matmul(a: list[list[Fraction]], b: list[list[Fraction]]) -> list[list[Fraction]]:
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), Fraction(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def identity(n: int) -> list[list[Fraction]]:
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def seq(matrix: list[list[Fraction]], count: int) -> tuple[Fraction, ...]:
    power = identity(len(matrix))
    values: list[Fraction] = []
    for _ in range(count):
        values.append(power[0][0])
        power = matmul(matrix, power)
    return tuple(values)


def exact_rank(matrix: list[list[Fraction]]) -> int:
    work = [row[:] for row in matrix]
    rows = len(work)
    cols = len(work[0]) if rows else 0
    r = 0
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if work[i][c] != 0), None)
        if pivot is None:
            continue
        work[r], work[pivot] = work[pivot], work[r]
        pv = work[r][c]
        work[r] = [x / pv for x in work[r]]
        for i in range(rows):
            if i == r or work[i][c] == 0:
                continue
            f = work[i][c]
            work[i] = [work[i][j] - f * work[r][j] for j in range(cols)]
        r += 1
        if r == rows:
            break
    return r


def hankel(values: tuple[Fraction, ...], rows: int, cols: int) -> list[list[Fraction]]:
    return [[values[i + j] for j in range(cols)] for i in range(rows)]


def words(alphabet: int, max_length: int) -> tuple[tuple[int, ...], ...]:
    return tuple(word for length in range(max_length + 1) for word in itertools.product(range(alphabet), repeat=length))


def transpose(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    return [list(row) for row in zip(*matrix)]


def row_times(row: list[Fraction], matrix: list[list[Fraction]]) -> list[Fraction]:
    return [sum((row[k] * matrix[k][j] for k in range(len(row))), Fraction(0)) for j in range(len(row))]


def column_after(word: tuple[int, ...], transitions: list[list[list[Fraction]]], dimension: int) -> list[Fraction]:
    matrix = identity(dimension)
    for symbol in word:
        matrix = matmul(matrix, transitions[symbol])
    return [matrix[i][0] for i in range(dimension)]


def finite_word_audit(filters: list[list[int]], order: int, z: int, prefix_depth: int, suffix_depth: int) -> tuple[int, int]:
    # Transposition matches chronological route composition with the standard
    # weighted-series convention alpha M_w omega.
    transitions = [transpose(evaluate(PolyphaseRoutePacket.from_filter(h, order), z)) for h in filters]
    prefixes = words(len(filters), prefix_depth)
    suffixes = words(len(filters), suffix_depth)
    rows: list[list[Fraction]] = []
    for word in prefixes:
        state = [Fraction(int(i == 0)) for i in range(order)]
        for symbol in word:
            state = row_times(state, transitions[symbol])
        rows.append(state)
    columns = [column_after(word, transitions, order) for word in suffixes]
    block = [[sum((r[k] * c[k] for k in range(order)), Fraction(0)) for c in columns] for r in rows]
    return exact_rank(block), len({tuple(row) for row in block})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rng = random.Random(SEED)
    failures: list[str] = []

    unary = []
    for order in range(2, 11):
        packet = PolyphaseRoutePacket.from_filter([1, 1], order)
        values = seq(evaluate(packet, 2), 150)
        rank = exact_rank(hankel(values, order, order))
        residuals = len({tuple(values[n + k] for k in range(65)) for n in range(65)})
        unary.append({"order": order, "hankel_rank": rank, "sampled_distinct_residuals": residuals})
        if rank != order or residuals != 65:
            failures.append(f"unary order {order}: rank={rank}, residuals={residuals}")

    packet2 = PolyphaseRoutePacket.from_filter([1, 1], 2)
    route_values = seq(evaluate(packet2, 2), 80)
    recurrence = LinearRecurrence((1, 2), (1, 1), name="evaluated route")
    recurrence_match = tuple(recurrence.first(80)) == route_values
    if not recurrence_match:
        failures.append("order-2 recurrence crosswalk mismatch")

    random_trials = 120
    full_rank_count = 0
    lower_rank_count = 0
    max_distinct = 0
    max_separation = 0.0
    examples: list[dict[str, object]] = []
    for trial in range(random_trials):
        order = rng.randint(2, 6)
        z = rng.choice((2, 3))
        # Long, nondegenerate filters make accidental unreachable phase slots
        # unlikely; this is a finite empirical stress test only.
        filters: list[list[int]] = []
        for _ in range(2):
            length = rng.randint(order + 1, 2 * order + 2)
            kernel = [rng.randint(-2, 2) for _ in range(length)]
            if not any(kernel):
                kernel[0] = 1
            filters.append(kernel)
        rank, distinct = finite_word_audit(filters, order, z, prefix_depth=4, suffix_depth=4)
        full_rank_count += int(rank == order)
        lower_rank_count += int(rank < order)
        max_distinct = max(max_distinct, distinct)
        max_separation = max(max_separation, distinct / max(rank, 1))
        if trial < 8:
            examples.append({"order": order, "z": z, "rank": rank, "distinct_sampled_residuals": distinct, "filters": filters})
        if rank > order:
            failures.append(f"trial {trial}: Hankel rank {rank} exceeded route dimension {order}")

    degenerate = PolyphaseRoutePacket.from_filter([1, 0], 6)
    degenerate_values = seq(evaluate(degenerate, 3), 30)
    degenerate_rank = exact_rank(hankel(degenerate_values, 10, 10))
    if degenerate_rank != 1:
        failures.append(f"degenerate rank expected 1, got {degenerate_rank}")

    result = {
        "schema": "komorebi-math.episode012-hankel-falsification",
        "schema_version": 1,
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "unary_two_tap": unary,
        "route_recurrence_crosswalk": {
            "matched": recurrence_match,
            "recurrence_coefficients": [1, 2],
            "initial": [1, 1],
        },
        "random_two_symbol_route_audit": {
            "trials": random_trials,
            "full_rank_count": full_rank_count,
            "lower_rank_count": lower_rank_count,
            "max_sampled_distinct_residuals": max_distinct,
            "max_sampled_distinct_to_rank_ratio": max_separation,
            "examples": examples,
        },
        "degenerate_route": {"phase_dimension": 6, "hankel_rank": degenerate_rank},
        "failures": failures,
        "passed": not failures,
        "non_claims": [
            "Random finite trials do not prove generic full Hankel rank.",
            "Finite suffix signatures give lower bounds on discrete future-equivalence classes only.",
            "The minimal-WFA/Hankel-rank theorem is an external classical theorem, not established by this computation.",
        ],
    }
    destination = Path(__file__).with_name("EPISODE012_HANKEL_PROBE.json")
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
