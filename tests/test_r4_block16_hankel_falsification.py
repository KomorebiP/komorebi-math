"""Episode 012: Hankel-rank falsification of discrete provenance state counts.

This file deliberately does not introduce a new public weighted-automaton API.
It audits existing polyphase route packets as exact linear realizations.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from komorebi_math.arithmetic.recurrence import LinearRecurrence
from komorebi_math.signals.route_packets import PolyphaseRoutePacket


def _fraction(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    return Fraction(value)  # type: ignore[arg-type]


def _evaluate(packet: PolyphaseRoutePacket, z: int | Fraction) -> list[list[Fraction]]:
    point = _fraction(z)
    return [
        [
            sum(
                (_fraction(coefficient) * point**lag for lag, coefficient in enumerate(packet.kernels[r][s])),
                Fraction(0),
            )
            for s in range(packet.order)
        ]
        for r in range(packet.order)
    ]


def _matmul(left: list[list[Fraction]], right: list[list[Fraction]]) -> list[list[Fraction]]:
    return [
        [sum((left[i][k] * right[k][j] for k in range(len(right))), Fraction(0)) for j in range(len(right[0]))]
        for i in range(len(left))
    ]


def _identity(size: int) -> list[list[Fraction]]:
    return [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]


def _sequence(matrix: list[list[Fraction]], count: int, source: int = 0, target: int = 0) -> tuple[Fraction, ...]:
    power = _identity(len(matrix))
    out: list[Fraction] = []
    for _ in range(count):
        out.append(power[target][source])
        power = _matmul(matrix, power)
    return tuple(out)


def _rank(matrix: list[list[Fraction]]) -> int:
    work = [row[:] for row in matrix]
    rows = len(work)
    columns = len(work[0]) if rows else 0
    pivot_row = 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, rows) if work[row][column] != 0), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        value = work[pivot_row][column]
        work[pivot_row] = [entry / value for entry in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row or work[row][column] == 0:
                continue
            factor = work[row][column]
            work[row] = [work[row][j] - factor * work[pivot_row][j] for j in range(columns)]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def _hankel(sequence: tuple[Fraction, ...], rows: int, columns: int) -> list[list[Fraction]]:
    return [[sequence[i + j] for j in range(columns)] for i in range(rows)]


def test_two_tap_route_has_infinite_discrete_residuals_but_rank_two() -> None:
    packet = PolyphaseRoutePacket.from_filter([1, 1], 2)
    matrix = _evaluate(packet, 2)
    assert matrix == [[Fraction(1), Fraction(2)], [Fraction(1), Fraction(1)]]

    values = _sequence(matrix, 140)
    assert values[:10] == tuple(map(Fraction, (1, 1, 3, 7, 17, 41, 99, 239, 577, 1393)))
    assert _rank(_hankel(values, 12, 12)) == 2

    # Prefix n is represented by residual k -> f(n+k).  The first 65 are
    # pairwise distinct already on the finite suffix window 0..64, hence the
    # exact discrete future quotient has at least 65 states.  The theorem in
    # the audit proves the infinite family is pairwise distinct.
    signatures = {tuple(values[n + k] for k in range(65)) for n in range(65)}
    assert len(signatures) == 65


def test_route_hankel_rank_matches_phase_dimension_for_generic_two_tap_family() -> None:
    # For h=(1,1), order N, evaluation at z=2 gives M=I+S with S^N=2I.
    # The N by N scalar Hankel block is nonsingular, so the existing N-phase
    # route state is already linearly minimal for this query.
    for order in range(2, 11):
        packet = PolyphaseRoutePacket.from_filter([1, 1], order)
        matrix = _evaluate(packet, 2)
        values = _sequence(matrix, 2 * order + 2)
        assert _rank(_hankel(values, order, order)) == order


def test_hankel_compression_recovers_existing_recurrence_coordinates() -> None:
    packet = PolyphaseRoutePacket.from_filter([1, 1], 2)
    values = _sequence(_evaluate(packet, 2), 40)
    recurrence = LinearRecurrence((1, 2), (1, 1), name="evaluated polyphase route")
    assert recurrence.first(40) == values
    assert recurrence.order == 2


def test_degenerate_phase_packet_can_reduce_by_standard_linear_realization() -> None:
    # h=(1,0) leaves only the identity transfer.  Six syntactic phase slots
    # therefore collapse to a rank-one scalar query.  This is standard
    # reachability/observability reduction, not a new packet theorem.
    packet = PolyphaseRoutePacket.from_filter([1, 0], 6)
    values = _sequence(_evaluate(packet, 3), 20)
    assert values == tuple(Fraction(1) for _ in range(20))
    assert _rank(_hankel(values, 8, 8)) == 1
