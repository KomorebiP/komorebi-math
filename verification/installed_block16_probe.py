"""Installed-wheel probe for Episode 012 route/Hankel crosswalk."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import komorebi_math
from komorebi_math.arithmetic.recurrence import LinearRecurrence
from komorebi_math.signals.route_packets import PolyphaseRoutePacket


def evaluate(packet: PolyphaseRoutePacket, z: int) -> list[list[Fraction]]:
    return [[sum((Fraction(int(c)) * Fraction(z) ** q for q, c in enumerate(packet.kernels[r][s])), Fraction(0))
             for s in range(packet.order)] for r in range(packet.order)]


def mm(a: list[list[Fraction]], b: list[list[Fraction]]) -> list[list[Fraction]]:
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), Fraction(0)) for j in range(len(b[0]))]
            for i in range(len(a))]


def sequence(matrix: list[list[Fraction]], count: int) -> tuple[Fraction, ...]:
    n = len(matrix)
    power = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    out: list[Fraction] = []
    for _ in range(count):
        out.append(power[0][0])
        power = mm(matrix, power)
    return tuple(out)


def main() -> int:
    imported = Path(komorebi_math.__file__).resolve()
    packet = PolyphaseRoutePacket.from_filter([1, 1], 2)
    matrix = evaluate(packet, 2)
    values = sequence(matrix, 24)
    recurrence = LinearRecurrence((1, 2), (1, 1), name="installed route")
    matched = values == tuple(recurrence.first(24))
    passed = "site-packages" in str(imported) and matrix == [[Fraction(1), Fraction(2)], [Fraction(1), Fraction(1)]] and matched
    result = {
        "schema": "komorebi-math.episode012-installed-probe",
        "schema_version": 1,
        "imported_from": str(imported),
        "matrix": [[int(x) for x in row] for row in matrix],
        "recurrence_match": matched,
        "passed": passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
