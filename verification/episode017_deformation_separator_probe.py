"""Deterministic exact audit of finite Frobenius trace separation."""
from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import sympy as sp

from komorebi_math.deformation import finite_frobenius_separator
from komorebi_math.exceptions import ValidationError

SEED = 20260924


def observed(rows, vector):
    return tuple(sum(Fraction(a) * Fraction(b) for a, b in zip(row, vector, strict=True)) for row in rows)


def run() -> dict:
    rng = random.Random(SEED)
    trials = 0
    rank_failures = 0
    certificate_failures = 0
    require_failures = 0
    recovery_failures = 0
    full_rank = 0
    deficient = 0

    for rational in (False, True):
        count = 1200 if not rational else 600
        for _ in range(count):
            width = rng.randint(1, 5)
            height = rng.randint(1, 7)
            rows = []
            for _row in range(height):
                values = []
                for _col in range(width):
                    numerator = rng.randint(-5, 5)
                    if rational:
                        denominator = rng.randint(1, 5)
                        values.append(Fraction(numerator, denominator))
                    else:
                        values.append(numerator)
                rows.append(tuple(values))
            traces = {101 + i: row for i, row in enumerate(rows)}
            result = finite_frobenius_separator(traces)
            exact_rank = int(sp.Matrix(rows).rank())
            expected_separating = exact_rank == width
            trials += 1

            if result.rank != exact_rank or result.separating != expected_separating:
                rank_failures += 1
            if result.certificate.valid != expected_separating:
                certificate_failures += 1

            if expected_separating:
                full_rank += 1
                try:
                    result.certificate.require()
                except Exception:
                    require_failures += 1
                vector = tuple(rng.randint(-4, 4) for _ in range(width))
                values = observed(result.trace_matrix, vector)
                try:
                    recovered = result.recover(values)
                except Exception:
                    recovery_failures += 1
                else:
                    if tuple(Fraction(x) for x in recovered) != tuple(Fraction(x) for x in vector):
                        recovery_failures += 1
            else:
                deficient += 1
                try:
                    result.certificate.require()
                except ValidationError:
                    pass
                else:
                    require_failures += 1

    explicit = finite_frobenius_separator({2: (1, 1), 3: (2, 2), 5: (3, 3)})
    explicit_require_rejected = False
    try:
        explicit.certificate.require()
    except ValidationError:
        explicit_require_rejected = True

    return {
        "schema": "komorebi-math.episode017-deformation-separator-probe",
        "schema_version": 1,
        "seed": SEED,
        "trials": trials,
        "full_rank_trials": full_rank,
        "rank_deficient_trials": deficient,
        "rank_or_status_failures": rank_failures,
        "certificate_validity_failures": certificate_failures,
        "certificate_require_failures": require_failures,
        "recovery_failures": recovery_failures,
        "explicit_rank_deficient_case": {
            "rank": explicit.rank,
            "constituent_count": explicit.constituent_count,
            "separating": explicit.separating,
            "certificate_valid": explicit.certificate.valid,
            "require_rejected": explicit_require_rejected,
        },
        "passed": not any((rank_failures, certificate_failures, require_failures, recovery_failures)) and explicit_require_rejected,
        "non_claims": [
            "This finite computation does not prove any Chebotarev-density or global Galois-representation statement.",
            "Random tables audit implementation consistency; the full-column-rank uniqueness theorem is proved separately by linear algebra.",
        ],
    }


if __name__ == "__main__":
    report = run()
    destination = Path("verification/EPISODE017_DEFORMATION_SEPARATOR_PROBE.json")
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)
