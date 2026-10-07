from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from komorebi_math.arithmetic import MahlerSeries, TruncatedPadic, UnitModeMahler, padic_valuation
from komorebi_math.generating import TruncatedPadicDomain

SEED = 20260923


def capped_exact(value: int, prime: int, precision: int) -> int:
    v = padic_valuation(value, prime)
    return precision if v == float("inf") else min(int(v), precision)


def run() -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    valuation_failures: list[dict[str, object]] = []
    generating_failures: list[dict[str, object]] = []
    refinement_failures: list[dict[str, object]] = []
    valuation_trials = 0

    for _ in range(6000):
        p = int(rng.choice([2, 3, 5, 7, 11]))
        n = int(rng.integers(1, 8))
        modulus = p**n
        value = int(rng.integers(-20 * modulus, 20 * modulus + 1))
        # Force exact multiples of the modulus regularly.
        if valuation_trials % 7 == 0:
            value = int(rng.integers(-20, 21)) * modulus
        expected = capped_exact(value, p, n)
        got = TruncatedPadic(p, n).valuation(value)
        got_generating = TruncatedPadicDomain(p, n).valuation(value)
        valuation_trials += 1
        if got != expected:
            valuation_failures.append({"p": p, "precision": n, "value": value, "got": got, "expected": expected})
        if got_generating != expected:
            generating_failures.append({"p": p, "precision": n, "value": value, "got": got_generating, "expected": expected})
        m = n + int(rng.integers(0, 5))
        coarse = TruncatedPadic(p, n).valuation(value)
        fine = TruncatedPadic(p, m).valuation(value)
        if coarse != min(int(fine), n):
            refinement_failures.append({"p": p, "coarse_precision": n, "fine_precision": m, "value": value, "coarse": coarse, "fine": fine})

    mahler_failures: list[dict[str, object]] = []
    certificate_contract_failures: list[dict[str, object]] = []
    mahler_trials = 0
    degenerate_low_degree_valid = 0
    nondegenerate_undertruncated_invalid = 0
    for _ in range(6000):
        base = int(rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]))
        jump = int(rng.integers(1, 6))
        residue = int(rng.integers(0, 6))
        maximum = int(rng.integers(0, 13))
        degree = int(rng.integers(0, 13))
        mode = UnitModeMahler(base=base, jump=jump, residue=residue)
        truncation = MahlerSeries(mode.coefficients(degree))
        expected_valid = True
        expected_first = None
        for x in range(maximum + 1):
            if truncation.evaluate(x) != base ** (jump * x + residue):
                expected_valid = False
                expected_first = x
                break
        certificate = mode.certificate(maximum, degree=degree)
        mahler_trials += 1
        if certificate.valid != expected_valid or certificate.diagnostics.get("first_failure") != expected_first:
            mahler_failures.append({
                "base": base, "jump": jump, "residue": residue, "maximum": maximum, "degree": degree,
                "got_valid": certificate.valid, "expected_valid": expected_valid,
                "got_first": certificate.diagnostics.get("first_failure"), "expected_first": expected_first,
            })
        if degree < maximum and mode.increment != 0:
            nondegenerate_undertruncated_invalid += int(not certificate.valid)
            if certificate.valid or expected_first != degree + 1:
                certificate_contract_failures.append({
                    "kind": "nondegenerate_undertruncation", "base": base, "jump": jump,
                    "maximum": maximum, "degree": degree, "increment": mode.increment,
                    "valid": certificate.valid, "first_failure": certificate.diagnostics.get("first_failure"),
                })
        if degree < maximum and mode.increment == 0:
            degenerate_low_degree_valid += int(certificate.valid)
            if not certificate.valid:
                certificate_contract_failures.append({
                    "kind": "zero_increment_should_be_exact", "base": base, "jump": jump,
                    "maximum": maximum, "degree": degree,
                })

    record: dict[str, object] = {
        "schema": "komorebi-math.episode015-arithmetic-semantic-probe",
        "schema_version": 1,
        "seed": SEED,
        "valuation": {
            "trials": valuation_trials,
            "arithmetic_failures": len(valuation_failures),
            "generating_failures": len(generating_failures),
            "refinement_failures": len(refinement_failures),
            "sample_failures": (valuation_failures + generating_failures + refinement_failures)[:5],
        },
        "mahler": {
            "trials": mahler_trials,
            "certificate_failures": len(mahler_failures),
            "structural_contract_failures": len(certificate_contract_failures),
            "nondegenerate_undertruncated_invalid_count": nondegenerate_undertruncated_invalid,
            "degenerate_low_degree_valid_count": degenerate_low_degree_valid,
            "sample_failures": (mahler_failures + certificate_contract_failures)[:5],
        },
        "non_claims": [
            "Randomized exact-integer trials do not prove the universal theorems.",
            "The probe does not audit all arithmetic modules or all p-adic precision conventions.",
        ],
    }
    record["passed"] = all(
        value == 0
        for value in (
            len(valuation_failures), len(generating_failures), len(refinement_failures),
            len(mahler_failures), len(certificate_contract_failures),
        )
    )
    return record


if __name__ == "__main__":
    result = run()
    output = Path(__file__).resolve().parents[1] / "verification" / "EPISODE015_ARITHMETIC_SEMANTIC_PROBE.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
