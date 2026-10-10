"""Run installed suites and an independent exact custom mathematical oracle.

Usage: python examples/math_stress_framework.py
No network, repository tests, optional scientific backend or pytest required.
"""

from __future__ import annotations

from math import comb

from komorebi_math.audit import CheckOutcome, StressCheck, builtin_checks, run_stress
from komorebi_math.certificates import Exactness
from komorebi_math.sequences.transforms import binomial_transform_values


def constant_sequence_oracle(context):
    """The transform of c,c,... equals c*2^n by the finite binomial theorem."""
    cases = []
    for count in range(1, context.parameters["maximum_count"] + 1):
        constant = context.rng.randrange(-(2**128), 2**128)
        actual = binomial_transform_values((constant,) * count)
        expected = tuple(constant * 2**n for n in range(count))
        if actual != expected:
            return CheckOutcome(False, {"count": count, "constant": constant}, "binomial theorem mismatch")
        cases.append({"count": count, "constant": constant, "last_term": actual[-1]})
    # A hand-checkable witness, separate from the random stream.
    if sum(comb(5, k) for k in range(6)) != 32:
        return CheckOutcome(False, {}, "binomial coefficient reference failed")
    return CheckOutcome(True, {"cases": cases})


def main() -> int:
    custom = StressCheck(
        "example.constant-binomial-v1",
        "custom",
        constant_sequence_oracle,
        Exactness.EXACT,
        {"counts": [1, 16], "coefficient_bits": 128},
        parameters={"maximum_count": 16},
    )
    report = run_stress(
        (*builtin_checks(samples=8), custom), seed=20261009, timeout_seconds=30, total_timeout_seconds=180
    )
    print(report.to_json())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
