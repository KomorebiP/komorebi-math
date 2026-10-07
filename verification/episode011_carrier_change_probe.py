from __future__ import annotations

import json
import math
from pathlib import Path
import random

from komorebi_math.core.carrier_change import (
    SignedMassPacket,
    TropicalResidualPacket,
    boolean_support_nonnegative,
)

SEED = 20260923
RNG = random.Random(SEED)


def stable_free_energy(scores: list[float], tau: float) -> float:
    if not scores:
        return -math.inf
    m = max(scores)
    return m + tau * math.log(sum(math.exp((x - m) / tau) for x in scores))


def close(a: float, b: float, tol: float = 2e-11) -> bool:
    if a == b:
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


boolean_failures = 0
for _ in range(3000):
    a = 0.0 if RNG.random() < 0.1 else RNG.random() * 20.0
    b = 0.0 if RNG.random() < 0.1 else RNG.random() * 20.0
    boolean_failures += int(
        boolean_support_nonnegative(a + b)
        != (boolean_support_nonnegative(a) or boolean_support_nonnegative(b))
    )
    boolean_failures += int(
        boolean_support_nonnegative(a * b)
        != (boolean_support_nonnegative(a) and boolean_support_nonnegative(b))
    )

signed_failures = 0
max_cancellation_ratio = 0.0
for _ in range(3000):
    left_values = [RNG.randint(-9, 9) for _ in range(RNG.randint(0, 8))]
    right_values = [RNG.randint(-9, 9) for _ in range(RNG.randint(0, 8))]
    left = SignedMassPacket.from_values(left_values)
    right = SignedMassPacket.from_values(right_values)
    merged = left.merge(right)
    multiplied = left.multiply(right)
    signed_failures += int(merged.value != sum(left_values) + sum(right_values))
    signed_failures += int(multiplied.value != sum(left_values) * sum(right_values))
    signed_failures += int(merged.cancellation_defect != merged.total_variation - abs(merged.value))
    if merged.total_variation:
        max_cancellation_ratio = max(
            max_cancellation_ratio,
            float(merged.cancellation_defect / merged.total_variation),
        )

# One explicit algebraic obstruction witness.
no_go_witness = SignedMassPacket.from_values([1, -1])
no_go_pass = bool(no_go_witness.value == 0 and no_go_witness.total_variation == 2)

tropical_failures = 0
max_free_energy_relative_error = 0.0
max_merge_relative_error = 0.0
max_product_relative_error = 0.0
max_bound_slack = 0.0
for _ in range(5000):
    tau = 10 ** RNG.uniform(-2.0, 0.6)
    left_scores = [RNG.uniform(-20.0, 20.0) for _ in range(RNG.randint(1, 7))]
    right_scores = [RNG.uniform(-20.0, 20.0) for _ in range(RNG.randint(1, 7))]
    left = TropicalResidualPacket.from_scores(left_scores, temperature=tau)
    right = TropicalResidualPacket.from_scores(right_scores, temperature=tau)

    expected_left = stable_free_energy(left_scores, tau)
    err = abs(left.free_energy - expected_left) / max(1.0, abs(expected_left))
    max_free_energy_relative_error = max(max_free_energy_relative_error, err)
    tropical_failures += int(not close(left.free_energy, expected_left))

    merged = left.merge(right)
    expected_merge = stable_free_energy(left_scores + right_scores, tau)
    err = abs(merged.free_energy - expected_merge) / max(1.0, abs(expected_merge))
    max_merge_relative_error = max(max_merge_relative_error, err)
    tropical_failures += int(not close(merged.free_energy, expected_merge))
    tropical_failures += int(merged.tropical_value != max(left.tropical_value, right.tropical_value))

    product_packet = left.multiply(right)
    pair_scores = [a + b for a in left_scores for b in right_scores]
    expected_product = stable_free_energy(pair_scores, tau)
    err = abs(product_packet.free_energy - expected_product) / max(1.0, abs(expected_product))
    max_product_relative_error = max(max_product_relative_error, err)
    tropical_failures += int(not close(product_packet.free_energy, expected_product))
    tropical_failures += int(
        not close(product_packet.tropical_value, left.tropical_value + right.tropical_value)
    )
    tropical_failures += int(
        not close(product_packet.correction, left.correction + right.correction)
    )

    slack = product_packet.correction_upper_bound - product_packet.correction
    max_bound_slack = max(max_bound_slack, slack)
    tropical_failures += int(slack < -2e-11)

# Repeated random expression trees.  Keep the raw score multiset in parallel.
expression_failures = 0
expression_cases = 1000
for _ in range(expression_cases):
    tau = 10 ** RNG.uniform(-1.5, 0.5)
    scores = [RNG.uniform(-8.0, 8.0)]
    packet = TropicalResidualPacket.from_scores(scores, temperature=tau)
    for _step in range(RNG.randint(4, 10)):
        other_scores = [RNG.uniform(-8.0, 8.0) for _ in range(RNG.randint(1, 4))]
        other = TropicalResidualPacket.from_scores(other_scores, temperature=tau)
        if RNG.random() < 0.6:
            packet = packet.merge(other)
            scores = scores + other_scores
        else:
            packet = packet.multiply(other)
            scores = [a + b for a in scores for b in other_scores]
        expected = stable_free_energy(scores, tau)
        if not close(packet.free_energy, expected, 5e-11):
            expression_failures += 1
            break
        if packet.tropical_value != max(scores):
            expression_failures += 1
            break

result = {
    "schema": "komorebi-math.episode011-carrier-change-probe",
    "schema_version": 1,
    "seed": SEED,
    "boolean_trials": 3000,
    "boolean_failures": boolean_failures,
    "signed_trials": 3000,
    "signed_failures": signed_failures,
    "max_cancellation_ratio": max_cancellation_ratio,
    "cancellation_no_go_witness_pass": no_go_pass,
    "tropical_trials": 5000,
    "tropical_failures": tropical_failures,
    "expression_tree_trials": expression_cases,
    "expression_tree_failures": expression_failures,
    "max_free_energy_relative_error": max_free_energy_relative_error,
    "max_merge_relative_error": max_merge_relative_error,
    "max_product_relative_error": max_product_relative_error,
    "max_correction_bound_slack": max_bound_slack,
    "claims": {
        "finite_assertion": "Implementation matches direct scalar semantics on the stated deterministic trials.",
        "non_claim": "Finite randomized success is not a proof of the universal algebraic theorems or a novelty result.",
    },
}

path = Path("verification/EPISODE011_CARRIER_CHANGE_PROBE.json")
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
if boolean_failures or signed_failures or tropical_failures or expression_failures or not no_go_pass:
    raise SystemExit(1)
