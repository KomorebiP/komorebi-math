#!/usr/bin/env python3
"""Deterministic finite audit for Episode 009 provenance packets."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

from komorebi_math.nonlinear.convex_packets import polyhedral_support_packet
from komorebi_math.nonlinear.provenance_packets import (
    compile_observer_convex_packet,
    compile_policy_packet,
    finite_provenance_observer,
)

SEED = 20260923
rng = np.random.default_rng(SEED)


def parity_observer():
    return finite_provenance_observer(
        ((2, 1), (3, 0), (0, 3), (1, 2)),
        ("even", "odd", "even", "odd"),
    )


def raw_outputs(packet, observer, direction):
    query = packet.query(direction)
    return sorted({observer.output_for_history(word) for word in query.active_provenance})


def observer_audit(trials: int = 400):
    failures = 0
    checked = 0
    reductions = []
    for _ in range(trials):
        n = 6
        transitions = rng.integers(0, n, size=(n, 2))
        outputs = tuple(int(v) for v in rng.integers(0, 3, size=n))
        observer = finite_provenance_observer(transitions, outputs)
        minimized = observer.minimize()
        reductions.append((observer.state_count, minimized.observer.state_count))
        for state in observer.reachable_states():
            mapped = minimized.old_to_new[state]
            for _ in range(12):
                length = int(rng.integers(0, 7))
                prefix = tuple(int(v) for v in rng.integers(0, 2, size=length))
                old_future = observer.advance_prefix(state, prefix)
                new_future = minimized.observer.advance_prefix(mapped, prefix)
                checked += 1
                if observer.outputs[old_future] != minimized.observer.outputs[new_future]:
                    failures += 1
    return failures, checked, reductions


def packet_audit(trials: int = 800):
    observer = parity_observer()
    value_failures = 0
    policy_failures = 0
    grouped_value_failures = 0
    grouped_policy_failures = 0
    queries = 0
    max_raw_atoms = 0
    max_compiled_atoms = 0
    for _ in range(trials):
        atoms = rng.normal(size=(5, 2))
        provenance = []
        for _i in range(5):
            length = int(rng.integers(0, 5))
            provenance.append(tuple(int(v) for v in rng.integers(0, 2, size=length)))
        raw = polyhedral_support_packet(atoms, provenance=tuple(provenance))
        compiled = compile_policy_packet(raw, observer)
        grouped = compile_observer_convex_packet(raw, observer)

        for _step in range(2):
            family = [rng.normal(size=(2, 2)), rng.normal(size=(2, 2))]
            raw = raw.bellman_envelope(family)
            compiled = compiled.bellman_envelope(family)
            grouped = grouped.bellman_envelope(family)

        max_raw_atoms = max(max_raw_atoms, raw.atom_count)
        max_compiled_atoms = max(max_compiled_atoms, compiled.atom_count)
        for _q in range(3):
            direction = rng.normal(size=2)
            if np.linalg.norm(direction) < 1e-12:
                direction[0] = 1.0
            queries += 1
            if not np.isclose(raw.support(direction), compiled.support(direction), rtol=1e-10, atol=1e-10):
                value_failures += 1
            expected_outputs = raw_outputs(raw, observer, direction)
            if expected_outputs != sorted(compiled.query(direction).active_outputs):
                policy_failures += 1
            if not np.isclose(raw.support(direction), grouped.support(direction), rtol=1e-10, atol=1e-10):
                grouped_value_failures += 1
            if expected_outputs != sorted(grouped.query(direction).active_outputs):
                grouped_policy_failures += 1
    return (value_failures, policy_failures, grouped_value_failures, grouped_policy_failures,
            queries, max_raw_atoms, max_compiled_atoms)


def exponential_compression(depth: int = 12):
    raw = polyhedral_support_packet(np.array([[1.0, 0.0]]))
    identity = np.eye(2)
    for _ in range(depth):
        raw = raw.bellman_envelope([identity, identity])
    compiled = compile_policy_packet(raw, parity_observer())
    return {
        "depth": depth,
        "raw_histories": raw.atom_count,
        "distinct_raw_histories": len(set(raw.provenance)),
        "compiled_atoms": compiled.atom_count,
        "compiled_observer_states": compiled.observer.state_count,
        "active_outputs": sorted(compiled.query(np.array([1.0, 0.0])).active_outputs),
    }


def resurrection_audit(trials: int = 500):
    observer = finite_provenance_observer(
        ((0, 1), (0, 1), (0, 1)),
        ("branch-0", "branch-1", "empty"),
        initial_state=2,
    )
    atoms = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]])
    provenance = ((0,), (0,), (0,), (0,), (1,))
    packet = compile_policy_packet(polyhedral_support_packet(atoms, provenance=provenance), observer)

    invertible_hidden_failures = 0
    checks = 0
    for _ in range(trials):
        while True:
            matrix = rng.normal(size=(2, 2))
            if abs(np.linalg.det(matrix)) > 0.1:
                break
        transformed = packet.linear_image(matrix)
        direction = rng.normal(size=2)
        if np.linalg.norm(direction) < 1e-12:
            direction[0] = 1.0
        checks += 1
        if "branch-1" in transformed.query(direction).active_outputs:
            invertible_hidden_failures += 1

    collapsed = packet.linear_image(np.zeros((2, 2)))
    resurrected = sorted(collapsed.query(np.array([1.0, 0.0])).active_outputs)
    return invertible_hidden_failures, checks, resurrected


def main():
    obs_fail, obs_checks, reductions = observer_audit()
    val_fail, pol_fail, grouped_val_fail, grouped_pol_fail, policy_checks, max_raw, max_compiled = packet_audit()
    inv_fail, inv_checks, resurrected = resurrection_audit()
    compression = exponential_compression()

    report = {
        "schema": "komorebi-math.episode009-provenance-probe",
        "schema_version": 1,
        "seed": SEED,
        "observer_minimization": {
            "trials": 400,
            "future_checks": obs_checks,
            "failures": obs_fail,
            "mean_minimized_states": float(np.mean([new for _, new in reductions])),
            "max_minimized_states": int(max(new for _, new in reductions)),
        },
        "policy_packet": {
            "trials": 800,
            "queries": policy_checks,
            "value_failures": val_fail,
            "policy_failures": pol_fail,
            "observer_convex_value_failures": grouped_val_fail,
            "observer_convex_policy_failures": grouped_pol_fail,
            "max_raw_atoms": max_raw,
            "max_compiled_atoms": max_compiled,
        },
        "exponential_compression": compression,
        "geometry_observability": {
            "invertible_trials": inv_checks,
            "interior_resurrection_failures_under_invertible_maps": inv_fail,
            "outputs_after_zero_map": resurrected,
        },
        "passed": (obs_fail == 0 and val_fail == 0 and pol_fail == 0 and grouped_val_fail == 0
                   and grouped_pol_fail == 0 and inv_fail == 0
                   and resurrected == ["branch-0", "branch-1"]),
        "non_claims": [
            "Random trials audit the implementation; they do not prove universal observer minimality or geometric observability theorems.",
            "The 2^12-to-2 compression is a parity-query example, not a universal provenance compression rate.",
            "PolicySupportPacket preserves set-valued observer outputs, not raw history identity or multiplicity.",
        ],
    }
    output = Path(__file__).with_name("EPISODE009_PROVENANCE_PROBE.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
