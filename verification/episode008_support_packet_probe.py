"""Deterministic Episode 008 probe for support/facet packets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from komorebi_math.nonlinear.convex_packets import polyhedral_support_packet


SEED = 20260922
TOL = 1e-10


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def hull_2d(points: np.ndarray) -> np.ndarray:
    """Return 2D convex-hull vertices in cyclic order without scipy."""
    pts = np.unique(np.round(np.asarray(points, dtype=float), 13), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 1e-13:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 1e-13:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def main() -> None:
    rng = np.random.default_rng(SEED)
    covariance_failures = 0
    bellman_failures = 0
    composition_failures = 0
    max_error = 0.0

    for _ in range(2000):
        atoms = rng.normal(size=(6, 3))
        packet = polyhedral_support_packet(atoms)
        direction = rng.normal(size=3)
        transform = rng.normal(size=(4, 3))
        image = packet.linear_image(transform)
        error = abs(image.support(rng.normal(size=4)) - image.support(np.zeros(4)))  # overwritten below
        target_direction = rng.normal(size=4)
        left = image.support(target_direction)
        right = packet.support(transform.T @ target_direction)
        error = abs(left - right) / max(1.0, abs(left), abs(right))
        max_error = max(max_error, error)
        if error > TOL:
            covariance_failures += 1

        matrices = [rng.normal(size=(3, 3)) for _ in range(3)]
        branch = packet.branch_query(matrices, direction)
        envelope = packet.bellman_envelope(matrices)
        error = abs(branch.value - envelope.support(direction)) / max(1.0, abs(branch.value))
        max_error = max(max_error, error)
        if error > TOL:
            bellman_failures += 1

        second = [rng.normal(size=(3, 3)) for _ in range(2)]
        sequential = envelope.bellman_envelope(second)
        direct_products = [a @ b for b in second for a in matrices]
        direct = packet.bellman_envelope(direct_products)
        error = abs(sequential.support(direction) - direct.support(direction)) / max(
            1.0, abs(sequential.support(direction)), abs(direct.support(direction))
        )
        max_error = max(max_error, error)
        if error > TOL:
            composition_failures += 1

    diamond = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    packet = polyhedral_support_packet(diamond)
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 1.0], [0.0, -1.0]])
    ando = {
        "x_dominant": list(packet.branch_query([first, second], np.array([2.0, 1.0])).active_branches),
        "y_dominant": list(packet.branch_query([first, second], np.array([1.0, 2.0])).active_branches),
        "tie": list(packet.branch_query([first, second], np.array([1.0, 1.0])).active_branches),
    }

    one_branch = packet.bellman_envelope([np.eye(2)])
    two_branch = packet.bellman_envelope([np.eye(2), -np.eye(2)])
    dirs = [rng.normal(size=2) for _ in range(500)]
    provenance_value_error = max(abs(one_branch.support(x) - two_branch.support(x)) for x in dirs)
    provenance_branch_sets = {
        "one": list(packet.branch_query([np.eye(2)], np.array([1.2, -0.7])).active_branches),
        "two": list(packet.branch_query([np.eye(2), -np.eye(2)], np.array([1.2, -0.7])).active_branches),
    }

    # A deterministic two-map example where raw branch histories grow as 2^k,
    # while exact convex-hull compaction keeps a much smaller semantic packet.
    matrices_2d = [
        np.array([[0.72986729, 0.66626973], [-0.65211401, 0.74571084]]),
        np.array([[0.94099071, -0.15070196], [0.14344742, 0.98857924]]),
    ]
    semantic = hull_2d(diamond)
    raw_history = np.array(diamond, copy=True)
    complexity = [{"step": 0, "raw_history_atoms": 4, "extreme_atoms": int(len(semantic)), "raw_history_hull_vertices": 4, "hull_match_error": 0.0}]
    for step in range(1, 8):
        raw = np.vstack([semantic @ matrix for matrix in matrices_2d])
        semantic = hull_2d(raw)
        raw_history = np.vstack([raw_history @ matrix for matrix in matrices_2d])
        raw_hull = hull_2d(raw_history)
        hull_match_error = 0.0
        for angle in np.linspace(0.0, 2.0 * np.pi, 361, endpoint=False):
            direction = np.array([np.cos(angle), np.sin(angle)])
            hull_match_error = max(
                hull_match_error,
                abs(float(np.max(raw_hull @ direction)) - float(np.max(semantic @ direction))),
            )
        complexity.append(
            {
                "step": step,
                "raw_history_atoms": int(len(raw_history)),
                "atoms_before_current_compaction": int(len(raw)),
                "extreme_atoms": int(len(semantic)),
                "raw_history_hull_vertices": int(len(raw_hull)),
                "hull_match_error": hull_match_error,
            }
        )

    # Future-complete value compaction check: interior atoms remain dispensable
    # after three additional Bellman steps.
    redundant = polyhedral_support_packet(np.vstack([diamond, [[0.2, 0.2], [0.0, 0.0]]]))
    compact = packet
    for _ in range(3):
        redundant = redundant.bellman_envelope(matrices_2d)
        compact = compact.bellman_envelope(matrices_2d)
    compaction_error = 0.0
    for angle in np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False):
        x = np.array([np.cos(angle), np.sin(angle)])
        compaction_error = max(compaction_error, abs(redundant.support(x) - compact.support(x)))

    result = {
        "schema": "komorebi-math.episode008-support-packet-probe",
        "seed": SEED,
        "random_trials": 2000,
        "coordinate_support_failures": covariance_failures,
        "bellman_value_failures": bellman_failures,
        "bellman_composition_failures": composition_failures,
        "maximum_relative_error": max_error,
        "ando_shih_active_branches": ando,
        "provenance_counterexample": {
            "maximum_value_difference": provenance_value_error,
            "active_branch_sets": provenance_branch_sets,
        },
        "semantic_compaction_growth": complexity,
        "future_compaction_max_support_error": compaction_error,
    }
    output = Path(__file__).with_name("EPISODE008_SUPPORT_PACKET_PROBE.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
