"""Deterministic Episode 007 probe for convex-body packets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from komorebi_math.nonlinear.convex_packets import (
    parallelotope_norm_packet,
    quadratic_operator_norm,
)

SEED = 20260922
CASES = 2000


def _well_conditioned_map(rng: np.random.Generator, n: int) -> np.ndarray:
    q1, _ = np.linalg.qr(rng.normal(size=(n, n)))
    q2, _ = np.linalg.qr(rng.normal(size=(n, n)))
    scales = np.exp(rng.uniform(-1.0, 1.0, size=n))
    return q1 @ np.diag(scales) @ q2


def _infinity_contraction(rng: np.random.Generator, n: int) -> np.ndarray:
    matrix = rng.normal(size=(n, n))
    row_sums = np.sum(np.abs(matrix), axis=1)
    factors = rng.uniform(0.05, 1.0, size=n) / np.maximum(row_sums, 1e-15)
    return factors[:, None] * matrix


def run() -> dict[str, object]:
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 1.0], [0.0, -1.0]])
    standard = parallelotope_norm_packet(np.eye(2))
    exact_example = {
        "infinity_bounds": [standard.operator_norm(first), standard.operator_norm(second)],
        "john_quadratic_bounds": [standard.john_operator_norm(first), standard.john_operator_norm(second)],
        "quadratic_optimum": float(np.sqrt(2.0)),
        "john_factor": standard.john_factor,
    }

    # Deterministic positive-metric scan.  The exact lower bound is proved in
    # the Block 11 contract; this scan is only an implementation pressure test.
    rng = np.random.default_rng(SEED)
    scan_min = float("inf")
    for _ in range(5000):
        raw = rng.normal(size=(2, 2))
        q = raw.T @ raw + 1e-5 * np.eye(2)
        bound = max(quadratic_operator_norm(first, q), quadratic_operator_norm(second, q))
        scan_min = min(scan_min, bound)

    covariance_failures = 0
    john_failures = 0
    max_covariance_relative_error = 0.0
    max_john_ratio = 0.0
    max_convex_bound = 0.0
    for _ in range(CASES):
        n = int(rng.integers(2, 9))
        f = _well_conditioned_map(rng, n)
        b = _infinity_contraction(rng, n)
        operator = np.linalg.solve(f, b @ f)
        packet = parallelotope_norm_packet(f)
        convex = packet.operator_norm(operator)
        quadratic = packet.john_operator_norm(operator)
        max_convex_bound = max(max_convex_bound, convex)
        ratio = quadratic / max(convex, 1e-15)
        max_john_ratio = max(max_john_ratio, ratio)
        if quadratic > packet.john_factor * convex * (1.0 + 2e-10):
            john_failures += 1

        gauge = _well_conditioned_map(rng, n)
        transported = packet.transport_coordinates(gauge)
        transformed = gauge @ operator @ np.linalg.inv(gauge)
        after = transported.operator_norm(transformed)
        relative = abs(after - convex) / max(1.0, abs(convex))
        max_covariance_relative_error = max(max_covariance_relative_error, relative)
        if relative > 2e-9:
            covariance_failures += 1

    unipotent = np.array([[1.0, 1.0], [0.0, 1.0]])
    escape_rows = []
    for scale in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        packet = parallelotope_norm_packet(np.diag([scale, 1.0]), tolerance=1e-15)
        escape_rows.append(
            {
                "scale": scale,
                "operator_bound": packet.operator_norm(unipotent),
                "coordinate_condition_number": packet.coordinate_condition_number,
            }
        )

    return {
        "schema": "komorebi-math.episode007-convex-probe",
        "schema_version": 1,
        "seed": SEED,
        "random_cases": CASES,
        "exact_separation_example": exact_example,
        "random_quadratic_scan_minimum": scan_min,
        "random_parallelotope_stress": {
            "covariance_failures": covariance_failures,
            "john_bound_failures": john_failures,
            "max_covariance_relative_error": max_covariance_relative_error,
            "max_john_ratio": max_john_ratio,
            "max_convex_operator_bound": max_convex_bound,
        },
        "unipotent_convex_escape": escape_rows,
        "passed": covariance_failures == 0 and john_failures == 0,
    }


def main() -> int:
    result = run()
    path = Path(__file__).with_name("EPISODE007_CONVEX_PROBE.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
