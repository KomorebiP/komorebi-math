from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from komorebi_math.nonlinear.distortion import metric_distortion_packet, spectral_distortion_lower_bound


def unipotent_probe() -> dict:
    h = np.array([[1.0, 1.0], [0.0, 1.0]])
    rows = []
    for s in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        q = np.diag([s * s, 1.0])
        packet = metric_distortion_packet(h, q, q, tolerance=1e-16)
        rows.append({
            "s": s,
            "metric_condition": float(np.linalg.cond(q)),
            "distortion": packet.distortion,
            "distortion_minus_one": packet.distortion - 1.0,
            "scaled_product": (packet.distortion - 1.0) * np.sqrt(np.linalg.cond(q)),
        })
    return {
        "spectral_lower_bound": spectral_distortion_lower_bound(h),
        "rows": rows,
    }


def joint_word_probe(t: float = 9.0) -> dict:
    a = np.array([[0.0, -1.0], [1.0, 0.0]])
    b = np.array([[0.0, -t], [1.0 / t, 0.0]])
    qs = np.logspace(-4, 4, 4001)
    best = None
    for ratio in qs:
        q = np.diag([ratio, 1.0])
        da = metric_distortion_packet(a, q, q).distortion
        db = metric_distortion_packet(b, q, q).distortion
        objective = max(da, db)
        if best is None or objective < best[0]:
            best = (objective, ratio, da, db)
    assert best is not None
    product = a @ b
    return {
        "t": t,
        "individual_spectral_bounds": [
            spectral_distortion_lower_bound(a),
            spectral_distortion_lower_bound(b),
        ],
        "mixed_word_spectral_bound": spectral_distortion_lower_bound(product),
        "predicted_minimax": t,
        "grid_minimax": best[0],
        "grid_metric_ratio": best[1],
        "grid_distortions": [best[2], best[3]],
        "exact_metric_ratio": 1.0 / t,
    }


def curvature_probe() -> dict:
    from scipy.linalg import expm
    # Constant connection one-form A = A_x dx + A_y dy on a trivial R^3 bundle.
    # A_x and A_y are skew, so the Euclidean metric is parallel.  Their
    # commutator is nonzero, so the curvature is nonzero.
    ax = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ay = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    curvature = ax @ ay - ay @ ax
    h = 0.2
    holonomy = expm(-h * ay) @ expm(-h * ax) @ expm(h * ay) @ expm(h * ax)
    return {
        "ax_skew_residual": float(np.linalg.norm(ax.T + ax)),
        "ay_skew_residual": float(np.linalg.norm(ay.T + ay)),
        "curvature_frobenius": float(np.linalg.norm(curvature)),
        "curvature_skew_residual": float(np.linalg.norm(curvature.T + curvature)),
        "curvature": curvature.tolist(),
        "finite_loop_holonomy_residual": float(np.linalg.norm(holonomy - np.eye(3), ord="fro")),
        "finite_loop_orthogonality_residual": float(np.linalg.norm(holonomy.T @ holonomy - np.eye(3), ord="fro")),
        "finite_loop_determinant": float(np.linalg.det(holonomy)),
    }


def covariance_stress(seed: int = 20260922, cases: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    max_relative_distortion_error = 0.0
    max_submultiplicative_ratio = 0.0
    covariance_failures = 0
    composition_failures = 0

    def spd(n: int) -> np.ndarray:
        x = rng.normal(size=(n, n))
        return x.T @ x + 0.5 * np.eye(n)

    for _ in range(cases):
        n = int(rng.integers(2, 5))
        q0, q1, q2 = spd(n), spd(n), spd(n)
        t1 = rng.normal(size=(n, n)) + 2.0 * np.eye(n)
        t2 = rng.normal(size=(n, n)) + 2.0 * np.eye(n)
        while abs(np.linalg.det(t1)) < 1e-3:
            t1 += 0.1 * np.eye(n)
        while abs(np.linalg.det(t2)) < 1e-3:
            t2 += 0.1 * np.eye(n)
        d1 = metric_distortion_packet(t1, q0, q1, tolerance=1e-14).distortion
        d2 = metric_distortion_packet(t2, q1, q2, tolerance=1e-14).distortion
        dc = metric_distortion_packet(t2 @ t1, q0, q2, tolerance=1e-14).distortion
        ratio = dc / (d2 * d1)
        max_submultiplicative_ratio = max(max_submultiplicative_ratio, ratio)
        if ratio > 1.0 + 2e-9:
            composition_failures += 1

        g0 = rng.normal(size=(n, n)) + 2.0 * np.eye(n)
        g1 = rng.normal(size=(n, n)) + 2.0 * np.eye(n)
        while abs(np.linalg.det(g0)) < 1e-3:
            g0 += 0.1 * np.eye(n)
        while abs(np.linalg.det(g1)) < 1e-3:
            g1 += 0.1 * np.eye(n)
        inv0 = np.linalg.inv(g0)
        inv1 = np.linalg.inv(g1)
        tp = g1 @ t1 @ inv0
        q0p = inv0.T @ q0 @ inv0
        q1p = inv1.T @ q1 @ inv1
        dp = metric_distortion_packet(tp, q0p, q1p, tolerance=1e-14).distortion
        rel = abs(dp - d1) / max(1.0, d1)
        max_relative_distortion_error = max(max_relative_distortion_error, rel)
        if rel > 2e-8:
            covariance_failures += 1

    return {
        "seed": seed,
        "cases": cases,
        "covariance_failures": covariance_failures,
        "composition_failures": composition_failures,
        "max_relative_distortion_error": max_relative_distortion_error,
        "max_composition_ratio": max_submultiplicative_ratio,
    }


result = {
    "unipotent": unipotent_probe(),
    "joint_word": joint_word_probe(),
    "nonzero_curvature_metric_connection": curvature_probe(),
    "stress": covariance_stress(),
}
path = Path(__file__).with_name("EPISODE006_DISTORTION_PROBE.json")
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
