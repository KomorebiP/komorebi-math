from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root.resolve()))

    import numpy as np
    import komorebi_math
    from komorebi_math.coordinates import AnalysisSynthesisGauge, AnalysisSynthesisSystem
    from komorebi_math.exceptions import ValidationError
    from komorebi_math.nonlinear import TransportEdge, TransportGraph

    checks: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    first = AnalysisSynthesisSystem.canonical_dual(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        source="physical",
        coordinates="first",
        tolerance=1e-10,
    )
    second = AnalysisSynthesisSystem.canonical_dual(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]]),
        source="physical",
        coordinates="second",
        tolerance=1e-10,
    )
    fg, sg = AnalysisSynthesisGauge(first), AnalysisSynthesisGauge(second)
    transition = fg.transition_to(sg)
    pullback_residual = float(
        np.linalg.norm(
            transition.conj().T @ sg.coefficient_metric() @ transition
            - fg.coefficient_metric()
        )
    )
    check(pullback_residual <= 1e-12, "frame bridge pulls back the quotient metric")

    coefficient = np.array([3.0, -4.0, 5.0])
    seminorm_error = abs(
        sg.coefficient_seminorm(transition @ coefficient)
        - fg.coefficient_seminorm(coefficient)
    )
    check(seminorm_error <= 1e-12, "frame bridge preserves the quotient seminorm")

    scale = 1e6
    G = np.diag([scale, 1.0, 1.0 / scale])
    Gi = np.linalg.inv(G)
    oblique = AnalysisSynthesisSystem(
        G @ first.analysis,
        first.synthesis @ Gi,
        source="physical",
        coordinates="reparameterized",
        tolerance=1e-10,
    )
    og = AnalysisSynthesisGauge(oblique)
    changed = G @ coefficient
    euclidean_ratio = float(np.linalg.norm(changed) / np.linalg.norm(coefficient))
    quotient_error = abs(og.coefficient_seminorm(changed) - fg.coefficient_seminorm(coefficient))
    check(euclidean_ratio > 1e5, "raw coefficient Euclidean norm can be strongly distorted")
    check(quotient_error <= 1e-12, "transported quotient metric removes the coordinate distortion")

    holonomy = np.array([[1.0, 1e-6], [0.0, 1.0]])
    graph = TransportGraph(
        ["v"],
        [TransportEdge("v", "v", holonomy, "loop")],
        tolerance=1e-12,
    )
    metric = np.array([[3.0, 0.4], [0.4, 1.5]])
    before = graph.metric_gluing_certificate(metric)
    root_gauge = np.array([[1e4, 2.0], [0.0, 1.0]])
    after_graph = graph.gauge_transform({"v": root_gauge}, gauge_regime="full_gl")
    after_metric = graph.transport_root_metric(metric, root_gauge)
    after = after_graph.metric_gluing_certificate(after_metric)
    metric_residual_error = abs(before.residuals["loop[0]"] - after.residuals["loop[0]"])
    check(metric_residual_error <= 1e-11, "metric-aware holonomy residual is full-GL coordinate covariant")

    bare_before = graph.gluing_certificate(gauge_regime="coordinate").residuals["loop[0]"]
    bare_after = after_graph.gluing_certificate(gauge_regime="coordinate").residuals["loop[0]"]
    check(bare_after > 1e3 * bare_before, "bare Frobenius residual remains coordinate dependent")

    try:
        graph.gluing_certificate(gauge_regime="full_gl")
    except ValidationError:
        checks.append("bare numerical full-GL certificate still fails closed")
    else:
        raise AssertionError("bare full-GL numerical certificate unexpectedly passed")

    result = {
        "schema": "komorebi-math.repair-r4-block9-metric-packet-audit",
        "schema_version": 1,
        "package_version": komorebi_math.__version__,
        "passed": True,
        "checks": checks,
        "frame_metric_pullback_residual": pullback_residual,
        "frame_quotient_seminorm_error": seminorm_error,
        "reparameterized_euclidean_ratio": euclidean_ratio,
        "reparameterized_quotient_error": quotient_error,
        "metric_holonomy_residual_before": before.residuals["loop[0]"],
        "metric_holonomy_residual_after": after.residuals["loop[0]"],
        "metric_holonomy_residual_error": metric_residual_error,
        "bare_coordinate_residual_before": bare_before,
        "bare_coordinate_residual_after": bare_after,
        "nonclaims": [
            "The quotient-metric identities are not claimed to be novel frame theory.",
            "A metric-aware holonomy residual is an invariant of the metric-decorated representation, not the bare conjugacy class.",
            "Metric covariance does not eliminate finite-precision conditioning of the transported metric itself.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
