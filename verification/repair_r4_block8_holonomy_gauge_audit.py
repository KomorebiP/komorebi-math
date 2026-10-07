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
    from komorebi_math.exceptions import ValidationError
    from komorebi_math.nonlinear import TransportEdge, TransportGraph

    checks: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    eps = 1e-12
    tol = 1e-10
    nilpotent = np.array([[0.0, 1.0], [0.0, 0.0]])
    graph = TransportGraph(
        ["a"],
        [TransportEdge("a", "a", np.eye(2) + eps * nilpotent, "loop")],
        tolerance=tol,
    )

    coordinate = graph.gluing_certificate(gauge_regime="coordinate")
    check(coordinate.valid, "near-unipotent is within the starting coordinate tolerance")
    check(coordinate.diagnostics["gauge_regime"] == "coordinate", "coordinate regime is recorded")

    try:
        graph.gluing_certificate(gauge_regime="full_gl")
    except ValidationError:
        checks.append("numerical full-GL tolerance certificate fails closed")
    else:
        raise AssertionError("numerical full-GL certificate must fail closed")

    gauge = np.diag([1e4, 1.0])
    envelope = graph.gluing_residual_envelope(gauge)
    transformed = graph.gauge_transform({"a": gauge})
    transformed_coordinate = transformed.gluing_certificate(gauge_regime="coordinate")
    transformed_residual = transformed_coordinate.residuals["loop[0]"]
    check(not transformed_coordinate.valid, "full-GL gauge can flip a coordinate tolerance decision")
    check(
        envelope.lower_bounds["loop[0]"] <= transformed_residual <= envelope.upper_bounds["loop[0]"],
        "condition-number envelope contains the transformed residual",
    )

    theta = 0.43
    unitary = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    nonflat = TransportGraph(
        ["a"],
        [TransportEdge("a", "a", np.array([[1.1, 0.2], [-0.3, 0.95]]), "loop")],
        tolerance=1e-12,
    )
    before = nonflat.gluing_certificate(gauge_regime="unitary")
    after = nonflat.gauge_transform({"a": unitary}, gauge_regime="unitary").gluing_certificate(
        gauge_regime="unitary"
    )
    check(
        bool(np.isclose(before.residuals["loop[0]"], after.residuals["loop[0]"], rtol=1e-12, atol=1e-12)),
        "unitary gauge preserves the Frobenius residual",
    )

    try:
        nonflat.gauge_transform({"a": gauge}, gauge_regime="unitary")
    except ValidationError:
        checks.append("unitary regime rejects nonunitary gauge")
    else:
        raise AssertionError("unitary regime accepted a nonunitary gauge")

    result = {
        "schema": "komorebi-math.repair-r4-block8-holonomy-gauge-audit",
        "schema_version": 1,
        "package_version": komorebi_math.__version__,
        "passed": True,
        "checks": checks,
        "source_coordinate_residual": coordinate.residuals["loop[0]"],
        "transformed_coordinate_residual": transformed_residual,
        "condition_number": envelope.condition_number,
        "envelope": {
            "lower": envelope.lower_bounds["loop[0]"],
            "upper": envelope.upper_bounds["loop[0]"],
        },
        "nonclaims": [
            "The full-GL envelope is not a gauge-invariant distance.",
            "The numerical envelope does not constitute a validated floating-point roundoff proof.",
            "No canonical representative for general simultaneous-conjugacy orbits is constructed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
