"""Deterministic tour of Komorebi Math.

The tour deliberately uses finite, inexpensive examples that span the stable
surface and the subject-specific integration and expression interfaces.  It is intended as the
first executable a reviewer runs after installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from typing import Any

import numpy as np

from measure_integration import run as run_integration_example
from typed_expression import build_record as run_expression_example
from komorebi_math import __version__
from komorebi_math.coordinates import AnalysisSynthesisSystem, euler_fibonacci_coordinates
from komorebi_math.export.persistence import dumps_portable, loads_portable
from komorebi_math.refinement import PolyphaseRefinement
from komorebi_math.signals import polyphase_analyze
from komorebi_math.stable import create_stable_api_manifest


def _json_value(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _canonical_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_record() -> dict[str, Any]:
    """Execute the deterministic public tour and return one evidence record."""

    signal = np.arange(29, dtype=np.int64)
    coarse = polyphase_analyze(signal, 4)
    refinement = PolyphaseRefinement(4, 12)
    fine = refinement.refine(coarse)
    refinement.certificate(coarse).require()
    assert np.array_equal(fine.reconstruct(), signal)
    assert np.array_equal(refinement.coarsen(fine).reconstruct(), signal)

    analysis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ]
    )
    system = AnalysisSynthesisSystem.canonical_dual(
        analysis,
        source="two-dimensional signal",
        coordinates="four redundant coordinates",
    )
    system.certificate().require()
    value = np.array([3.0, -2.0])
    assert np.allclose(system.reconstruct(value), value)
    restored = loads_portable(dumps_portable(system))
    restored.certificate().require()
    assert np.allclose(restored.reconstruct(value), value)

    coordinates = euler_fibonacci_coordinates(18)
    coordinates.certificate().require()
    reconstructed = coordinates.reconstruct()
    assert len(reconstructed) == 18

    integration = run_integration_example()
    assert integration["passed"] is True
    assert integration["exact_integral"] == Fraction(106, 5)
    assert integration["adversarial_guard"]["false_exactness_certificate_blocked"] is True

    expression = run_expression_example()
    assert expression["pipeline_is_I4"] is True
    assert expression["result"] == Fraction(5, 4)
    assert expression["portable"]["roundtrip_equal_normal_form"] is True

    stable_manifest = create_stable_api_manifest()
    assert len(stable_manifest.symbols) == 131

    record: dict[str, Any] = {
        "schema": "komorebi-math.library-tour",
        "schema_version": 1,
        "package_version": __version__,
        "passed": True,
        "capabilities": {
            "polyphase_refinement": {
                "coarse_order": 4,
                "fine_order": 12,
                "signal_length": int(signal.size),
                "roundtrip": True,
            },
            "analysis_synthesis": {
                "source_dimension": system.source_dimension,
                "coordinate_dimension": system.coordinate_dimension,
                "redundant": system.redundant,
                "portable_roundtrip": True,
            },
            "euler_fibonacci_coordinates": {
                "length": len(reconstructed),
                "initial": tuple(str(item) for item in coordinates.initial),
                "defect_count": len(coordinates.defects),
                "reconstruction_verified": True,
            },
            "finite_integration": {
                "exact_integral": integration["exact_integral"],
                "coordinate_families": tuple(sorted(integration["coordinate_pairings"])),
                "adversarial_false_certificate_blocked": True,
            },
            "typed_expression": {
                "exact_integral": expression["result"],
                "pipeline_is_identity": expression["pipeline_is_I4"],
                "portable_roundtrip": expression["portable"]["roundtrip_equal_normal_form"],
                "optimal_chain_cost": expression["execution_plan"]["chain_optimal_cost"],
                "left_associative_cost": expression["execution_plan"][
                    "chain_left_associative_cost"
                ],
            },
            "stable_contract": {
                "symbol_count": len(stable_manifest.symbols),
                "manifest_digest": stable_manifest.digest,
            },
        },
    }
    record["record_sha256"] = _canonical_digest(record)
    return _json_value(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--output", help="write the JSON record to this path")
    arguments = parser.parse_args(argv)
    record = build_record()
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        from pathlib import Path

        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
    if arguments.json or not arguments.output:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
