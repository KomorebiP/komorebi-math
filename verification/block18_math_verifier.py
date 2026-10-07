"""Independent source-tree verification for Block 18 frame algebra."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

import komorebi_math
from komorebi_math import beta, stable
from komorebi_math.compatibility import (
    compare_public_api,
    load_public_api_manifest,
)
from komorebi_math.coordinates import (
    AnalysisSynthesisGauge,
    AnalysisSynthesisMap,
    AnalysisSynthesisSystem,
    CoefficientPartition,
    ExactAnalysisSynthesisMap,
    ExactAnalysisSynthesisSystem,
)
from komorebi_math.coordinates.exact import identity_matrix, matrix_multiply
from komorebi_math.exceptions import ValidationError
from komorebi_math.stable_semantics import run_stable_semantics_fixture


def _oblique(*, source: str = "physical") -> AnalysisSynthesisSystem:
    return AnalysisSynthesisSystem(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([[0.5, -0.5, 0.5], [-0.25, 0.75, 0.25]]),
        source=source,
        coordinates="oblique",
    )


def _canonical(rows: object, *, source: str) -> AnalysisSynthesisSystem:
    return AnalysisSynthesisSystem.canonical_dual(
        np.asarray(rows, dtype=float),
        source=source,
    )


def _check(name: str, function: Callable[[], None]) -> dict[str, object]:
    try:
        function()
        return {"name": name, "passed": True, "detail": ""}
    except Exception as exc:
        return {
            "name": name,
            "passed": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def run(root: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def version_and_boundaries() -> None:
        assert komorebi_math.__version__ == "1.0.0b2"
        assert len(stable.__all__) == 131
        assert len(beta.__all__) == 83
        names = {
            "AnalysisSynthesisGauge",
            "AnalysisSynthesisMap",
            "CoefficientPartition",
            "ExactAnalysisSynthesisSystem",
        }
        assert names.isdisjoint(stable.__all__)

    checks.append(_check("version and frozen API boundaries", version_and_boundaries))

    def oblique_gauge() -> None:
        system = _oblique()
        gauge = AnalysisSynthesisGauge(system)
        gauge.certificate().require()
        projection = gauge.coefficient_projection()
        kernel = np.array([-2.0, -1.0, 1.0])
        assert np.allclose(projection @ projection, projection)
        assert np.linalg.norm(projection - projection.T) > 0.4
        assert np.allclose(system.synthesis @ kernel, 0.0)
        value = np.array([3.0, -4.0, 5.0])
        assert gauge.gauge_equivalent(value, value + kernel)
        assert np.allclose(
            gauge.canonical_coefficients(value) + gauge.gauge_part(value),
            value,
        )

    checks.append(_check("oblique coefficient gauge", oblique_gauge))

    def map_closure() -> None:
        domain = _oblique(source="domain")
        codomain = _canonical(
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
            source="codomain",
        )
        operator = np.array([[2.0, -1.0], [3.0, 4.0]])
        mapping = AnalysisSynthesisMap(domain, codomain, operator)
        mapping.certificate().require()
        lift = mapping.coefficient_lift()
        assert np.allclose(codomain.synthesis @ lift @ domain.analysis, operator)
        kernel = np.array([-2.0, -1.0, 1.0])
        coefficient = np.array([1.0, 2.0, -1.0])
        assert np.allclose(
            mapping.apply_coefficients(coefficient + kernel),
            mapping.apply_coefficients(coefficient),
        )

    checks.append(_check("coefficient-map sandwich and quotient", map_closure))

    def bridged_composition() -> None:
        domain = _oblique(source="domain")
        middle_left = _canonical(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            source="middle",
        )
        middle_right = _canonical(
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]],
            source="middle",
        )
        codomain = _canonical(
            [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]],
            source="codomain",
        )
        first = AnalysisSynthesisMap(
            domain,
            middle_left,
            np.array([[2.0, -1.0], [3.0, 4.0]]),
        )
        second = AnalysisSynthesisMap(
            middle_right,
            codomain,
            np.array([[1.0, 2.0], [-2.0, 1.0]]),
        )
        bridge = AnalysisSynthesisGauge(middle_left).transition_to(middle_right)
        composite = first.then(second)
        assert np.allclose(
            composite.coefficient_lift(),
            second.coefficient_lift() @ bridge @ first.coefficient_lift(),
        )

    checks.append(_check("cross-presentation composition bridge", bridged_composition))

    def block_law_and_counterexample() -> None:
        system = _oblique()
        partition = CoefficientPartition(3, ((0, 2), (1,)))
        first = AnalysisSynthesisMap(
            system, system, np.array([[2.0, -1.0], [3.0, 4.0]])
        )
        second = AnalysisSynthesisMap(
            system, system, np.array([[1.0, 2.0], [-2.0, 1.0]])
        )
        first_blocks = first.blocks(partition, partition)
        second_blocks = second.blocks(partition, partition)
        first_blocks.certificate().require()
        second_blocks.certificate().require()
        target = 1
        source = 0
        block_product = sum(
            second_blocks.blocks[target][middle]
            @ first_blocks.blocks[middle][source]
            for middle in range(len(partition.groups))
        )
        lift_product = second.coefficient_lift() @ first.coefficient_lift()
        assert np.allclose(
            block_product,
            lift_product[
                np.ix_(partition.groups[target], partition.groups[source])
            ],
        )
        components = system.components()
        naive = sum(
            (components[1] @ second.operator @ components[middle])
            @ (components[middle] @ first.operator @ components[0])
            for middle in range(len(components))
        )
        expected = components[1] @ second.operator @ first.operator @ components[0]
        assert np.linalg.norm(naive - expected) > 0.8

    checks.append(_check("coefficient blocks and physical-block counterexample", block_law_and_counterexample))

    def adjoint_laws() -> None:
        domain = _canonical(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
            source="domain",
        )
        codomain = _canonical(
            [[2.0, 0.0], [0.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            source="codomain",
        )
        mapping = AnalysisSynthesisMap(
            domain, codomain, np.array([[1.0, 2.0], [-3.0, 4.0]])
        )
        lift = mapping.coefficient_lift()
        physical = mapping.adjoint().coefficient_lift()
        dual = mapping.dual_adjoint().coefficient_lift()
        left_metric = AnalysisSynthesisGauge(domain).coefficient_metric()
        right_metric = AnalysisSynthesisGauge(codomain).coefficient_metric()
        assert np.allclose(lift.T @ right_metric, left_metric @ physical)
        assert np.allclose(dual, lift.T)
        assert not np.allclose(physical, lift.T)

    checks.append(_check("physical and dual adjoints", adjoint_laws))

    def exact_closure() -> None:
        system = ExactAnalysisSynthesisSystem.canonical_dual(
            ((1, 0), (0, 1), (1, 1)),
            source="exact",
        )
        assert system.synthesis == (
            (Fraction(2, 3), Fraction(-1, 3), Fraction(1, 3)),
            (Fraction(-1, 3), Fraction(2, 3), Fraction(1, 3)),
        )
        system.certificate().require()
        projection = system.coefficient_projection()
        assert matrix_multiply(projection, projection) == projection
        first = ExactAnalysisSynthesisMap(
            system, system, ((2, -1), (3, 4))
        )
        second = ExactAnalysisSynthesisMap(
            system, system, ((1, 2), (-2, 1))
        )
        assert first.then(second).coefficient_lift() == matrix_multiply(
            second.coefficient_lift(), first.coefficient_lift()
        )
        assert matrix_multiply(system.synthesis, system.analysis) == identity_matrix(2)

    checks.append(_check("exact rational frame algebra", exact_closure))

    def threshold_regression() -> None:
        try:
            AnalysisSynthesisSystem.canonical_dual(np.diag([1.0, 9e-16]))
        except ValidationError:
            return
        raise AssertionError("near-singular canonical dual was accepted")

    checks.append(_check("consistent SVD pseudoinverse threshold", threshold_regression))

    def research_additivity() -> None:
        reference = load_public_api_manifest(
            root / "src/komorebi_math/data/public_api_1_0_0b1.json"
        )
        report = compare_public_api(reference)
        assert not report.removed
        assert not report.kind_changed
        expected = {
            (namespace, name)
            for namespace in ("komorebi_math", "komorebi_math.coordinates")
            for name in (
                "AnalysisSynthesisCoefficientBlocks",
                "AnalysisSynthesisGauge",
                "AnalysisSynthesisMap",
                "CoefficientPartition",
                "ExactAnalysisSynthesisMap",
                "ExactAnalysisSynthesisSystem",
            )
        }
        assert set(report.added) == expected

    checks.append(_check("research API additive against 1.0.0b1", research_additivity))

    def frozen_semantics() -> None:
        report = run_stable_semantics_fixture()
        report.certificate.require()
        assert len(report.results) == 10

    checks.append(_check("ten frozen stable semantics", frozen_semantics))

    passed = sum(bool(check["passed"]) for check in checks)
    return {
        "schema": "komorebi-math.block18-math-verification",
        "schema_version": 1,
        "package_version": komorebi_math.__version__,
        "passed": passed == len(checks),
        "passed_count": passed,
        "total_count": len(checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run(root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
