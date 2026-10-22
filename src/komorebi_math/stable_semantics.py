"""Closed semantic regression fixture for the stable 1.0 API.

Structural API manifests detect missing names and signature changes.  They do
not detect an implementation that retains the same signature while changing a
mathematical law.  This module executes a small closed set of exact examples
through the curated stable namespace and compares canonical observed values
with committed expected values.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .certificates import Certificate, Exactness
from .exceptions import ValidationError

STABLE_SEMANTICS_SCHEMA = "komorebi-math.stable-semantics"
STABLE_SEMANTICS_SCHEMA_VERSION = 1
DEFAULT_STABLE_SEMANTICS_FIXTURE = (
    Path(__file__).with_name("data") / "stable_semantics_1_0.json"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def _scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (bool, type(None), str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValidationError("stable semantic fixtures cannot contain nonfinite values")
        return number
    return str(value)


def _vector(values: Any) -> list[Any]:
    return [_scalar(value) for value in values]


@dataclass(frozen=True, slots=True)
class StableSemanticCaseResult:
    """Observed and expected canonical values for one semantic law."""

    name: str
    kind: str
    expected: Any
    observed: Any
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "passed": self.passed,
            "expected": self.expected,
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class StableSemanticReport:
    """Aggregate result for the committed stable semantic fixture."""

    source: Path
    fixture_digest: str
    results: tuple[StableSemanticCaseResult, ...]
    certificate: Certificate

    @property
    def passed(self) -> bool:
        return self.certificate.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "fixture_digest": self.fixture_digest,
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
            "certificate": self.certificate.to_dict(),
        }


def load_stable_semantics_fixture(
    path: str | Path | None = None,
) -> tuple[Path, Mapping[str, Any], str]:
    """Load, validate, and digest one stable semantic fixture."""

    source = DEFAULT_STABLE_SEMANTICS_FIXTURE if path is None else Path(path)
    try:
        raw_text = source.read_text(encoding="utf-8")
        raw = json.loads(raw_text)
    except OSError as exc:
        raise ValidationError(f"could not read stable semantic fixture {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError("stable semantic fixture is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError("stable semantic fixture root must be an object")
    if set(raw) != {"schema", "schema_version", "cases"}:
        raise ValidationError("stable semantic fixture fields do not match the schema")
    if raw["schema"] != STABLE_SEMANTICS_SCHEMA:
        raise ValidationError("unknown stable semantic fixture schema")
    if int(raw["schema_version"]) != STABLE_SEMANTICS_SCHEMA_VERSION:
        raise ValidationError("unsupported stable semantic fixture schema version")
    cases = raw["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValidationError("stable semantic fixture must contain a nonempty case list")
    names: list[str] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValidationError("stable semantic cases must be objects")
        if not {"name", "kind", "input", "expected"} <= set(case):
            raise ValidationError("stable semantic case is missing required fields")
        name = str(case["name"]).strip()
        kind = str(case["kind"]).strip()
        if not name or not kind:
            raise ValidationError("stable semantic case names and kinds must be nonempty")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValidationError("stable semantic case names must be unique")
    digest = sha256(_canonical_json(raw).encode("utf-8")).hexdigest()
    return source, raw, digest


def _result(case: Mapping[str, Any], observed: Any) -> StableSemanticCaseResult:
    expected = case["expected"]
    passed = _canonical_json(expected) == _canonical_json(observed)
    return StableSemanticCaseResult(
        name=str(case["name"]),
        kind=str(case["kind"]),
        expected=expected,
        observed=observed,
        passed=passed,
        detail="" if passed else "canonical observed value differs from the committed expectation",
    )



def _run_analysis_synthesis(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import AnalysisSynthesisSystem

    spec = case["input"]
    analysis = np.asarray(spec["analysis"], dtype=float)
    value = np.asarray(spec["value"], dtype=float)
    operator = np.asarray(spec["operator"], dtype=float)
    groups = tuple(tuple(int(index) for index in group) for group in spec["groups"])
    system = AnalysisSynthesisSystem.canonical_dual(analysis)
    reconstructed = np.round(system.reconstruct(value), 12)
    components = system.components(groups)
    blocks = system.operator_blocks(operator, groups=groups)
    observed = {
        "certificate_valid": system.certificate().valid,
        "redundant": system.redundant,
        "reconstructed": _vector(reconstructed.tolist()),
        "component_sum_close": bool(
            np.allclose(sum(components), np.eye(system.source_dimension), atol=1e-12, rtol=1e-12)
        ),
        "operator_reconstructed": bool(
            np.allclose(blocks.reconstruct(), operator, atol=1e-12, rtol=1e-12)
        ),
        "operator_certificate_valid": blocks.certificate().valid,
    }
    return _result(case, observed)

def _run_signal_polyphase(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import PolyphaseRefinement, signal_polyphase_analyze

    spec = case["input"]
    values = np.asarray(spec["values"], dtype=np.int64)
    coarse = int(spec["coarse"])
    fine = int(spec["fine"])
    coordinates = signal_polyphase_analyze(values, coarse)
    refinement = PolyphaseRefinement(coarse, fine)
    refined = refinement.refine(coordinates)
    direct = signal_polyphase_analyze(values, fine)
    restored = refinement.coarsen(refined).reconstruct()
    observed = {
        "children": [list(refinement.children(index)) for index in range(coarse)],
        "direct_match": all(
            np.array_equal(left, right)
            for left, right in zip(refined.channels, direct.channels, strict=True)
        ),
        "reconstructed": _vector(restored.tolist()),
        "certificate_valid": refinement.certificate(coordinates).valid,
    }
    return _result(case, observed)


def _run_arithmetic_polyphase(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import PolyphaseRefinement, arithmetic_polyphase_analyze

    spec = case["input"]
    values = tuple(_fraction(value) for value in spec["values"])
    coarse = int(spec["coarse"])
    fine = int(spec["fine"])
    coordinates = arithmetic_polyphase_analyze(values, coarse)
    refinement = PolyphaseRefinement(coarse, fine)
    refined = refinement.refine(coordinates)
    direct = arithmetic_polyphase_analyze(values, fine)
    observed = {
        "direct_match": refined == direct,
        "reconstructed": _vector(refined.reconstruct()),
        "coarsened": _vector(refinement.coarsen(refined).reconstruct()),
        "certificate_valid": refinement.certificate(coordinates).valid,
    }
    return _result(case, observed)


def _run_lattice_polyphase(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import LatticePolyphaseRefinement, lattice_polyphase_analyze

    spec = case["input"]
    shape = tuple(int(value) for value in spec["shape"])
    coarse = tuple(int(value) for value in spec["coarse"])
    fine = tuple(int(value) for value in spec["fine"])
    values = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    coordinates = lattice_polyphase_analyze(values, coarse)
    refinement = LatticePolyphaseRefinement(coarse, fine)
    refined = refinement.refine(coordinates)
    direct = lattice_polyphase_analyze(values, fine)
    observed = {
        "channel_count": len(refined.channels),
        "addresses": [list(address) for address in sorted(refined.channels)],
        "direct_match": set(refined.channels) == set(direct.channels)
        and all(np.array_equal(refined.channels[key], direct.channels[key]) for key in direct.channels),
        "roundtrip": np.array_equal(refinement.coarsen(refined).reconstruct(), values),
        "certificate_valid": refinement.certificate(coordinates).valid,
    }
    return _result(case, observed)


def _run_operator_grammar(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import OperatorGrammar, OperatorRealization

    spec = case["input"]
    diagonal = np.asarray(spec["diagonal"], dtype=float)
    grammar = OperatorGrammar(["X"], {"T": ("X", "X")})
    t = grammar.word("T")
    grammar.relation("idempotent", [(1, grammar.word("T", "T")), (-1, t)])
    realization = OperatorRealization(grammar, {"X": len(diagonal)}, {"T": np.diag(diagonal)})
    certificate = realization.relation_certificate()
    observed = {
        "valid": certificate.valid,
        "relation_count": len(grammar.relations),
        "maximum_residual": max(certificate.residuals.values(), default=0.0),
    }
    return _result(case, observed)


def _run_coordinate_basis(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import basis_from_matrix, standard_basis

    spec = case["input"]
    source = standard_basis(len(spec["vector"]), name="source")
    target = basis_from_matrix("target", spec["target_matrix"])
    vector = tuple(_fraction(value) for value in spec["vector"])
    transform = source.connection_to(target)
    source_coordinates = source.analyze(vector)
    target_coordinates = target.analyze(vector)
    observed = {
        "matrix": [_vector(row) for row in transform.matrix],
        "source_coordinates": _vector(source_coordinates),
        "target_coordinates": _vector(target_coordinates),
        "roundtrip": transform.inverse().apply(target_coordinates) == source_coordinates,
        "certificate_valid": transform.certificate().valid,
    }
    return _result(case, observed)


def _run_normalized_series(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import FormalSeries, factorial_power_basis

    spec = case["input"]
    basis = factorial_power_basis(int(spec["power"]))
    left = FormalSeries(basis, tuple(_fraction(value) for value in spec["left"]))
    right = FormalSeries(basis, tuple(_fraction(value) for value in spec["right"]))
    product = left.multiply(right, degree=int(spec["degree"]))
    observed = {
        "basis": basis.name,
        "coefficients": _vector(product.coefficients),
        "ordinary_coefficients": _vector(product.ordinary_coefficients()),
    }
    return _result(case, observed)


def _run_document_render(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import DocumentMetadata, MathDocument, render_latex, render_markdown

    spec = case["input"]
    document = MathDocument(DocumentMetadata(title=str(spec["title"]), author=str(spec["author"])))
    document.paragraph(str(spec["paragraph"]))
    document.equation(str(spec["equation"]), numbered=True, label="eq:stable")
    markdown = render_markdown(document)
    latex = render_latex(document)
    observed = {
        "markdown_sha256": sha256(markdown.encode("utf-8")).hexdigest(),
        "latex_sha256": sha256(latex.encode("utf-8")).hexdigest(),
        "markdown_has_equation": str(spec["equation"]) in markdown,
        "latex_has_equation": str(spec["equation"]) in latex,
    }
    return _result(case, observed)


def _run_settings_scope(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import get_settings, settings_context

    spec = case["input"]
    before = get_settings()
    with settings_context(
        backend=str(spec["backend"]),
        tolerance=float(spec["tolerance"]),
        validate_on_construction=bool(spec["validate_on_construction"]),
    ) as active:
        inside = active
    after = get_settings()
    observed = {
        "inside": {
            "backend": inside.backend,
            "tolerance": inside.tolerance,
            "validate_on_construction": inside.validate_on_construction,
        },
        "restored": after == before,
    }
    return _result(case, observed)


def _run_portable_refinement(case: Mapping[str, Any]) -> StableSemanticCaseResult:
    from .stable import PolyphaseRefinementTower, dumps_portable, loads_portable

    spec = case["input"]
    value = PolyphaseRefinementTower(tuple(int(order) for order in spec["orders"]))
    text = dumps_portable(value)
    envelope = json.loads(text)
    restored = loads_portable(text)
    payload = envelope["payload"]
    observed = {
        "equal": restored == value,
        "restored_type": type(restored).__name__,
        "schema": envelope["schema"],
        "codec_tag": payload["__komorebi_type__"],
        "codec_version": payload["__version__"],
    }
    return _result(case, observed)


_RUNNERS: dict[str, Callable[[Mapping[str, Any]], StableSemanticCaseResult]] = {
    "analysis_synthesis": _run_analysis_synthesis,
    "signal_polyphase": _run_signal_polyphase,
    "arithmetic_polyphase": _run_arithmetic_polyphase,
    "lattice_polyphase": _run_lattice_polyphase,
    "operator_grammar": _run_operator_grammar,
    "coordinate_basis": _run_coordinate_basis,
    "normalized_series": _run_normalized_series,
    "document_render": _run_document_render,
    "settings_scope": _run_settings_scope,
    "portable_refinement": _run_portable_refinement,
}


def run_stable_semantics_fixture(
    path: str | Path | None = None,
) -> StableSemanticReport:
    """Execute every committed stable 1.0 semantic case."""

    source, fixture, digest = load_stable_semantics_fixture(path)
    results: list[StableSemanticCaseResult] = []
    for case in fixture["cases"]:
        kind = str(case["kind"])
        try:
            runner = _RUNNERS[kind]
        except KeyError as exc:
            raise ValidationError(f"unknown stable semantic case kind {kind!r}") from exc
        try:
            results.append(runner(case))
        except Exception as exc:
            results.append(
                StableSemanticCaseResult(
                    name=str(case["name"]),
                    kind=kind,
                    expected=case["expected"],
                    observed=None,
                    passed=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
    failed = tuple(result.name for result in results if not result.passed)
    certificate = Certificate(
        valid=not failed,
        exactness=Exactness.EXACT,
        residuals={"failed_cases": float(len(failed))},
        assumptions_checked=(
            "fixture schema and canonical digest",
            "closed stable semantic case registry",
            "exact committed expected values",
            "major stable API families represented",
        ),
        assumptions_unchecked=(
            "fixture coverage is representative rather than exhaustive",
            "optional hardware backend equivalence",
        ),
        diagnostics={
            "cases": len(results),
            "failed": failed,
            "fixture_digest": digest,
        },
        theorem_id="KM-STABLE-SEM-016",
        message=(
            f"stable semantic fixture passed {len(results)}/{len(results)} cases"
            if not failed
            else f"stable semantic fixture failed {len(failed)} of {len(results)} cases"
        ),
    )
    return StableSemanticReport(source, digest, tuple(results), certificate)


__all__ = [
    "DEFAULT_STABLE_SEMANTICS_FIXTURE",
    "STABLE_SEMANTICS_SCHEMA",
    "STABLE_SEMANTICS_SCHEMA_VERSION",
    "StableSemanticCaseResult",
    "StableSemanticReport",
    "load_stable_semantics_fixture",
    "run_stable_semantics_fixture",
]
