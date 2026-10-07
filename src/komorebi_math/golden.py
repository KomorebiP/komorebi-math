"""Deterministic behavioral fixtures for release-level semantic regression.

Public API manifests detect structural changes.  Golden fixtures complement them
with a small, explicit set of mathematical behaviors whose inputs and expected
outputs are committed as data.  The fixture language is closed: JSON selects one
of the built-in case kinds and never names an importable Python callable.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .certificates import Certificate, Exactness
from .core import PacketSystem
from .exceptions import ValidationError
from .export.persistence import dumps_portable, from_portable_data, loads_portable, to_portable_data
from .nonlinear.execution import compile_packet_jet_plan, execute_packet_jet_plan
from .nonlinear.jets import PacketJetAddress, PacketJetFamily
from .nonlinear.snapshots import PacketJetCoefficient, PacketJetSnapshot
from .nonlinear.sparse import SparseSymmetricTensor
from .nonlinear.streaming import save_packet_jet_snapshot_stream
from .nonlinear.derivatives import PolynomialDerivativeProvider

BEHAVIORAL_GOLDEN_SCHEMA = "komorebi-math.behavioral-golden"
BEHAVIORAL_GOLDEN_SCHEMA_VERSION = 1
DEFAULT_BEHAVIORAL_GOLDEN_FIXTURE = (
    Path(__file__).with_name("data") / "behavioral_golden_0_14.json"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValidationError(f"behavioral golden JSON contains duplicate key {key!r}")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> Any:
    raise ValidationError(f"behavioral golden JSON contains nonfinite constant {value}")


def _scalar(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    if text.endswith("j"):
        return complex(text)
    return int(text)


def _canonical_scalar(value: Any) -> str:
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, complex):
        return f"{value.real:.17g}{value.imag:+.17g}j"
    if isinstance(value, np.generic):
        return _canonical_scalar(value.item())
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def _canonical_vector(value: Sequence[Any]) -> list[str]:
    return [_canonical_scalar(entry) for entry in value]


def _exact_snapshot(case: Mapping[str, Any]) -> PacketJetSnapshot:
    source_dimension = int(case["source_dimension"])
    target_dimension = int(case["target_dimension"])
    address_data = case["address"]
    if not isinstance(address_data, Mapping):
        raise ValidationError("golden snapshot address must be an object")
    coefficients = tuple(
        PacketJetCoefficient(
            tuple(int(index) for index in coefficient["indices"]),
            tuple(_scalar(value) for value in coefficient["value"]),
        )
        for coefficient in case["coefficients"]
    )
    certificate = Certificate(
        valid=True,
        exactness=Exactness.EXACT,
        assumptions_checked=("committed finite golden coefficient table",),
        theorem_id="KM-GOLDEN-011",
        message="committed exact golden snapshot",
    )
    return PacketJetSnapshot(
        address=PacketJetAddress(
            str(address_data["output"]),
            tuple(str(value) for value in address_data["inputs"]),
        ),
        source_dimension=source_dimension,
        target_dimension=target_dimension,
        provider_name=str(case.get("provider_name", "behavioral golden fixture")),
        exactness=Exactness.EXACT,
        point=tuple(_scalar(value) for value in case["point"]),
        coefficients=coefficients,
        capture_certificate=certificate,
    )


@dataclass(frozen=True, slots=True)
class BehavioralGoldenCaseResult:
    """Observed and expected data for one closed golden case."""

    name: str
    kind: str
    observed: Any
    expected: Any
    certificate: Certificate

    @property
    def passed(self) -> bool:
        return self.certificate.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "observed": self.observed,
            "expected": self.expected,
            "certificate": self.certificate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BehavioralGoldenReport:
    """Aggregate deterministic semantic regression report."""

    fixture_path: Path
    fixture_digest: str
    results: tuple[BehavioralGoldenCaseResult, ...]
    certificate: Certificate

    @property
    def passed(self) -> bool:
        return self.certificate.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_path": str(self.fixture_path),
            "fixture_digest": self.fixture_digest,
            "results": tuple(result.to_dict() for result in self.results),
            "certificate": self.certificate.to_dict(),
        }


def load_behavioral_golden_fixture(
    path: str | Path | None = None,
) -> tuple[Path, dict[str, Any], str]:
    """Load and validate the closed behavioral fixture document."""

    source = DEFAULT_BEHAVIORAL_GOLDEN_FIXTURE if path is None else Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except OSError as exc:
        raise ValidationError(f"could not read behavioral golden fixture {source}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError("behavioral golden fixture is not valid UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError("behavioral golden fixture root must be an object")
    required = {"schema", "schema_version", "cases"}
    if set(raw) != required:
        raise ValidationError("behavioral golden fixture fields do not match the schema")
    if raw["schema"] != BEHAVIORAL_GOLDEN_SCHEMA:
        raise ValidationError("behavioral golden fixture uses an unknown schema")
    if raw["schema_version"] != BEHAVIORAL_GOLDEN_SCHEMA_VERSION:
        raise ValidationError("behavioral golden fixture schema version is unsupported")
    cases = raw["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValidationError("behavioral golden fixture requires a nonempty case array")
    names: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping) or not {"name", "kind", "expected"} <= set(case):
            raise ValidationError("behavioral golden case is malformed")
        name = str(case["name"])
        if not name or name in names:
            raise ValidationError("behavioral golden case names must be nonempty and unique")
        names.add(name)
    document = dict(raw)
    return source, document, sha256(_canonical_json(document).encode("utf-8")).hexdigest()


def _result(
    case: Mapping[str, Any],
    observed: Any,
    *,
    exactness: Exactness = Exactness.EXACT,
    tolerance: float | None = None,
) -> BehavioralGoldenCaseResult:
    expected = case["expected"]
    if tolerance is None:
        passed = observed == expected
        residual = 0.0 if passed else 1.0
    else:
        observed_array = np.asarray(observed, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
        residual = float(np.linalg.norm(observed_array - expected_array))
        passed = observed_array.shape == expected_array.shape and residual <= tolerance
    certificate = Certificate(
        valid=passed,
        exactness=exactness,
        residuals={"golden_residual": residual},
        tolerance=tolerance,
        assumptions_checked=(
            "closed built-in golden case evaluator",
            "committed explicit inputs and expected output",
        ),
        assumptions_unchecked=(),
        diagnostics={"kind": str(case["kind"])},
        theorem_id="KM-GOLDEN-011",
        message=(
            f"behavioral golden case {case['name']} passed"
            if passed
            else f"behavioral golden case {case['name']} failed"
        ),
    )
    return BehavioralGoldenCaseResult(
        str(case["name"]),
        str(case["kind"]),
        observed,
        expected,
        certificate,
    )


def _run_exact_snapshot(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    snapshot = _exact_snapshot(case)
    directions = tuple(
        np.asarray([_scalar(value) for value in direction], dtype=object)
        for direction in case["directions"]
    )
    observed = _canonical_vector(snapshot.evaluate(directions))
    return _result(case, observed)


def _run_sparse_contraction(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    mapping = {
        tuple(int(value) for value in entry["key"]): _scalar(entry["value"])
        for entry in case["entries"]
    }
    tensor = SparseSymmetricTensor.from_mapping(
        int(case["order"]),
        int(case["source_dimension"]),
        int(case["target_dimension"]),
        mapping,
    )
    directions = tuple(
        np.asarray([_scalar(value) for value in direction], dtype=object)
        for direction in case["directions"]
    )
    observed = _canonical_vector(tensor.contract(directions))
    return _result(case, observed)


def _run_execution_plan(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    tensor = np.zeros(tuple(int(value) for value in case["tensor_shape"]), dtype=float)
    for entry in case["tensor_entries"]:
        tensor[tuple(int(value) for value in entry["indices"])] = float(entry["value"])
    source_dimension = int(case["source_dimension"])
    target_dimension = int(case["target_dimension"])
    provider = PolynomialDerivativeProvider(
        {int(case["order"]): tensor},
        source_dimension=source_dimension,
        target_dimension=target_dimension,
    )
    source_projectors = tuple(np.asarray(value, dtype=float) for value in case["source_projectors"])
    target_projectors = tuple(np.asarray(value, dtype=float) for value in case["target_projectors"])
    source = PacketSystem(source_projectors, names=tuple(case["source_names"]))
    target = PacketSystem(target_projectors, names=tuple(case["target_names"]))
    family = PacketJetFamily(
        provider,
        np.asarray(case["point"], dtype=float),
        source,
        target,
        int(case["order"]),
    )
    plan = compile_packet_jet_plan(family)
    result = execute_packet_jet_plan(
        plan,
        family,
        tuple(np.asarray(value, dtype=float) for value in case["directions"]),
    )
    observed = {
        "full_addresses": plan.full_address_count,
        "active_addresses": plan.active_address_count,
        "derivative_calls": plan.derivative_call_count,
        "reconstructed": [
            float(value)
            for value in np.real_if_close(np.asarray(result.reconstructed)).reshape(-1)
        ],
        "direct": [
            float(value)
            for value in np.real_if_close(np.asarray(result.direct)).reshape(-1)
        ],
    }
    return _result(case, observed)


def _run_snapshot_migration(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    snapshot = _exact_snapshot(case)
    envelope = json.loads(dumps_portable(snapshot))
    payload = envelope["payload"]
    payload["__version__"] = 1
    data = from_portable_data(payload["data"])
    data["provider_name"] = data.pop("provider")
    data.pop("coefficient_ordering")
    payload["data"] = to_portable_data(data)
    envelope["payload_sha256"] = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    migrated = loads_portable(_canonical_json(envelope))
    if not isinstance(migrated, PacketJetSnapshot):
        raise ValidationError("golden v1 snapshot did not migrate to PacketJetSnapshot")
    directions = tuple(
        np.asarray([_scalar(value) for value in direction], dtype=object)
        for direction in case["directions"]
    )
    current_envelope = json.loads(dumps_portable(migrated))
    observed = {
        "provider_name": migrated.provider_name,
        "nnz": migrated.nnz,
        "evaluation": _canonical_vector(migrated.evaluate(directions)),
        "current_codec_version": current_envelope["payload"]["__version__"],
    }
    return _result(case, observed)


def _run_normalized_series_product(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate one exact normalized-basis product fixture."""

    from .generating import FormalSeries, factorial_power_basis

    basis = factorial_power_basis(int(case["factorial_power"]))
    left = FormalSeries(basis, tuple(_scalar(value) for value in case["left"]))
    right = FormalSeries(basis, tuple(_scalar(value) for value in case["right"]))
    observed = _canonical_vector(left.multiply(right).coefficients)
    return _result(case, observed)


def _run_normalized_packet_transfer(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate one exact normalized residue-packet transfer fixture."""

    from .generating import FormalSeries, factorial_power_basis, ordinary_basis

    basis = factorial_power_basis(int(case["factorial_power"]))
    target_name = str(case["target_basis"])
    if target_name != "ordinary":
        raise ValidationError("golden packet transfer supports only the ordinary target basis")
    target = ordinary_basis()
    source = FormalSeries(basis, tuple(_scalar(value) for value in case["coefficients"]))
    modulus = int(case["modulus"])
    residue = int(case["residue"])
    packet = source.packet(modulus, residue, target_basis=target)
    certificate = source.packet_identity(modulus, residue, target_basis=target)
    observed = {
        "packet_coefficients": _canonical_vector(packet.coefficients),
        "source_basis": basis.name,
        "target_basis": target.name,
        "valid": certificate.valid,
    }
    return _result(case, observed)


def _run_finite_mask_evaluation(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate one exact finite-mask carry fixture."""

    from .generating import compile_finite_mask_carry_plan

    plan = compile_finite_mask_carry_plan(
        tuple(_scalar(value) for value in case["mask"]),
        int(case["radix"]),
    )
    observed = _canonical_vector(plan.sequence(int(case["count"])))
    return _result(case, observed)


def _run_prime_power_kernel(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate one exact factorial-power kernel fixture modulo a prime power."""

    from .generating import PrimePowerKernelCompiler

    compiler = PrimePowerKernelCompiler(
        int(case["prime"]),
        int(case["exponent"]),
        int(case["factorial_power"]),
    )
    value = compiler.kernel(int(case["index"]), int(case["subindex"]))
    observed = {
        "valuation": value.valuation,
        "unit": value.unit,
        "residue": value.residue,
        "survives": value.survives,
    }
    return _result(case, observed)


def _run_recurrence_coordinates(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate one exact recurrence defect coordinate fixture."""

    from .coordinates import euler_fibonacci_coordinates, variation_of_constants_prefix

    record = euler_fibonacci_coordinates(int(case["count"]))
    observed = {
        "initial": _canonical_vector(record.initial),
        "defects": _canonical_vector(record.defects),
        "reconstructed": _canonical_vector(variation_of_constants_prefix(record)),
        "homogeneous": record.homogeneous,
    }
    return _result(case, observed)


def _run_polynomial_connection(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate one Euler to Fibonacci polynomial connection fixture."""

    from .coordinates import euler_to_fibonacci_transform

    transform = euler_to_fibonacci_transform(int(case["degree"]))
    observed = {
        "matrix": [_canonical_vector(row) for row in transform.matrix],
        "determinant": _canonical_scalar(transform.determinant),
        "roundtrip": transform.roundtrip_valid(),
    }
    return _result(case, observed)


def _run_arbitrary_basis_connection(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate one exact connection between arbitrary finite bases."""

    from .coordinates import basis_from_matrix

    source = basis_from_matrix(
        "golden source basis",
        tuple(tuple(_scalar(value) for value in row) for row in case["source_matrix"]),
    )
    target = basis_from_matrix(
        "golden target basis",
        tuple(tuple(_scalar(value) for value in row) for row in case["target_matrix"]),
    )
    vector = tuple(_scalar(value) for value in case["vector"])
    transform = source.connection_to(target)
    source_coordinates = source.analyze(vector)
    target_coordinates = target.analyze(vector)
    observed = {
        "matrix": [_canonical_vector(row) for row in transform.matrix],
        "source_coordinates": _canonical_vector(source_coordinates),
        "target_coordinates": _canonical_vector(target_coordinates),
        "roundtrip": transform.apply(source_coordinates) == target_coordinates,
    }
    return _result(case, observed)


def _run_combinatorial_coordinate_transform(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate one exact combinatorial coordinate transform."""

    from .coordinates import forward_difference_transform

    kind = str(case["transform"])
    if kind != "forward-difference":
        raise ValidationError(f"unsupported golden combinatorial transform {kind!r}")
    values = tuple(_scalar(value) for value in case["values"])
    observed = _canonical_vector(forward_difference_transform(len(values)).apply(values))
    return _result(case, observed)


def _run_classical_polynomial_connection(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate a connection between two built-in polynomial families."""

    from .coordinates import connection_transform, get_polynomial_family

    transform = connection_transform(
        get_polynomial_family(str(case["source"])),
        get_polynomial_family(str(case["target"])),
        int(case["degree"]),
    )
    observed = {
        "matrix": [_canonical_vector(row) for row in transform.matrix],
        "determinant": _canonical_scalar(transform.determinant),
        "roundtrip": transform.roundtrip_valid(),
    }
    return _result(case, observed)


def _run_formal_coordinate_roundtrip(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    """Evaluate one exact formal-coordinate pullback and inverse."""

    from .coordinates import formal_coordinate_from_iterable

    change = formal_coordinate_from_iterable(
        tuple(_scalar(value) for value in case["coordinate"]),
        source="u",
        target="z",
    )
    values = tuple(_scalar(value) for value in case["series"])
    pulled = change.pullback_coefficients(values)
    observed = {
        "inverse": _canonical_vector(change.inverse().coefficients),
        "pullback": _canonical_vector(pulled),
        "recovered": _canonical_vector(change.pushforward_coefficients(pulled)),
        "valid": change.certificate().valid,
    }
    return _result(case, observed)


def _run_stream_snapshot(case: Mapping[str, Any]) -> BehavioralGoldenCaseResult:
    snapshot = _exact_snapshot(case)
    directions = tuple(
        np.asarray([_scalar(value) for value in direction], dtype=object)
        for direction in case["directions"]
    )
    with tempfile.TemporaryDirectory() as directory:
        result = save_packet_jet_snapshot_stream(
            snapshot,
            Path(directory) / "golden.kmstream",
            chunk_size=int(case["chunk_size"]),
        )
        from .nonlinear.streaming import open_packet_jet_snapshot_stream

        reader = open_packet_jet_snapshot_stream(result.destination, verify_chunks=True)
        evaluation = reader.evaluate(directions)
        observed = {
            "chunks": len(reader.manifest.chunks),
            "coefficients": reader.manifest.coefficients_completed,
            "stream_sha256": reader.manifest.stream_sha256,
            "evaluation": _canonical_vector(evaluation.value),
        }
    return _result(case, observed)


_RUNNERS = {
    "exact_packet_snapshot": _run_exact_snapshot,
    "sparse_symmetric_contraction": _run_sparse_contraction,
    "packet_execution_plan": _run_execution_plan,
    "packet_snapshot_v1_migration": _run_snapshot_migration,
    "streamed_packet_snapshot": _run_stream_snapshot,
    "normalized_series_product": _run_normalized_series_product,
    "normalized_packet_transfer": _run_normalized_packet_transfer,
    "finite_mask_evaluation": _run_finite_mask_evaluation,
    "prime_power_kernel": _run_prime_power_kernel,
    "recurrence_coordinates": _run_recurrence_coordinates,
    "polynomial_connection": _run_polynomial_connection,
    "arbitrary_basis_connection": _run_arbitrary_basis_connection,
    "combinatorial_coordinate_transform": _run_combinatorial_coordinate_transform,
    "classical_polynomial_connection": _run_classical_polynomial_connection,
    "formal_coordinate_roundtrip": _run_formal_coordinate_roundtrip,
}


def run_behavioral_golden_fixture(
    path: str | Path | None = None,
) -> BehavioralGoldenReport:
    """Execute every committed semantic fixture through its closed evaluator."""

    source, fixture, digest = load_behavioral_golden_fixture(path)
    results: list[BehavioralGoldenCaseResult] = []
    for case in fixture["cases"]:
        kind = str(case["kind"])
        try:
            runner = _RUNNERS[kind]
        except KeyError as exc:
            raise ValidationError(f"unknown behavioral golden case kind {kind!r}") from exc
        results.append(runner(case))
    failed = tuple(result.name for result in results if not result.passed)
    certificate = Certificate(
        valid=not failed,
        exactness=Exactness.EXACT,
        residuals={"failed_cases": float(len(failed))},
        assumptions_checked=(
            "fixture schema and digest",
            "closed case-kind registry",
            "deterministic committed expected values",
        ),
        assumptions_unchecked=(
            "the fixture set is representative rather than exhaustive",
        ),
        diagnostics={
            "cases": len(results),
            "failed": failed,
            "fixture_digest": digest,
        },
        theorem_id="KM-GOLDEN-011",
        message=(
            f"behavioral golden fixture passed {len(results)}/{len(results)} cases"
            if not failed
            else f"behavioral golden fixture failed {len(failed)} of {len(results)} cases"
        ),
    )
    return BehavioralGoldenReport(source, digest, tuple(results), certificate)


__all__ = [
    "BEHAVIORAL_GOLDEN_SCHEMA",
    "BEHAVIORAL_GOLDEN_SCHEMA_VERSION",
    "DEFAULT_BEHAVIORAL_GOLDEN_FIXTURE",
    "BehavioralGoldenCaseResult",
    "BehavioralGoldenReport",
    "load_behavioral_golden_fixture",
    "run_behavioral_golden_fixture",
]
