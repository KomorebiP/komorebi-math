from __future__ import annotations

import json
import platform
import random
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import numpy as np

import komorebi_math
from komorebi_math import Exactness, PacketSystem
from komorebi_math.backends.numpy import NumpyBackend
from komorebi_math.compatibility import (
    check_beta_api,
    check_public_api,
    create_beta_api_manifest,
    create_public_api_manifest,
    load_beta_api_manifest,
    load_public_api_manifest,
)
from komorebi_math.exceptions import ValidationError
from komorebi_math.export import dumps_portable, from_portable_data, loads_portable, to_portable_data
from komorebi_math.golden import run_behavioral_golden_fixture
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    capture_packet_jet_stream,
    cupy_runtime_status,
    estimate_packet_jet_complexity,
    open_packet_jet_snapshot_stream,
    packet_jet_support,
    profile_operation,
    save_packet_jet_snapshot_stream,
    snapshot_packet_jet,
)


@dataclass
class GroupResult:
    name: str
    checks: int
    failures: list[str]
    skipped: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def run_group(
    name: str,
    function: Callable[[Callable[[bool, str], None], list[str]], None],
) -> GroupResult:
    checks = 0
    failures: list[str] = []
    skipped: list[str] = []

    def check(condition: bool, detail: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(detail)

    try:
        function(check, skipped)
    except Exception as exc:
        failures.append(f"unexpected {type(exc).__name__}: {exc}")
    return GroupResult(name, checks, failures, skipped)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class _ExactObjectBackend(NumpyBackend):
    """Small exact Fraction backend used only by the randomized audit."""

    name = "block11-exact-object"
    exact = True

    def __init__(self) -> None:
        super().__init__(dtype=object)

    def norm(self, value: Any) -> float:
        entries = np.asarray(value, dtype=object).reshape(-1)
        return float(sum(abs(complex(entry)) ** 2 for entry in entries) ** 0.5)

    def is_zero(self, value: Any, *, tolerance: float | None = None) -> bool:
        del tolerance
        return all(entry == 0 for entry in np.asarray(value, dtype=object).reshape(-1))


def _one_packet(
    dimension: int,
    *,
    name: str = "all",
    exact: bool = False,
) -> PacketSystem:
    if exact:
        return PacketSystem(
            (np.eye(dimension, dtype=object),),
            names=(name,),
            backend=_ExactObjectBackend(),
        )
    return PacketSystem((np.eye(dimension),), names=(name,))


def _random_tensor(
    rng: random.Random,
    target_dimension: int,
    source_dimension: int,
    order: int,
    *,
    exact: bool,
) -> np.ndarray:
    shape = (target_dimension,) + (source_dimension,) * order
    tensor = np.empty(shape, dtype=object if exact else float)
    for index in np.ndindex(shape):
        if rng.random() < 0.35:
            value: Any = 0
        elif exact:
            value = Fraction(rng.randrange(-7, 8), rng.randrange(1, 8))
        else:
            value = rng.uniform(-2.0, 2.0)
        tensor[index] = value
    return tensor


def _family(
    rng: random.Random,
    *,
    exact: bool = False,
    dimension: int | None = None,
    target_dimension: int | None = None,
    order: int | None = None,
) -> PacketJetFamily:
    source_dimension = dimension or rng.randrange(1, 4)
    target = target_dimension or rng.randrange(1, 4)
    derivative_order = order or rng.randrange(1, 4)
    tensor = _random_tensor(
        rng,
        target,
        source_dimension,
        derivative_order,
        exact=exact,
    )
    provider = PolynomialDerivativeProvider(
        {derivative_order: tensor},
        source_dimension=source_dimension,
        target_dimension=target,
        exactness=Exactness.EXACT if exact else Exactness.NUMERICAL,
        name="block11 deep-audit polynomial",
    )
    point = np.array(
        [Fraction(rng.randrange(-3, 4), 5) for _ in range(source_dimension)],
        dtype=object,
    ) if exact else np.array([rng.uniform(-1.0, 1.0) for _ in range(source_dimension)])
    return PacketJetFamily(
        provider,
        point,
        _one_packet(source_dimension, exact=exact),
        _one_packet(target, exact=exact),
        derivative_order,
    )


def _directions(rng: random.Random, family: PacketJetFamily, *, exact: bool) -> tuple[np.ndarray, ...]:
    if exact:
        return tuple(
            np.array(
                [Fraction(rng.randrange(-6, 7), rng.randrange(1, 7)) for _ in range(family.source.dimension)],
                dtype=object,
            )
            for _ in range(family.order)
        )
    return tuple(
        np.array([rng.uniform(-2.0, 2.0) for _ in range(family.source.dimension)])
        for _ in range(family.order)
    )


def snapshot_stream_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    rng = random.Random(11001)
    for case in range(420):
        exact = case % 4 == 0
        family = _family(rng, exact=exact)
        jet = family.jet("all", ("all",) * family.order)
        snapshot = snapshot_packet_jet(jet)
        directions = _directions(rng, family, exact=exact)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "snapshot.kmstream"
            chunk_size = rng.randrange(1, 5)
            if snapshot.nnz and case % 3 != 0:
                budget = rng.randrange(snapshot.nnz)
                partial = save_packet_jet_snapshot_stream(
                    snapshot,
                    destination,
                    chunk_size=chunk_size,
                    coefficient_budget=budget,
                )
                check(not partial.completed, f"snapshot partial state case={case}")
                incomplete = open_packet_jet_snapshot_stream(
                    destination,
                    allow_incomplete=True,
                )
                check(incomplete.verify().valid, f"snapshot partial integrity case={case}")
                result = save_packet_jet_snapshot_stream(
                    snapshot,
                    destination,
                    chunk_size=chunk_size,
                    resume=True,
                )
                check(result.resumed, f"snapshot resumed case={case}")
            else:
                result = save_packet_jet_snapshot_stream(
                    snapshot,
                    destination,
                    chunk_size=chunk_size,
                )
            check(result.completed, f"snapshot completion case={case}")
            reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
            check(reader.verify().valid, f"snapshot full integrity case={case}")
            restored = reader.load_snapshot(max_coefficients=max(1, snapshot.nnz))
            observed = restored.evaluate(directions)
            expected = snapshot.evaluate(directions)
            if exact:
                check(np.array_equal(observed, expected), f"exact stream evaluation case={case}")
            else:
                check(np.allclose(observed, expected, atol=1e-11), f"stream evaluation case={case}")
            split = rng.randrange(snapshot.nnz + 1)
            left = reader.evaluate(directions, start=0, stop=split)
            right = reader.evaluate(directions, start=split)
            combined = left.value + right.value
            if exact:
                check(np.array_equal(combined, expected), f"exact partial sum case={case}")
            else:
                check(np.allclose(combined, expected, atol=1e-11), f"partial sum case={case}")
            check(
                reader.manifest.coefficients_completed == snapshot.nnz,
                f"snapshot nnz case={case}",
            )
            check(
                reader.manifest.evaluations_completed == snapshot.dense_coefficient_count,
                f"snapshot basis count case={case}",
            )


def direct_stream_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    rng = random.Random(11002)
    for case in range(320):
        exact = case % 5 == 0
        family = _family(rng, exact=exact)
        jet = family.jet("all", ("all",) * family.order)
        directions = _directions(rng, family, exact=exact)
        total = family.source.dimension**family.order
        budget = rng.randrange(total) if total > 1 else 0
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "direct.kmstream"
            partial = capture_packet_jet_stream(
                jet,
                destination,
                chunk_size=rng.randrange(1, 5),
                evaluation_budget=budget,
            )
            check(not partial.completed, f"direct pause case={case}")
            check(
                partial.manifest.evaluations_completed == budget,
                f"direct pause ordinal case={case}",
            )
            incomplete = open_packet_jet_snapshot_stream(
                destination,
                allow_incomplete=True,
            )
            check(incomplete.verify().valid, f"direct partial integrity case={case}")
            complete = capture_packet_jet_stream(
                jet,
                destination,
                chunk_size=partial.manifest.chunk_size,
                resume=True,
            )
            check(complete.completed and complete.resumed, f"direct resume case={case}")
            reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
            observed = reader.evaluate(directions)
            expected = jet.evaluate(directions)
            check(observed.complete_snapshot, f"direct full coverage case={case}")
            if exact:
                check(np.array_equal(observed.value, expected), f"exact direct evaluation case={case}")
            else:
                check(np.allclose(observed.value, expected, atol=1e-11), f"direct evaluation case={case}")
            check(reader.verify().valid, f"direct full integrity case={case}")


def tamper_and_resume_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    rng = random.Random(11003)
    for case in range(180):
        family = _family(rng)
        snapshot = snapshot_packet_jet(family.jet("all", ("all",) * family.order))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tamper.kmstream"
            save_packet_jet_snapshot_stream(snapshot, destination, chunk_size=2)
            chunk = next((destination / "chunks").glob("chunk-*.json"))
            raw = json.loads(chunk.read_text(encoding="utf-8"))
            raw["payload"]["coefficient_start"] += 1
            chunk.write_text(json.dumps(raw), encoding="utf-8")
            try:
                open_packet_jet_snapshot_stream(destination, verify_chunks=True)
            except ValidationError:
                check(True, f"chunk tamper rejection case={case}")
            else:
                check(False, f"chunk tamper rejection case={case}")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "partial-tamper.kmstream"
            tamper_family = _family(
                rng,
                dimension=2,
                target_dimension=family.target.dimension,
                order=2,
            )
            tamper_jet = tamper_family.jet("all", ("all", "all"))
            partial = capture_packet_jet_stream(
                tamper_jet,
                destination,
                chunk_size=1,
                evaluation_budget=1,
            )
            chunk = next((partial.active_path / "chunks").glob("chunk-*.json"))
            raw = json.loads(chunk.read_text(encoding="utf-8"))
            raw["payload"]["basis_stop"] += 1
            chunk.write_text(json.dumps(raw), encoding="utf-8")
            try:
                capture_packet_jet_stream(
                    tamper_jet,
                    destination,
                    chunk_size=1,
                    resume=True,
                )
            except ValidationError:
                check(True, f"tampered committed-prefix rejection case={case}")
            else:
                check(False, f"tampered committed-prefix rejection case={case}")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resume.kmstream"
            jet = family.jet("all", ("all",) * family.order)
            capture_packet_jet_stream(jet, destination, chunk_size=1, evaluation_budget=0)
            changed = _family(
                rng,
                dimension=family.source.dimension,
                target_dimension=family.target.dimension,
                order=family.order,
            ).jet("all", ("all",) * family.order)
            try:
                capture_packet_jet_stream(changed, destination, chunk_size=1, resume=True)
            except ValidationError:
                check(True, f"resume source mismatch rejection case={case}")
            else:
                check(False, f"resume source mismatch rejection case={case}")


def migration_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    rng = random.Random(11004)
    for case in range(520):
        family = _family(rng, exact=case % 3 == 0)
        snapshot = snapshot_packet_jet(family.jet("all", ("all",) * family.order))
        envelope = json.loads(dumps_portable(snapshot, indent=None))
        payload = envelope["payload"]
        check(payload["__version__"] == 2, f"current snapshot codec version case={case}")
        payload["__version__"] = 1
        data = from_portable_data(payload["data"])
        data["provider_name"] = data.pop("provider")
        data.pop("coefficient_ordering")
        payload["data"] = to_portable_data(data)
        envelope["payload_sha256"] = sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        migrated = loads_portable(_canonical_json(envelope))
        check(migrated == snapshot, f"v1 to v2 snapshot migration case={case}")
        check(
            json.loads(dumps_portable(migrated, indent=None))["payload"]["__version__"] == 2,
            f"migrated re-encoding version case={case}",
        )


def semantics_api_complexity_group(
    check: Callable[[bool, str], None],
    skipped: list[str],
) -> None:
    del skipped
    golden = run_behavioral_golden_fixture()
    check(golden.passed, "behavioral golden fixture")
    check(len(golden.results) == 5, "behavioral golden case count")
    check(len(golden.fixture_digest) == 64, "behavioral golden fixture digest")

    legacy_reference = load_public_api_manifest()
    beta_reference = load_beta_api_manifest()
    legacy = create_public_api_manifest()
    beta = create_beta_api_manifest()
    check(komorebi_math.__version__ == "0.11.0a1", "package version")
    check(legacy == create_public_api_manifest(), "legacy manifest determinism")
    check(beta == create_beta_api_manifest(), "beta manifest determinism")
    check(legacy.digest == legacy_reference.digest, "legacy manifest digest")
    check(beta.digest == beta_reference.digest, "beta manifest digest")
    legacy_report = check_public_api(strict_signatures=True)
    beta_report = check_beta_api()
    check(legacy_report.compatible, "legacy strict API compatibility")
    check(beta_report.compatible, "beta strict API compatibility")
    check(not legacy_report.signature_changed, "legacy signatures unchanged")
    check(not beta_report.signature_changed, "beta signatures unchanged")

    rng = random.Random(11005)
    for case in range(600):
        dimension = rng.randrange(1, 4)
        order = rng.randrange(1, 4)
        target = rng.randrange(1, 4)
        family = _family(
            rng,
            dimension=dimension,
            target_dimension=target,
            order=order,
        )
        support = packet_jet_support(family)
        estimate = estimate_packet_jet_complexity(family, support)
        check(estimate.certificate.valid, f"complexity certificate case={case}")
        check(
            estimate.family_addresses == len(family.target) * len(family.source) ** order,
            f"family address count case={case}",
        )
        check(
            estimate.active_addresses + estimate.pruned_addresses == estimate.family_addresses,
            f"active/pruned partition case={case}",
        )
        check(
            estimate.planned_derivative_calls <= estimate.naive_derivative_calls,
            f"derivative call bound case={case}",
        )
        check(
            estimate.dense_basis_evaluations == dimension**order,
            f"dense basis count case={case}",
        )
        check(
            estimate.dense_tensor_elements == target * dimension**order,
            f"dense tensor count case={case}",
        )


def profile_boundary_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    for case in range(120):
        profile = profile_operation(
            f"no-op-{case}",
            lambda: None,
            repetitions=1,
            warmup=0,
            scope="one local no-op for observation record validation",
        )
        check(profile.repetitions == 1, f"profile repetitions case={case}")
        check(profile.minimum_duration_ns >= 0, f"profile duration case={case}")
        check(profile.maximum_peak_bytes >= 0, f"profile peak case={case}")
        check(bool(profile.scope), f"profile scope case={case}")

    status = cupy_runtime_status()
    check(status.device_count >= 0, "CuPy nonnegative device count")
    if status.runtime_available:
        check(status.installed and status.device_count > 0, "CuPy available runtime consistency")
        skipped.append("No performance claim was made by the device boundary audit")
    else:
        check(status.device_count == 0, "CuPy unavailable runtime has zero devices")
        check(bool(status.error), "CuPy unavailable runtime reports reason")
        skipped.append("Device-backed validation not executed because no usable CuPy CUDA runtime exists")


def main() -> int:
    groups = (
        run_group("snapshot_streams", snapshot_stream_group),
        run_group("direct_streams", direct_stream_group),
        run_group("tamper_and_resume_guards", tamper_and_resume_group),
        run_group("codec_migrations", migration_group),
        run_group("semantic_api_complexity", semantics_api_complexity_group),
        run_group("profile_and_device_boundary", profile_boundary_group),
    )
    payload = {
        "version": komorebi_math.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "checks": sum(group.checks for group in groups),
        "failures": sum(len(group.failures) for group in groups),
        "groups": [asdict(group) | {"passed": group.passed} for group in groups],
    }
    output = Path(__file__).with_name("BLOCK11_DEEP_AUDIT.json")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Block 11 deep audit: {payload['checks']} checks, "
        f"{payload['failures']} failures"
    )
    for group in groups:
        state = "PASS" if group.passed else "FAIL"
        print(f"[{state}] {group.name}: {group.checks} checks")
        for failure in group.failures[:10]:
            print(f"  {failure}")
        for skipped in group.skipped:
            print(f"  SKIP: {skipped}")
    return 0 if payload["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
