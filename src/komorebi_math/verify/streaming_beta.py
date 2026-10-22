"""Installed verification for streaming persistence and compatibility."""

from __future__ import annotations

import json
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np

from ..compatibility import check_beta_api, check_public_api
from ..core import PacketSystem
from ..exceptions import ValidationError
from ..export import dumps_portable, from_portable_data, loads_portable, to_portable_data
from ..golden import run_behavioral_golden_fixture
from ..nonlinear import (
    PacketJetFamily,
    PacketJetSnapshot,
    PolynomialDerivativeProvider,
    capture_packet_jet_stream,
    estimate_packet_jet_complexity,
    open_packet_jet_snapshot_stream,
    packet_jet_support,
    profile_operation,
    save_packet_jet_snapshot_stream,
    snapshot_packet_jet,
)
from .core import VerificationCheck, VerificationReport, _capture


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _coordinate_family() -> tuple[PacketJetFamily, tuple[np.ndarray, np.ndarray]]:
    tensor = np.zeros((2, 2, 2), dtype=float)
    tensor[0, 0, 0] = 2.0
    tensor[1, 1, 1] = -3.0
    provider = PolynomialDerivativeProvider(
        {2: tensor},
        source_dimension=2,
        target_dimension=2,
        name="streaming coordinate quadratic",
    )
    packets = PacketSystem(
        (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
        names=("x", "y"),
    )
    return (
        PacketJetFamily(provider, np.array([0.25, -0.5]), packets, packets, 2),
        (np.array([2.0, 3.0]), np.array([-1.0, 4.0])),
    )


def run_streaming_beta_verification() -> VerificationReport:
    """Run streaming, migration, and namespace compatibility checks."""

    checks: list[VerificationCheck] = []

    def direct_resume_check() -> None:
        family, directions = _coordinate_family()
        jet = family.jet("x", ("x", "x"))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "direct.kmstream"
            partial = capture_packet_jet_stream(
                jet,
                destination,
                chunk_size=1,
                evaluation_budget=2,
            )
            assert not partial.completed
            resumed = capture_packet_jet_stream(
                jet,
                destination,
                chunk_size=1,
                resume=True,
            )
            resumed.certificate.require()
            assert resumed.completed and resumed.resumed
            reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
            observed = reader.evaluate(directions)
            observed.certificate.require()
            assert np.allclose(observed.value, jet.evaluate(directions))

    checks.append(_capture("resumable direct packet-jet stream", direct_resume_check))

    def snapshot_stream_check() -> None:
        family, directions = _coordinate_family()
        snapshot = snapshot_packet_jet(family.jet("x", ("x", "x")))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "snapshot.kmstream"
            result = save_packet_jet_snapshot_stream(snapshot, destination, chunk_size=1)
            result.certificate.require()
            reader = open_packet_jet_snapshot_stream(destination, verify_chunks=True)
            reader.verify().require()
            restored = reader.load_snapshot(max_coefficients=10)
            assert np.allclose(restored.evaluate(directions), snapshot.evaluate(directions))

    checks.append(_capture("bounded packet-jet snapshot stream", snapshot_stream_check))

    def tamper_check() -> None:
        family, _ = _coordinate_family()
        snapshot = snapshot_packet_jet(family.jet("x", ("x", "x")))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tamper.kmstream"
            save_packet_jet_snapshot_stream(snapshot, destination, chunk_size=1)
            chunk = next((destination / "chunks").glob("chunk-*.json"))
            raw = json.loads(chunk.read_text(encoding="utf-8"))
            raw["payload"]["coefficient_stop"] += 1
            chunk.write_text(json.dumps(raw), encoding="utf-8")
            try:
                open_packet_jet_snapshot_stream(destination, verify_chunks=True)
            except ValidationError:
                return
            raise AssertionError("modified stream chunk passed integrity verification")

    checks.append(_capture("stream chunk tamper detection", tamper_check))

    def codec_migration_check() -> None:
        family, directions = _coordinate_family()
        snapshot = snapshot_packet_jet(family.jet("x", ("x", "x")))
        envelope = json.loads(dumps_portable(snapshot))
        payload = envelope["payload"]
        payload["__version__"] = 1
        data = from_portable_data(payload["data"])
        data["provider_name"] = data.pop("provider")
        data.pop("coefficient_ordering")
        payload["data"] = to_portable_data(data)
        envelope["payload_sha256"] = sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        migrated = loads_portable(_canonical_json(envelope))
        assert isinstance(migrated, PacketJetSnapshot)
        assert np.allclose(migrated.evaluate(directions), snapshot.evaluate(directions))
        assert json.loads(dumps_portable(migrated))["payload"]["__version__"] == 2

    checks.append(_capture("versioned snapshot codec migration", codec_migration_check))

    def golden_check() -> None:
        report = run_behavioral_golden_fixture()
        report.certificate.require()
        assert len(report.results) == 15

    checks.append(_capture("committed behavioral golden semantics", golden_check))

    def complexity_check() -> None:
        family, _ = _coordinate_family()
        support = packet_jet_support(family)
        estimate = estimate_packet_jet_complexity(family, support)
        estimate.certificate.require()
        assert estimate.family_addresses == 8
        assert estimate.active_addresses == 2
        assert estimate.planned_derivative_calls == 2
        assert estimate.dense_basis_evaluations == 4

    checks.append(_capture("exact packet-jet operation counts", complexity_check))

    def observational_profile_check() -> None:
        profile = profile_operation(
            "verification no-op",
            lambda: None,
            repetitions=1,
            warmup=0,
            scope="one local Python no-op used only to validate the observation record",
        )
        assert profile.repetitions == 1
        assert profile.minimum_duration_ns >= 0
        assert profile.maximum_peak_bytes >= 0

    checks.append(_capture("environment-scoped runtime observation", observational_profile_check))

    def beta_api_check() -> None:
        beta = check_beta_api()
        beta.certificate.require()
        assert beta.strict_signatures
        legacy = check_public_api()
        legacy.certificate.require()

    checks.append(_capture("strict curated beta API contract", beta_api_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_streaming_beta_verification"]
