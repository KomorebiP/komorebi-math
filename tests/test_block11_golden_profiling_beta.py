from __future__ import annotations

import json
import tracemalloc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import komorebi_math.beta as beta
from komorebi_math import PacketSystem
from komorebi_math.compatibility import (
    BETA_API_STABILITY_LEVEL,
    BETA_PUBLIC_NAMESPACES,
    PublicApiManifest,
    PublicApiSymbol,
    check_beta_api,
    compare_public_api,
    create_beta_api_manifest,
    load_beta_api_manifest,
    save_public_api_manifest,
)
from komorebi_math.exceptions import ValidationError
from komorebi_math.golden import (
    load_behavioral_golden_fixture,
    run_behavioral_golden_fixture,
)
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    estimate_packet_jet_complexity,
    packet_jet_support,
    profile_operation,
    profile_packet_jet_runtime,
)


def _family() -> tuple[PacketJetFamily, tuple[np.ndarray, np.ndarray]]:
    tensor = np.zeros((2, 2, 2))
    tensor[0, 0, 0] = 2.0
    tensor[1, 1, 1] = -3.0
    provider = PolynomialDerivativeProvider(
        {2: tensor},
        source_dimension=2,
        target_dimension=2,
    )
    packets = PacketSystem(
        (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
        names=("x", "y"),
    )
    return (
        PacketJetFamily(provider, np.zeros(2), packets, packets, 2),
        (np.array([2.0, 3.0]), np.array([-1.0, 4.0])),
    )


def test_behavioral_golden_fixture_passes_all_committed_cases() -> None:
    source, fixture, digest = load_behavioral_golden_fixture()
    assert source.is_file()
    assert len(digest) == 64
    assert len(fixture["cases"]) == 15
    report = run_behavioral_golden_fixture()
    report.certificate.require()
    assert report.passed
    assert len(report.results) == 15
    assert all(result.passed for result in report.results)


def test_behavioral_golden_fixture_rejects_unknown_case_kind(tmp_path: Path) -> None:
    _, fixture, _ = load_behavioral_golden_fixture()
    mutated = json.loads(json.dumps(fixture))
    mutated["cases"][0]["kind"] = "dynamic.import.selector"
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValidationError, match="unknown behavioral golden case kind"):
        run_behavioral_golden_fixture(path)


def test_complexity_estimate_reports_exact_structural_counts() -> None:
    family, _ = _family()
    support = packet_jet_support(family)
    estimate = estimate_packet_jet_complexity(family, support)
    estimate.certificate.require()
    assert estimate.family_addresses == 8
    assert estimate.active_addresses == 2
    assert estimate.pruned_addresses == 6
    assert estimate.naive_derivative_calls == 8
    assert estimate.planned_derivative_calls == 2
    assert estimate.derivative_call_reduction == pytest.approx(0.75)
    assert estimate.projected_input_vectors == 4
    assert estimate.output_projections == 2
    assert estimate.dense_basis_evaluations == 4
    assert estimate.dense_tensor_elements == 8


def test_profile_operation_and_packet_runtime_record_scope_not_speed_claims() -> None:
    sample = profile_operation("addition", lambda: 1 + 1, repetitions=2, warmup=0)
    assert sample.repetitions == 2
    assert sample.minimum_duration_ns >= 0
    assert sample.maximum_peak_bytes >= 0

    family, directions = _family()
    support = packet_jet_support(family)
    report = profile_packet_jet_runtime(
        family,
        directions,
        support=support,
        jet=family.jet("x", ("x", "x")),
        repetitions=1,
        warmup=0,
        include_stream_write=True,
        stream_chunk_size=1,
    )
    report.certificate.require()
    assert report.certificate.exactness.value == "heuristic"
    assert [stage.name for stage in report.stages] == [
        "address_enumeration",
        "plan_compilation",
        "plan_execution",
        "snapshot_capture",
        "portable_serialization",
        "stream_snapshot_write",
    ]
    assert "platform" in report.environment
    assert report.estimate.active_addresses == 2
    assert report.certificate.assumptions_unchecked



def test_profile_operation_preserves_an_existing_tracemalloc_session() -> None:
    tracemalloc.start()
    try:
        report = profile_operation("nested tracing", lambda: [0] * 32, repetitions=1, warmup=0)
        assert tracemalloc.is_tracing()
        assert report.maximum_peak_bytes >= 0
    finally:
        tracemalloc.stop()

def test_curated_beta_namespace_is_explicit_and_deterministic() -> None:
    manifest = create_beta_api_manifest()
    assert manifest.namespaces == BETA_PUBLIC_NAMESPACES
    assert manifest.stability_level == BETA_API_STABILITY_LEVEL
    assert manifest == create_beta_api_manifest()
    assert tuple(beta.__all__) == tuple(dict.fromkeys(beta.__all__))
    assert all(hasattr(beta, name) for name in beta.__all__)
    assert any(symbol.name == "capture_packet_jet_stream" for symbol in manifest.symbols)
    assert any(symbol.name == "check_beta_api" for symbol in manifest.symbols)


def test_beta_policy_rejects_signature_changes_even_when_additive_policy_would_not() -> None:
    reference = PublicApiManifest(
        package_version="0.11",
        namespaces=("example",),
        symbols=(PublicApiSymbol("example", "f", "function", "example", "f", "(x)"),),
        stability_level=BETA_API_STABILITY_LEVEL,
    )
    current = PublicApiManifest(
        package_version="0.12",
        namespaces=("example",),
        symbols=(PublicApiSymbol("example", "f", "function", "example", "f", "(x, y=0)"),),
        stability_level=BETA_API_STABILITY_LEVEL,
    )
    assert compare_public_api(reference, current).compatible
    assert not compare_public_api(reference, current, strict_signatures=True).compatible


def test_beta_manifest_roundtrip_and_check(tmp_path: Path) -> None:
    manifest = create_beta_api_manifest()
    path = tmp_path / "beta.json"
    save_public_api_manifest(manifest, path)
    assert load_beta_api_manifest(path) == manifest
    report = check_beta_api(path)
    report.certificate.require()
    assert report.strict_signatures

    symbol = manifest.symbols[0]
    changed = replace(symbol, signature=(symbol.signature or "") + " changed")
    altered = PublicApiManifest(
        package_version=manifest.package_version,
        namespaces=manifest.namespaces,
        symbols=tuple(sorted((changed, *manifest.symbols[1:]))),
        stability_level=manifest.stability_level,
    )
    altered_path = tmp_path / "altered.json"
    save_public_api_manifest(altered, altered_path)
    assert not check_beta_api(altered_path).compatible
