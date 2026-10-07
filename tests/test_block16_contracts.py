from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from komorebi_math.compatibility import (
    create_beta_api_manifest,
    create_public_api_manifest,
    create_stable_api_manifest,
)
from komorebi_math.coordinates import AnalysisSynthesisSystem
from komorebi_math.exceptions import ValidationError
from komorebi_math.persistence_contract import (
    PortableCodecContract,
    PortableCodecContractEntry,
    compare_portable_codec_contract,
    create_portable_codec_contract,
)
from komorebi_math.stable_contract import check_stable_contract
from komorebi_math.stable_semantics import run_stable_semantics_fixture
from komorebi_math.support import SupportMatrix, load_support_matrix


def test_redundant_analysis_synthesis_uses_additive_components() -> None:
    system = AnalysisSynthesisSystem.canonical_dual(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    )
    components = system.components()
    assert np.allclose(sum(components), np.eye(2))
    assert any(np.linalg.norm(component @ component - component) > 1e-6 for component in components)
    for index, component in enumerate(components):
        assert np.allclose(system.projector(index), component)


def test_stable_contract_layers_pass_together() -> None:
    report = check_stable_contract()
    report.certificate.require()
    assert report.api.compatible
    assert report.semantics.passed
    assert report.persistence.compatible


def test_api_factory_versions_follow_runtime() -> None:
    versions = {
        create_public_api_manifest().package_version,
        create_beta_api_manifest().package_version,
        create_stable_api_manifest().package_version,
    }
    assert versions == {"1.0.0b4"}


def test_semantic_fixture_detects_changed_expected_value(tmp_path: Path) -> None:
    source = files("komorebi_math").joinpath("data/stable_semantics_0_16.json")
    fixture = json.loads(source.read_text(encoding="utf-8"))
    fixture["cases"][0]["expected"]["reconstructed"][0] = 999.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    report = run_stable_semantics_fixture(path)
    assert not report.passed
    assert report.certificate.residuals["failed_cases"] == 1.0


def test_codec_contract_detects_removed_tag() -> None:
    reference = create_portable_codec_contract()
    current = PortableCodecContract("1.0.0b1", reference.entries[1:])
    report = compare_portable_codec_contract(reference, current)
    assert not report.compatible
    assert report.removed == (reference.entries[0].tag,)


def test_codec_contract_detects_missing_adjacent_migration() -> None:
    reference = create_portable_codec_contract()
    first = reference.entries[0]
    changed = PortableCodecContractEntry(first.tag, first.version + 20, first.implementation_type)
    current = PortableCodecContract(
        "1.0.0b1",
        tuple(sorted((changed, *reference.entries[1:]))),
    )
    report = compare_portable_codec_contract(reference, current)
    assert not report.compatible
    assert report.missing_migrations


def test_codec_contract_rejects_nonobject_entries() -> None:
    with pytest.raises(ValidationError, match="entries must be objects"):
        PortableCodecContract.from_dict(
            {
                "schema": "komorebi-math.portable-codec-contract",
                "schema_version": 1,
                "package_version": "0.16.0a2",
                "entries": [None],
            }
        )


def test_support_matrix_is_machine_readable_and_strict() -> None:
    matrix = load_support_matrix()
    assert matrix.python["minimum"] == "3.11"
    assert {entry.identifier for entry in matrix.validated_environments} == {
        "windows-0.16.0a2-inherited",
        "linux-1.0.0b4-publication-gate",
    }
    payload = matrix.to_dict()
    payload["validated_environments"] = [None]
    with pytest.raises(ValidationError, match="must contain objects"):
        SupportMatrix.from_dict(payload)
