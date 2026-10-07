from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from komorebi_math.compatibility import (
    STABLE_API_STABILITY_LEVEL,
    check_beta_api,
    check_public_api,
    check_stable_api,
    create_stable_api_manifest,
)
from komorebi_math.export.persistence import load_portable
from komorebi_math.qualification import (
    RELEASE_QUALIFICATION_SCHEMA,
    run_release_qualification,
    run_stable_semantic_verification,
    runtime_inventory,
)
from komorebi_math.refinement import PolyphaseRefinement, PolyphaseRefinementTower
from komorebi_math.verify import run_stable_verification


def test_all_api_manifests_match_block16_runtime() -> None:
    check_public_api(strict_signatures=True).certificate.require()
    check_beta_api().certificate.require()
    check_stable_api().certificate.require()
    manifest = create_stable_api_manifest()
    assert manifest.stability_level == STABLE_API_STABILITY_LEVEL
    assert len(manifest.symbols) == 131


def test_stable_semantic_fixture_passes() -> None:
    report = run_stable_semantic_verification()
    assert report.passed
    assert report.passed_count == report.total_count == 10


def test_installed_verification_layers_pass() -> None:
    stable = run_stable_verification()
    assert stable.passed
    assert stable.total_count == 6
    completed = subprocess.run(
        [sys.executable, "-m", "komorebi_math", "verify", "block16"],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "verification passed" in completed.stdout


def test_release_qualification_without_hardware_requirement(tmp_path: Path) -> None:
    report = run_release_qualification()
    assert report.passed
    assert report.gpu is None
    assert report.certificate().valid
    path = report.write(tmp_path / "qualification.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == RELEASE_QUALIFICATION_SCHEMA
    assert payload["version"] == "1.0.0b4"
    assert payload["certificate"]["valid"] is True


def test_runtime_inventory_is_data_only() -> None:
    inventory = runtime_inventory()
    assert inventory["package_version"] == "1.0.0b4"
    assert inventory["numpy"]
    assert inventory["python_executable"]
    assert set(inventory["distributions"]) >= {"torch", "cupy-cuda12x", "numba"}


def test_0_15_portable_fixtures_load_under_block16() -> None:
    root = Path(__file__).parent / "fixtures" / "portable"
    refinement = load_portable(root / "polyphase_refinement_0_15.json")
    tower = load_portable(root / "polyphase_tower_0_15.json")
    assert refinement == PolyphaseRefinement(4, 12)
    assert tower == PolyphaseRefinementTower((2, 6, 24))
