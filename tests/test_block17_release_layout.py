"""Block 17 source-tree layout and version identity."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_block17_required_source_contract_files_exist() -> None:
    required = (
        "README_BLOCK17.md",
        "Run_Block17_Source_Validation.ps1",
        "Run_Block17_Upgrade_Validation.ps1",
        "Run_Block17_GPU_Validation.ps1",
        "Install_And_Verify_Block17.ps1",
        "Build_Block17_Distributions.ps1",
        "run_block17_source_validation.sh",
        "run_block17_upgrade_validation.sh",
        "run_block17_gpu_validation.sh",
        "install_and_verify_block17.sh",
        "build_block17_distributions.sh",
        "docs/BLOCK_17_RELEASE.md",
        "docs/UPGRADE_CONTRACT.md",
        "docs/BETA_TO_RC.md",
        "src/komorebi_math/data/release_contract_1_0.json",
        "src/komorebi_math/data/upgrade_contract_1_0.json",
        "src/komorebi_math/data/validation_evidence_1_0.json",
        "src/komorebi_math/data/stable_documentation_1_0.json",
        "verification/block17_package_audit.py",
        "verification/block17_static_audit.py",
        "verification/build_block17_distributions.py",
        "verification/block17_distribution_audit.py",
        "verification/run_block17_upgrade_validation.py",
    )
    missing = [item for item in required if not (ROOT / item).is_file()]
    assert not missing


def test_pyproject_and_runtime_version_are_public_beta() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version\s*=\s*"1\.0\.0b4"$', text, re.MULTILINE)
    assert 'Development Status :: 4 - Beta' in text
