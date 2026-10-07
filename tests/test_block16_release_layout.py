"""Block 16 release-tree layout and version consistency."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_block16_release_files_exist() -> None:
    required = (
        "README_BLOCK16.md",
        "Run_Block16_Source_Validation.ps1",
        "Run_Block16_Batched_Validation.ps1",
        "Install_And_Verify_Block16.ps1",
        "Run_GPU_Validation.ps1",
        "Build_Block16_Distributions.ps1",
        "docs/BLOCK_16_RELEASE.md",
        "docs/STABLE_API.md",
        "docs/SUPPORTED_PLATFORMS.md",
        "docs/GPU_VALIDATION.md",
        "docs/MIGRATION_TO_1_0.md",
        "docs/RELEASE_PROCESS.md",
        "docs/COMPANION_C_INTEGRATION.md",
        "constraints/base-py311.txt",
        "constraints/base-py312.txt",
        "constraints/base-py313.txt",
        "environment/windows-cuda126.yml",
        ".github/workflows/ci.yml",
        "src/komorebi_math/data/release_contract_0_16.json",
        "verification/COMPANION_C_BLOCK16_CROSSWALK.json",
    )
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    assert missing == []


def test_canonical_block16_records_identify_historical_version() -> None:
    # Block 16 remains a retained predecessor record after the package advances.
    # Current entry-point documentation is intentionally versioned for Block 17.
    for relative in (
        "README_BLOCK16.md",
        "docs/BLOCK_16_RELEASE.md",
        "src/komorebi_math/data/release_contract_0_16.json",
    ):
        assert "0.16.0a2" in (ROOT / relative).read_text(encoding="utf-8"), relative


def test_release_scripts_reference_current_wheel() -> None:
    for relative in ("Install_And_Verify_Block16.ps1", "install_and_verify_block16.sh"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "komorebi_math-0.16.0a2-*.whl" in text
        assert "0.16.0a2" in text
