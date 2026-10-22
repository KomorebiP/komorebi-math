"""Release source selection and retained compatibility inputs."""
from __future__ import annotations
import json
from pathlib import Path
from verification.source_layout import source_files
ROOT = Path(__file__).resolve().parents[1]


def test_required_release_sources_are_selected() -> None:
    required = {
        "README.md", "LICENSE.md", "pyproject.toml", "RELEASE_ID.json", "MANIFEST.in",
        "verification/build_distributions.py", "verification/build_source_release.py",
        "verification/check_release.py", "verification/source_audit.py",
        "verification/package_audit.py", "verification/source_layout.py",
        "docs/USER_GUIDE.md", "docs/STABLE_API.md", "docs/RELEASE_PROCESS.md",
        "constraints/base-py311.txt", "constraints/base-py312.txt", "constraints/base-py313.txt",
        "environment/windows-cuda126.yml", ".github/workflows/ci.yml",
        "src/komorebi_math/data/release_contract_1_0.json",
    }
    assert required <= source_files(ROOT)


def test_predecessor_contract_keeps_its_historical_identity() -> None:
    record = ROOT / "src/komorebi_math/data/release_contract_0_16.json"
    assert json.loads(record.read_text(encoding="utf-8"))["package_version"] == "0.16.0a2"


def test_obsolete_launchers_and_generated_releases_are_not_selected(tmp_path: Path) -> None:
    paths = (
        "README.md", "check_release.sh", "Build_Block16_Distributions.ps1",
        "verify_repair.sh", "development/history/README.md",
        "release/dist/previous.whl", "dist/current.whl", "qualification/report.json",
    )
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    assert source_files(tmp_path) == {"README.md", "check_release.sh"}
