"""Release identity consistency and public documentation entry points."""
from __future__ import annotations
import json
import re
import tomllib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def test_current_documentation_and_contract_entry_points_exist() -> None:
    required = (
        "docs/index.md", "docs/USER_GUIDE.md", "docs/PUBLIC_API.md",
        "docs/STABLE_API.md", "docs/RELEASE_PROCESS.md",
        "src/komorebi_math/data/release_contract_1_0.json",
        "src/komorebi_math/data/upgrade_contract_1_0.json",
        "src/komorebi_math/data/validation_evidence_1_0.json",
        "src/komorebi_math/data/stable_documentation_1_0.json",
    )
    assert [name for name in required if not (ROOT / name).is_file()] == []


def test_runtime_build_and_contract_versions_are_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    runtime = (ROOT / "src/komorebi_math/_version.py").read_text(encoding="utf-8")
    identity = json.loads((ROOT / "RELEASE_ID.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "src/komorebi_math/data/release_contract_1_0.json").read_text(encoding="utf-8"))
    version = project["version"]
    assert version == identity["package_version"] == contract["package_version"]
    runtime_match = re.search(r'^__version__\s*=\s*"([^"]+)"$', runtime, re.MULTILINE)
    assert runtime_match is not None and runtime_match.group(1) == version
    if version == "1.0.0":
        stage = "final"
    elif re.fullmatch(r"1\.0\.0rc[1-9]\d*", version):
        stage = "release-candidate"
    else:
        assert re.fullmatch(r"1\.0\.0b[1-9]\d*", version)
        stage = "beta"
    assert contract["stage"] == stage
    if stage in {"release-candidate", "final"}:
        assert identity["wheel_build_tag"] == ""


def test_public_development_status_remains_beta() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    # The version and contract stage identify the prepared artifacts. They do
    # not promote the current public project beyond its declared beta status.
    assert {value for value in project["classifiers"] if value.startswith("Development Status ::")} == {
        "Development Status :: 4 - Beta"
    }
