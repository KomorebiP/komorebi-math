"""Release archives keep executable sources and reject obsolete build output."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from verification.build_distributions import _audit_sdist
from verification import build_source_release
from verification.release_metadata import check_wheel_install_contract, wheel_filename
from verification.source_layout import source_files


def _write(root: Path, name: str, content: str = "fixture\n") -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _source(root: Path) -> None:
    for name in ("README.md", "LICENSE.md", "verification/build_distributions.py",
                 "verification/build_source_release.py", "src/komorebi_math/__init__.py"):
        _write(root, name)
    _write(root, "pyproject.toml", '[project]\nname = "komorebi-math"\nversion = "1.0.0"\n')
    _write(root, "RELEASE_ID.json", json.dumps({"release_id": "1.0.0", "wheel_build_tag": ""}))


def _sdist(root: Path, destination: Path, extra: str | None = None) -> None:
    with tarfile.open(destination, "w:gz") as archive:
        for name in sorted(source_files(root)):
            archive.add(root / name, arcname="komorebi_math-1.0.0/" + name)
        if extra:
            payload = b"old binary or generated report"
            info = tarfile.TarInfo("komorebi_math-1.0.0/" + extra)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_standard_wheel_has_no_historical_build_identifier(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "# fixture\n")
    _write(tmp_path, "pyproject.toml", '[project]\nname = "komorebi-math"\nversion = "1.0.0"\nreadme = "README.md"\n')
    prefix = "komorebi_math-1.0.0.dist-info/"
    files = {
        prefix + "METADATA": b"Metadata-Version: 2.4\nName: komorebi-math\nVersion: 1.0.0\n\n# fixture\n",
        prefix + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        prefix + "RECORD": b"",
        prefix + "top_level.txt": b"komorebi_math\n",
        prefix + "entry_points.txt": b"",
    }
    assert wheel_filename("1.0.0") == "komorebi_math-1.0.0-py3-none-any.whl"
    assert check_wheel_install_contract(tmp_path, files, "")["passed"]
    files[prefix + "WHEEL"] += b"Build: 4b21\n"
    with pytest.raises(ValueError, match="Build"):
        check_wheel_install_contract(tmp_path, files, "")


def test_historical_build_identifiers_remain_readable() -> None:
    assert wheel_filename("1.0.0b4", "4b21") == "komorebi_math-1.0.0b4-4b21-py3-none-any.whl"
    with pytest.raises(ValueError, match="build tag"):
        wheel_filename("1.0.0", "../old")


def test_source_inventory_keeps_runtime_inputs_and_excludes_development_output(tmp_path: Path) -> None:
    _source(tmp_path)
    required = {
        "src/komorebi_math/data/public_api_1_0_0b4.json",
        "verification/installed_runtime_profile.json",
        "tests/fixtures/numerical_control.json",
        "docs/assets/mathjax/LICENSE",
    }
    excluded = {
        "release/predecessors/old.whl", "development/history/old.md",
        "qualification/old.json", "dist/old.tar.gz", ".venv-test/bin/python",
        "README_BLOCK21.md", "RELEASE_REPAIR_R1.md", "build_block21_handoff.sh",
        "verification/BLOCK14_FINAL_VALIDATION.json", "src/komorebi_math/__pycache__/x.pyc",
    }
    for name in required | excluded:
        _write(tmp_path, name)
    selected = source_files(tmp_path)
    assert required <= selected
    assert not (excluded & selected)


@pytest.mark.parametrize("extra", ["release/predecessors/old.whl", "RELEASE_REPAIR_R1.md",
                                  "qualification/old.json", "development/history/audit.md"])
def test_sdist_rejects_leftover_binary_or_development_output(tmp_path: Path, extra: str) -> None:
    root = tmp_path / "source"
    root.mkdir()
    _source(root)
    archive = tmp_path / "source.tar.gz"
    _sdist(root, archive)
    assert _audit_sdist(archive, "1.0.0", root)["members"] > 0
    _sdist(root, archive, extra)
    with pytest.raises(SystemExit, match="stale source layout"):
        _audit_sdist(archive, "1.0.0", root)


def test_source_release_rejects_distributions_outside_qualified_inventory(tmp_path: Path, monkeypatch) -> None:
    _source(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "komorebi_math-1.0.0-py3-none-any.whl").write_bytes(b"changed wheel")
    (dist / "komorebi_math-1.0.0.tar.gz").write_bytes(b"changed sdist")
    gate = {"package_version": "1.0.0", "environment": {"release_binding": {"artifacts": [
        {"name": "komorebi_math-1.0.0-py3-none-any.whl", "sha256": "0" * 64, "bytes": 13},
        {"name": "komorebi_math-1.0.0.tar.gz", "sha256": "0" * 64, "bytes": 13},
    ]}}}
    monkeypatch.setattr(build_source_release, "_load_passing_gate", lambda *args: gate)
    with pytest.raises(SystemExit, match="differs from the qualified distributions"):
        build_source_release.build(tmp_path, dist, dist, tmp_path / "gate.json", python="unused")


def test_source_inventory_rejects_symlink_ancestor_of_workflows(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside"
    _write(outside, "workflows/ci.yml", "private external content\n")
    try:
        (root / ".github").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        build_source_release._collect(root)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special-file control")
def test_source_inventory_rejects_selected_fifo_without_opening_it(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "README.md")
    with pytest.raises(ValueError, match="special file"):
        source_files(tmp_path)


def test_source_inventory_rejects_regular_file_in_place_of_source_directory(tmp_path: Path) -> None:
    _write(tmp_path, "docs", "not a directory\n")
    with pytest.raises(ValueError, match="directory"):
        source_files(tmp_path)


def test_source_inventory_ignores_excluded_history_and_cache_links(tmp_path: Path) -> None:
    _write(tmp_path, "README.md")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "docs").mkdir()
    try:
        (tmp_path / "docs/history").symlink_to(outside, target_is_directory=True)
        (tmp_path / ".venv-test").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    assert source_files(tmp_path) == {"README.md"}
