"""Distribution identity, snapshot isolation and transient-member controls."""

import io
import json
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from verification import build_block21_distributions as builder
from verification import release_integrity
from verification.build_block21_distributions import _audit_sdist, _audit_wheel, _snapshot_source

INFO = "komorebi_math-1.0.0b4.dist-info"
VERSION = "1.0.0b4"


def _wheel(tmp_path: Path, metadata: bytes) -> Path:
    path = tmp_path / "fixture.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{INFO}/METADATA", metadata)
        archive.writestr(f"{INFO}/RECORD", b"")
    return path


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_wheel_version_header_accepts_platform_line_endings(tmp_path: Path, newline: bytes) -> None:
    metadata = newline.join(
        [b"Metadata-Version: 2.4", b"Name: komorebi-math", b"Version: 1.0.0b4", b"", b"Exact math"]
    )
    report = _audit_wheel(_wheel(tmp_path, metadata), VERSION)
    assert report["metadata"] == f"{INFO}/METADATA"


@pytest.mark.parametrize(
    "metadata",
    [
        b"Metadata-Version: 2.4\nName: komorebi-math\n\n",
        b"Metadata-Version: 2.4\nVersion: 0.0.0\n\n",
        b"Metadata-Version: 2.4\nVersion: 1.0.0b4\nVersion: 1.0.0b4\n\n",
        b"Metadata-Version: 2.4\nVersion: 1.0.0b4\nVersion: 0.0.0\n\n",
        b"Metadata-Version: 2.4\nName: komorebi-math\n\nVersion: 1.0.0b4\n",
    ],
    ids=["missing", "wrong", "duplicate", "conflicting", "body-only"],
)
def test_wheel_version_header_rejects_missing_or_ambiguous_identity(tmp_path: Path, metadata: bytes) -> None:
    with pytest.raises(SystemExit, match="wheel metadata version mismatch"):
        _audit_wheel(_wheel(tmp_path, metadata), VERSION)


def test_source_snapshot_preserves_inputs_and_modes_without_generated_trees(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    legitimate = {
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        ".gitignore",
        ".github/workflows/check.yml",
        "src/komorebi_math/core.py",
        "src/komorebi_math/py.typed",
        "src/komorebi_math/data/catalog.json",
        "src/komorebi_math/build/config.json",
        "docs/build/legitimate.md",
        "environment/requirements.yml",
        "examples/example.py",
        "tests/test_example.py",
        "tests/fixtures/metadata.egg-info/PKG-INFO",
        "verification/check.py",
        "release/predecessors/previous.whl",
        "verify.sh",
    }
    excluded = {
        ".git/index",
        ".pytest_cache/state",
        ".ruff_cache/state",
        "build/lib/cache.py",
        "dist/output.whl",
        "site/index.html",
        "env/bin/python",
        "venv/bin/python",
        ".venv/bin/python",
        ".venv-build/bin/python",
        "src/komorebi_math.egg-info/SOURCES.txt",
        "src/komorebi_math/__pycache__/core.pyc",
        "src/komorebi_math/data/.rsync-tmp/catalog.json",
        "docs/.rsync-tmp/temporary.md",
        "release/dist/current.whl",
        f"komorebi_math-{VERSION}/PKG-INFO",
        "paper_sdist_staging_success/PKG-INFO",
    }
    for relative in legitimate | excluded:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    (root / "verify.sh").chmod(0o755)
    snapshot = _snapshot_source(root, tmp_path / "private" / "source", VERSION)
    assert all((snapshot / name).read_bytes() == name.encode() for name in legitimate)
    assert not any((snapshot / name).exists() for name in excluded)
    assert stat.S_IMODE((snapshot / "verify.sh").stat().st_mode) == 0o755
    (root / "src/komorebi_math/core.py").write_bytes(b"changed after snapshot")
    assert (snapshot / "src/komorebi_math/core.py").read_bytes() == b"src/komorebi_math/core.py"
    with pytest.raises(SystemExit, match="outside the original checkout"):
        _snapshot_source(root, root / "recursive-copy", VERSION)


def test_build_replays_private_snapshot_and_compares_original_source(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text(f'version = "{VERSION}"\n', encoding="utf-8")
    (root / "RELEASE_ID.json").write_text(json.dumps({"wheel_build_tag": "4b21"}), encoding="utf-8")
    cache = root / "build" / "existing-cache"
    cache.parent.mkdir()
    cache.write_bytes(b"original cache must remain")
    calls = []

    def fake_build_once(source, destination, version):
        assert source != root and root not in source.parents
        assert not (source / "build" / "existing-cache").exists()
        assert (source / "pyproject.toml").read_text(encoding="utf-8") == f'version = "{VERSION}"\n'
        calls.append(source)
        destination.mkdir()
        wheel = destination / f"komorebi_math-{version}-4b21-py3-none-any.whl"
        sdist = destination / f"komorebi_math-{version}.tar.gz"
        wheel.write_bytes(b"identical wheel")
        sdist.write_bytes(b"identical sdist")
        return wheel, sdist

    def fake_parity(source, distribution):
        assert source == root
        assert distribution.name == "first"
        return {"passed": True}

    monkeypatch.setattr(builder, "_build_once", fake_build_once)
    monkeypatch.setattr(builder, "_audit_wheel", lambda *_: {"members": 1})
    monkeypatch.setattr(builder, "_audit_sdist", lambda *_: {"members": 1})
    monkeypatch.setattr(release_integrity, "verify_release", fake_parity)
    report = builder.build(root, tmp_path / "artifacts", tmp_path / "report.json")
    assert report["passed"] and report["byte_reproducible_rebuild"]
    assert len(calls) == 2 and calls[0] == calls[1]
    assert cache.read_bytes() == b"original cache must remain"


@pytest.mark.parametrize(
    "member", ["komorebi_math/.rsync-tmp/py.typed", "komorebi_math/data/.rsync-tmp/catalog.json"]
)
def test_wheel_rejects_synchronization_members(tmp_path: Path, member: str) -> None:
    wheel = _wheel(tmp_path, b"Metadata-Version: 2.4\nVersion: 1.0.0b4\n\n")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, b"temporary duplicate")
    with pytest.raises(SystemExit, match="transient synchronization member"):
        _audit_wheel(wheel, VERSION)


@pytest.mark.parametrize("suffix", ["src/komorebi_math/data/.rsync-tmp/catalog.json", "docs/.rsync-tmp"])
def test_sdist_rejects_synchronization_members(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / "fixture.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(f"komorebi_math-{VERSION}/{suffix}")
        payload = b"temporary duplicate"
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(SystemExit, match="transient synchronization member"):
        _audit_sdist(path, VERSION)
