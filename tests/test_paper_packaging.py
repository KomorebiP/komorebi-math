"""Packaging excludes synchronization staging without losing real files."""

import tomllib
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools._distutils.filelist import FileList

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_excludes_transients_after_all_support_includes():
    legitimate = {
        "src/komorebi_math/data/evidence.json",
        "docs/mathematics.md",
        "docs/assets/fixture.bin",
        "examples/finite_example.py",
        "tests/test_finite_example.py",
        "verification/check.py",
        "verification/report.json",
    }
    transients = {
        "src/komorebi_math/data/.rsync-tmp/evidence.json",
        "docs/.rsync-tmp/mathematics.md",
        "docs/nested/.rsync-tmp/mathematics.md",
        "docs/assets/.rsync-tmp/fixture.bin",
        "examples/.rsync-tmp/finite_example.py",
        "tests/.rsync-tmp/test_finite_example.py",
        "verification/.rsync-tmp/check.py",
        "verification/.rsync-tmp/report.json",
    }
    manifest = FileList()
    manifest.allfiles = sorted(legitimate | transients)
    for raw_line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            manifest.process_template_line(line)
    included = set(manifest.files)
    assert legitimate <= included
    assert not transients & included


def test_namespace_discovery_excludes_transient_packages(tmp_path):
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    discovery = config["tool"]["setuptools"]["packages"]["find"]
    source_root = tmp_path / discovery["where"][0]
    paths = (
        "komorebi_math/arithmetic/real_module.py",
        "komorebi_math/data/evidence.json",
        "komorebi_math/.rsync-tmp/duplicate.py",
        "komorebi_math/.rsync-tmp/nested/duplicate.py",
        "komorebi_math/data/.rsync-tmp/evidence.json",
        "komorebi_math/rsync-tmp/duplicate.py",
        "komorebi_math/rsync-tmp/nested/duplicate.py",
    )
    for relative in paths:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic packaging fixture\n", encoding="utf-8")
    unfiltered = find_namespace_packages(where=source_root)
    # Dot directories are already ignored by setuptools. The undotted fixture
    # also exercises the configured namespace exclusions and their descendants.
    assert "komorebi_math.rsync-tmp" in unfiltered
    assert "komorebi_math.rsync-tmp.nested" in unfiltered
    packages = find_namespace_packages(where=source_root, exclude=discovery["exclude"])
    assert {"komorebi_math", "komorebi_math.arithmetic", "komorebi_math.data"} <= set(packages)
    assert not any("rsync-tmp" in name for name in packages)
    assert (source_root / "komorebi_math/arithmetic/real_module.py").is_file()
