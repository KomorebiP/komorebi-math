"""Build, canonicalize, replay, and audit the Block 21 distributions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any
from unicodedata import normalize

from setuptools.build_meta import build_sdist

try:
    from .reproducible_wheel import build_canonical_wheel, reproducible_metadata
except ImportError:
    from reproducible_wheel import build_canonical_wheel, reproducible_metadata

SOURCE_DATE_EPOCH = 1787356800


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if match is None:
        raise SystemExit("package version is absent from pyproject.toml")
    return match.group(1)


def _clean(root: Path) -> None:
    for relative in ("build", "src/komorebi_math.egg-info"):
        path = (root / relative).resolve()
        if root not in path.parents:
            raise SystemExit(f"unsafe build-cache path: {path}")
        if path.exists():
            shutil.rmtree(path)


def _safe_name(name: str) -> None:
    path = PurePosixPath(name.rstrip("/"))
    text = path.as_posix()
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
        or (path.parts and ":" in path.parts[0])
        or normalize("NFC", text) != text
    ):
        raise SystemExit(f"unsafe distribution member: {name!r}")


def _canonicalize_sdist(root: Path, path: Path) -> None:
    sys.path.insert(0, str(root / "verification"))
    try:
        from canonicalize_block20_sdist import canonicalize
    finally:
        sys.path.pop(0)
    canonicalize(path, epoch=SOURCE_DATE_EPOCH)


def _build_once(root: Path, destination: Path, version: str) -> tuple[Path, Path]:
    _clean(root)
    destination.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    try:
        os.chdir(root)
        identity = json.loads((root / "RELEASE_ID.json").read_text(encoding="utf-8"))
        build_tag = identity["wheel_build_tag"]
        wheel_name = build_canonical_wheel(str(destination), build_tag)
        with reproducible_metadata():
            sdist_name = build_sdist(str(destination))
    finally:
        os.chdir(old)
    wheel = destination / wheel_name
    sdist = destination / sdist_name
    expected = {
        f"komorebi_math-{version}-{build_tag}-py3-none-any.whl",
        f"komorebi_math-{version}.tar.gz",
    }
    observed = {path.name for path in destination.iterdir() if path.is_file()}
    if observed != expected:
        raise SystemExit(f"unexpected distribution outputs: {sorted(observed)}")
    _canonicalize_sdist(root, sdist)
    return wheel, sdist


def _audit_wheel(path: Path, version: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise SystemExit("wheel contains duplicate members")
        for name in names:
            _safe_name(name)
        folded = [normalize("NFC", name).casefold() for name in names]
        if len(folded) != len(set(folded)):
            raise SystemExit("wheel contains case-colliding members")
        metadata_name = next(
            (name for name in names if name.endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_name is None:
            raise SystemExit("wheel has no METADATA")
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        if metadata.get_all("Version", []) != [version]:
            raise SystemExit("wheel metadata version mismatch")
        if not any(name.endswith(".dist-info/RECORD") for name in names):
            raise SystemExit("wheel has no RECORD")
        return {"members": len(names), "metadata": metadata_name}


def _audit_sdist(path: Path, version: str) -> dict[str, Any]:
    expected_root = f"komorebi_math-{version}"
    names: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            _safe_name(member.name)
            if not (member.isfile() or member.isdir()):
                raise SystemExit(f"unsupported sdist member type: {member.name}")
            names.append(member.name.rstrip("/"))
    if len(names) != len(set(names)):
        raise SystemExit("sdist contains duplicate members")
    folded = [normalize("NFC", name).casefold() for name in names]
    if len(folded) != len(set(folded)):
        raise SystemExit("sdist contains case-colliding members")
    roots = {PurePosixPath(name).parts[0] for name in names}
    if roots != {expected_root}:
        raise SystemExit(f"sdist root mismatch: {sorted(roots)}")
    required = {
        f"{expected_root}/README.md",
        f"{expected_root}/README_BLOCK21.md",
        f"{expected_root}/REPRODUCIBILITY.md",
        f"{expected_root}/SECURITY.md",
        f"{expected_root}/CONTRIBUTING.md",
        f"{expected_root}/verification/block21_publication_gate.py",
        f"{expected_root}/verification/build_block21_handoff.py",
        f"{expected_root}/examples/block21_public_capability_tour.py",
    }
    missing = sorted(required - set(names))
    if missing:
        raise SystemExit(f"sdist is missing Block 21 files: {missing}")
    return {"members": len(names), "root": expected_root}


def build(root: Path, destination: Path, report_path: Path) -> dict[str, Any]:
    root = root.resolve()
    destination = destination.resolve()
    report_path = report_path.resolve()
    version = _version(root)
    previous_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = str(SOURCE_DATE_EPOCH)
    try:
        with tempfile.TemporaryDirectory(prefix="komorebi-block21-build-") as temporary:
            workspace = Path(temporary)
            first_wheel, first_sdist = _build_once(root, workspace / "first", version)
            replay_wheel, replay_sdist = _build_once(root, workspace / "replay", version)
            if first_wheel.read_bytes() != replay_wheel.read_bytes():
                raise SystemExit("wheel build is not byte reproducible")
            if first_sdist.read_bytes() != replay_sdist.read_bytes():
                raise SystemExit("sdist build is not byte reproducible")
            wheel_audit = _audit_wheel(first_wheel, version)
            sdist_audit = _audit_sdist(first_sdist, version)
            try:
                from .release_integrity import verify_release
            except ImportError:
                from release_integrity import verify_release
            integrity = verify_release(root, workspace / "first")
            if not integrity["passed"]:
                raise SystemExit("source/distribution parity failed: " + json.dumps(integrity))
            destination.mkdir(parents=True, exist_ok=True)
            for existing in destination.glob(f"komorebi_math-{version}*"):
                if existing.is_file() or existing.is_symlink():
                    existing.unlink()
                else:
                    shutil.rmtree(existing)
            final_wheel = destination / first_wheel.name
            final_sdist = destination / first_sdist.name
            shutil.copy2(first_wheel, final_wheel)
            shutil.copy2(first_sdist, final_sdist)
    finally:
        if previous_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous_epoch
        _clean(root)

    def artifact(path: Path, audit: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
            "audit": audit,
        }

    record = {
        "schema": "komorebi-math.block21-distribution-build",
        "schema_version": 1,
        "package_version": version,
        "release_identity": json.loads((root / "RELEASE_ID.json").read_text(encoding="utf-8")),
        "source_distribution_integrity": integrity,
        "passed": True,
        "byte_reproducible_rebuild": True,
        "wheel_envelope": "sorted ZIP_STORED members, fixed POSIX metadata; member bytes preserved",
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "toolchain": {
            "python": platform.python_version(),
            "setuptools": importlib.metadata.version("setuptools"),
            "wheel_backend": "setuptools.command.bdist_wheel",
        },
        "artifacts": [
            artifact(final_wheel, wheel_audit),
            artifact(final_sdist, sdist_audit),
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-directory", default="release/dist")
    parser.add_argument("--report", default="release/BLOCK21_DISTRIBUTION_BUILD.json")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    destination = Path(args.output_directory)
    if not destination.is_absolute():
        destination = root / destination
    report = Path(args.report)
    if not report.is_absolute():
        report = root / report
    print(json.dumps(build(root, destination, report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
