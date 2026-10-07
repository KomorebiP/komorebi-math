"""Build and replay-audit a deterministic Block 21 publication handoff.

The handoff is a source repository archive, not merely a blind ZIP operation.
It binds every included payload file by path, mode, size, and SHA-256, rejects
unsafe or case-colliding paths, embeds qualified distributions, and replays the
quick publication gate from a clean extraction before reporting success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from unicodedata import normalize

SCHEMA = "komorebi-math.block21-handoff-manifest"
SCHEMA_VERSION = 1
FIXED_TIMESTAMP = (2026, 8, 22, 0, 0, 0)

_EXCLUDED_DIRECTORIES = {
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "env",
    "node_modules",
    "site",
    "temp",
    "temporary",
    "tmp",
    "venv",
    "virtualenv",
}
_EXCLUDED_NAMES = {
    "REPAIR_R2_HANDOFF_MANIFEST.json",
    "REPAIR_R2_HANDOFF_BUILD.json",
}


@dataclass(frozen=True, slots=True)
class PayloadEntry:
    path: str
    size: int
    sha256: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "mode": self.mode,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if match is None:
        raise SystemExit("package version is absent from pyproject.toml")
    return match.group(1)


def _safe_relative(relative: PurePosixPath) -> None:
    text = relative.as_posix()
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in text
        or (relative.parts and ":" in relative.parts[0])
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
        or normalize("NFC", text) != text
    ):
        raise SystemExit(f"unsafe handoff path: {text!r}")


def _excluded(relative: PurePosixPath) -> bool:
    parts = relative.parts
    if any(
        part.casefold() in _EXCLUDED_DIRECTORIES
        or part.casefold().startswith((".venv-", ".venv_"))
        or part.casefold().endswith(".egg-info")
        for part in parts
    ):
        return True
    if parts and parts[0].casefold() == "dist":
        return True
    name = relative.name
    if name in _EXCLUDED_NAMES:
        return True
    if name.endswith((".pyc", ".pyo")):
        return True
    folded = name.casefold()
    if "handoff" in folded and folded.endswith((".zip", ".zip.sha256")):
        return True
    return False


def _mode(path: Path) -> str:
    # Normalize permissions so a handoff built from Windows and one built from
    # POSIX do not diverge merely because their checkout preserved different
    # executable bits. Shell launchers remain executable; all other payload
    # files are ordinary read/write source artifacts.
    return "0755" if path.suffix == ".sh" else "0644"


def _collect(root: Path) -> tuple[PayloadEntry, ...]:
    entries: list[PayloadEntry] = []
    folded_paths: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        kept: list[str] = []
        for name in sorted(directory_names):
            child = directory_path / name
            relative = PurePosixPath(child.relative_to(root).as_posix())
            if _excluded(relative):
                continue
            if child.is_symlink():
                raise SystemExit(f"handoff source contains a symlink: {relative}")
            if not stat.S_ISDIR(child.stat(follow_symlinks=False).st_mode):
                raise SystemExit(f"handoff source contains a special directory entry: {relative}")
            kept.append(name)
        directory_names[:] = kept
        for name in sorted(file_names):
            path = directory_path / name
            relative = PurePosixPath(path.relative_to(root).as_posix())
            if _excluded(relative):
                continue
            _safe_relative(relative)
            if path.is_symlink():
                raise SystemExit(f"handoff source contains a symlink: {relative}")
            if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                raise SystemExit(f"handoff source contains a special file: {relative}")
            key = normalize("NFC", relative.as_posix()).casefold()
            prior = folded_paths.get(key)
            if prior is not None and prior != relative.as_posix():
                raise SystemExit(f"case-colliding handoff paths: {prior!r}, {relative.as_posix()!r}")
            folded_paths[key] = relative.as_posix()
            entries.append(
                PayloadEntry(
                    path=relative.as_posix(),
                    size=path.stat().st_size,
                    sha256=_sha256(path),
                    mode=_mode(path),
                )
            )
    return tuple(sorted(entries, key=lambda entry: entry.path))


def _artifact_inventory(directory: Path) -> tuple[dict[str, Any], ...]:
    artifacts: list[dict[str, Any]] = []
    if not directory.is_dir():
        raise SystemExit(f"artifact directory does not exist: {directory}")
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if not candidates:
        raise SystemExit(f"no wheel or source distribution found in {directory}")
    for path in candidates:
        artifacts.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return tuple(artifacts)


def _load_passing_gate(path: Path, root: Path | None = None) -> dict[str, Any]:
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not load publication-gate record {path}: {exc}") from exc
    if gate.get("schema") != "komorebi-math.block21-publication-gate":
        raise SystemExit("publication-gate record has the wrong schema")
    if gate.get("passed") is not True or gate.get("mode") != "full":
        raise SystemExit("a passing full Block 21 publication gate is required")
    full = gate.get("full_test_record")
    if not isinstance(full, dict) or full.get("returncode") != 0:
        raise SystemExit("publication-gate record does not bind a passing full test suite")
    digest=gate.get("report_sha256")
    content={key:value for key,value in gate.items() if key != "report_sha256"}
    if not isinstance(digest,str) or digest != _canonical_sha256(content):
        raise SystemExit("publication-gate self-digest does not match its content")
    checks=gate.get("checks")
    if not isinstance(checks,list) or not checks or any(row.get("passed") is not True for row in checks):
        raise SystemExit("publication-gate checks contain an omitted or failing result")
    if root is not None:
        try:
            from .release_integrity import release_binding
        except ImportError:
            from release_integrity import release_binding
        if gate.get("environment",{}).get("release_binding") != release_binding(root):
            raise SystemExit("publication-gate evidence is stale for the current source or artifacts")
    return gate


def _tree_digest(entries: Iterable[PayloadEntry]) -> str:
    payload = [entry.to_dict() for entry in entries]
    return _canonical_sha256(payload)


def _zip_info(path: str, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, FIXED_TIMESTAMP)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _write_zip(root: Path, archive_root: str, entries: tuple[PayloadEntry, ...], manifest: bytes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.temporary")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for entry in entries:
                source = root / entry.path
                mode = int(entry.mode, 8)
                archive.writestr(
                    _zip_info(f"{archive_root}/{entry.path}", mode),
                    source.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
            archive.writestr(
                _zip_info(f"{archive_root}/release/REPAIR_R2_HANDOFF_MANIFEST.json", 0o644),
                manifest,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replay(archive_path: Path, manifest: dict[str, Any], *, python: str) -> dict[str, Any]:
    archive_root = manifest["archive_root"]
    expected = {entry["path"]: entry for entry in manifest["payload"]}
    expected_names = {f"{archive_root}/{name}" for name in expected}
    manifest_name = f"{archive_root}/release/REPAIR_R2_HANDOFF_MANIFEST.json"
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise SystemExit("handoff ZIP contains duplicate members")
        folded: dict[str, str] = {}
        for name in names:
            relative = PurePosixPath(name)
            _safe_relative(relative)
            key = normalize("NFC", name).casefold()
            if key in folded and folded[key] != name:
                raise SystemExit(f"handoff ZIP contains case-colliding members: {folded[key]!r}, {name!r}")
            folded[key] = name
        if set(names) != expected_names | {manifest_name}:
            missing = sorted((expected_names | {manifest_name}) - set(names))
            extra = sorted(set(names) - (expected_names | {manifest_name}))
            raise SystemExit(f"handoff ZIP inventory mismatch; missing={missing[:5]}, extra={extra[:5]}")
        for path, entry in expected.items():
            name = f"{archive_root}/{path}"
            data = archive.read(name)
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise SystemExit(f"handoff ZIP payload mismatch: {path}")
            encoded_mode = (archive.getinfo(name).external_attr >> 16) & 0o7777
            if f"{encoded_mode:04o}" != entry["mode"]:
                raise SystemExit(f"handoff ZIP mode mismatch: {path}")
        embedded = json.loads(archive.read(manifest_name))
        if embedded != manifest:
            raise SystemExit("embedded handoff manifest does not match the qualified manifest")

        with tempfile.TemporaryDirectory(prefix="komorebi-block21-replay-") as temporary:
            archive.extractall(temporary)
            extracted = Path(temporary) / archive_root
            # ZipFile.extractall ignores stored execute bits. Restore exactly
            # the modes already validated above, rather than fabricating them.
            if os.name != "nt":
                for relative, entry in expected.items():
                    (extracted / relative).chmod(int(entry["mode"], 8))
                (extracted / "release/REPAIR_R2_HANDOFF_MANIFEST.json").chmod(0o644)
            gate_output = Path(temporary) / "replay-gate.json"
            gate_markdown = Path(temporary) / "replay-gate.md"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(extracted / "src")
            environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            process = subprocess.run(
                (
                    python,
                    str(extracted / "verification/block21_publication_gate.py"),
                    "--root",
                    str(extracted),
                    "--skip-mkdocs",
                    "--output",
                    str(gate_output),
                    "--markdown",
                    str(gate_markdown),
                ),
                cwd=extracted,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if process.returncode != 0:
                raise SystemExit(
                    "clean-extraction publication-gate replay failed:\n"
                    + process.stdout[-4000:]
                    + process.stderr[-4000:]
                )
            replay_gate = json.loads(gate_output.read_text(encoding="utf-8"))
            if replay_gate.get("passed") is not True:
                raise SystemExit("clean-extraction publication-gate replay returned a failing record")
            return {
                "passed": True,
                "quick_gate_sha256": replay_gate["report_sha256"],
                "capability_record_sha256": replay_gate["capability_record"]["record_sha256"],
                "member_count": len(names),
            }


def build(
    root: Path,
    output_directory: Path,
    artifact_directory: Path,
    gate_path: Path,
    *,
    python: str,
) -> tuple[Path, Path, Path]:
    root = root.resolve()
    output_directory = output_directory.resolve()
    artifact_directory = artifact_directory.resolve()
    version = _version(root)
    gate = _load_passing_gate(gate_path.resolve(), root)
    if gate.get("package_version") != version:
        raise SystemExit("publication-gate version does not match pyproject.toml")

    entries = _collect(root)
    identity = json.loads((root / "RELEASE_ID.json").read_text(encoding="utf-8"))
    archive_root = f"Komorebi_Math_Complete_Handoff_{identity['release_id']}"
    archive_path = output_directory / f"{archive_root}.zip"
    checksum_path = output_directory / f"{archive_root}.zip.sha256"
    build_path = output_directory / f"{archive_root}.build.json"
    artifacts = _artifact_inventory(artifact_directory)

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "package_version": version,
        "release_identity": identity,
        "archive_root": archive_root,
        "payload_file_count": len(entries),
        "payload_bytes": sum(entry.size for entry in entries),
        "payload_sha256": _tree_digest(entries),
        "payload": [entry.to_dict() for entry in entries],
        "publication_gate": {
            "path": gate_path.resolve().relative_to(root).as_posix(),
            "report_sha256": gate["report_sha256"],
            "full_test_summary": gate["full_test_record"]["summary"],
        },
        "capability_record_sha256": gate["capability_record"]["record_sha256"],
        "artifacts": list(artifacts),
        "determinism": {
            "fixed_zip_timestamp_utc": "2026-08-22T00:00:00Z",
            "lexicographic_member_order": True,
            "normalized_permissions": True,
            "manifest_excluded_from_payload_digest": True,
        },
        "security": {
            "regular_files_only": True,
            "symlinks_rejected": True,
            "absolute_and_parent_paths_rejected": True,
            "unicode_nfc_required": True,
            "casefold_collisions_rejected": True,
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path = root / "release/REPAIR_R2_HANDOFF_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)

    _write_zip(root, archive_root, entries, manifest_bytes, archive_path)
    first_digest = _sha256(archive_path)
    with tempfile.TemporaryDirectory(prefix="komorebi-block21-byte-rebuild-") as temporary:
        rebuilt = Path(temporary) / archive_path.name
        _write_zip(root, archive_root, entries, manifest_bytes, rebuilt)
        if rebuilt.read_bytes() != archive_path.read_bytes():
            raise SystemExit("Block 21 handoff is not byte reproducible")

    replay = _replay(archive_path, manifest, python=python)
    checksum_path.write_text(f"{first_digest}  {archive_path.name}\n", encoding="utf-8", newline="\n")
    build_record = {
        "schema": "komorebi-math.block21-handoff-build",
        "schema_version": 1,
        "package_version": version,
        "release_identity": identity,
        "passed": True,
        "archive": {
            "name": archive_path.name,
            "size": archive_path.stat().st_size,
            "sha256": first_digest,
        },
        "manifest": {
            "name": manifest_path.name,
            "size": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "payload_sha256": manifest["payload_sha256"],
        },
        "artifacts": list(artifacts),
        "byte_reproducible_rebuild": True,
        "replay": replay,
    }
    build_path.write_text(json.dumps(build_record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return archive_path, checksum_path, build_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-directory", default=".")
    parser.add_argument("--artifact-directory", default="release/dist")
    parser.add_argument("--gate", default="release/REPAIR_R2_PUBLICATION_GATE.json")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    output = Path(args.output_directory)
    if not output.is_absolute():
        output = root / output
    artifacts = Path(args.artifact_directory)
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    gate = Path(args.gate)
    if not gate.is_absolute():
        gate = root / gate

    archive, checksum, record = build(root, output, artifacts, gate, python=args.python)
    print(archive)
    print(checksum)
    print(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
