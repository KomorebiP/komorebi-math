"""Select release sources without development output or old distribution binaries."""
from __future__ import annotations

from pathlib import Path
import stat
from typing import Iterable

ROOT_FILES = frozenset({
    ".editorconfig", ".gitattributes", ".gitignore", "CHANGELOG.md", "CITATION.cff",
    "CONTRIBUTING.md", "LICENSE.md", "MANIFEST.in", "README.md", "RELEASE_ID.json",
    "REPRODUCIBILITY.md", "SECURITY.md", "mkdocs.yml", "pyproject.toml",
})
INPUT_RECORDS = frozenset({
    "verification/installed_runtime_profile.json",
    "verification/release_manual_profile.json",
    "verification/companion_c_crosswalk.json",
})
_CACHES = frozenset({
    ".git", ".hg", ".svn", ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".nox", ".venv", "__pycache__", "node_modules", "venv", "virtualenv",
})
_GENERATED = frozenset({
    "PKG-INFO", "setup.cfg",
    "src/komorebi_math.egg-info/PKG-INFO",
    "src/komorebi_math.egg-info/SOURCES.txt",
    "src/komorebi_math.egg-info/dependency_links.txt",
    "src/komorebi_math.egg-info/entry_points.txt",
    "src/komorebi_math.egg-info/requires.txt",
    "src/komorebi_math.egg-info/top_level.txt",
})


def _cache(parts: tuple[str, ...]) -> bool:
    return any(part.casefold() in _CACHES
               or part.casefold().startswith((".venv-", ".venv_"))
               or part.casefold().endswith(".egg-info") for part in parts)


def _selected(relative: Path) -> bool:
    parts = relative.parts
    if not parts or _cache(parts) or relative.suffix in {".pyc", ".pyo"}:
        return False
    name = relative.as_posix()
    if len(parts) == 1:
        return name in ROOT_FILES or (relative.suffix in {".sh", ".ps1"}
                and not any(word in name.casefold() for word in ("block", "repair")))
    if parts[:2] == ("src", "komorebi_math"):
        return relative.suffix in {".py", ".json"} or relative.name == "py.typed"
    if parts[0] == "docs":
        return len(parts) < 2 or parts[1] not in {"history", "development"}
    if parts[0] in {"examples", "tests"}:
        return relative.suffix in {".py", ".ps1", ".json"}
    if parts[0] == "verification":
        return relative.suffix == ".py" or name in INPUT_RECORDS
    if parts[0] == "constraints":
        return relative.suffix == ".txt"
    if parts[0] == "environment":
        return relative.suffix in {".yml", ".yaml"}
    return parts[:2] == (".github", "workflows") and relative.suffix in {".yml", ".yaml"}


def _require_directories(root: Path, relative: Path) -> bool:
    """Check each lexical ancestor before traversing a source directory."""
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return False
        name = current.relative_to(root).as_posix()
        if stat.S_ISLNK(mode):
            raise ValueError(f"release source directory is a symlink: {name}")
        if not stat.S_ISDIR(mode):
            raise ValueError(f"release source directory is not a directory: {name}")
    return True


def source_files(root: Path) -> set[str]:
    """Return selected regular files, rejecting links and special source nodes."""
    root = root.resolve()
    selected: set[str] = set()
    folders = ("src", "docs", "examples", "tests", "verification", "constraints",
               "environment", ".github/workflows")
    candidates = list(root.iterdir())
    for folder in folders:
        base = root / folder
        if _require_directories(root, Path(folder)):
            candidates.extend(base.rglob("*"))
    for path in candidates:
        relative = path.relative_to(root)
        if _selected(relative):
            _require_directories(root, relative.parent)
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"release source contains a symlink: {relative}")
            if stat.S_ISREG(mode):
                selected.add(relative.as_posix())
            elif not stat.S_ISDIR(mode):
                raise ValueError(f"release source contains a special file: {relative}")
    return selected


def support_files(root: Path) -> set[str]:
    """Return non-importable inputs that must survive the sdist unchanged."""
    return {name for name in source_files(root) if not name.startswith("src/komorebi_math/")}


def audit_sdist_layout(root: Path, files: Iterable[str]) -> dict[str, object]:
    """Reject omitted source files and unexpected development payloads."""
    expected = source_files(root)
    actual = set(files)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected - _GENERATED)
    return {"passed": not (missing or unexpected), "source_files": len(expected),
            "missing": missing, "unexpected": unexpected}
