"""Dependency-free static and import audit for the Block 16 release tree."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import os
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class StaticAuditReport:
    package_version: str
    root: str
    checks: tuple[AuditCheck, ...]
    diagnostics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload["sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
        return payload


_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in ("src", "tests", "verification"):
        files.extend(sorted((root / relative).rglob("*.py")))
    return files


def _parse_python(files: list[Path]) -> tuple[list[str], list[str]]:
    syntax_errors: list[str] = []
    forbidden_imports: list[str] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            syntax_errors.append(f"{path}: {exc}")
            continue
        if "src/komorebi_math" not in path.as_posix():
            continue
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            for module in modules:
                if module == "tests" or module.startswith("tests.") or module == "verification" or module.startswith("verification."):
                    forbidden_imports.append(f"{path}:{node.lineno}: {module}")
    return syntax_errors, forbidden_imports


def _module_names(root: Path) -> list[str]:
    package_root = root / "src" / "komorebi_math"
    names: set[str] = set()
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(root / "src")
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        if name and name != "komorebi_math.__main__":
            names.add(name)
    return sorted(names)


def _import_modules(root: Path) -> tuple[list[str], list[str]]:
    names = _module_names(root)
    worker = r"""
import importlib
import json
import sys
names = json.loads(sys.argv[1])
imported = []
failures = []
for name in names:
    try:
        importlib.import_module(name)
        imported.append(name)
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
print("__KOMOREBI_IMPORT_AUDIT__" + json.dumps({"imported": imported, "failures": failures}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", worker, json.dumps(names)],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    marker = "__KOMOREBI_IMPORT_AUDIT__"
    payload_line = next((line for line in completed.stdout.splitlines() if line.startswith(marker)), None)
    if payload_line is None:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"child return code {completed.returncode}"
        return [], [f"import audit child failed: {detail[-2000:]}"]
    payload = json.loads(payload_line[len(marker) :])
    failures = list(payload["failures"])
    if completed.returncode != 0:
        failures.append(f"import audit child returned {completed.returncode}")
    return list(payload["imported"]), failures


def _stable_docs() -> tuple[list[str], list[str]]:
    import komorebi_math.stable as stable

    manifest_path = Path(stable.__file__).resolve().parent / "data" / "public_api_stable_0_16.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_signatures = {
        symbol["name"]
        for symbol in manifest["symbols"]
        if symbol.get("signature") is not None
    }
    missing: list[str] = []
    signature_failures: list[str] = []
    for name in stable.__all__:
        value = getattr(stable, name)
        if callable(value):
            if not inspect.getdoc(value):
                missing.append(name)
            if name in required_signatures:
                try:
                    inspect.signature(value)
                except (TypeError, ValueError):
                    signature_failures.append(name)
    return missing, signature_failures


def _markdown_links(root: Path) -> list[str]:
    failures: list[str] = []
    markdown = [root / "README.md", root / "README_BLOCK16.md", *sorted((root / "docs").glob("*.md"))]
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        for raw in _LINK.findall(text):
            target = raw.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            suffix = Path(target).suffix.lower()
            if suffix not in {".md", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".json", ".py", ".ps1", ".sh", ".yml", ".yaml", ".txt"}:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                failures.append(f"{path.relative_to(root)} -> {raw}")
    return failures


def _json_files(root: Path) -> list[str]:
    failures: list[str] = []
    paths = [
        *sorted((root / "src" / "komorebi_math" / "data").glob("*.json")),
        *sorted((root / "verification").glob("*.json")),
        *sorted((root / "environment").glob("*.json")),
    ]
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path.relative_to(root)}: {type(exc).__name__}: {exc}")
    return failures


def _shell_mode_diagnostics(
    project: Path,
    *,
    operating_system: str | None = None,
) -> tuple[list[Path], list[str], bool]:
    """Return shell runners, mode failures, and whether the mode gate applies.

    The optional operating-system argument exists so the cross-platform rule
    can be regression-tested without changing ``os.name`` globally.
    """

    shell_scripts = sorted(project.glob("*.sh"))
    observed_os = os.name if operating_system is None else str(operating_system)
    applicable = observed_os == "posix"
    non_executable = (
        [path.name for path in shell_scripts if not path.stat().st_mode & 0o111]
        if applicable
        else []
    )
    return shell_scripts, non_executable, applicable


def run_audit(root: Path | None = None) -> StaticAuditReport:
    project = (root or Path(__file__).resolve().parents[1]).resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import komorebi_math

    files = _python_files(project)
    syntax_errors, forbidden_imports = _parse_python(files)
    imported, import_failures = _import_modules(project)
    missing_docs, signature_failures = _stable_docs()
    link_failures = _markdown_links(project)
    json_failures = _json_files(project)
    # Windows does not implement POSIX executable permission bits.  A ZIP
    # extracted by ``Expand-Archive`` therefore reports every ``.sh`` file as
    # non-executable even though the exact same archive is valid and the files
    # can be executed normally after transfer to a POSIX host.  Enforce the
    # mode-bit gate only where those bits have operating-system meaning.
    shell_scripts, non_executable, posix_mode_applicable = _shell_mode_diagnostics(project)

    checks = (
        AuditCheck("python syntax", not syntax_errors, f"{len(files)} files parsed; {len(syntax_errors)} errors"),
        AuditCheck("production dependency boundary", not forbidden_imports, f"{len(forbidden_imports)} forbidden imports"),
        AuditCheck("production module imports", not import_failures, f"{len(imported)} imported; {len(import_failures)} failures"),
        AuditCheck("stable callable documentation", not missing_docs, f"{len(missing_docs)} missing docstrings"),
        AuditCheck("stable callable signatures", not signature_failures, f"{len(signature_failures)} unavailable signatures"),
        AuditCheck("markdown links", not link_failures, f"{len(link_failures)} unresolved relative links"),
        AuditCheck("JSON payloads", not json_failures, f"{len(json_failures)} invalid JSON files"),
        AuditCheck(
            "POSIX runner modes",
            not non_executable,
            (
                f"{len(non_executable)} non-executable scripts"
                if posix_mode_applicable
                else f"not applicable on {os.name}; {len(shell_scripts)} shell scripts inspected"
            ),
        ),
    )
    return StaticAuditReport(
        package_version=komorebi_math.__version__,
        root=str(project),
        checks=checks,
        diagnostics={
            "syntax_errors": syntax_errors,
            "forbidden_imports": forbidden_imports,
            "import_failures": import_failures,
            "missing_stable_docstrings": missing_docs,
            "stable_signature_failures": signature_failures,
            "unresolved_links": link_failures,
            "invalid_json": json_failures,
            "non_executable_scripts": non_executable,
            "posix_mode_check_applicable": posix_mode_applicable,
            "production_modules_imported": imported,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = run_audit()
    payload = report.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
