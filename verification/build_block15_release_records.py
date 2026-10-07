"""Build deterministic Block 15 release records from validated local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pkgutil
import py_compile
from pathlib import Path
from typing import Any, Iterable

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PACKAGE_ROOT / "release"
BATCH_RECORDS = (
    "BLOCK15_SOURCE_BATCH_A.json",
    "BLOCK15_SOURCE_BATCH_B.json",
    "BLOCK15_SOURCE_BATCH_C1.json",
    "BLOCK15_SOURCE_BATCH_C2.json",
    "BLOCK15_SOURCE_BATCH_C3.json",
    "BLOCK15_SOURCE_BATCH_D.json",
    "BLOCK15_SOURCE_BATCH_E.json",
    "BLOCK15_SOURCE_BATCH_F.json",
)
TRANSIENT_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "release",
    "site",
}
TRANSIENT_SUFFIXES = {".pyc", ".pyo"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8", newline="\n")


def is_tracked_file(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in TRANSIENT_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return False
    if path.suffix in TRANSIENT_SUFFIXES:
        return False
    return path.is_file()


def file_map(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not is_tracked_file(path, root):
            continue
        relative = path.relative_to(root).as_posix()
        records[relative] = {"sha256": sha256(path), "size": path.stat().st_size}
    return records


def aggregate_source_validation() -> dict[str, Any]:
    """Validate and normalize the current source-test evidence.

    The hotfix handoff records a complete single-process run in
    ``BLOCK15A2_SOURCE_VALIDATION.json``.  Legacy bounded batch records remain
    supported for maintainers rebuilding a release on constrained hosts.
    """

    current = RELEASE_ROOT / "BLOCK15A2_SOURCE_VALIDATION.json"
    if current.is_file():
        payload = json.loads(current.read_text(encoding="utf-8"))
        if (
            payload.get("package_version") != "0.15.0a2"
            or int(payload.get("collected", -1)) != 960
            or int(payload.get("passed", -1)) != 960
            or int(payload.get("failed", -1)) != 0
            or not payload.get("complete", False)
        ):
            raise RuntimeError("current Block 15a2 source-validation record is not passing")
        return payload

    logical_batches: list[dict[str, Any]] = []
    selected: set[str] = set()
    total = 0
    for name in BATCH_RECORDS:
        path = RELEASE_ROOT / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("passed"):
            raise RuntimeError(f"source validation record is not passing: {name}")
        for batch in payload["batches"]:
            label = str(batch["label"])
            if label in selected:
                raise RuntimeError(f"duplicate source batch: {label}")
            selected.add(label)
            logical_batches.append(
                {
                    "label": label,
                    "passed": int(batch["passed"]),
                    "returncode": int(batch["returncode"]),
                    "seconds": float(batch["seconds"]),
                    "timed_out": bool(batch["timed_out"]),
                    "record": f"release/{name}",
                }
            )
            total += int(batch["passed"])
    if total != 960 or len(logical_batches) != 15:
        raise RuntimeError(
            f"expected 960 tests in 15 logical batches, found {total} in {len(logical_batches)}"
        )
    payload = {
        "schema": "komorebi-math.block15-source-validation",
        "schema_version": 2,
        "package_version": "0.15.0a2",
        "collected": 960,
        "logical_batch_count": len(logical_batches),
        "execution_record_count": len(BATCH_RECORDS),
        "passed": total,
        "failed": 0,
        "errors": 0,
        "complete": True,
        "execution_mode": "bounded-independent-processes",
        "batches": logical_batches,
    }
    write_json(RELEASE_ROOT / "BLOCK15A2_SOURCE_VALIDATION.json", payload)
    return payload


def build_api_records() -> dict[str, Any]:
    manifest_path = PACKAGE_ROOT / "src/komorebi_math/data/public_api_stable_0_15.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_record = {
        "schema": "komorebi-math.block15-public-api-record",
        "schema_version": 1,
        "package_version": manifest["package_version"],
        "stability_level": manifest["stability_level"],
        "namespace": manifest["namespaces"][0],
        "symbol_count": len(manifest["symbols"]),
        "manifest_sha256": sha256(manifest_path),
        "manifest_path": "src/komorebi_math/data/public_api_stable_0_15.json",
        "symbols": manifest["symbols"],
    }
    write_json(RELEASE_ROOT / "BLOCK15_PUBLIC_API.json", public_record)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for symbol in manifest["symbols"]:
        grouped.setdefault(symbol["implementation_module"], []).append(symbol)
    lines = [
        "# Block 15 Curated Public API",
        "",
        "Package version: `0.15.0a2`  ",
        f"Namespace: `{manifest['namespaces'][0]}`  ",
        f"Stability level: `{manifest['stability_level']}`  ",
        f"Recorded symbols: **{len(manifest['symbols'])}**  ",
        f"Manifest SHA-256: `{public_record['manifest_sha256']}`",
        "",
        "The complete research namespace remains available. This document lists the smaller compatibility surface intended for the 1.0 freeze.",
        "",
    ]
    for module in sorted(grouped):
        lines += [f"## `{module}`", "", "| Export | Kind | Signature |", "|---|---|---|"]
        for symbol in sorted(grouped[module], key=lambda item: item["name"]):
            signature = symbol.get("signature") or ""
            signature = signature.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{symbol['name']}` | {symbol['kind']} | `{signature}` |")
        lines.append("")
    (RELEASE_ROOT / "BLOCK15_PUBLIC_API.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return public_record


def build_change_manifest(baseline: Path) -> dict[str, Any]:
    old = file_map(baseline)
    new = file_map(PACKAGE_ROOT)
    old_names = set(old)
    new_names = set(new)
    added = sorted(new_names - old_names)
    deleted = sorted(old_names - new_names)
    modified = sorted(name for name in old_names & new_names if old[name]["sha256"] != new[name]["sha256"])
    unchanged = sorted(name for name in old_names & new_names if old[name]["sha256"] == new[name]["sha256"])
    payload = {
        "schema": "komorebi-math.block15-change-manifest",
        "schema_version": 1,
        "from_version": "0.14.0a2",
        "to_version": "0.15.0a2",
        "scope": "source handoff excluding generated release files and transient build/cache material",
        "counts": {
            "added": len(added),
            "modified": len(modified),
            "deleted": len(deleted),
            "unchanged": len(unchanged),
        },
        "added": [{"path": name, **new[name]} for name in added],
        "modified": [
            {"path": name, "before": old[name], "after": new[name]} for name in modified
        ],
        "deleted": [{"path": name, **old[name]} for name in deleted],
        "unchanged_paths": unchanged,
    }
    write_json(RELEASE_ROOT / "BLOCK15_CHANGE_MANIFEST.json", payload)
    return payload


def build_static_audit() -> dict[str, Any]:
    source_root = PACKAGE_ROOT / "src"
    production_files = sorted((source_root / "komorebi_math").rglob("*.py"))
    compile_errors: list[dict[str, str]] = []
    for path in production_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:  # pragma: no cover - release audit path
            compile_errors.append({"path": path.relative_to(PACKAGE_ROOT).as_posix(), "error": str(exc)})

    import sys
    sys.path.insert(0, str(source_root))
    package = importlib.import_module("komorebi_math")
    discovered = [info.name for info in pkgutil.walk_packages(package.__path__, package.__name__ + ".")]
    import_errors: list[dict[str, str]] = []
    for name in discovered:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - release audit path
            import_errors.append({"module": name, "error": f"{type(exc).__name__}: {exc}"})

    stable = importlib.import_module("komorebi_math.stable")
    payload = {
        "schema": "komorebi-math.block15-static-audit",
        "schema_version": 1,
        "package_version": getattr(package, "__version__", None),
        "production_python_files": len(production_files),
        "package_modules_discovered": len(discovered),
        "package_modules_imported": len(discovered) - len(import_errors),
        "stable_export_count": len(stable.__all__),
        "compile_errors": compile_errors,
        "import_errors": import_errors,
        "passed": not compile_errors and not import_errors,
    }
    write_json(PACKAGE_ROOT / "STATIC_AUDIT.json", payload)
    lines = [
        "# Static and import audit",
        "",
        f"Status: **{'passed' if payload['passed'] else 'failed'}**",
        "",
        f"* Package version: `{payload['package_version']}`",
        f"* Production Python files: {payload['production_python_files']}",
        f"* Package modules discovered: {payload['package_modules_discovered']}",
        f"* Package modules imported: {payload['package_modules_imported']}",
        f"* Curated stable exports: {payload['stable_export_count']}",
        f"* Compilation errors: {len(compile_errors)}",
        f"* Import errors: {len(import_errors)}",
        "",
    ]
    if compile_errors or import_errors:
        lines += ["## Errors", "", "```json", canonical_json({"compile": compile_errors, "import": import_errors}).rstrip(), "```", ""]
    else:
        lines += ["## Errors", "", "None.", ""]
    (PACKAGE_ROOT / "STATIC_AUDIT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    if not (baseline / "pyproject.toml").is_file():
        parser.error(f"baseline package root is invalid: {baseline}")
    source = aggregate_source_validation()
    api = build_api_records()
    changes = build_change_manifest(baseline)
    static = build_static_audit()
    summary = {
        "source_tests": int(source["passed"]),
        "stable_symbols": api["symbol_count"],
        "changes": changes["counts"],
        "static_passed": static["passed"],
    }
    print(canonical_json(summary), end="")
    return 0 if source["complete"] and static["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
