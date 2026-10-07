"""Run each pytest module in a fresh interpreter and write a release ledger."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

_SUMMARY = re.compile(r"(?P<passed>\d+) passed(?:, (?P<skipped>\d+) skipped)?")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run(path: Path, *, root: Path, timeout: float) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", str(path)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return {
            "path": str(path.relative_to(root)),
            "sha256": _digest(path),
            "passed": 0,
            "skipped": 0,
            "returncode": None,
            "timed_out": True,
            "duration_seconds": elapsed,
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
        }
    elapsed = time.perf_counter() - start
    output = completed.stdout + "\n" + completed.stderr
    match = _SUMMARY.search(output)
    return {
        "path": str(path.relative_to(root)),
        "sha256": _digest(path),
        "passed": int(match.group("passed")) if match else 0,
        "skipped": int(match.group("skipped") or 0) if match else 0,
        "returncode": int(completed.returncode),
        "timed_out": timed_out,
        "duration_seconds": elapsed,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.paths:
        files = [path if path.is_absolute() else root / path for path in args.paths]
    else:
        files = sorted((root / "tests").glob("test_*.py"))
    output = args.output or root / "release" / "BLOCK16_ISOLATED_SOURCE_VALIDATION.json"
    results: list[dict[str, Any]] = []
    started = time.time()
    for index, path in enumerate(files, start=1):
        result = _run(path.resolve(), root=root, timeout=args.timeout)
        results.append(result)
        state = "PASS" if result["returncode"] == 0 else "FAIL"
        print(
            f"[{index:03d}/{len(files):03d}] {state} {result['path']} "
            f"({result['passed']} passed, {result['skipped']} skipped, "
            f"{result['duration_seconds']:.2f}s)",
            flush=True,
        )
    failed = [result["path"] for result in results if result["returncode"] != 0]
    ledger = {
        "schema": "komorebi-math.isolated-pytest-ledger",
        "schema_version": 1,
        "python": sys.version,
        "python_executable": sys.executable,
        "root": str(root),
        "started_unix": started,
        "duration_seconds": time.time() - started,
        "files": len(results),
        "passed_tests": sum(result["passed"] for result in results),
        "skipped_tests": sum(result["skipped"] for result in results),
        "failed_files": failed,
        "passed": not failed,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"isolated source validation: {ledger['passed_tests']} passed, "
        f"{ledger['skipped_tests']} skipped, {len(failed)} failed files",
        flush=True,
    )
    print(output)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
