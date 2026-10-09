"""Replay bounded release stress campaigns with input and output provenance.

Passing records concern the pinned finite checks, not universal mathematical
theorems or untested devices. Every child has a wall-time budget; there is no
operating-system memory or CPU-affinity cap.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

CAMPAIGNS = {
    "certificates": ("test_stress_certificates.py",),
    "directed_tails": ("test_stress_directed_tails.py",),
    "exact_arithmetic": ("test_stress_exact_arithmetic.py",),
    "numerical": ("test_stress_numerical.py",),
    "autodiff_contracts": ("test_stress_autodiff_contracts.py",),
    "closeout_contracts": ("test_release_closeout_contracts.py",),
    "optional_backends": ("test_stress_optional_backends.py",
                          "test_nonlinear_sparse_autodiff.py",
                          "test_block10_sparse_autodiff.py"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pytest_counts(path: Path) -> dict[str, int]:
    """Validate the concrete pytest JUnit cases, including failed/skipped cases."""
    tree = ET.parse(path)
    if tree.getroot().tag not in {"testsuite", "testsuites"}:
        raise ValueError("unsupported JUnit document")
    suites = list(tree.iter("testsuite"))
    if not suites:
        raise ValueError("JUnit document has no test suites")
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        if suite.findall("testsuite"):
            raise ValueError("nested JUnit suites are not supported")
        values = {}
        for key in counts:
            value = suite.get(key, "")
            if not value.isascii() or not value.isdigit():
                raise ValueError(f"invalid JUnit {key} count")
            values[key] = int(value)
        cases = suite.findall("testcase")
        observed = {"tests": len(cases), **{
            key: sum(case.find(tag) is not None for case in cases)
            for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))
        }}
        if values != observed:
            raise ValueError("JUnit counts disagree with recorded test cases")
        for key in counts:
            counts[key] += values[key]
    return counts


def _resource_result_valid(path: Path) -> bool:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = ({("finite join validation and quotient", n) for n in (4, 8, 16, 32, 64)}
                | {("directed stored-factor spectrum", n) for n in (4, 8, 16)}
                | {("bounded directed positive tail", n) for n in (4, 8, 16)})
    records = report["records"]
    return (report["status"] == "passed" and len(records) == len(expected)
            and {(row["operation"], row["size"]) for row in records} == expected
            and all(row.get("passed") is True for row in records))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path, default=Path("release/STRESS_AUDIT"))
    parser.add_argument("--revision", default="unavailable")
    parser.add_argument("--campaign", action="append", choices=tuple(CAMPAIGNS))
    parser.add_argument("--include-optional-backends", action="store_true")
    parser.add_argument("--skip-resources", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--total-timeout", type=int, default=900)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("stress replay requires enabled Python assertions")
    if args.timeout <= 0 or args.total_timeout <= 0:
        parser.error("time budgets must be positive")
    root = args.root.resolve()
    output = (root / args.output_directory).resolve()
    if not output.is_relative_to(root) or output == root:
        parser.error("output directory must be strictly below the project root")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "manifest.json"
    target.unlink(missing_ok=True)
    selected = list(dict.fromkeys(args.campaign or [n for n in CAMPAIGNS if n != "optional_backends"]))
    if args.include_optional_backends and "optional_backends" not in selected:
        selected.append("optional_backends")
    tests = sorted({root / "tests" / name for c in selected for name in CAMPAIGNS[c]})
    if any(not p.is_file() for p in tests):
        parser.error("requested stress test files are missing")
    scripts = [Path(__file__).resolve()]
    if not args.skip_resources:
        scripts.append(root / "verification/stress_resource_probe.py")
    pinned = sorted({root / "pyproject.toml", *tests, *scripts}
                    | {p for p in (root / "src/komorebi_math").rglob("*")
                       if p.is_file() and p.suffix in {".py", ".json"}})
    inputs = [{"path": p.relative_to(root).as_posix(), "sha256": digest(p)} for p in pinned]
    environment = dict(os.environ, PYTHONPATH=str(root / "src"),
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # A user's normal test filters must not narrow release qualification.
    environment.pop("PYTEST_ADDOPTS", None)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    commands = []
    results = []
    declared = []

    def run(name: str, command: list[str], scientific_result: Path, log: Path) -> dict:
        remaining = args.total_timeout - (time.monotonic() - clock)
        budget = max(0, min(args.timeout, remaining))
        commands.append(command)
        declared.extend([scientific_result, log])
        for stale in (scientific_result, log):
            if stale.exists():
                stale.unlink()
        item = {"name": name, "command": command, "timeout_seconds": budget}
        t0 = time.monotonic()
        if budget <= 0:
            item.update(status="not-run-total-budget", exit_status=124)
        else:
            with log.open("w", encoding="utf-8") as stream:
                try:
                    process = subprocess.run(command, cwd=root, env=environment,
                                             stdout=stream, stderr=subprocess.STDOUT,
                                             timeout=budget, check=False)
                    item.update(status="completed" if process.returncode == 0 else "failed",
                                exit_status=process.returncode)
                except subprocess.TimeoutExpired:
                    item.update(status="timeout", exit_status=124)
                except OSError as exc:
                    item.update(status="launch-error", exit_status=1,
                                error=f"{type(exc).__name__}: {exc}")
        item["runtime_seconds"] = time.monotonic() - t0
        if item["status"] == "completed" and not scientific_result.is_file():
            item["status"] = "result-missing"
        return item

    for campaign in selected:
        junit = output / f"{campaign}.xml"
        command = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--junitxml={junit}"]
        if campaign == "optional_backends":
            command.extend(["-m", "optional_backend"])
        command.extend(f"tests/{name}" for name in CAMPAIGNS[campaign])
        item = run(campaign, command, junit, output / f"{campaign}.log")
        counts = {k: 0 for k in ("tests", "failures", "errors", "skipped")}
        if junit.is_file():
            try:
                counts = _pytest_counts(junit)
            except (ET.ParseError, OSError, ValueError) as exc:
                item.update(status="result-invalid", error=f"{type(exc).__name__}: {exc}")
        item["pytest_counts"] = counts
        if item["status"] == "completed" and (not counts["tests"] or any(counts[k] for k in ("failures", "errors", "skipped"))):
            item["status"] = "result-invalid"
        results.append(item)
    if not args.skip_resources:
        result = output / "resources.json"
        command = [sys.executable, "verification/stress_resource_probe.py", "--output", str(result)]
        item = run("resources", command, result, output / "resources.log")
        if item["status"] == "completed":
            try:
                if not _resource_result_valid(result):
                    item["status"] = "result-invalid"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                item.update(status="result-invalid", error=f"{type(exc).__name__}: {exc}")
        results.append(item)
    for entry, path in zip(inputs, pinned, strict=True):
        entry["sha256_after"] = digest(path) if path.is_file() else None
    unchanged = all(x["sha256"] == x["sha256_after"] for x in inputs)
    passed = unchanged and all(x["status"] == "completed" for x in results)
    versions = [{"name": "Python", "version": platform.python_version()}]
    for name in ("numpy", "scipy", "sympy", "mpmath", "pytest", "jax", "jaxlib", "torch"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        versions.append({"name": name, "version": version})
    manifest = {
        "schema_version": 2,
        "claim_id": "KOMOREBI-RELEASE-STRESS-2026-10-08",
        "repository": {"commit": args.revision, "dirty": True},
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "environment": {"software": versions, "hardware": platform.machine(),
                        "native_thread_environment": {k: environment[k] for k in
                                                      ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "mathematics": {
            "assertion_tested": "The requested pinned finite stress cases satisfy their independent exact, interval, numerical, persistence and failure contracts.",
            "coefficient_domain": "Python integers and rationals; binary floating and complex arrays; qualified mpmath directed intervals.",
            "conventions": "Literal outward bounds carry directed enclosure claims. Finite numerical consistency and exact arithmetic are distinguished in each test.",
            "inputs": [p.relative_to(root).as_posix() for p in tests],
            "bounds": {"campaigns": results, "range_source": "Explicit finite loops, parameters, seeds and controls in the pinned stress tests; resource probe sizes are in resources.json."},
            "non_claims": ["Finite tests do not re-prove historical theorem dependencies.",
                           "No GPU or untested platform/dependency-version qualification.",
                           "Ordinary numerical output is not a rigorous enclosure.",
                           "Optional backends are tested only when explicitly requested and all requested cases execute."]},
        "randomness": {"used": True, "generator": "Per-test explicitly seeded NumPy generators and deterministic enumeration", "seed": "See the pinned test files"},
        "run": {"started_at": started, "runtime_seconds": time.monotonic() - clock,
                "exit_status": 0 if passed else 1, "status": "completed" if passed else "failed",
                "timeout_seconds": args.total_timeout,
                "resource_limits": {"memory_bytes": None, "cpu_seconds": None, "max_cores": None, "max_threads": None},
                "missing_results": [p.relative_to(root).as_posix() for p in declared if not p.is_file()],
                "invalid_results": [x["name"] for x in results if x["status"] != "completed"]},
        "input_artifacts": inputs,
        "execution_artifacts": [x["path"] for x in inputs],
        "declared_results": [p.relative_to(root).as_posix() for p in declared],
        "output_directory": output.relative_to(root).as_posix(),
        "outputs": [{"path": p.relative_to(root).as_posix(), "sha256": digest(p), "kind": "result"}
                    for p in declared if p.is_file()],
        "checks": [{"name": "pinned inputs unchanged", "passed": unchanged},
                   {"name": "all requested campaigns completed without skips", "passed": all(x["status"] == "completed" for x in results)}],
        "result": "completed" if passed else "failed; inspect campaign status and logs",
        "residual_risks": ["Directed arithmetic depends on the documented mpmath trust base.",
                           "Numerical cases retain their model, tolerance and conditioning assumptions.",
                           "Thread environment caps are cooperative; no OS memory or affinity cap is asserted."]}
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["run"]["status"], "campaigns": results, "manifest": str(target)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
