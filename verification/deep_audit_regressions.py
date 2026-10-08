"""Replay the October 2026 mathematical audit with pinned input evidence.

This finite regression run records its inputs, outputs and bounds. It does not
prove universal mathematical claims or qualify optional hardware backends.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

PATTERNS = (
    "test_certificate_finite_evidence.py",
    "test_deep_*.py",
    "test_quantum_numerics_adversarial.py",
    "test_memory_laplace_domains.py",
    "test_quadratic_forms*.py",
    "test_sequences_oeis*.py",
    "test_special_functions*.py",
    "test_fourier_recovery*.py",
    "test_finite_join_semilattice.py",
    "test_dirac_modes.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path, default=Path("release/DEEP_AUDIT"))
    parser.add_argument("--revision", default="unavailable")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output_directory).resolve()
    if not output.is_relative_to(root) or output == root:
        parser.error("output directory must be strictly below the project root")
    if args.timeout < 1:
        parser.error("timeout must be positive")
    output.mkdir(parents=True, exist_ok=True)
    tests = sorted({p for pattern in PATTERNS for p in (root / "tests").glob(pattern)})
    if len(tests) < 10:
        parser.error("audit regression files are missing")
    pinned = sorted(
        {root / "pyproject.toml", Path(__file__).resolve(), *tests}
        | {p for p in (root / "src/komorebi_math").rglob("*") if p.suffix in {".py", ".json"}}
    )
    artifacts = [{"path": p.relative_to(root).as_posix(), "sha256": digest(p)} for p in pinned]
    log = output / "pytest.log"
    junit = output / "pytest.xml"
    if junit.exists():
        junit.unlink()
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--junitxml={junit}"]
    command.extend(p.relative_to(root).as_posix() for p in tests)
    environment = dict(os.environ, PYTHONPATH=str(root / "src"),
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    status = "completed"
    with log.open("w", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(command, cwd=root, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
            exit_status = completed.returncode
            if exit_status:
                status = "failed"
        except subprocess.TimeoutExpired:
            exit_status = 124
            status = "timeout"
    for artifact, path in zip(artifacts, pinned, strict=True):
        artifact["sha256_after"] = digest(path) if path.is_file() else None
    unchanged = all(a["sha256"] == a["sha256_after"] for a in artifacts)
    if exit_status == 0 and not unchanged:
        status = "inputs-changed"
    counts = {}
    if junit.is_file():
        for suite in ET.parse(junit).iter("testsuite"):
            for key in ("tests", "failures", "errors", "skipped"):
                counts[key] = counts.get(key, 0) + int(suite.get(key, "0"))
    all_executed = bool(counts.get("tests", 0)) and counts.get("skipped", 0) == 0
    if exit_status == 0 and status == "completed" and not all_executed:
        status = "result-invalid"
    software = [{"name": "Python", "version": platform.python_version()}]
    for name in ("numpy", "sympy", "scipy", "mpmath", "pytest"):
        try:
            software.append({"name": name, "version": importlib.metadata.version(name)})
        except importlib.metadata.PackageNotFoundError:
            software.append({"name": name, "version": "not installed"})
    declared = [log.relative_to(root).as_posix(), junit.relative_to(root).as_posix()]
    outputs = [{"path": p.relative_to(root).as_posix(), "sha256": digest(p), "kind": "result"}
               for p in (log, junit) if p.is_file()]
    if exit_status == 0 and not junit.is_file():
        status = "result-missing"
    manifest = {
        "schema_version": 2,
        "claim_id": "KOMOREBI-PYTHON-DEEP-AUDIT-2026-10-08",
        "repository": {"commit": args.revision, "dirty": True},
        "command": command,
        "environment": {"software": software, "hardware": platform.machine(),
                        "native_thread_environment": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}},
        "mathematics": {
            "assertion_tested": "The named adversarial regressions and independent finite controls satisfy their declared exact or numerical contracts.",
            "coefficient_domain": "Unbounded integers and rationals for exact tests; binary64/complex128 and mpmath directed interval arithmetic for numerical tests.",
            "conventions": "Schrodinger derivative memory kernel has two -i coupling factors; Laplace integrals require the convergence half-plane; interval literals, not rounded display floats, bound K-COS tails.",
            "inputs": [p.relative_to(root).as_posix() for p in tests],
            "bounds": {"test_files": len(tests), "executed_cases": counts,
                       "range_source": "The pinned test files state every matrix, rank, coefficient, parameter, precision and finite loop range."},
            "non_claims": ["No universal theorem follows from the finite regression run.",
                           "No current GPU/autodiff qualification or replay of all source Lean proofs.",
                           "No rigorous enclosure claim for ordinary floating-point special-function or Fourier computations."]},
        "randomness": {"used": True, "generator": "Per-test NumPy generators, explicitly seeded in the pinned tests", "seed": "See each pinned test; there is no unseeded audit driver randomness"},
        "run": {"started_at": started, "runtime_seconds": time.monotonic() - clock,
                "exit_status": exit_status, "status": status, "timeout_seconds": args.timeout,
                "resource_limits": {"memory_bytes": None, "cpu_seconds": None, "max_cores": None, "max_threads": None},
                "missing_results": [name for name in declared if not (root / name).is_file()], "invalid_results": []},
        "input_artifacts": artifacts,
        "execution_artifacts": [a["path"] for a in artifacts],
        "declared_results": declared,
        "output_directory": output.relative_to(root).as_posix(),
        "outputs": outputs,
        "checks": [{"name": "pinned inputs unchanged", "passed": unchanged},
                   {"name": "pytest succeeded", "passed": exit_status == 0},
                   {"name": "all audit cases executed", "passed": all_executed}],
        "result": f"{status}: {counts}",
        "residual_risks": ["K-COS interval guarantees depend on the documented mpmath interval arithmetic trust base.",
                           "Numerical validity depends on the declared tolerance, model and conditioning; it is not exact identity.",
                           "Native thread environment settings are recorded; Python concurrency controls may use multiple threads and no operating-system resource cap is asserted."],
    }
    target = output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "counts": counts, "manifest": str(target)}))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
