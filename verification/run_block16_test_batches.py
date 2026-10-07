"""Run all Block 16 source tests in bounded independent processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from verification.run_block15_test_batches import BATCHES as BLOCK15_BATCHES

BLOCK16_BATCH = (
    "block16-analysis-contract-release-cli-gpu-layout-companion-static",
    48,
    (
        "tests/test_block16_analysis_synthesis.py",
        "tests/test_block16_contracts.py",
        "tests/test_block16_release_qualification.py",
        "tests/test_block16_cli.py",
        "tests/test_block16_release_contract.py",
        "tests/test_block16_gpu_qualification.py",
        "tests/test_block16_companion_crosswalk.py",
        "tests/test_block16_release_layout.py",
        "tests/test_block16_static_audit.py",
        "tests/test_block16_package_audit.py",
    ),
)
BATCHES = (*BLOCK15_BATCHES, BLOCK16_BATCH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--batch", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for label, expected_count, _ in BATCHES:
            print(f"{label}\t{expected_count}")
        return 0

    selected = BATCHES
    if args.batch:
        requested = set(args.batch)
        known = {label for label, _, _ in BATCHES}
        unknown = sorted(requested - known)
        if unknown:
            parser.error(f"unknown batch label(s): {', '.join(unknown)}")
        selected = tuple(batch for batch in BATCHES if batch[0] in requested)

    expected_total = args.expected if args.expected is not None else sum(batch[1] for batch in selected)
    environment = dict(os.environ)
    source_root = str(PACKAGE_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = source_root if not existing_pythonpath else source_root + os.pathsep + existing_pythonpath
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    results: list[dict[str, object]] = []
    total = 0
    all_passed = True
    for label, expected_count, files in selected:
        command = [sys.executable, "-m", "pytest", "-q", "--tb=short", *files]
        print(f"\n=== {label} ===", flush=True)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=PACKAGE_ROOT,
                env=environment,
                timeout=args.timeout,
                check=False,
            )
            returncode = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = 124
            timed_out = True
        elapsed = time.monotonic() - started
        passed = returncode == 0 and not timed_out
        count = expected_count if passed else 0
        total += count
        all_passed = all_passed and passed
        results.append(
            {
                "label": label,
                "expected": expected_count,
                "passed": count,
                "returncode": returncode,
                "seconds": round(elapsed, 6),
                "timed_out": timed_out,
            }
        )
        if not passed:
            break

    payload = {
        "schema": "komorebi-math.block16-test-batch-run.v1",
        "package_version": "0.16.0a2",
        "expected": expected_total,
        "total_passed": total,
        "passed": all_passed and total == expected_total,
        "selected_batches": [batch[0] for batch in selected],
        "batches": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        destination = args.output if args.output.is_absolute() else PACKAGE_ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
