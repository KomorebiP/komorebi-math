"""Run installed mathematical stress suites offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import run_stress
from .suites import BUILTIN_SUITES, builtin_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list installed suite names")
    parser.add_argument(
        "--coverage", action="store_true", help="also validate the paper-to-machinery inventory"
    )
    parser.add_argument(
        "--papers", action="store_true", help="append finite paper machinery and special-function checks"
    )
    parser.add_argument("--suite", action="append", choices=BUILTIN_SUITES)
    parser.add_argument("--samples", type=int, default=8, help="sampled cases per suite, 1 to 128")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--timeout", type=float, default=30.0, help="hard wall seconds per isolated check")
    parser.add_argument(
        "--total-timeout", type=float, default=180.0, help="total wall seconds for check execution"
    )
    parser.add_argument("--output", type=Path, help="save the full provenance report as JSON")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(BUILTIN_SUITES))
        return 0
    try:
        checks = builtin_checks(suites=args.suite, samples=args.samples)
        if args.papers:
            from .continuation_checks import continuation_checks
            from .paper_suites import paper_checks
            from .special_function_checks import special_function_checks

            checks = (
                *checks,
                *paper_checks(args.samples),
                *special_function_checks(args.samples),
                *continuation_checks(args.samples),
            )
        report = run_stress(
            checks, seed=args.seed, timeout_seconds=args.timeout, total_timeout_seconds=args.total_timeout
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    data = report.to_dict()
    data["replay_digest"] = report.replay_digest
    coverage_valid = True
    if args.coverage:
        from .coverage import validate_coverage_catalog

        coverage = validate_coverage_catalog().to_dict()
        data["paper_coverage"] = coverage
        coverage_valid = coverage.get("valid") is True
        data["stress_status"] = data["status"]
        if not coverage_valid:
            data["status"] = "failed"
    import json

    serialized = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            print(f"cannot write stress report: {exc}", file=sys.stderr)
            return 2
    print(serialized, end="")
    return 0 if report.passed and coverage_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
