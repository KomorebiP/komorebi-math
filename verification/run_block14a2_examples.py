"""Run a deterministic slice of packaged Python examples."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    examples = sorted((ROOT / "examples").glob("*.py"))[args.start : args.end]
    results: list[dict[str, object]] = []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    for path in examples:
        started = time.monotonic()
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        result = {
            "name": path.name,
            "returncode": completed.returncode,
            "seconds": round(time.monotonic() - started, 6),
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
        results.append(result)
        if completed.returncode != 0:
            break
    payload = {
        "schema": "komorebi-math.block14a2-example-slice.v1",
        "package_version": __import__("komorebi_math").__version__,
        "start": args.start,
        "end": args.end,
        "count": len(results),
        "passed": len(results) == len(examples) and all(item["returncode"] == 0 for item in results),
        "results": results,
    }
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("start", "end", "count", "passed")}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
