"""Replay every Python example in the current reader guides independently."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def check(root: Path) -> dict:
    root = root.resolve()
    if sys.flags.optimize:
        raise ValueError("manual checks require Python assertions")
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(root / "src"),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    rows = []
    sources = {}
    for relative in ("docs/USER_GUIDE.md", "docs/NEW_MACHINERY_GUIDE.md"):
        source = root / relative
        content = source.read_text(encoding="utf-8")
        sources[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        for index, match in enumerate(re.finditer(r"^```python\n(.*?)^```[ \t]*$", content, re.M | re.S), 1):
            program = match.group(1)
            row = {
                "source": relative,
                "example": index,
                "line": content.count("\n", 0, match.start()) + 1,
                "sha256": hashlib.sha256(program.encode()).hexdigest(),
            }
            with tempfile.TemporaryDirectory(prefix="komorebi-manual-") as work:
                try:
                    result = subprocess.run(
                        [sys.executable, "-B", "-c", program],
                        cwd=work, env=environment, text=True, capture_output=True, timeout=60,
                    )
                    row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                               passed=result.returncode == 0)
                except subprocess.TimeoutExpired:
                    row.update(returncode=None, passed=False, error="example timed out after 60 seconds")
            rows.append(row)
    if not rows:
        raise ValueError("no manual examples found")
    return {
        "schema": "komorebi-math.release-manual-examples",
        "schema_version": 1,
        "python": sys.version,
        "source_sha256": sources,
        "examples": rows,
        "example_count": len(rows),
        "passed": all(row["passed"] for row in rows),
        "scope": "Each fenced Python example runs in a fresh interpreter and temporary directory. "
                 "Installation commands and optional hardware are covered separately.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['example_count']} manual examples: {'passed' if report['passed'] else 'FAILED'}")
    for row in report["examples"]:
        if not row["passed"]:
            print(json.dumps(row, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
