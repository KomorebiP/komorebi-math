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


def _python_examples(content: str, source: str):
    """Read explicitly labelled Markdown fences without entering other blocks."""
    opening = None
    lines = []
    for number, line in enumerate(content.splitlines(keepends=True), 1):
        if opening is None:
            match = re.fullmatch(r"( {0,3})(`{3,}|~{3,})([^\r\n]*)[\r\n]*", line)
            if match:
                indent, fence, info = match.groups()
                language = info.strip().split(maxsplit=1)[0].casefold() if info.strip() else ""
                opening = (number, len(indent), fence, language in {"python", "py", "python3"})
                lines = []
            continue
        start, indent, fence, python = opening
        if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*[\r\n]*", line):
            if python:
                yield start, "".join(lines)
            opening = None
        elif python:
            # CommonMark removes up to the opening fence's indentation.
            leading = len(line) - len(line.lstrip(" "))
            lines.append(line[min(leading, indent):])
    if opening is not None and opening[3]:
        raise ValueError(f"unterminated Python fence in {source} at line {opening[0]}")


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
        data = source.read_bytes()
        content = data.decode("utf-8")
        sources[relative] = hashlib.sha256(data).hexdigest()
        examples = list(_python_examples(content, relative))
        if not examples:
            raise ValueError(f"no Python examples found in {relative}")
        for index, (line, program) in enumerate(examples, 1):
            row = {
                "source": relative,
                "example": index,
                "line": line,
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
    try:
        report = check(args.root)
    except (OSError, ValueError) as exc:
        report = {"schema": "komorebi-math.release-manual-examples", "schema_version": 1,
                  "passed": False, "example_count": 0, "examples": [],
                  "error": f"{type(exc).__name__}: {exc}"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['example_count']} manual examples: {'passed' if report['passed'] else 'FAILED'}")
    if "error" in report:
        print(report["error"])
    for row in report["examples"]:
        if not row["passed"]:
            print(json.dumps(row, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
