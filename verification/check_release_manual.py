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


def check(root: Path, *, profile: Path | None = None) -> dict:
    root = root.resolve()
    if sys.flags.optimize:
        raise ValueError("manual checks require Python assertions")
    environment = os.environ.copy()
    environment.pop("PYTHONOPTIMIZE", None)
    environment.update(
        PYTHONPATH=str(root / "src"),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    rows = []
    sources = {}
    excluded = []
    selections = {name: {"python_fences": None, "exclude": []} for name in
                  ("docs/USER_GUIDE.md", "docs/NEW_MACHINERY_GUIDE.md")}
    profile_digest = None
    if profile is not None:
        data = (root / profile).read_bytes()
        profile_digest = hashlib.sha256(data).hexdigest()
        config = json.loads(data)
        if not isinstance(config, dict) or config.get("schema_version") != 1:
            raise ValueError("unsupported manual-example profile")
        selections = config.get("sources")
        if not isinstance(selections, dict) or not selections:
            raise ValueError("manual-example profile requires sources")
    for relative, selection in selections.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or not (root / relative).resolve().is_relative_to(root)):
            raise ValueError("manual-example source must be below the project root")
        if not isinstance(selection, dict):
            raise ValueError(f"invalid manual-example selection: {relative}")
        source = root / relative
        data = source.read_bytes()
        content = data.decode("utf-8")
        sources[relative] = hashlib.sha256(data).hexdigest()
        examples = list(_python_examples(content, relative))
        if not examples:
            raise ValueError(f"no Python examples found in {relative}")
        expected = selection.get("python_fences")
        if profile is not None and (type(expected) is not int or expected != len(examples)):
            raise ValueError(f"Python-fence count changed in {relative}; review the profile")
        omissions = selection.get("exclude", [])
        if not isinstance(omissions, list):
            raise ValueError(f"invalid exclusions for {relative}")
        skip = {}
        for omission in omissions:
            if not isinstance(omission, dict):
                raise ValueError(f"invalid exclusion for {relative}")
            index, reason = omission.get("example"), omission.get("reason")
            if (type(index) is not int or not 1 <= index <= len(examples)
                    or not isinstance(reason, str) or not reason.strip()):
                raise ValueError(f"invalid exclusion for {relative}")
            if index in skip:
                raise ValueError(f"duplicate exclusion for {relative}")
            skip[index] = reason
        for index, (line, program) in enumerate(examples, 1):
            row = {
                "source": relative,
                "example": index,
                "line": line,
                "sha256": hashlib.sha256(program.encode()).hexdigest(),
            }
            if index in skip:
                excluded.append({**row, "reason": skip[index]})
                continue
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
        "profile_sha256": profile_digest,
        "excluded_examples": excluded,
        "examples": rows,
        "example_count": len(rows),
        "passed": all(row["passed"] for row in rows),
        "scope": "Each selected fenced Python example runs in a fresh interpreter and temporary directory. "
                 "Explicit exclusions identify reference fragments and separate optional-runtime checks. "
                 "Installation commands and optional hardware are covered separately.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", type=Path, help="Explicit reference-example selection with documented exclusions")
    args = parser.parse_args()
    try:
        report = check(args.root, profile=args.profile)
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
