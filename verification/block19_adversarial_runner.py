"""Run the fixture-free Block 19 tests without requiring pytest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import traceback
import types
from pathlib import Path


class _Raises:
    def __init__(self, expected: type[BaseException], match: str | None = None) -> None:
        self.expected = expected
        self.match = match

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback_object: object,
    ) -> bool:
        if kind is None:
            raise AssertionError(f"did not raise {self.expected.__name__}")
        if not issubclass(kind, self.expected):
            return False
        if self.match is not None and re.search(self.match, str(value)) is None:
            raise AssertionError(f"{value} does not match {self.match!r}")
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    pytest = types.ModuleType("pytest")
    pytest.raises = lambda expected, match=None: _Raises(expected, match)  # type: ignore[attr-defined]
    pytest.mark = types.SimpleNamespace(optional_backend=lambda function: function)  # type: ignore[attr-defined]
    sys.modules["pytest"] = pytest
    root = Path(__file__).resolve().parents[1]
    source = root / "tests/test_block19_integration.py"
    spec = importlib.util.spec_from_file_location("block19_integration_tests", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the Block 19 test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results: list[dict[str, object]] = []
    for name, value in vars(module).items():
        if not name.startswith("test_") or not callable(value):
            continue
        try:
            value()
            results.append({"name": name, "passed": True, "detail": ""})
        except Exception as exc:
            traceback.print_exc()
            results.append(
                {
                    "name": name,
                    "passed": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    passed = sum(bool(result["passed"]) for result in results)
    report = {
        "schema": "komorebi-math.block19-adversarial-tests",
        "schema_version": 1,
        "passed": passed == len(results),
        "passed_count": passed,
        "total_count": len(results),
        "results": results,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
