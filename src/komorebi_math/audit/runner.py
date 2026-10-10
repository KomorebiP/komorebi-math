"""Finite mathematical stress checks with explicit evidence and execution status.

Reports record bounded computations. They are neither proof objects nor live
package Certificates, and loading a report does not revalidate its contents.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import multiprocessing
import pickle
import platform
import random
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..certificates import Exactness


def _json_value(value: Any, *, exact: bool = False) -> Any:
    """Own strict JSON data without lossy stringification of unknown objects."""
    if value is None or isinstance(value, str) or type(value) is bool:
        return value
    if type(value) is int:
        return value
    if isinstance(value, Fraction):
        return {"rational": [int(value.numerator), int(value.denominator)]}
    if type(value) is float:
        if exact:
            raise ValueError("exact check evidence cannot contain floating-point values")
        if not math.isfinite(value):
            raise ValueError("evidence must contain only finite numerical values")
        return value
    if type(value) is complex:
        return {"complex": [_json_value(value.real, exact=exact), _json_value(value.imag, exact=exact)]}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON evidence mappings require string keys")
        return {key: _json_value(item, exact=exact) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item, exact=exact) for item in value]
    raise TypeError(f"unsupported evidence type: {type(value).__name__}")


def _encoded(value: Any, *, exact: bool = False) -> str:
    return json.dumps(_json_value(value, exact=exact), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _integer(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}")
    return value


def _duration(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


class SkipCheck(Exception):
    """Explicitly mark a check as unexecuted; a skipped run never passes."""


@dataclass(frozen=True, slots=True, init=False)
class CheckOutcome:
    """A Boolean decision and owned finite evidence from one check."""

    passed: bool
    detail: str
    _evidence_json: str = field(repr=False)

    def __init__(self, passed: bool, evidence: Any = None, detail: str = "") -> None:
        if type(passed) is not bool:
            raise TypeError("passed must be a Python Boolean")
        if not isinstance(detail, str):
            raise TypeError("detail must be text")
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(self, "_evidence_json", _encoded({} if evidence is None else evidence))

    @property
    def evidence(self) -> Any:
        return json.loads(self._evidence_json)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "evidence": json.loads(self._evidence_json), "detail": self.detail}


@dataclass(frozen=True, slots=True)
class StressContext:
    """The check's independent random stream and its declared parameters."""

    seed: int
    rng: random.Random
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True, init=False)
class StressCheck:
    """An executable finite assertion with explicit arithmetic and tested range.

    A hard timeout requires a picklable module-level callable. Local callables
    remain useful for explicitly unbounded, in-process interactive execution.
    """

    identifier: str
    suite: str
    function: Callable[[StressContext], CheckOutcome | bool]
    exactness: Exactness
    tolerance: float | None
    assumptions: tuple[str, ...]
    _range_json: str = field(repr=False)
    _parameters_json: str = field(repr=False)

    def __init__(
        self,
        identifier: str,
        suite: str,
        function: Callable[[StressContext], CheckOutcome | bool],
        exactness: Exactness,
        tested_range: Mapping[str, Any],
        *,
        parameters: Mapping[str, Any] | None = None,
        tolerance: float | None = None,
        assumptions: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("check identifier must be nonempty text")
        if not isinstance(suite, str) or not suite.strip():
            raise ValueError("suite must be nonempty text")
        if not callable(function):
            raise TypeError("check function must be callable")
        if exactness not in (Exactness.EXACT, Exactness.NUMERICAL):
            raise ValueError("stress check arithmetic must be exact or numerical")
        exactness = Exactness(exactness)
        if not isinstance(tested_range, Mapping) or not tested_range:
            raise ValueError("a nonempty finite tested range is required")
        parameters = {} if parameters is None else parameters
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        if exactness is Exactness.EXACT and tolerance is not None:
            raise ValueError("an exact check cannot declare a numerical tolerance")
        if exactness is Exactness.NUMERICAL:
            tolerance = _duration(tolerance, "tolerance")
        if any(not isinstance(value, str) for value in assumptions):
            raise TypeError("assumptions must be text")
        for name, value in (
            ("identifier", identifier),
            ("suite", suite),
            ("function", function),
            ("exactness", exactness),
            ("tolerance", tolerance),
            ("assumptions", tuple(assumptions)),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_range_json", _encoded(tested_range))
        object.__setattr__(self, "_parameters_json", _encoded(parameters))

    @property
    def tested_range(self) -> dict[str, Any]:
        return json.loads(self._range_json)

    @property
    def parameters(self) -> dict[str, Any]:
        return json.loads(self._parameters_json)

    def metadata(self) -> dict[str, Any]:
        function = self.function
        return {
            "identifier": self.identifier,
            "suite": self.suite,
            "exactness": self.exactness.value,
            "tolerance": self.tolerance,
            "tested_range": json.loads(self._range_json),
            "parameters": json.loads(self._parameters_json),
            "assumptions": list(self.assumptions),
            "callable": {
                "module": getattr(function, "__module__", None),
                "qualname": getattr(function, "__qualname__", type(function).__qualname__),
            },
        }


def _execute(check: StressCheck, seed: int) -> dict[str, Any]:
    try:
        # A spawned interpreter can inherit PYTHONOPTIMIZE changed after the
        # parent started. Check this process, before executing any callback.
        if sys.flags.optimize:
            raise ValueError("stress execution requires enabled Python assertions in the worker")
        context = StressContext(seed, random.Random(seed), json.loads(check._parameters_json))
        outcome = check.function(context)
        if type(outcome) is bool:
            outcome = CheckOutcome(outcome)
        if not isinstance(outcome, CheckOutcome):
            raise TypeError("a check must return CheckOutcome or a Python Boolean")
        data = outcome.to_dict()
        _encoded(data["evidence"], exact=check.exactness is Exactness.EXACT)
        return {"status": "passed" if data.pop("passed") else "failed", **data}
    except SkipCheck as exc:
        return {"status": "skipped", "detail": str(exc) or "explicit skip", "evidence": {}}
    except AssertionError as exc:
        return {"status": "failed", "detail": f"AssertionError: {exc}", "evidence": {}}
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}", "evidence": {}}


def _worker(sender: Any, check: StressCheck, seed: int) -> None:
    stream = io.StringIO()
    try:
        with redirect_stdout(stream), redirect_stderr(stream):
            result = _execute(check, seed)
        if stream.tell():
            result["captured_output"] = stream.getvalue()[:10000]
            result["captured_output_truncated"] = stream.tell() > 10000
        sender.send(result)
    finally:
        sender.close()


def _isolated(check: StressCheck, seed: int, timeout: float) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, check, seed))
    try:
        # Fail visibly before trying to start an unpickleable interactive closure.
        pickle.dumps(check)
        process.start()
        sender.close()
        if receiver.poll(timeout):
            try:
                result = receiver.recv()
            except EOFError:
                result = {"status": "error", "detail": "worker exited without a result", "evidence": {}}
        else:
            result = {"status": "timeout", "detail": "wall-time budget exhausted", "evidence": {}}
        process.join(timeout=0.2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
        if result["status"] != "timeout" and process.exitcode != 0:
            result = {"status": "error", "detail": f"worker exit status {process.exitcode}", "evidence": {}}
        return result
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}", "evidence": {}}
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join()


def _sources(checks: tuple[StressCheck, ...]) -> list[tuple[str, Path]]:
    package = Path(__file__).resolve().parents[1]
    paths = {
        ("package/" + path.relative_to(package).as_posix(), path)
        for path in package.rglob("*")
        if path.is_file() and path.suffix in (".py", ".json")
    }
    for check in checks:
        try:
            path = inspect.getsourcefile(check.function)
        except TypeError:
            path = None
        if path and Path(path).is_file():
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(package):
                paths.add(("custom/" + str(resolved), resolved))
    return sorted(paths)


@dataclass(frozen=True, slots=True)
class StressReport:
    """An immutable historical report; dictionary reads return fresh copies."""

    _json: str

    @property
    def passed(self) -> bool:
        return self.to_dict()["status"] == "passed"

    @property
    def counts(self) -> dict[str, int]:
        return self.to_dict()["counts"]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @property
    def replay_digest(self) -> str:
        """Hash scientific decisions, inputs and code, excluding clocks and paths."""
        data = self.to_dict()
        checks = [
            {key: value for key, value in item.items() if key != "runtime_seconds"} for item in data["checks"]
        ]
        source = sorted(
            ({key: item[key] for key in ("sha256_before", "sha256_after")} for item in data["source_inputs"]),
            key=lambda item: item["sha256_before"],
        )
        return _hash(
            _encoded(
                {"seed": data["seed"], "checks": checks, "sources": source, "software": data["software"]}
            )
        )


def run_stress(
    checks: Iterable[StressCheck],
    *,
    seed: int = 20261009,
    timeout_seconds: float | None = None,
    total_timeout_seconds: float | None = None,
) -> StressReport:
    """Execute every requested check and retain all failures and omissions.

    Without timeout_seconds execution is in-process and has no hard resource
    limit. With it each check is isolated in a spawn child and terminated on
    timeout. A total budget also requires isolation. No CPU/memory cap is made.
    """
    if sys.flags.optimize:
        raise ValueError("stress execution requires enabled Python assertions")
    seed = _integer(seed, "seed", 2**64 - 1)
    if timeout_seconds is not None:
        timeout_seconds = _duration(timeout_seconds, "timeout_seconds")
    if total_timeout_seconds is not None:
        total_timeout_seconds = _duration(total_timeout_seconds, "total_timeout_seconds")
        if timeout_seconds is None:
            raise ValueError("a total wall-time budget requires subprocess isolation")
    selected = tuple(checks)
    if not selected or any(not isinstance(check, StressCheck) for check in selected):
        raise ValueError("at least one StressCheck is required")
    if len({check.identifier for check in selected}) != len(selected):
        raise ValueError("duplicate check identifiers are not allowed")
    sources = _sources(selected)
    hashes = [
        {
            "path": name,
            "local_path": str(path),
            "sha256_before": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in sources
    ]
    software = {"Python": platform.python_version()}
    for distribution in ("komorebi-math", "numpy", "sympy", "scipy", "mpmath"):
        try:
            software[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            software[distribution] = "not installed"
    started = datetime.now(UTC).isoformat()
    clock = time.monotonic()
    records = []
    for check in selected:
        metadata = check.metadata()
        stream_seed = int(_hash(f"komorebi-stress-v1:{seed}:{check.identifier}"), 16)
        before = time.monotonic()
        if timeout_seconds is None:
            result = _execute(check, stream_seed)
        else:
            budget = timeout_seconds
            if total_timeout_seconds is not None:
                budget = min(budget, total_timeout_seconds - (before - clock))
            if budget <= 0:
                result = {"status": "not_run", "detail": "total wall-time budget exhausted", "evidence": {}}
            else:
                result = _isolated(check, stream_seed, budget)
        records.append(
            {
                **metadata,
                "stream_seed": stream_seed,
                "input_sha256": _hash(_encoded({**metadata, "stream_seed": stream_seed})),
                **result,
                "runtime_seconds": time.monotonic() - before,
            }
        )
    for row, (_, path) in zip(hashes, sources, strict=True):
        row["sha256_after"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    unchanged = all(row["sha256_before"] == row["sha256_after"] for row in hashes)
    counts = {
        status: sum(row["status"] == status for row in records)
        for status in ("passed", "failed", "error", "skipped", "timeout", "not_run")
    }
    data = {
        "schema_version": 1,
        "kind": "finite_mathematical_stress_report",
        "status": "passed" if unchanged and counts["passed"] == len(records) else "failed",
        "counts": counts,
        "seed": seed,
        "software": software,
        "randomness": "Python random.Random, independent SHA256-derived stream per check identifier",
        "started_at": started,
        "runtime_seconds": time.monotonic() - clock,
        "resource_limits": {
            "per_check_wall_seconds": timeout_seconds,
            "total_wall_seconds": total_timeout_seconds,
            "memory_bytes": None,
            "cpu_seconds": None,
        },
        "source_inputs": hashes,
        "source_inputs_unchanged": unchanged,
        "checks": records,
        "non_claims": [
            "Finite sampled/enumerated checks are not universal mathematical proofs.",
            "Numerical tests do not supply rigorous interval enclosures.",
            "Declarations and custom check oracles are supplied by the caller, not authenticated.",
            "Reports are historical evidence and do not grant live recovery or theorem certificates.",
            "Wall-time isolation does not enforce CPU, memory, thread or descendant-process limits.",
        ],
    }
    return StressReport(_encoded(data))
