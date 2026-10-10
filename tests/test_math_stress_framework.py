"""Adversarial execution semantics and independent mathematical witnesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError
from fractions import Fraction
from math import pi

import pytest

from komorebi_math.audit import (
    CheckOutcome,
    SkipCheck,
    StressCheck,
    builtin_checks,
    run_stress,
)
from komorebi_math.certificates import Exactness


def _true(context):
    return CheckOutcome(True, {"draws": [context.rng.randrange(2**80) for _ in range(8)]})


def _false(context):
    # The proposed divisibility statement has the concrete counterexample n=2.
    return CheckOutcome(2**2 % 3 == 0, {"n": 2, "remainder": 2**2 % 3}, "bounded counterexample")


def _assertion(context):
    raise AssertionError("incorrect mathematical identity")


def _assert_based_check(context):
    assert 4 % 3 == 0, "incorrect divisibility assertion"
    return CheckOutcome(True, {"n": 2, "remainder": 1})


def _error(context):
    raise ZeroDivisionError("singular test formula")


def _skip(context):
    raise SkipCheck("external oracle unavailable")


def _sleep(context):
    time.sleep(10)
    return True


def _slow_interpreter_teardown(context):
    from multiprocessing.util import Finalize

    Finalize(None, time.sleep, args=(0.4,), exitpriority=0)
    return True


def _crash(context):
    os._exit(23)


def _none(context):
    return None


def _float_exact(context):
    return CheckOutcome(True, {"rounded_rational": 0.5})


def _numerical_nonfinite(context):
    return CheckOutcome(True, {"residual": float("nan")})


def check(function, *, name=None, exactness=Exactness.EXACT, tolerance=None):
    return StressCheck(
        name or function.__name__,
        "test",
        function,
        exactness,
        {"case": "explicit finite witness"},
        tolerance=tolerance,
    )


def test_false_assertion_exception_skip_and_missing_return_are_distinct():
    report = run_stress([check(function) for function in (_true, _false, _assertion, _error, _skip, _none)])
    assert not report.passed
    assert report.counts == {"passed": 1, "failed": 2, "error": 2, "skipped": 1, "timeout": 0, "not_run": 0}
    rows = {row["identifier"]: row for row in report.to_dict()["checks"]}
    assert rows["_false"]["evidence"] == {"n": 2, "remainder": 1}
    assert rows["_error"]["detail"].startswith("ZeroDivisionError:")
    assert rows["_skip"]["detail"] == "external oracle unavailable"


def test_nonfinite_or_rounded_exact_evidence_cannot_grant_pass():
    report = run_stress(
        [check(_float_exact), check(_numerical_nonfinite, exactness=Exactness.NUMERICAL, tolerance=1e-9)]
    )
    assert report.counts["error"] == 2
    assert not report.passed
    assert "floating-point" in report.to_dict()["checks"][0]["detail"]
    assert "finite" in report.to_dict()["checks"][1]["detail"]
    with pytest.raises(TypeError):
        CheckOutcome(1)
    with pytest.raises(TypeError):
        CheckOutcome(True, {"unordered": {1, 2}})
    with pytest.raises(ValueError):
        CheckOutcome(True, {"infinity": complex(float("inf"), 0)})


def test_parameters_ranges_evidence_and_reports_own_their_inputs():
    parameters, bounds, evidence = {"n": [7]}, {"indices": [0, 7]}, {"value": [Fraction(2, 3)]}
    outcome = CheckOutcome(True, evidence)
    custom = StressCheck(
        "owned", "custom", lambda context: outcome, Exactness.EXACT, bounds, parameters=parameters
    )
    parameters["n"].append(99)
    bounds["indices"].append(99)
    evidence["value"].append(float("nan"))
    custom.parameters["n"].append(999)
    outcome.evidence["value"].append(999)
    report = run_stress([custom])
    row = report.to_dict()["checks"][0]
    assert row["parameters"] == {"n": [7]}
    assert row["tested_range"] == {"indices": [0, 7]}
    assert row["evidence"] == {"value": [{"rational": [2, 3]}]}
    report.to_dict()["checks"][0]["status"] = "error"
    assert report.passed
    with pytest.raises(FrozenInstanceError):
        custom.identifier = "changed"


def test_seeded_replay_and_per_check_streams_are_stable():
    selected = (check(_true, name="A"), check(_true, name="B"))
    first, second = run_stress(selected, seed=57), run_stress(selected, seed=57)
    assert first.replay_digest == second.replay_digest
    assert first.replay_digest != run_stress(selected, seed=58).replay_digest
    reversed_report = run_stress(tuple(reversed(selected)), seed=57)
    original = {r["identifier"]: r["evidence"] for r in first.to_dict()["checks"]}
    reverse = {r["identifier"]: r["evidence"] for r in reversed_report.to_dict()["checks"]}
    assert original == reverse
    assert original["A"] != original["B"]
    assert first.to_dict()["source_inputs_unchanged"]
    assert "Python" in first.to_dict()["software"]
    assert json.loads(first.to_json())["counts"]["passed"] == 2


def test_isolated_timeout_and_budget_omission_are_visible():
    report = run_stress([check(_sleep), check(_true)], timeout_seconds=0.05, total_timeout_seconds=0.01)
    assert report.counts["timeout"] == 1
    assert report.counts["not_run"] == 1
    assert not report.passed
    assert len(report.to_dict()["checks"]) == 2


def test_isolated_crash_and_unpickleable_callable_are_errors():
    report = run_stress([check(_crash)], timeout_seconds=10)
    assert report.counts["error"] == 1
    local = StressCheck("closure", "custom", lambda context: True, Exactness.EXACT, {"n": 1})
    report = run_stress([local], timeout_seconds=10)
    assert report.counts["error"] == 1
    assert "pickle" in report.to_dict()["checks"][0]["detail"].lower()


def test_isolated_success_matches_inprocess_scientific_result():
    a = run_stress([check(_true)], seed=73)
    b = run_stress([check(_true)], seed=73, timeout_seconds=10)
    assert b.passed
    assert a.replay_digest == b.replay_digest


@pytest.mark.parametrize(
    "teardown,exit_status,expected",
    [(0.5, 0, "passed"), (2.0, 0, "timeout"), (0.5, 23, "error")],
)
def test_isolated_teardown_uses_original_deadline(monkeypatch, teardown, exit_status, expected):
    from komorebi_math.audit import runner

    clock = [0.0]
    joins = []

    class Connection:
        def close(self):
            pass

        def poll(self, timeout):
            assert timeout == pytest.approx(0.9)
            clock[0] += 0.1
            return True

        def recv(self):
            return {"status": "passed", "detail": "", "evidence": {"n": 1}}

    class Process:
        pid = None
        exitcode = None

        def start(self):
            self.pid = 123
            clock[0] += 0.1
            self.finished_at = clock[0] + 0.1 + teardown

        def is_alive(self):
            return self.pid is not None and self.exitcode is None

        def join(self, timeout=None):
            joins.append(timeout)
            if self.exitcode is None:
                clock[0] = min(clock[0] + timeout, self.finished_at)
                if clock[0] >= self.finished_at:
                    self.exitcode = exit_status

        def terminate(self):
            self.exitcode = -15

        def kill(self):
            self.exitcode = -9

    process = Process()

    class Context:
        def Pipe(self, duplex):
            assert duplex is False
            return Connection(), Connection()

        def Process(self, **kwargs):
            return process

    monkeypatch.setattr(runner.multiprocessing, "get_context", lambda _: Context())
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    result = runner._isolated(check(_true), 57, 1.0)
    assert result["status"] == expected
    assert joins[0] == pytest.approx(0.8)
    if expected == "passed":
        assert result["evidence"] == {"n": 1}
        assert process.exitcode == 0
    elif expected == "timeout":
        assert "wall-time budget" in result["detail"]
        assert process.exitcode == -15
    else:
        assert result["detail"] == "worker exit status 23"


def test_spawned_worker_can_finish_slow_teardown_inside_check_budget():
    report = run_stress([check(_slow_interpreter_teardown)], timeout_seconds=10)
    assert report.passed, report.to_dict()["checks"]
    assert report.to_dict()["checks"][0]["runtime_seconds"] >= 0.4


def test_spawn_rejects_child_optimization_inherited_after_parent_start(monkeypatch):
    assert sys.flags.optimize == 0
    selected = [check(_assert_based_check)]
    assert run_stress(selected).counts["failed"] == 1
    # This does not change the parent's flag, but does optimize a fresh child.
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    report = run_stress(selected, timeout_seconds=10)
    assert not report.passed
    assert report.counts["error"] == 1
    assert "enabled Python assertions in the worker" in report.to_dict()["checks"][0]["detail"]


@pytest.mark.parametrize("seed", [True, -1, 1.0, 2**64])
def test_seed_must_be_a_bounded_integer(seed):
    with pytest.raises(ValueError):
        run_stress([check(_true)], seed=seed)


def test_empty_duplicate_and_resource_contracts_cannot_silently_narrow_scope():
    with pytest.raises(ValueError):
        run_stress([])
    with pytest.raises(ValueError):
        run_stress([check(_true), check(_true)])
    with pytest.raises(ValueError):
        run_stress([check(_true)], total_timeout_seconds=1)
    for budget in (True, 0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            run_stress([check(_true)], timeout_seconds=budget)
    with pytest.raises(ValueError):
        builtin_checks(samples=0)
    with pytest.raises(ValueError):
        builtin_checks(suites=["not-a-suite"])
    with pytest.raises(ValueError):
        builtin_checks(suites=["core", "core"])
    with pytest.raises(ValueError):
        StressCheck("invalid", "custom", _true, Exactness.NUMERICAL, {"n": 1})


def test_all_independent_suites_and_known_obstruction_witnesses():
    report = run_stress(builtin_checks(samples=4), seed=20261009)
    assert report.passed, [(row["identifier"], row["detail"]) for row in report.to_dict()["checks"]]
    rows = {row["suite"]: row for row in report.to_dict()["checks"]}
    assert rows["arithmetic"]["evidence"]["fixed_sequence_comparisons"] == 385
    assert rows["special_functions"]["evidence"]["rank_one_table"] == [[{"rational": [1, 1]}] * 2] * 2
    assert rows["finite_joins"]["evidence"]["candidate_partitions_per_case"] == 15
    assert rows["signals"]["evidence"]["pointwise_error"] > 0.69
    assert rows["signals"]["evidence"]["candidate_order"] == 2
    assert rows["quantum"]["evidence"]["constraint_nonclosure"] == [True, True, True]
    assert {row["exactness"] for row in rows.values()} == {"exact", "numerical"}


def test_signal_fixture_segment_integration_has_hand_computable_normalization():
    from komorebi_math.audit.suites import _measurements

    # -1 on [0,1), +1 on [1,3), -1 on [3,4). Direct elementary integration
    # gives the first coefficient -2/pi and the third coefficient 2/(3*pi).
    measured = _measurements([1.0, 3.0], [2.0, -2.0], 4.0, degree=0, bound=3)
    assert measured[1] == pytest.approx(-2 / pi, abs=1e-14)
    assert measured[2] == pytest.approx(0, abs=1e-14)
    assert measured[3] == pytest.approx(2 / (3 * pi), abs=1e-14)


def test_independent_sequence_suite_detects_an_injected_bad_formula(monkeypatch):
    import komorebi_math.sequences.families as families

    monkeypatch.setattr(families, "kbonacci_closed_form", lambda order, index: 0)
    report = run_stress(builtin_checks(suites=["arithmetic"], samples=1))
    assert not report.passed
    assert report.counts["failed"] == 1
    assert "k-bonacci" in report.to_dict()["checks"][0]["detail"]


def test_cli_json_report_failure_exit_and_assertion_mode(tmp_path):
    target = tmp_path / "stress.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "komorebi_math.audit",
            "--suite",
            "finite_joins",
            "--samples",
            "2",
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout == json.loads(target.read_text())
    assert stdout["counts"]["passed"] == 1
    assert len(stdout["replay_digest"]) == 64
    failed = subprocess.run(
        [sys.executable, "-m", "komorebi_math.audit", "--suite", "finite_joins", "--timeout", ".001"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["counts"]["timeout"] == 1
    optimized = subprocess.run(
        [sys.executable, "-O", "-m", "komorebi_math.audit", "--suite", "core"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert optimized.returncode != 0
    assert "enabled Python assertions" in optimized.stderr
