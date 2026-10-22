"""External developer test filters must not narrow release qualification."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from verification import block21_publication_gate as publication
from verification import installed_runtime_suite as installed
from verification import release_artifact_gate as artifacts


def toy_tests(root: Path):
    (root / "tests").mkdir()
    (root / "tests/test_qualification.py").write_text(
        "def test_keep():\n    assert True\ndef test_required_failure():\n    assert False\n",
        encoding="utf-8",
    )


def test_full_publication_tests_ignore_external_selection(tmp_path, monkeypatch):
    toy_tests(tmp_path)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k keep")
    report = publication._run_full_tests(tmp_path)
    assert report["returncode"] == 1
    assert "1 failed, 1 passed" in report["summary"]
    assert report["junit"]["counts"] == {"tests": 2, "errors": 0, "failures": 1, "skipped": 0}
    assert report["passed"] is False
    assert os.environ["PYTEST_ADDOPTS"] == "-k keep"


def test_missing_source_junit_cannot_pass_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(publication.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, "20 passed\n", ""))
    report = publication._run_full_tests(tmp_path)
    assert report["passed"] is False
    assert report["junit"]["passed"] is False
    assert "FileNotFoundError" in report["junit"]["error"]


def test_rejected_source_junit_preserves_exact_producer_bytes(tmp_path, monkeypatch):
    raw = b'<testsuite tests="4" errors="0" failures="0" skipped="0"><testcase name="parent"/></testsuite>\r\n'
    def mismatched_report(command, **kwargs):
        path = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        path.write_bytes(raw)
        return subprocess.CompletedProcess(command, 0, "1 passed, 3 subtests passed\n", "")
    monkeypatch.setattr(publication.subprocess, "run", mismatched_report)
    report = publication._run_full_tests(tmp_path)
    assert report["passed"] is False
    assert report["junit"]["xml"].encode("utf-8") == raw
    assert report["junit"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "counts disagree" in report["junit"]["error"]


@pytest.mark.parametrize("fails", [False, True])
def test_qualified_unittest_subtests_produce_concrete_parent_evidence(tmp_path, fails):
    """All passing assertions execute; a failed subTest fails its parent."""
    tests = tmp_path / "tests"
    tests.mkdir()
    assertion = "self.assertNotEqual(value, 1)" if fails else "self.assertGreaterEqual(value, 0)"
    (tests / "test_subtests.py").write_text(
        "import unittest\nclass TestCases(unittest.TestCase):\n"
        " def test_parent(self):\n  for value in range(3):\n"
        "   with self.subTest(value=value): " + assertion + "\n", encoding="utf-8")
    report = publication._run_full_tests(tmp_path)
    assert report["returncode"] == (1 if fails else 0)
    assert report["passed"] is (not fails)
    assert report["junit"]["counts"] == {"tests": 1, "errors": 0, "failures": int(fails), "skipped": 0}


@pytest.mark.parametrize("mode", ["inherited", "isolated"])
def test_artifact_child_environment_ignores_external_selection(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k keep")
    monkeypatch.setattr(artifacts, "verify_release", lambda root: {"passed": True})
    monkeypatch.setattr(artifacts, "release_binding", lambda root: {"source_sha256": "fixture"})
    monkeypatch.setattr(artifacts, "_tar_files", lambda path: ({}, {}))
    (tmp_path / "RELEASE_ID.json").write_text(
        json.dumps({"release_id": "fixture", "wheel_build_tag": "1"}), encoding="utf-8"
    )
    dist = tmp_path / "release/dist"
    dist.mkdir(parents=True)
    (dist / "fixture.whl").write_bytes(b"fixture")
    (dist / "fixture.tar.gz").write_bytes(b"fixture")
    seen = []

    def inspect_environment(command, cwd, environment, report, timeout=180):
        seen.append(environment.get("PYTEST_ADDOPTS"))
        raise RuntimeError("bounded fixture stops before artifact execution")

    monkeypatch.setattr(artifacts, "checked_run", inspect_environment)
    report = artifacts.qualify(tmp_path, dependency_mode=mode,
                               wheelhouse=dist if mode == "isolated" else None)
    assert seen
    assert seen[0] is None
    assert "bounded fixture" in report["error"]
    assert os.environ["PYTEST_ADDOPTS"] == "-k keep"


@pytest.mark.parametrize("child_outcome", ["failure", "exception"])
def test_standalone_installed_profile_ignores_and_restores_selection(tmp_path, monkeypatch, child_outcome):
    toy_tests(tmp_path)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"name": "fixture", "test_files": ["test_qualification.py"]}), encoding="utf-8")
    result = tmp_path / "result.json"
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k keep")
    # This subprocess tests filtering and cleanup independently of the package
    # location guard. Real installed-package provenance is checked by the full gate.
    program = """
import importlib.util, json, os, sys
from pathlib import Path
script, profile, workspace, output, outcome = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parents[1] / 'src'))
spec = importlib.util.spec_from_file_location('qualification_fixture', script)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.installed_module_audit = lambda: {'passed': True}
if outcome == 'exception':
    import pytest
    def fail(*args, **kwargs):
        assert 'PYTEST_ADDOPTS' not in os.environ
        raise RuntimeError('bounded pytest fixture exception')
    pytest.main = fail
original_directory = Path.cwd()
sys.argv = [script, '--profile', profile, '--workspace', workspace, '--output', output]
code = runner.main()
assert os.environ['PYTEST_ADDOPTS'] == '-k keep'
assert Path.cwd() == original_directory
raise SystemExit(code)
"""
    completed = subprocess.run([sys.executable, "-c", program, str(Path(installed.__file__).resolve()),
                                str(profile), str(tmp_path), str(result), child_outcome],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 1
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["passed"] is False
    if child_outcome == "failure":
        assert report["counts"] == {"tests": 2, "failures": 1, "errors": 0, "skipped": 0}
    else:
        assert report["error"] == "RuntimeError: bounded pytest fixture exception"
