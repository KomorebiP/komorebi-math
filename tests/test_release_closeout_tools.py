"""Release qualification must fail closed when controls do not execute."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from verification import check_release_manual as manual
from verification import check_math_docs as math_docs
from verification import stress_release_candidate as stress
from verification import stress_resource_probe as resources


@pytest.fixture
def toy_root(tmp_path, monkeypatch):
    root = tmp_path / "candidate"
    (root / "tests").mkdir(parents=True)
    (root / "verification").mkdir()
    (root / "src/komorebi_math").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    for name in ("certificates", "exact_arithmetic"):
        (root / "tests" / stress.CAMPAIGNS[name][0]).write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
    script = root / "verification/stress_release_candidate.py"
    shutil.copyfile(stress.__file__, script)
    monkeypatch.setattr(stress, "__file__", str(script))
    monkeypatch.setattr(sys, "argv", [str(script), "--root", str(root),
                                     "--campaign", "certificates", "--skip-resources"])
    return root


def result_path(command):
    return Path(next(value.split("=", 1)[1] for value in command if value.startswith("--junitxml=")))


def junit(tests=1, failures=0, skipped=0):
    cases = "".join(
        '<testcase name="case%d">%s</testcase>' % (
            i, "<failure/>" if i < failures else "<skipped/>" if i < failures + skipped else ""
        ) for i in range(max(tests, 0))
    )
    return (f'<testsuites><testsuite tests="{tests}" failures="{failures}" '
            f'errors="0" skipped="{skipped}">{cases}</testsuite></testsuites>')


def read_manifest(root):
    return json.loads((root / "release/STRESS_AUDIT/manifest.json").read_text(encoding="utf-8"))


def test_stress_replay_ignores_external_test_selection(toy_root, monkeypatch):
    test = toy_root / "tests/test_stress_certificates.py"
    test.write_text("def test_keep():\n    assert True\ndef test_must_execute():\n    assert False\n",
                    encoding="utf-8")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k keep")
    completed = subprocess.run([sys.executable, *sys.argv], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 1
    manifest = read_manifest(toy_root)
    assert manifest["mathematics"]["bounds"]["campaigns"][0]["pytest_counts"]["tests"] == 2
    assert manifest["run"]["status"] == "failed"


@pytest.mark.parametrize("payload", ["<testsuites>", junit(-1), junit(0),
                                    junit(1, failures=1), junit(1, skipped=1),
                                    '<testsuite tests="1" failures="0" errors="0" skipped="0"/>'])
def test_stress_invalid_result_replaces_previous_success(toy_root, monkeypatch, payload):
    output = toy_root / "release/STRESS_AUDIT"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text('{"run":{"status":"completed"}}', encoding="utf-8")

    def emit(command, **kwargs):
        result_path(command).write_text(payload, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stress.subprocess, "run", emit)
    assert stress.main() == 1
    assert read_manifest(toy_root)["run"]["status"] == "failed"


@pytest.mark.parametrize("outcome", ["missing", "timeout", "launch-error", "nonzero"])
def test_stress_child_failure_is_recorded(toy_root, monkeypatch, outcome):
    def emit(command, **kwargs):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "launch-error":
            raise OSError("fixture launch error")
        if outcome == "nonzero":
            result_path(command).write_text(junit(), encoding="utf-8")
        return SimpleNamespace(returncode=7 if outcome == "nonzero" else 0)

    monkeypatch.setattr(stress.subprocess, "run", emit)
    assert stress.main() == 1
    record = read_manifest(toy_root)
    assert record["run"]["status"] == "failed"
    assert record["run"]["invalid_results"] == ["certificates"]


def test_stress_total_budget_exhaustion_is_recorded(toy_root, monkeypatch):
    sys.argv.extend(["--campaign", "exact_arithmetic", "--total-timeout", "1"])
    elapsed = [0.0]
    monkeypatch.setattr(stress.time, "monotonic", lambda: elapsed[0])

    def emit(command, **kwargs):
        result_path(command).write_text(junit(), encoding="utf-8")
        elapsed[0] = 2.0
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stress.subprocess, "run", emit)
    assert stress.main() == 1
    campaigns = read_manifest(toy_root)["mathematics"]["bounds"]["campaigns"]
    assert campaigns[0]["status"] == "completed"
    assert campaigns[1]["status"] == "not-run-total-budget"


@pytest.mark.parametrize("fence", ["```python ", "```py", "~~~python", "````python", "   ```python3"])
def test_manual_checker_executes_valid_python_fences(tmp_path, fence):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "USER_GUIDE.md").write_text("```python\nassert True\n```\n", encoding="utf-8")
    closing = fence.lstrip()[0] * len(fence.lstrip().split("python")[0].split("py")[0])
    (docs / "WORKED_EXAMPLES.md").write_text(f"{fence}\nassert False\n{closing}\n", encoding="utf-8")
    report = manual.check(tmp_path)
    assert report["example_count"] == 2
    assert report["passed"] is False


def test_manual_checker_rejects_guide_without_examples(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "USER_GUIDE.md").write_text("```python\nassert True\n```\n", encoding="utf-8")
    (docs / "WORKED_EXAMPLES.md").write_text("No executable examples.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no.*examples"):
        manual.check(tmp_path)


def test_resource_probe_rejects_disabled_assertions(monkeypatch, tmp_path):
    completed = subprocess.run([sys.executable, "-O", str(resources.__file__), "--output",
                                str(tmp_path / "report.json")], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 2
    assert "requires enabled Python assertions" in completed.stderr
    assert not (tmp_path / "report.json").exists()


def test_manual_failure_replaces_stale_success(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text('{"passed":true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(manual.__file__), "--root", str(tmp_path), "--output", str(target)])
    assert manual.main() == 1
    assert json.loads(target.read_text(encoding="utf-8"))["passed"] is False


def test_manual_examples_inside_non_python_fence_are_ignored():
    text = "````text\n```python\nraise AssertionError\n```\n````\n```python\nassert True\n```\n"
    assert list(manual._python_examples(text, "fixture")) == [(6, "assert True\n")]


def test_manual_children_keep_assertions_with_inherited_optimization(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("USER_GUIDE.md", "WORKED_EXAMPLES.md"):
        (docs / name).write_text("```python\nassert False, 'required control'\n```\n", encoding="utf-8")
    # A parent started with -E, or an embedding application, can have enabled
    # assertions while its child inherits this environment setting.
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    report = manual.check(tmp_path)
    assert report["example_count"] == 2
    assert report["passed"] is False
    assert all(row["returncode"] != 0 for row in report["examples"])


def test_math_docs_failure_replaces_stale_success(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text('{"passed":true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(math_docs.__file__), "--root", str(tmp_path),
                                     "--site-dir", str(tmp_path / "absent-site"),
                                     "--output", str(target)])
    assert math_docs.main() == 1
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert "documentation site is empty" in report["error"]


def test_manual_profile_records_exclusions_and_rejects_added_fences(tmp_path):
    source = tmp_path / "README.md"
    source.write_text("```python\nassert True\n```\n```python\nundefined_context\n```\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema_version": 1, "sources": {
        "README.md": {"python_fences": 2, "exclude": [{"example": 2, "reason": "Contextual fragment"}]}
    }}), encoding="utf-8")
    report = manual.check(tmp_path, profile=profile)
    assert report["passed"] and report["example_count"] == 1
    assert report["excluded_examples"][0]["reason"] == "Contextual fragment"
    source.write_text(source.read_text(encoding="utf-8") + "```python\nassert False\n```\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Python-fence count changed"):
        manual.check(tmp_path, profile=profile)


def test_manual_profile_rejects_duplicate_exclusions(tmp_path):
    (tmp_path / "README.md").write_text("```python\nassert True\n```\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema_version": 1, "sources": {
        "README.md": {"python_fences": 1, "exclude": [
            {"example": 1, "reason": "Context"}, {"example": 1, "reason": "Repeated context"}]}
    }}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate exclusion"):
        manual.check(tmp_path, profile=profile)


@pytest.mark.parametrize("report", [{"status": "passed", "records": []}, {"status": "passed"}, []])
def test_stress_incomplete_resources_cannot_pass(toy_root, monkeypatch, report):
    sys.argv.remove("--skip-resources")
    (toy_root / "verification/stress_resource_probe.py").write_text("# fixture\n", encoding="utf-8")

    def emit(command, **kwargs):
        if "-m" in command:
            result_path(command).write_text(junit(), encoding="utf-8")
        else:
            Path(command[-1]).write_text(json.dumps(report), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stress.subprocess, "run", emit)
    assert stress.main() == 1
    assert read_manifest(toy_root)["run"]["invalid_results"] == ["resources"]
