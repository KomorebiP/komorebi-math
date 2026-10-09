"""Negative controls for external candidate identity and CPU promotion evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from verification import candidate_evidence as ce


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "verification").mkdir()
    (root / "src/komorebi_math").mkdir(parents=True)
    (root / "src/komorebi_math/__init__.py").write_text("# fixture\n")
    write(root / "verification/installed_runtime_profile.json",
          {"name": "full-test-profile", "test_files": ["test_a.py", "test_b.py"]})
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Candidate Test",
                    "-c", "user.email=candidate@example.invalid", "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    binding = {"release_id": "1.0.0b4-repair-r4-block21", "source_files": 2,
               "source_sha256": "a" * 64, "artifacts": [
                   {"name": "komorebi_math-1.0.0b4-4b21-py3-none-any.whl", "bytes": 20, "sha256": "b" * 64},
                   {"name": "komorebi_math-1.0.0b4.tar.gz", "bytes": 30, "sha256": "c" * 64}]}
    monkeypatch.setattr(ce, "release_binding", lambda _: copy.deepcopy(binding))
    monkeypatch.setattr(ce, "verify_release", lambda _: {"passed": True})
    manifest = ce.create_candidate(root, revision)
    path = write(tmp_path / "candidate.json", manifest)
    environment = {"platform": "Linux", "machine": "x86_64", "python": "3.12.9"}
    monkeypatch.setattr(ce, "_host", lambda: environment.copy())
    return root, path, binding, environment


def artifact_report(binding, environment):
    integrity = {"passed": True, "failures": [], "release_id": binding["release_id"],
                 "package_version": "1.0.0b4", "artifacts": copy.deepcopy(binding["artifacts"])}
    for name in ("wheel_record", "wheel_install_contract", "wheel_package_parity", "sdist_package_parity",
                 "sdist_support_parity", "sdist_shell_modes", "source_shell_modes", "build_configuration",
                 "sdist_metadata"):
        integrity[name] = {"passed": True}
    wheel = binding["artifacts"][0]["sha256"]
    report = {"schema": "komorebi-math.installed-artifact-gate", "schema_version": 2, "passed": True,
              "runtime_tests_requested": True, "dependency_isolated": True, "dependencies_inherited": False,
              "dependency_mode_requested": "isolated", "network_used": False,
              "sdist_rebuilt_wheel_byte_identical": True, "environment": environment.copy(),
              "binding_before": copy.deepcopy(binding), "binding_after": copy.deepcopy(binding),
              "integrity": integrity, "release_id": binding["release_id"],
              "direct_wheel_sha256": wheel, "sdist_rebuilt_wheel_sha256": wheel,
              "byte_reproducibility": {"passed": True, "direct_sha256": wheel, "rebuilt_sha256": wheel},
              "sdist_rebuilt_package_parity": {"passed": True},
              "commands": [{"returncode": 0, "timed_out": False}]}
    for name in ("installed_wheel", "installed_sdist_rebuilt_wheel"):
        report[name] = {"passed": True, "require_installed": True, "cli_help_passed": True,
                        "package_version": "1.0.0b4",
                        "tests": [{"name": p, "passed": True} for p in (
                            "recurrence_synthesize", "direct_mapping_roundtrip", "direct_ndarray_roundtrip")],
                        "runtime_suite": {"schema": "komorebi-math.installed-runtime-suite", "passed": True,
                            "returncode": 0, "profile": "full-test-profile", "test_files": 2,
                            "package_version": "1.0.0b4",
                            "counts": {"tests": 12, "errors": 0, "failures": 0, "skipped": 0},
                            "module_audit": {"passed": True, "module_count": 7, "wrong_locations": []},
                            "dependency_versions": {p: "1.0" for p in ("numpy", "pytest", "sympy", "scipy", "mpmath")}}}
    return report


def test_actual_git_revision_cleanliness_and_generated_files(candidate):
    root, path, _, _ = candidate
    manifest = ce.verify_candidate(root, path)
    with pytest.raises(ValueError, match="HEAD"):
        ce.create_candidate(root, "0" * 40)
    (root / "generated.log").write_text("generated output")
    assert ce.create_candidate(root, manifest["revision"]) == manifest
    (root / "verification/installed_runtime_profile.json").write_text("modified")
    with pytest.raises(ValueError, match="tracked modifications"):
        ce.create_candidate(root, manifest["revision"])
    with pytest.raises(ValueError, match="tracked modifications"):
        ce.verify_candidate(root, path)


def test_offline_verification_checks_content_without_claiming_git_authentication(candidate):
    root, path, binding, _ = candidate
    shutil.rmtree(root / ".git")
    assert ce.candidate_identity(root, path)["release_binding"] == binding
    binding["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="stale"):
        ce.verify_candidate(root, path)


def test_untracked_source_cannot_claim_clean_revision(candidate):
    root, path, _, _ = candidate
    (root / "src/komorebi_math/untracked.py").write_text("# uncommitted source\n")
    with pytest.raises(ValueError, match="untracked qualification inputs"):
        ce.verify_candidate(root, path)


@pytest.mark.parametrize("field,value", [("revision", "short"), ("git_tree", "g" * 40),
    ("schema_version", True), ("schema", "unrelated")])
def test_manifest_schema_is_strict_even_with_recomputed_digest(candidate, field, value):
    root, path, _, _ = candidate
    manifest = json.loads(path.read_text())
    manifest[field] = value
    manifest["content_sha256"] = ce._digest({k: v for k, v in manifest.items() if k != "content_sha256"})
    write(path, manifest)
    with pytest.raises(ValueError):
        ce.verify_candidate(root, path)


def test_candidate_integrity_failure_is_not_hidden(candidate, monkeypatch):
    root, path, _, _ = candidate
    monkeypatch.setattr(ce, "verify_release", lambda _: {"passed": False, "failures": ["parity"]})
    with pytest.raises(ValueError, match="parity"):
        ce.verify_candidate(root, path)


def test_complete_cpu_matrix_and_provenance(candidate, tmp_path, monkeypatch):
    root, path, binding, _ = candidate
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    records = []
    for os_name, minor in sorted(ce.REQUIRED_CELLS):
        env = {"platform": os_name, "machine": "AMD64" if os_name == "Windows" else "x86_64",
               "python": minor + ".9"}
        monkeypatch.setattr(ce, "_host", lambda env=env: env)
        raw = write(tmp_path / "artifact.json", artifact_report(binding, env))
        record = ce.create_cpu_record(root, path, raw)
        records.append(write(tmp_path / f"cpu-{os_name}-{minor}.json", record))
    result = ce.aggregate_cpu(root, path, records)
    assert result["cpu_candidate_qualified"] and len(result["cells"]) == 6
    assert all(row["ci_context"]["GITHUB_RUN_ID"] == "12345" for row in result["cells"])
    assert any("CUDA" in limit for limit in result["limits"])
    with pytest.raises(ValueError, match="missing CPU"):
        ce.aggregate_cpu(root, path, records[:-1])
    with pytest.raises(ValueError, match="duplicate CPU"):
        ce.aggregate_cpu(root, path, [*records, records[0]])
    forged = json.loads(records[0].read_text())
    forged["artifact_report"]["installed_wheel"]["runtime_suite"]["counts"]["failures"] = 1
    forged["artifact_report_sha256"] = ce._digest(forged["artifact_report"])
    forged["content_sha256"] = ce._digest({k: v for k, v in forged.items() if k != "content_sha256"})
    write(records[0], forged)
    with pytest.raises(ValueError, match="suite failed"):
        ce.aggregate_cpu(root, path, records)


@pytest.mark.parametrize("change", [
    lambda r: r.update(runtime_tests_requested=False),
    lambda r: r.update(dependency_isolated=False),
    lambda r: r["binding_before"].update(source_sha256="d" * 64),
    lambda r: r["binding_after"].update(source_sha256="d" * 64),
    lambda r: r["environment"].update(python="3.11.9"),
    lambda r: r["installed_wheel"]["runtime_suite"]["counts"].update(skipped=1),
    lambda r: r["installed_wheel"]["runtime_suite"]["counts"].update(tests=0),
    lambda r: r["installed_wheel"]["runtime_suite"].update(test_files=1),
    lambda r: r["installed_sdist_rebuilt_wheel"].pop("runtime_suite"),
    lambda r: r["installed_wheel"]["runtime_suite"]["module_audit"].update(wrong_locations=["src"]),
    lambda r: r["integrity"]["wheel_record"].update(passed=False),
    lambda r: r["commands"][0].update(returncode=1),
    lambda r: r["commands"][0].update(timed_out=True),
    lambda r: r.update(direct_wheel_sha256="d" * 64),
])
def test_outer_success_cannot_hide_invalid_nested_evidence(candidate, tmp_path, change):
    root, path, binding, env = candidate
    report = artifact_report(binding, env)
    change(report)
    raw = write(tmp_path / "artifact.json", report)
    with pytest.raises(ValueError):
        ce.create_cpu_record(root, path, raw)


@pytest.mark.parametrize("action", ["create", "cpu", "aggregate"])
def test_failed_cli_replaces_stale_success(candidate, tmp_path, action):
    root, path, _, _ = candidate
    output = write(tmp_path / "result.json", {"passed": True})
    args = [action, "--root", str(root), "--output", str(output)]
    if action == "create":
        args += ["--revision", "0" * 40]
    else:
        args += ["--candidate", str(path)]
        args += ["--artifact-report" if action == "cpu" else "--records", str(tmp_path / "missing.json")]
    assert ce.main(args) == 1
    failed = json.loads(output.read_text())
    assert failed["passed"] is False and failed["action"] == action and failed["error"]
