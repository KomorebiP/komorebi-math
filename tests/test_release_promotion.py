"""Adversarial promotion controls runnable without CUDA or a release tag."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from verification import candidate_evidence as ce
from verification import release_promotion as promotion
from tests.test_hardware_qualification import _hardware_result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def hardware(tmp_path):
    root = tmp_path / "root"
    dist = root / "release/dist"
    dist.mkdir(parents=True)
    wheel = dist / "candidate.whl"
    payload = b"__version__ = '1.0.0b4'\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("komorebi_math/__init__.py", payload)
    identity = {"revision": "1" * 40, "manifest_sha256": "2" * 64,
                "release_binding": {"artifacts": [{"name": wheel.name, "bytes": wheel.stat().st_size,
                                                    "sha256": promotion._sha(wheel.read_bytes())}]}}
    inventory = {"komorebi_math/__init__.py": promotion._sha(payload)}
    installed = {"passed": True, "package_directory": "/fixture/venv/komorebi_math", "package_files": 1,
                 "package_sha256": promotion._sha(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())}
    report = {"schema": "komorebi-math.hardware-qualification", "schema_version": 1, "passed": True,
              "candidate": deepcopy(identity), "candidate_after": deepcopy(identity), "candidate_unchanged": True,
              "dependency_isolated": False, "dependencies_inherited": True, "network_access_requested": False,
              "required_backends": ["torch", "cupy"],
              "host": {"system": "Windows", "machine": "AMD64", "python": "3.13.2"},
              "wheel": deepcopy(identity["release_binding"]["artifacts"][0]),
              "commands": [{"returncode": 0, "timed_out": False} for _ in range(4)],
              "installed": {"passed": True, "package_version": "1.0.0b4", "installed_before": deepcopy(installed),
                            "installed_after": deepcopy(installed), "gpu": _hardware_result(),
                            "imports_before": {"passed": True, "modules_checked": 2},
                            "imports_after": {"passed": True, "modules_checked": 3}}}
    return root, identity, report, tmp_path / "hardware.json"


def test_candidate_bound_hardware_revalidates_actual_wheel_payload(hardware):
    root, identity, report, path = hardware
    write(path, report)
    assert promotion.verify_hardware(root, identity, path, "1.0.0b4")["package_files"] == 1


@pytest.mark.parametrize("change", [
    lambda r: r.update(passed=False),
    lambda r: r.update(schema_version=True),
    lambda r: r["candidate_after"].update(revision="3" * 40),
    lambda r: r.update(candidate_unchanged=False),
    lambda r: r.update(required_backends=["torch"]),
    lambda r: r["commands"][0].update(returncode=False),
    lambda r: r["commands"][0].update(timed_out=True),
    lambda r: r["wheel"].update(sha256="3" * 64),
    lambda r: r["installed"].update(package_version="1.0.0b3"),
    lambda r: r["installed"]["installed_after"].update(package_sha256="3" * 64),
    lambda r: r["installed"]["installed_before"].update(package_files=True),
    lambda r: r["installed"]["imports_after"].update(modules_checked=0),
    lambda r: r["installed"]["gpu"]["qualification"]["backends"].pop(),
    lambda r: r["installed"]["gpu"]["qualification"]["backends"][0].update(returncode=False),
    lambda r: r["installed"]["gpu"]["verification"]["checks"][0].update(passed=False),
])
def test_outer_hardware_success_cannot_hide_failed_or_stale_evidence(hardware, change):
    root, identity, report, path = hardware
    change(report)
    write(path, report)
    with pytest.raises((ValueError, RuntimeError)):
        promotion.verify_hardware(root, identity, path, "1.0.0b4")


def publication_record(identity):
    output = "20 passed in 1.00s\n"
    result = {"schema": "komorebi-math.block21-publication-gate", "schema_version": 1,
              "package_version": "1.0.0b4", "mode": "full", "passed": True,
              "environment": {"release_binding": deepcopy(identity["release_binding"])},
              "checks": [{"name": name, "passed": True} for name in (
                  "strict MkDocs build", "complete source test suite",
                  "source and artifact snapshot unchanged during qualification")],
              "full_test_record": {"returncode": 0, "summary": output.strip(), "output": output,
                                   "output_sha256": promotion._sha(output.encode())}}
    xml = '<testsuite tests="20" errors="0" failures="0" skipped="0">' + "".join(
        f'<testcase classname="tests.example" name="test_{i}"/>' for i in range(20)) + '</testsuite>'
    result["full_test_record"]["junit"] = {"passed": True, "xml": xml, "sha256": promotion._sha(xml.encode()),
        "counts": {"tests": 20, "errors": 0, "failures": 0, "skipped": 0}}
    return result


def sign(result):
    result["report_sha256"] = ce._digest({k: v for k, v in result.items() if k != "report_sha256"})
    return result


def junit(path, *, skipped=None):
    count = 20 + len(skipped or [])
    cases = [f'<testcase classname="tests.example" name="test_{i}"/>' for i in range(20)]
    cases += [f'<testcase classname="{module}" name="{name}"><skipped message="{reason}"/></testcase>'
              for module, name, reason in (skipped or [])]
    path.write_text(f'<testsuite tests="{count}" errors="0" failures="0" skipped="{len(skipped or [])}">' +
                    "".join(cases) + '</testsuite>')
    return path


def bind_source(record, xml):
    full = record["full_test_record"]
    full["output"] = full["summary"] + "\n"
    full["output_sha256"] = promotion._sha(full["output"].encode())
    full["junit"] = {"passed": True, "xml": xml.read_text(), "sha256": promotion._sha(xml.read_bytes()),
                     "counts": promotion._pytest_counts(xml)}


def test_full_source_record_keeps_documented_optional_tex_exclusion(hardware, tmp_path):
    _, identity, _, _ = hardware
    record = publication_record(identity)
    record["full_test_record"]["summary"] = "20 passed, 1 skipped in 1.00s"
    xml = junit(tmp_path / "source.xml", skipped=[("tests.test_export_files", "test_pdf_export_and_direct_compile",
                                                  "LaTeX is not installed")])
    bind_source(record, xml)
    result = promotion.verify_publication(identity, write(tmp_path / "publication.json", sign(record)), "1.0.0b4", xml)
    assert result["counts"]["skipped"] == 1
    assert result["source_passed"] == 20


@pytest.mark.parametrize("case", ["quick", "stale", "empty", "filtered", "changed_output", "missing_docs", "unapproved_skip", "contradictory_output", "missing_junit"])
def test_incomplete_source_or_docs_cannot_promote(hardware, tmp_path, case):
    _, identity, _, _ = hardware
    record = publication_record(identity)
    xml = junit(tmp_path / "source.xml")
    if case == "quick":
        record["mode"] = "quick"
    elif case == "stale":
        record["environment"]["release_binding"] = {}
    elif case == "empty":
        record["full_test_record"]["summary"] = "no tests ran"
    elif case == "filtered":
        record["full_test_record"]["summary"] = "20 passed, 2 deselected"
    elif case == "changed_output":
        record["full_test_record"]["output"] = "changed"
    elif case == "missing_docs":
        record["checks"].pop(0)
    elif case == "unapproved_skip":
        record["full_test_record"]["summary"] = "20 passed, 1 skipped"
        junit(xml, skipped=[("tests.test_release_contract", "test_important", "LaTeX is not installed")])
        bind_source(record, xml)
    elif case == "missing_junit":
        record["full_test_record"].pop("junit")
    else:
        record["full_test_record"]["output"] = "1 failed, 19 passed\n"
        record["full_test_record"]["output_sha256"] = promotion._sha(record["full_test_record"]["output"].encode())
    with pytest.raises(ValueError):
        promotion.verify_publication(identity, write(tmp_path / "publication.json", sign(record)), "1.0.0b4", xml)


@pytest.mark.parametrize("version,stage", [("1.0.0b4", "beta"), ("1.0.0rc1", "release-candidate"), ("1.0.0", "final")])
def test_contract_accepts_only_matching_stage_version_pairs(version, stage):
    from komorebi_math.release import load_release_contract
    contract = load_release_contract()
    assert replace(contract, package_version=version, stage=stage).stage == stage
    with pytest.raises(ValueError):
        replace(contract, package_version=version, stage="final" if stage != "final" else "beta")


def test_unsupported_version_and_none_stage_cannot_match_each_other():
    from komorebi_math.release import load_release_contract
    with pytest.raises(ValueError):
        replace(load_release_contract(), package_version="2.0.0", stage=None)


def test_promoted_runtime_cannot_fall_back_to_beta_evidence(monkeypatch):
    from komorebi_math.release import readiness
    calls = []
    passed = SimpleNamespace(certificate=SimpleNamespace(require=lambda: None))
    for name in ("check_stable_contract", "check_release_contract", "check_upgrade_contract", "check_stable_documentation"):
        monkeypatch.setattr(readiness, name, lambda: passed)
    def evidence(**kwargs):
        calls.append(kwargs)
        return passed
    monkeypatch.setattr(readiness, "__version__", "1.0.0rc1")
    monkeypatch.setattr(readiness, "check_validation_evidence", evidence)
    monkeypatch.setattr(readiness, "qualify_gpu", lambda: SimpleNamespace(passed=False,
        certificate=SimpleNamespace(message="missing hardware"), to_dict=lambda: {"passed": False}))
    report = readiness.check_beta_readiness()
    assert calls == [{"require_current_all_platforms": True}]
    assert report.mode == "release-candidate" and not report.passed
    assert any(c.name == "CUDA hardware qualification" and not c.passed for c in report.checks)


def test_preserved_registry_history_cannot_claim_current_rc_qualification(monkeypatch):
    from komorebi_math.release import evidence
    monkeypatch.setattr(evidence, "__version__", "1.0.0rc1")
    historical = evidence._check_historical_validation_evidence()
    assert historical.passed
    assert historical.certificate.diagnostics["current_candidate_qualification"] is False
    assert historical.certificate.diagnostics["historical_target"] == "1.0.0b4"
    assert not evidence.check_validation_evidence(require_current_all_platforms=True).passed


def test_metadata_only_final_comparison_rejects_code_change(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "src/komorebi_math").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nversion="1.0.0rc1"\nclassifiers=["Development Status :: 4 - Beta"]\n')
    (root / "src/komorebi_math/_version.py").write_text('__version__ = "1.0.0rc1"\n')
    (root / "src/komorebi_math/code.py").write_text('ANSWER = 42\n')
    monkeypatch.setattr(promotion, "package_files", lambda _: {"komorebi_math/_version.py": b"", "komorebi_math/code.py": b""})
    monkeypatch.setattr(promotion, "support_files", lambda _: {"pyproject.toml"})
    before = promotion._metadata_snapshot(root, "1.0.0rc1")
    (root / "pyproject.toml").write_text('[project]\nversion="1.0.0"\nclassifiers=["Development Status :: 5 - Production/Stable"]\n')
    (root / "src/komorebi_math/_version.py").write_text('__version__ = "1.0.0"\n')
    assert promotion._metadata_snapshot(root, "1.0.0") == before
    (root / "src/komorebi_math/code.py").write_text('ANSWER = 43\n')
    assert promotion._metadata_snapshot(root, "1.0.0") != before


def test_realistic_final_metadata_refresh_preserves_history_and_source(tmp_path, monkeypatch):
    root = tmp_path / "root"
    data = root / "src/komorebi_math/data"
    data.mkdir(parents=True)
    historical = {"target_version": "1.0.0b4", "records": [{"package_version": "0.16.0a2"}]}
    write(data / "validation_evidence_1_0.json", historical)
    write(data / "stable_semantics_1_0.json", {"cases": [{"answer": "5/4"}]})
    def update(version, stage):
        (root / "pyproject.toml").write_text('[project]\nversion="' + version + '"\n')
        (root / "src/komorebi_math/_version.py").write_text('__version__ = "' + version + '"\n')
        write(root / "RELEASE_ID.json", {"package_version": version, "release_id": version + "-repair-r4-block21",
                                         "release_channel": "stable" if stage == "final" else "release_candidate"})
        for name in ("public_api_1_0.json", "public_api_beta_1_0.json", "public_api_stable_1_0.json",
                     "portable_codec_contract_1_0.json", "stable_documentation_1_0.json"):
            write(data / name, {"package_version": version, "protected_payload": ["unchanged"]})
        write(data / "upgrade_contract_1_0.json", {"target_version": version, "fixtures": ["unchanged"]})
        def pin(name):
            return {"filename": name, "sha256": promotion._sha((data / name).read_bytes()), "count": 1}
        write(data / "release_contract_1_0.json", {"package_version": version, "stage": stage,
            "predecessor_version": "1.0.0rc1" if stage == "final" else "1.0.0b4",
            "api": {"research": pin("public_api_1_0.json"), "beta": pin("public_api_beta_1_0.json"),
                    "stable": pin("public_api_stable_1_0.json")},
            "semantics": pin("stable_semantics_1_0.json"), "persistence": pin("portable_codec_contract_1_0.json"),
            "upgrade": pin("upgrade_contract_1_0.json"), "evidence": pin("validation_evidence_1_0.json"),
            "documentation": pin("stable_documentation_1_0.json")})
    monkeypatch.setattr(promotion, "package_files", lambda _: {
        p.relative_to(root / "src").as_posix(): b"" for p in (root / "src").rglob("*") if p.is_file()})
    monkeypatch.setattr(promotion, "support_files", lambda _: {"pyproject.toml", "RELEASE_ID.json"})
    update("1.0.0rc1", "release-candidate")
    before = promotion._metadata_snapshot(root, "1.0.0rc1")
    update("1.0.0", "final")
    assert promotion._metadata_snapshot(root, "1.0.0") == before
    assert ce._load(data / "validation_evidence_1_0.json") == historical
    write(data / "stable_documentation_1_0.json", {"package_version": "1.0.0", "protected_payload": ["changed"]})
    with pytest.raises(ValueError, match="metadata pin"):
        promotion._metadata_snapshot(root, "1.0.0")


def test_stress_scientific_outputs_must_be_the_files_actually_hash_bound(tmp_path, monkeypatch):
    root = tmp_path / "root"
    output = root / "release/STRESS"
    output.mkdir(parents=True)
    monkeypatch.setattr(promotion, "CAMPAIGNS", {"toy": ("test_toy.py",)})
    monkeypatch.setattr(promotion, "package_files", lambda _: {})
    inputs = []
    for name in ("pyproject.toml", "verification/stress_release_candidate.py", "verification/stress_resource_probe.py",
                 "tests/test_toy.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
        digest = promotion._sha(path.read_bytes())
        inputs.append({"path": name, "sha256": digest, "sha256_after": digest})
    xml = junit(output / "toy.xml")
    expected = ({("finite join validation and quotient", n) for n in (4, 8, 16, 32, 64)}
                | {("directed stored-factor spectrum", n) for n in (4, 8, 16)}
                | {("bounded directed positive tail", n) for n in (4, 8, 16)})
    write(output / "resources.json", {"status": "passed", "records": [
        {"operation": name, "size": size, "passed": True} for name, size in expected]})
    for name in ("toy.log", "resources.log"):
        (output / name).write_text("fixture output\n")
    outputs = [{"path": p.relative_to(root).as_posix(), "sha256": promotion._sha(p.read_bytes())}
               for p in sorted(output.iterdir())]
    report = {"schema_version": 2, "claim_id": "KOMOREBI-RELEASE-STRESS-2026-10-08",
              "run": {"exit_status": 0, "status": "completed", "missing_results": [], "invalid_results": []},
              "input_artifacts": inputs, "output_directory": "release/STRESS", "outputs": outputs,
              "declared_results": [row["path"] for row in outputs],
              "mathematics": {"bounds": {"campaigns": [
                  {"name": "toy", "status": "completed", "exit_status": 0, "pytest_counts": promotion._pytest_counts(xml)},
                  {"name": "resources", "status": "completed", "exit_status": 0}]}}}
    path = write(root / "stress.json", report)
    assert promotion.verify_stress(root, path)["campaigns"] == ["resources", "toy"]
    unrelated = root / "unrelated.txt"
    unrelated.write_text("different scientific output\n")
    report["outputs"][0] = {"path": "unrelated.txt", "sha256": promotion._sha(unrelated.read_bytes())}
    report["declared_results"] = [row["path"] for row in report["outputs"]]
    write(path, report)
    with pytest.raises(ValueError, match="output inventory"):
        promotion.verify_stress(root, path)


def test_failed_cli_replaces_stale_success(tmp_path):
    output = write(tmp_path / "promotion.json", {"passed": True})
    missing = str(tmp_path / "missing.json")
    assert promotion.main(["--root", str(tmp_path), "--candidate", missing, "--records", missing,
                           "--hardware", missing, "--publication", missing, "--stress", missing,
                           "--source-junit", missing, "--output", str(output)]) == 1
    assert ce._load(output)["passed"] is False


@pytest.mark.parametrize("failure", [RuntimeError("failed nested GPU"), promotion.ET.ParseError("malformed source JUnit")])
def test_nested_failure_cli_also_replaces_stale_success(tmp_path, monkeypatch, failure):
    output = write(tmp_path / "promotion.json", {"passed": True})
    def rejected(*args, **kwargs):
        raise failure
    monkeypatch.setattr(promotion, "review", rejected)
    missing = str(tmp_path / "input.json")
    assert promotion.main(["--root", str(tmp_path), "--candidate", missing, "--records", missing,
                           "--hardware", missing, "--publication", missing, "--stress", missing,
                           "--output", str(output)]) == 1
    result = ce._load(output)
    assert result["passed"] is False and type(failure).__name__ in result["error"]
