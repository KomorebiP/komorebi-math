"""Fail-closed promotion review of one exact candidate, without publishing it.

The installed beta registry remains historical. This repository gate rechecks
the six external CPU records, actual CUDA record, full-source/documentation
record and all bounded stress campaigns against the candidate being reviewed.
Hashes establish content identity, not operator or CI authenticity. Retain the
original execution provenance. A successful result never creates a tag, changes
a version, merges a PR, changes visibility, or publishes a release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

try:
    from . import candidate_evidence as ce
    from .hardware_qualification import _require_hardware_success
    from .release_integrity import package_files, support_files
    from .stress_release_candidate import CAMPAIGNS, _pytest_counts, _resource_result_valid
except ImportError:
    import candidate_evidence as ce
    from hardware_qualification import _require_hardware_success
    from release_integrity import package_files, support_files
    from stress_release_candidate import CAMPAIGNS, _pytest_counts, _resource_result_valid

SCHEMA = "komorebi-math.release-promotion-review"
METADATA_JSON = {
    "RELEASE_ID.json",
    "src/komorebi_math/data/public_api_1_0.json",
    "src/komorebi_math/data/public_api_beta_1_0.json",
    "src/komorebi_math/data/public_api_stable_1_0.json",
    "src/komorebi_math/data/portable_codec_contract_1_0.json",
    "src/komorebi_math/data/release_contract_1_0.json",
    "src/komorebi_math/data/upgrade_contract_1_0.json",
    "src/komorebi_math/data/stable_documentation_1_0.json",
}


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stage(version: str) -> str:
    if re.fullmatch(r"1\.0\.0b[1-9]\d*", version):
        return "beta"
    if re.fullmatch(r"1\.0\.0rc[1-9]\d*", version):
        return "release-candidate"
    if version == "1.0.0":
        return "final"
    raise ValueError("unsupported 1.0 release version")


def _project_version(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    _require(isinstance(version, str), "project version must be a string")
    _stage(version)
    return version


def _path(root: Path, name: Any) -> Path:
    _require(isinstance(name, str) and bool(name) and "\\" not in name,
             "invalid evidence-relative path")
    relative = PurePosixPath(name)
    _require(not relative.is_absolute() and not set(relative.parts) & {"..", "."},
             "unsafe evidence-relative path")
    result = root / name
    _require(result.resolve().is_relative_to(root.resolve()) and not result.is_symlink(),
             "evidence path escaped source root")
    return result


def _signed_record(path: Path, digest_field: str) -> dict[str, Any]:
    value = ce._load(path)
    _require(value.get(digest_field) == ce._digest({k: v for k, v in value.items() if k != digest_field}),
             f"{path.name}: content digest mismatch")
    return value


def verify_hardware(root: Path, identity: dict[str, Any], path: Path,
                    version: str) -> dict[str, Any]:
    """Revalidate nested results and installed payload, not merely outer PASS."""
    report = ce._load(path)
    _require(report.get("schema") == "komorebi-math.hardware-qualification"
             and type(report.get("schema_version")) is int and report["schema_version"] == 1,
             "unsupported hardware record")
    _require(report.get("passed") is True and report.get("candidate_unchanged") is True
             and report.get("candidate") == report.get("candidate_after") == identity,
             "hardware record is failed, stale or changed")
    _require(report.get("dependency_isolated") is False
             and report.get("dependencies_inherited") is True
             and report.get("network_access_requested") is False
             and report.get("required_backends") == ["torch", "cupy"],
             "hardware environment contract differs")
    host = report.get("host", {})
    _require(isinstance(host, dict) and host.get("system") in {"Linux", "Windows"}
             and str(host.get("machine", "")).lower() in {"amd64", "x86_64"}
             and re.fullmatch(r"3\.(11|12|13)\.\d+", str(host.get("python", ""))) is not None,
             "hardware host is outside the supported matrix")
    commands = report.get("commands")
    _require(isinstance(commands, list) and len(commands) >= 4, "hardware command history is incomplete")
    for command in commands:
        _require(isinstance(command, dict) and type(command.get("returncode")) is int
                 and command["returncode"] == 0 and command.get("timed_out") is False,
                 "hardware command failed or timed out")
    wheel = next(row for row in identity["release_binding"]["artifacts"] if row["name"].endswith(".whl"))
    _require(report.get("wheel") == wheel, "hardware wheel identity differs")
    with zipfile.ZipFile(root / "release/dist" / wheel["name"]) as archive:
        inventory = {i.filename: _sha(archive.read(i)) for i in archive.infolist()
                     if i.filename.startswith("komorebi_math/") and not i.is_dir()}
    _require(bool(inventory), "hardware wheel has no package inventory")
    inventory_digest = _sha(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())
    installed = report.get("installed", {})
    _require(isinstance(installed, dict) and installed.get("passed") is True
             and installed.get("package_version") == version, "installed hardware package identity differs")
    before, after = installed.get("installed_before", {}), installed.get("installed_after", {})
    _require(before == after and before.get("passed") is True
             and type(before.get("package_files")) is int and before["package_files"] == len(inventory)
             and before.get("package_sha256") == inventory_digest,
             "hardware installed payload does not match the candidate wheel")
    for name in ("imports_before", "imports_after"):
        audit = installed.get(name, {})
        _require(isinstance(audit, dict) and audit.get("passed") is True
                 and type(audit.get("modules_checked")) is int and audit["modules_checked"] > 0,
                 "hardware module-origin audit absent")
    gpu = installed.get("gpu", {})
    _require_hardware_success(gpu)
    for backend in gpu["qualification"]["backends"]:
        _require(type(backend.get("returncode")) is int and backend["returncode"] == 0,
                 "hardware backend did not return a genuine integer success status")
    ce._all_passed(report)
    return {"sha256": _sha(path.read_bytes()), "host": host, "wheel": wheel,
            "package_files": len(inventory), "required_backends": ["torch", "cupy"]}


def _source_skip(case: ET.Element) -> bool:
    """Retain existing optional export/backend and Windows POSIX exclusions."""
    module, name = case.get("classname", ""), case.get("name", "")
    skipped = case.find("skipped")
    reason = skipped.get("message", "") if skipped is not None else ""
    if module == "tests.test_export_files":
        if name in {"test_pdf_figure_asset_compiles", "test_export_many_latex_and_pdf_share_generated_source",
                    "test_pdf_export_and_direct_compile"}:
            return reason == "LaTeX is not installed"
        return name == "test_yaml_optional_dependency" and reason == "Python executable unavailable"
    if module == "tests.test_release_repair":
        return (name, reason) in {("test_source_shell_mode_loss_rejected", "POSIX execute permissions only"),
                                  ("test_handoff_replay_restores_verified_zip_modes", "POSIX execute permission replay")}
    optional = {
        "tests.test_nonlinear_sparse_autodiff": {"test_jax_adapter_matches_polynomial_through_order_three",
                                                 "test_torch_adapter_matches_polynomial_through_order_three"},
        "tests.test_block10_sparse_autodiff": {"test_jax_adapter_through_third_order",
                                               "test_torch_adapter_through_third_order"},
        "tests.test_stress_optional_backends": {"test_optional_cpu_ad_matches_independent_analytic_controls[jax]",
                                                 "test_optional_cpu_ad_matches_independent_analytic_controls[torch]",
                                                 "test_torch_cpu_controls_allow_cuda_build_metadata"},
    }
    return name in optional.get(module, set()) and reason.startswith(("jax backend unavailable", "torch backend unavailable",
                                                                     "jax unavailable:", "torch unavailable:"))


def verify_publication(identity: dict[str, Any], path: Path, version: str,
                       source_junit: Path | None = None) -> dict[str, Any]:
    report = _signed_record(path, "report_sha256")
    _require(report.get("schema") == "komorebi-math.block21-publication-gate"
             and type(report.get("schema_version")) is int and report["schema_version"] == 1
             and report.get("passed") is True and report.get("mode") == "full"
             and report.get("package_version") == version, "full publication qualification absent")
    _require(report.get("environment", {}).get("release_binding") == identity["release_binding"],
             "publication source/artifact binding is stale")
    checks = report.get("checks", [])
    _require(isinstance(checks, list) and all(isinstance(x, dict) and x.get("passed") is True for x in checks),
             "publication check failed")
    required = {"strict MkDocs build", "complete source test suite",
                "source and artifact snapshot unchanged during qualification"}
    names = [x.get("name") for x in checks]
    _require(len(names) == len(set(names)) and required <= set(names),
             "publication qualification omitted required checks")
    full = report.get("full_test_record", {})
    _require(isinstance(full, dict) and type(full.get("returncode")) is int and full["returncode"] == 0,
             "full source tests failed or are missing")
    output = full.get("output")
    _require(isinstance(output, str) and full.get("output_sha256") == _sha(output.encode()),
             "source test output digest differs")
    summary = full.get("summary", "")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    _require(bool(lines) and summary == lines[-1], "source summary differs from its actual command output")
    match = re.search(r"(?:^|\s)([1-9]\d*) passed(?:[,\s]|$)", summary)
    _require(bool(match) and not re.search(r"\b\d+ (?:failed|errors?|deselected|xfailed|xpassed)\b", summary),
             "source tests were failed, filtered or empty")
    junit = full.get("junit", {})
    _require(isinstance(junit, dict) and junit.get("passed") is True
             and isinstance(junit.get("xml"), str) and junit.get("sha256") == _sha(junit["xml"].encode()),
             "publication lacks the exact successful source-run JUnit")
    if source_junit is not None:
        _require(source_junit.read_bytes().decode("utf-8") == junit["xml"], "external JUnit differs from publication run")
    document = ET.fromstring(junit["xml"])
    # Reuse the concrete-case parser so counts cannot be asserted without cases.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="komorebi-promotion-junit-") as directory:
        xml_path = Path(directory) / "source.xml"
        xml_path.write_bytes(junit["xml"].encode("utf-8"))
        counts = _pytest_counts(xml_path)
    _require(junit.get("counts") == counts, "source JUnit summary differs from its concrete cases")
    _require(counts["errors"] == counts["failures"] == 0, "source JUnit contains failures")
    subtests = re.search(r"([1-9]\d*) subtests? passed", summary)
    passed = int(match.group(1)) + (int(subtests.group(1)) if subtests else 0)
    _require(counts["tests"] == passed + counts["skipped"], "source summary and concrete JUnit counts disagree")
    skipped_cases = [case for case in document.iter("testcase") if case.find("skipped") is not None]
    _require(all(_source_skip(case) for case in skipped_cases), "source JUnit contains an unapproved skipped case")
    ce._all_passed(report)
    return {"sha256": _sha(path.read_bytes()), "source_junit_sha256": junit["sha256"],
            "source_passed": passed, "counts": counts, "strict_documentation": True,
            "optional_source_exclusions": [{"module": c.get("classname"), "name": c.get("name"),
                                             "reason": c.find("skipped").get("message")} for c in skipped_cases]}


def verify_stress(root: Path, path: Path) -> dict[str, Any]:
    report = ce._load(path)
    _require(type(report.get("schema_version")) is int and report["schema_version"] == 2
             and report.get("claim_id") == "KOMOREBI-RELEASE-STRESS-2026-10-08",
             "unsupported stress record")
    run = report.get("run", {})
    _require(type(run.get("exit_status")) is int and run["exit_status"] == 0
             and run.get("status") == "completed" and run.get("missing_results") == []
             and run.get("invalid_results") == [], "stress execution is incomplete")
    inputs = report.get("input_artifacts", [])
    _require(isinstance(inputs, list) and bool(inputs), "stress input bindings absent")
    paths = set()
    for row in inputs:
        path_name = row.get("path")
        _require(path_name not in paths, "duplicate stress input")
        paths.add(path_name)
        _require(row.get("sha256") == row.get("sha256_after") == _sha(_path(root, path_name).read_bytes()),
                 "stress input source is stale or changed")
    required = {"pyproject.toml", "verification/stress_release_candidate.py",
                "verification/stress_resource_probe.py"}
    required.update("tests/" + name for names in CAMPAIGNS.values() for name in names)
    required.update("src/" + name for name in package_files(root) if name.endswith((".py", ".json")))
    _require(paths == required, "stress input inventory differs from the complete configured campaigns")
    campaigns = report.get("mathematics", {}).get("bounds", {}).get("campaigns", [])
    names = [row.get("name") for row in campaigns]
    _require(len(names) == len(set(names)) and set(names) == set(CAMPAIGNS) | {"resources"},
             "required stress campaigns or resource controls are absent")
    directory = _path(root, report.get("output_directory"))
    for row in campaigns:
        _require(row.get("status") == "completed" and type(row.get("exit_status")) is int
                 and row["exit_status"] == 0, "stress campaign failed")
        if row["name"] == "resources":
            _require(_resource_result_valid(directory / "resources.json"), "resource evidence failed")
        else:
            counts = _pytest_counts(directory / (row["name"] + ".xml"))
            _require(row.get("pytest_counts") == counts and counts["tests"] > 0
                     and all(counts[k] == 0 for k in ("errors", "failures", "skipped")),
                     "stress JUnit evidence failed, skipped or empty")
    outputs = report.get("outputs", [])
    declared = report.get("declared_results", [])
    expected_outputs = {str(PurePosixPath(report["output_directory"]) / (name + suffix))
                        for name in CAMPAIGNS for suffix in (".xml", ".log")}
    expected_outputs.update(str(PurePosixPath(report["output_directory"]) / name)
                            for name in ("resources.json", "resources.log"))
    _require(isinstance(outputs, list) and isinstance(declared, list)
             and len(outputs) == len(declared) == len(set(declared)) == 2 * len(names)
             and {row.get("path") for row in outputs} == set(declared) == expected_outputs,
             "stress output inventory is incomplete")
    for row in outputs:
        _require(row.get("sha256") == _sha(_path(root, row.get("path")).read_bytes()),
                 "stress output digest differs")
    ce._all_passed(report)
    return {"sha256": _sha(path.read_bytes()), "campaigns": sorted(names), "inputs": len(paths)}


def _metadata_snapshot(root: Path, version: str) -> dict[str, str]:
    """Allow only version labels and checked metadata pins in an RC→final diff."""
    files = {"src/" + name for name in package_files(root)} | support_files(root)
    modes = {}
    if (root / ".git").exists():
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--stage", "-z"],
                                 capture_output=True, text=True, timeout=30, check=False)
        _require(tracked.returncode == 0, "could not inventory tracked promotion source")
        for row in tracked.stdout.split("\0"):
            if not row:
                continue
            metadata, name = row.split("\t", 1)
            mode, _, stage = metadata.split()
            _require(stage == "0", "unmerged path in promotion source")
            files.add(name)
            modes[name] = mode
    result = {}
    for name in sorted(files):
        raw = _path(root, name).read_bytes()
        if name == "pyproject.toml":
            value = tomllib.loads(raw.decode())
            value["project"]["version"] = "<VERSION>"
            value["project"]["classifiers"] = ["<DEVELOPMENT-STATUS>" if x.startswith("Development Status :: ") else x
                                                  for x in value["project"].get("classifiers", [])]
            raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        elif name in METADATA_JSON:
            value = json.loads(raw)
            for field in ("package_version", "target_version"):
                if field in value:
                    _require(value[field] == version, f"stale promotion metadata: {name}:{field}")
                    value[field] = "<VERSION>"
            if name.endswith("release_contract_1_0.json"):
                _require(value["stage"] == _stage(version), "release-contract stage differs from version")
                value["stage"] = "<STAGE>"
                value["predecessor_version"] = "<PREDECESSOR>"
                for field in ("semantics", "persistence", "upgrade", "evidence", "documentation"):
                    artifact = value[field]
                    _require(artifact["sha256"] == _sha(_path(root, "src/komorebi_math/data/" + artifact["filename"]).read_bytes()),
                             "release-contract metadata pin is stale")
                    artifact["sha256"] = "<CHECKED-PIN>"
                for tier in ("stable", "beta", "research"):
                    artifact = value["api"][tier]
                    _require(artifact["sha256"] == _sha(_path(root, "src/komorebi_math/data/" + artifact["filename"]).read_bytes()),
                             "release-contract API metadata pin is stale")
                    artifact["sha256"] = "<CHECKED-PIN>"
            if name == "RELEASE_ID.json":
                _require(isinstance(value.get("release_id"), str) and value["release_id"].startswith(version),
                         "release identifier does not begin with the current version")
                value["release_id"] = "<VERSION>" + value["release_id"][len(version):]
                # Only the declared channel may change at promotion. The scope,
                # provenance, license, repair identifier suffix and build tag
                # remain protected by the normalized source comparison.
                channels = {"beta": {"development_checkpoint", "beta"},
                            "release-candidate": {"release_candidate"}, "final": {"stable"}}
                _require(value.get("release_channel") in channels[_stage(version)],
                         "release channel differs from package stage")
                value["release_channel"] = "<CHANNEL>"
            raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        elif name == "src/komorebi_math/_version.py" or name.startswith("docs/") or name in {"README.md", "CITATION.cff"}:
            raw = raw.replace(version.encode(), b"<VERSION>")
        result[name] = _sha(modes.get(name, "").encode() + b"\0" + raw)
    return result


def _clean_tag(root: Path, identity: dict[str, Any], tag: str, version: str) -> None:
    _require(tag == "v" + version, "release tag must be v plus the exact package version")
    # candidate_identity has already checked actual clean HEAD and tree. Resolve
    # the exact tag independently, accepting annotated or lightweight tags.
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "refs/tags/" + tag + "^{commit}"],
                            capture_output=True, text=True, timeout=30, check=False)
    _require(result.returncode == 0 and result.stdout.strip() == identity["revision"],
             "release tag is absent or does not point to the candidate")


def review(root: Path, candidate: Path, records: list[Path], hardware: Path,
           publication: Path, stress: Path, source_junit: Path | None = None, *, tag: str | None = None,
           predecessor: Path | None = None) -> dict[str, Any]:
    if sys.flags.optimize:
        raise ValueError("promotion qualification requires Python assertions enabled")
    root = root.resolve()
    _require((root / ".git").exists(), "promotion requires a real clean Git checkout")
    identity = ce.candidate_identity(root, candidate)
    version = _project_version(root)
    stage = _stage(version)
    cpu = ce.aggregate_cpu(root, candidate, records)
    hardware_result = verify_hardware(root, identity, hardware, version)
    publication_result = verify_publication(identity, publication, version, source_junit)
    stress_result = verify_stress(root, stress)
    snapshot = _metadata_snapshot(root, version)
    if stage != "beta":
        _require(tag is not None, "RC/final qualification requires the exact clean release tag")
        _clean_tag(root, identity, tag, version)
    if stage == "final":
        _require(predecessor is not None, "final promotion requires its successful RC review")
        previous = _signed_record(predecessor, "content_sha256")
        _require(previous.get("schema") == SCHEMA and previous.get("passed") is True
                 and previous.get("stage") == "release-candidate"
                 and _stage(previous.get("version", "")) == "release-candidate",
                 "final predecessor was not a successful release candidate")
        _require(previous.get("metadata_normalized_source") == snapshot,
                 "source changed beyond the permitted RC-to-final metadata transition")
        contract = ce._load(root / "src/komorebi_math/data/release_contract_1_0.json")
        _require(contract.get("predecessor_version") == previous["version"],
                 "final release contract does not name the qualified RC predecessor")
    _require(ce.candidate_identity(root, candidate) == identity, "candidate changed during promotion review")
    result = {"schema": SCHEMA, "schema_version": 1, "passed": True, "version": version, "stage": stage,
              "candidate": identity, "tag": tag, "cpu": cpu, "hardware": hardware_result,
              "publication": publication_result, "stress": stress_result,
              "metadata_normalized_source": snapshot,
              "release_publishable": stage in {"release-candidate", "final"},
              "limits": ["No version, tag, merge, repository visibility or publication was changed.",
                         "Digest checks identify content; preserve CI and hardware operator provenance.",
                         "Every new candidate requires its own artifacts and execution records."]}
    result["content_sha256"] = ce._digest(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--source-junit", type=Path, help="Optional external copy, required to match embedded source-run XML")
    parser.add_argument("--tag")
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = review(args.root, args.candidate, args.records, args.hardware,
                        args.publication, args.stress, args.source_junit, tag=args.tag, predecessor=args.predecessor)
        code = 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError, zipfile.BadZipFile, ET.ParseError, RuntimeError) as exc:
        result = {"schema": SCHEMA, "schema_version": 1, "passed": False,
                  "error": f"{type(exc).__name__}: {exc}"}
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": code == 0, "output": str(args.output), "error": result.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
