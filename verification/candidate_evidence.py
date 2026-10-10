"""External revision/artifact-bound CPU evidence, separate from beta history.

Hashes establish content identity, not who executed a report. Preserve the CI
run or hardware operator provenance alongside this bundle. Candidate creation
requires a clean tracked Git checkout; verification also supports an offline
source archive whose candidate manifest was obtained from that trusted run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    from .release_integrity import package_files, release_binding, support_files, verify_release
    from .release_artifact_gate import require_installed_math_campaign
except ImportError:
    from release_integrity import package_files, release_binding, support_files, verify_release
    from release_artifact_gate import require_installed_math_campaign

MANIFEST_SCHEMA = "komorebi-math.candidate-manifest"
CPU_SCHEMA = "komorebi-math.candidate-cpu-record"
AGGREGATE_SCHEMA = "komorebi-math.candidate-cpu-qualification"
REQUIRED_CELLS = {(os_name, minor) for os_name in ("Linux", "Windows")
                  for minor in ("3.11", "3.12", "3.13")}
LIMITS = ["CUDA hardware qualification is separate", "clean release tag is not established",
          "this CPU report does not authorize RC or final release promotion",
          "digests establish content identity, not execution authenticity"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _hex(value: Any, size: int) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(size) + "}", value) is not None


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _load(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"duplicate JSON field: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=constant)
    _require(isinstance(value, dict), "JSON record must be an object")
    return value


def _git_identity(root: Path) -> tuple[str, str]:
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, check=False, timeout=30)
        _require(result.returncode == 0, "candidate creation requires an accessible Git checkout")
        return result.stdout.strip()
    _require(Path(git("rev-parse", "--show-toplevel")).resolve() == root.resolve(),
             "candidate root must be the Git checkout root")
    _require(not git("status", "--porcelain", "--untracked-files=no"),
             "candidate checkout has tracked modifications")
    tracked = set(git("ls-files", "-z").split("\0"))
    bound = {f"src/{name}" for name in package_files(root)}
    bound.update(name for name in support_files(root) if (root / name).is_file())
    _require(not bound - tracked, "candidate contains untracked qualification inputs")
    revision, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    _require(_hex(revision, 40) and _hex(tree, 40), "Git identity must use full 40-digit hashes")
    return revision, tree


def _validate_binding(binding: Any) -> None:
    _require(isinstance(binding, dict) and set(binding) == {
        "release_id", "source_files", "source_sha256", "artifacts"}, "invalid release binding fields")
    _require(isinstance(binding["release_id"], str) and bool(binding["release_id"].strip()),
             "release identifier is required")
    _require(_integer(binding["source_files"], 1) and _hex(binding["source_sha256"], 64),
             "invalid source binding")
    artifacts = binding["artifacts"]
    _require(isinstance(artifacts, list) and len(artifacts) == 2, "one wheel and one sdist required")
    names = []
    for row in artifacts:
        _require(isinstance(row, dict) and set(row) == {"name", "sha256", "bytes"},
                 "invalid artifact binding fields")
        name = row["name"]
        _require(isinstance(name, str) and name not in ("", ".", "..")
                 and "/" not in name and "\\" not in name and ":" not in name,
                 "artifact name must be a basename")
        _require(_hex(row["sha256"], 64) and _integer(row["bytes"], 1), "invalid artifact identity")
        names.append(name)
    _require(len(set(names)) == 2 and sum(n.endswith(".whl") for n in names) == 1
             and sum(n.endswith(".tar.gz") for n in names) == 1, "one distinct wheel and sdist required")


def _check_manifest(value: dict[str, Any]) -> None:
    _require(set(value) == {"schema", "schema_version", "revision", "git_tree",
                            "release_binding", "content_sha256"}, "invalid candidate manifest fields")
    _require(value["schema"] == MANIFEST_SCHEMA and type(value["schema_version"]) is int
             and value["schema_version"] == 1, "unsupported candidate manifest schema")
    _require(_hex(value["revision"], 40) and _hex(value["git_tree"], 40), "invalid candidate Git identity")
    _validate_binding(value["release_binding"])
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    _require(_hex(value["content_sha256"], 64) and value["content_sha256"] == _digest(content),
             "candidate manifest content digest mismatch")


def _integrity(root: Path) -> dict[str, Any]:
    report = verify_release(root)
    _require(report.get("passed") is True, f"source/artifact parity failed: {report.get('failures')}")
    return report


def create_candidate(root: Path, revision: str) -> dict[str, Any]:
    root = Path(root).resolve()
    _require(_hex(revision, 40), "revision must be a full lowercase 40-digit Git hash")
    before = _git_identity(root)
    _require(before[0] == revision, "requested revision does not match actual Git HEAD")
    _integrity(root)
    binding = release_binding(root)
    _validate_binding(binding)
    _require(_git_identity(root) == before and release_binding(root) == binding,
             "candidate changed while creating manifest")
    value = {"schema": MANIFEST_SCHEMA, "schema_version": 1, "revision": revision,
             "git_tree": before[1], "release_binding": binding}
    value["content_sha256"] = _digest(value)
    return value


def verify_candidate(root: Path, candidate_path: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = _load(Path(candidate_path))
    _check_manifest(manifest)
    if (root / ".git").exists():
        _require(_git_identity(root) == (manifest["revision"], manifest["git_tree"]),
                 "candidate revision/tree differs from the current Git checkout")
    _integrity(root)
    _require(release_binding(root) == manifest["release_binding"],
             "candidate source or artifact binding is stale")
    return manifest


def candidate_identity(root: Path, candidate_path: Path) -> dict[str, Any]:
    path = Path(candidate_path)
    before = path.read_bytes()
    manifest = verify_candidate(root, path)
    _require(path.read_bytes() == before, "candidate manifest changed during verification")
    return {"manifest_sha256": hashlib.sha256(before).hexdigest(), "revision": manifest["revision"],
            "release_binding": manifest["release_binding"]}


def _host() -> dict[str, str]:
    return {"platform": platform.system(), "machine": platform.machine(),
            "python": platform.python_version()}


def _ci_context() -> dict[str, str]:
    """Record observed execution context; these environment values are not signatures."""
    names = ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SERVER_URL", "GITHUB_REPOSITORY",
             "GITHUB_JOB", "GITHUB_WORKFLOW", "GITHUB_SHA")
    return {name: os.environ[name] for name in names if name in os.environ}


def _cell(environment: Any) -> tuple[str, str]:
    _require(isinstance(environment, dict) and set(environment) == {"platform", "machine", "python"},
             "invalid observed host environment")
    _require(all(isinstance(v, str) for v in environment.values()), "host values must be strings")
    _require(environment["machine"].lower() in {"amd64", "x86_64"}, "unsupported CPU architecture")
    _require(re.fullmatch(r"3\.\d+\.\d+", environment["python"]) is not None, "invalid Python version")
    cell = environment["platform"], ".".join(environment["python"].split(".")[:2])
    _require(cell in REQUIRED_CELLS, "host is outside the declared CPU matrix")
    return cell


def _all_passed(value: Any, *, campaign_counts: bool = False) -> None:
    if isinstance(value, dict):
        if "passed" in value:
            if campaign_counts:
                # Stress campaign status counters use "passed" as a count,
                # while result records require an actual Boolean certificate.
                _require(_integer(value["passed"], 0), "invalid nested passed count")
            else:
                _require(value["passed"] is True, "nested qualification result failed")
        for name, item in value.items():
            _all_passed(item, campaign_counts=(name == "counts"
                        and value.get("kind") == "finite_mathematical_stress_report"))
    elif isinstance(value, list):
        for item in value:
            _all_passed(item)


def _validate_artifact_report(root: Path, report: dict[str, Any], identity: dict[str, Any],
                              environment: dict[str, str]) -> dict[str, Any]:
    _require(report.get("schema") == "komorebi-math.installed-artifact-gate"
             and type(report.get("schema_version")) is int and report["schema_version"] == 2,
             "unsupported installed artifact report")
    _all_passed(report)
    for key in ("passed", "runtime_tests_requested", "dependency_isolated", "sdist_rebuilt_wheel_byte_identical"):
        _require(report.get(key) is True, f"required artifact qualification absent: {key}")
    _require(report.get("dependency_mode_requested") == "isolated"
             and report.get("dependencies_inherited") is False and report.get("network_used") is False,
             "artifact qualification must use offline isolated dependencies")
    _require(report.get("environment") == environment, "artifact report host does not match observed host")
    binding = identity["release_binding"]
    _require(report.get("binding_before") == binding and report.get("binding_after") == binding,
             "artifact execution source/artifact binding is stale or changed")
    integrity = report.get("integrity", {})
    _require(isinstance(integrity, dict) and integrity.get("passed") is True
             and integrity.get("failures") == [], "artifact integrity did not pass")
    _require(integrity.get("release_id") == binding["release_id"] == report.get("release_id"),
             "artifact release identity mismatch")
    _require(sorted(integrity.get("artifacts", []), key=lambda v: v["name"])
             == sorted(binding["artifacts"], key=lambda v: v["name"]), "artifact inventory mismatch")
    for name in ("wheel_record", "wheel_install_contract", "wheel_package_parity", "sdist_package_parity",
                 "sdist_support_parity", "sdist_shell_modes", "source_shell_modes", "build_configuration",
                 "sdist_metadata"):
        _require(isinstance(integrity.get(name), dict) and integrity[name].get("passed") is True,
                 f"missing integrity check: {name}")
    wheel_hash = next(row["sha256"] for row in binding["artifacts"] if row["name"].endswith(".whl"))
    _require(report.get("direct_wheel_sha256") == report.get("sdist_rebuilt_wheel_sha256") == wheel_hash,
             "tested wheel identity mismatch")
    replay = report.get("byte_reproducibility", {})
    _require(replay.get("passed") is True and replay.get("direct_sha256") == wheel_hash
             and replay.get("rebuilt_sha256") == wheel_hash, "byte reproducibility evidence mismatch")
    _require(report.get("sdist_rebuilt_package_parity", {}).get("passed") is True,
             "rebuilt package parity absent")
    commands = report.get("commands")
    _require(isinstance(commands, list) and bool(commands), "artifact command history is missing")
    for command in commands:
        _require(isinstance(command, dict) and type(command.get("returncode")) is int
                 and command["returncode"] == 0 and command.get("timed_out") is False,
                 "artifact command failed or timed out")
    profile = _load(root / "verification/installed_runtime_profile.json")
    files = profile["test_files"]
    _require(isinstance(files, list) and bool(files) and len(files) == len(set(files)), "invalid runtime profile")
    summaries = {}
    expected_probes = {"recurrence_synthesize", "direct_mapping_roundtrip", "direct_ndarray_roundtrip"}
    for name in ("installed_wheel", "installed_sdist_rebuilt_wheel"):
        artifact = report.get(name, {})
        _require(isinstance(artifact, dict) and artifact.get("passed") is True
                 and artifact.get("require_installed") is True and artifact.get("cli_help_passed") is True,
                 f"missing installed artifact probes: {name}")
        try:
            require_installed_math_campaign(artifact.get("mathematical_campaign"))
        except RuntimeError as exc:
            raise ValueError(f"invalid installed math campaign in {name}: {exc}") from exc
        probes = artifact.get("tests", [])
        _require(isinstance(probes, list) and len(probes) == len(expected_probes)
                 and all(isinstance(p, dict) and p.get("passed") is True for p in probes)
                 and {p.get("name") for p in probes} == expected_probes,
                 "installed regression probes are incomplete")
        suite = artifact.get("runtime_suite", {})
        _require(isinstance(suite, dict) and suite.get("passed") is True
                 and suite.get("schema") == "komorebi-math.installed-runtime-suite"
                 and type(suite.get("returncode")) is int and suite["returncode"] == 0,
                 "installed runtime suite did not pass")
        _require(suite.get("profile") == profile["name"] and type(suite.get("test_files")) is int
                 and suite["test_files"] == len(files), "installed runtime profile was incomplete")
        _require(suite.get("package_version") == artifact.get("package_version") == integrity.get("package_version"),
                 "installed package version mismatch")
        counts = suite.get("counts", {})
        _require(isinstance(counts, dict) and set(counts) == {"tests", "errors", "failures", "skipped"}
                 and _integer(counts["tests"], 1)
                 and all(type(counts[key]) is int and counts[key] == 0
                         for key in ("errors", "failures", "skipped")), "installed suite failed, skipped, or ran no tests")
        audit = suite.get("module_audit", {})
        _require(isinstance(audit, dict) and audit.get("passed") is True
                 and _integer(audit.get("module_count"), 1) and audit.get("wrong_locations") == [],
                 "installed module-origin audit failed")
        versions = suite.get("dependency_versions", {})
        _require(isinstance(versions, dict) and all(isinstance(versions.get(p), str) and versions[p]
                 for p in ("numpy", "pytest", "sympy", "scipy", "mpmath")), "dependency versions missing")
        summaries[name] = {"counts": counts, "profile": suite["profile"], "test_files": len(files),
                           "dependency_versions": versions}
    return summaries


def create_cpu_record(root: Path, candidate_path: Path, artifact_report: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    identity = candidate_identity(root, candidate_path)
    environment = _host()
    _cell(environment)
    report = _load(Path(artifact_report))
    summaries = _validate_artifact_report(root, report, identity, environment)
    _require(candidate_identity(root, candidate_path) == identity, "candidate changed during CPU record creation")
    record = {"schema": CPU_SCHEMA, "schema_version": 1, "passed": True, "candidate": identity,
              "environment": environment, "artifact_report": report, "artifact_report_sha256": _digest(report),
              "installed_results": summaries, "ci_context": _ci_context(), "limits": LIMITS}
    record["content_sha256"] = _digest(record)
    return record


def aggregate_cpu(root: Path, candidate_path: Path, record_paths: list[Path]) -> dict[str, Any]:
    root = Path(root).resolve()
    identity = candidate_identity(root, candidate_path)
    cells = {}
    for path in record_paths:
        record = _load(Path(path))
        _require(record.get("schema") == CPU_SCHEMA and type(record.get("schema_version")) is int
                 and record["schema_version"] == 1 and record.get("passed") is True, "invalid CPU record")
        _require(record.get("content_sha256") == _digest({k: v for k, v in record.items() if k != "content_sha256"}),
                 "CPU record content digest mismatch")
        _require(record.get("candidate") == identity, "CPU record belongs to a different candidate")
        context = record.get("ci_context")
        _require(isinstance(context, dict) and all(isinstance(k, str) and isinstance(v, str)
                 for k, v in context.items()), "invalid recorded CI context")
        cell = _cell(record.get("environment"))
        _require(cell not in cells, f"duplicate CPU matrix cell: {cell}")
        report = record.get("artifact_report")
        _require(isinstance(report, dict) and record.get("artifact_report_sha256") == _digest(report),
                 "artifact report content digest mismatch")
        summary = _validate_artifact_report(root, report, identity, record["environment"])
        _require(summary == record.get("installed_results"), "CPU summary disagrees with nested evidence")
        cells[cell] = {"environment": record["environment"], "record_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                       "installed_results": summary, "ci_context": context}
    _require(set(cells) == REQUIRED_CELLS, f"missing CPU matrix cells: {sorted(REQUIRED_CELLS - cells.keys())}")
    _require(candidate_identity(root, candidate_path) == identity, "candidate changed during aggregation")
    result = {"schema": AGGREGATE_SCHEMA, "schema_version": 1, "passed": True,
              "cpu_candidate_qualified": True, "candidate": identity,
              "cells": [cells[cell] for cell in sorted(cells)], "limits": LIMITS}
    result["content_sha256"] = _digest(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("create", "cpu", "aggregate"):
        action = actions.add_parser(name)
        action.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
        action.add_argument("--output", type=Path, required=True)
        if name == "create":
            action.add_argument("--revision", required=True)
        else:
            action.add_argument("--candidate", type=Path, required=True)
        if name == "cpu":
            action.add_argument("--artifact-report", type=Path, required=True)
        if name == "aggregate":
            action.add_argument("--records", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "create":
            result = create_candidate(args.root, args.revision)
        elif args.action == "cpu":
            result = create_cpu_record(args.root, args.candidate, args.artifact_report)
        else:
            result = aggregate_cpu(args.root, args.candidate, args.records)
        code = 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as exc:
        result = {"schema": "komorebi-math.candidate-evidence-error", "schema_version": 1,
                  "passed": False, "action": args.action, "error": f"{type(exc).__name__}: {exc}"}
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"action": args.action, "passed": code == 0, "output": str(args.output),
                      **({"error": result["error"]} if code else {})}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
