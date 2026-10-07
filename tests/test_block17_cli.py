"""Command-line contract for Block 17 public-beta qualification."""

from __future__ import annotations

import json
from pathlib import Path

from komorebi_math.cli.main import build_parser, main


def test_block17_verification_routes_are_parseable() -> None:
    parser = build_parser()
    for suite in ("upgrade", "evidence", "documentation", "beta", "block17", "all"):
        assert parser.parse_args(["verify", suite]).suite == suite


def test_block17_focused_verification_routes_pass(capsys) -> None:
    for suite in ("upgrade", "evidence", "documentation", "beta"):
        assert main(["verify", suite]) == 0
        assert "verification passed" in capsys.readouterr().out


def test_release_upgrade_and_documentation_commands(capsys) -> None:
    assert main(["release", "upgrade"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert main(["release", "documentation"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_release_evidence_command_can_persist_beta_report(tmp_path: Path, capsys) -> None:
    output = tmp_path / "evidence.json"
    assert main(["release", "evidence", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert printed == persisted
    assert printed["passed"] is True
    assert printed["release_candidate_ready"] is False


def test_release_evidence_current_mode_exposes_open_windows_gate(capsys) -> None:
    assert main(["release", "evidence", "--current"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["missing_current_platforms"] == ["windows-x86_64"]


def test_release_readiness_beta_passes_and_final_mode_remains_closed(capsys) -> None:
    assert main(["release", "readiness"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert main(["release", "readiness", "--final"]) == 1
    final = json.loads(capsys.readouterr().out)
    assert final["passed"] is False
    assert final["mode"] == "release-candidate"
