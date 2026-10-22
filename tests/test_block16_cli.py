from __future__ import annotations

from komorebi_math import __version__

import json
from pathlib import Path

import pytest

from komorebi_math.cli.main import build_parser, main


def test_block16_verification_commands(capsys) -> None:
    for suite in ("stable", "release"):
        assert main(["verify", suite]) == 0
        assert "verification passed" in capsys.readouterr().out
    parser = build_parser()
    for suite in ("block15", "block16", "all"):
        assert parser.parse_args(["verify", suite]).suite == suite


def test_stable_api_check_and_snapshot(tmp_path: Path, capsys) -> None:
    assert main(["api", "check", "--stable"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["compatible"] is True
    output = tmp_path / "stable.json"
    assert main(["api", "snapshot", "--stable", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert Path(result["path"]) == output
    assert output.exists()


def test_api_tiers_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["api", "check", "--beta", "--stable"])


def test_release_qualify_cli_writes_record(tmp_path: Path, capsys) -> None:
    output = tmp_path / "qualification.json"
    assert main(["release", "qualify", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert printed["certificate"]["valid"] is True
    assert persisted["version"] == __version__
