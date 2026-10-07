from __future__ import annotations

import json
from pathlib import Path

from komorebi_math.cli.main import main
from komorebi_math.compatibility import check_public_api


def test_committed_public_api_manifest_matches_strictly() -> None:
    report = check_public_api(strict_signatures=True)
    report.certificate.require()
    assert not report.added
    assert not report.removed
    assert not report.kind_changed
    assert not report.signature_changed


def test_cli_api_check(capsys) -> None:
    assert main(["api", "check", "--strict-signatures"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["certificate"]["valid"] is True
    assert payload["signature_changed"] == []


def test_cli_api_snapshot_roundtrip(tmp_path: Path, capsys) -> None:
    output = tmp_path / "candidate.json"
    assert main(["api", "snapshot", "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["path"]) == output
    assert len(payload["digest"]) == 64
    assert output.is_file()
    assert main(["api", "check", "--manifest", str(output), "--strict-signatures"]) == 0
    json.loads(capsys.readouterr().out)


def test_cli_api_snapshot_refuses_overwrite(tmp_path: Path, capsys) -> None:
    output = tmp_path / "candidate.json"
    assert main(["api", "snapshot", "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["api", "snapshot", "--output", str(output)]) == 1
    error = capsys.readouterr().err
    assert "refusing to overwrite" in error
    assert main(["api", "snapshot", "--output", str(output), "--force"]) == 0


def test_cli_hardening_verification(capsys) -> None:
    assert main(["verify", "hardening"]) == 0
    output = capsys.readouterr().out
    assert "verification passed 7/7 checks" in output
