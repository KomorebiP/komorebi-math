"""Public final-release commands and the publication boundary."""

from komorebi_math import __version__

import json

import pytest

from komorebi_math.cli.main import build_parser, main
from komorebi_math.release import check_validation_evidence


def test_public_info_and_verification_help_use_subject_names(capsys):
    assert main(["info"]) == 0
    info = capsys.readouterr().out
    assert f"komorebi-math {__version__}" in info
    assert "signal processing" in info
    assert "block" not in info.lower()
    with pytest.raises(SystemExit) as stopped:
        main(["verify", "--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "expression" in help_text and "streaming" in help_text
    assert "block" not in help_text.lower()


def test_compatibility_suite_names_remain_accepted():
    parser = build_parser()
    assert parser.parse_args(["verify", "block20"]).suite == "block20"
    assert parser.parse_args(["verify", "streaming-beta"]).suite == "streaming-beta"
    assert parser.parse_args(["verify", "streaming"]).suite == "streaming"


def test_installed_check_does_not_claim_publication_evidence(capsys):
    assert main(["release", "check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["version"] == __version__
    assert report["certificate"]["valid"] is True
    assert report["gpu"] is None and report["gpu_required"] is False
    current = check_validation_evidence(require_current_all_platforms=True)
    assert not current.passed
    assert current.missing_current_platforms == ("linux-x86_64", "windows-x86_64")


def test_all_runs_functional_suites_without_external_publication_records(capsys):
    assert main(["verify", "all"]) == 0
    output = capsys.readouterr().out
    assert "verification passed" in output
    assert "typed" in output.lower()
    assert "integration" in output.lower()
    assert "CUDA hardware qualification" not in output


def test_optional_gpu_diagnostics_do_not_claim_hardware_qualification(monkeypatch):
    from komorebi_math import qualification

    monkeypatch.setattr(qualification, "gpu_diagnostics", lambda: {
        "available": False, "passed": False, "detail": "backend unavailable",
    })
    optional = qualification.run_release_qualification(include_gpu=True)
    assert optional.passed
    assert optional.gpu["available"] is False
    assert "CUDA hardware qualification" not in optional.certificate().assumptions_checked
    assert "hardware-backed CUDA qualification did not pass" in optional.certificate().assumptions_unchecked
    required = qualification.run_release_qualification(require_gpu=True)
    assert not required.passed
    assert any(check.name == "CUDA hardware qualification" and not check.passed
               for check in required.verification.checks)
