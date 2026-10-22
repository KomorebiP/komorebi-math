from __future__ import annotations

from komorebi_math import __version__

from komorebi_math.cli.main import main
from komorebi_math.compatibility import (
    BETA_PUBLIC_NAMESPACES,
    DEFAULT_PUBLIC_NAMESPACES,
    check_beta_api,
    check_public_api,
    create_public_api_manifest,
)


def test_block12_public_manifest_covers_signal_namespace() -> None:
    manifest = create_public_api_manifest()
    assert "komorebi_math.signals" in DEFAULT_PUBLIC_NAMESPACES
    assert manifest.namespaces == DEFAULT_PUBLIC_NAMESPACES
    assert len(manifest.symbols) >= 699
    assert any(
        symbol.namespace == "komorebi_math.signals" and symbol.name == "stft"
        for symbol in manifest.symbols
    )
    check_public_api(strict_signatures=True).certificate.require()


def test_beta_contract_remains_curated_and_unchanged() -> None:
    assert BETA_PUBLIC_NAMESPACES == ("komorebi_math.beta",)
    report = check_beta_api()
    report.certificate.require()
    assert not report.added
    assert not report.removed
    assert not report.signature_changed


def test_dsp_cli_verification(capsys) -> None:
    assert main(["verify", "dsp"]) == 0
    output = capsys.readouterr().out
    assert "verification passed 11/11 checks" in output


def test_block12_cli_verification(capsys) -> None:
    assert main(["verify", "block12"]) == 0
    output = capsys.readouterr().out
    assert "verification passed 133/133 checks" in output


def test_info_reports_block12_version(capsys) -> None:
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert f"komorebi-math {__version__}" in output
    assert "signal processing" in output
