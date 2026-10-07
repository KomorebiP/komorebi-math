from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from komorebi_math import PacketSystem
from komorebi_math.cli.main import main
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    save_packet_jet_snapshot_stream,
    snapshot_packet_jet,
)


def _snapshot():
    tensor = np.array([[[2.0]]])
    provider = PolynomialDerivativeProvider(
        {2: tensor},
        source_dimension=1,
        target_dimension=1,
        name="block11 CLI quadratic",
    )
    packets = PacketSystem((np.eye(1),), names=("all",))
    family = PacketJetFamily(provider, np.zeros(1), packets, packets, 2)
    return snapshot_packet_jet(family.jet("all", ("all", "all")))


def test_cli_beta_manifest_check_and_snapshot(tmp_path: Path, capsys) -> None:
    assert main(["api", "check", "--beta"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["certificate"]["valid"] is True
    assert checked["strict_signatures"] is True

    path = tmp_path / "beta-api.json"
    assert main(["api", "snapshot", "--beta", "--output", str(path)]) == 0
    created = json.loads(capsys.readouterr().out)
    assert Path(created["path"]) == path
    assert len(created["digest"]) == 64
    assert main(["api", "check", "--beta", "--manifest", str(path)]) == 0
    roundtrip = json.loads(capsys.readouterr().out)
    assert roundtrip["certificate"]["valid"] is True


def test_cli_behavioral_golden_fixture(capsys) -> None:
    assert main(["golden"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["certificate"]["valid"] is True
    assert len(payload["results"]) == 15
    assert all(result["certificate"]["valid"] for result in payload["results"])


def test_cli_stream_inspect_and_verify(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "cli.kmstream"
    save_packet_jet_snapshot_stream(_snapshot(), destination, chunk_size=1)

    assert main(["stream", "inspect", str(destination), "--verify-chunks"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["state"] == "complete"
    assert manifest["coefficients_completed"] == 1

    assert main(["stream", "verify", str(destination)]) == 0
    certificate = json.loads(capsys.readouterr().out)
    assert certificate["valid"] is True


def test_cli_streaming_beta_verification(capsys) -> None:
    assert main(["verify", "streaming-beta"]) == 0
    output = capsys.readouterr().out
    assert "verification passed 8/8 checks" in output
