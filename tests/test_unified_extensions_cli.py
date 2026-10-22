from __future__ import annotations

import json

import numpy as np

from komorebi_math import (
    OperatorGrammar,
    OperatorRealization,
    PacketSystem,
    SeedSystem,
    UnifiedSystem,
)
from komorebi_math.cli.main import main
from komorebi_math.extensions import discover_extensions
from komorebi_math.verify import run_core_verification


def build_system() -> UnifiedSystem:
    grammar = OperatorGrammar(["X"], {"S": ("X", "X")})
    shift = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    realization = OperatorRealization(grammar, {"X": 3}, {"S": shift})
    seeds = SeedSystem({"S": shift}, [np.array([1, 0, 0], dtype=float)])
    packets = PacketSystem(
        [np.diag([1, 0, 0]), np.diag([0, 1, 0]), np.diag([0, 0, 1])],
        names=["0", "1", "2"],
    )
    return UnifiedSystem(
        grammar=grammar,
        realization=realization,
        seed_system=seeds,
        packet_system=packets,
        operators={"shift": shift},
    )


def test_unified_system_validation_and_blocking() -> None:
    system = build_system()
    assert system.validate().valid
    assert system.summary()["packet_names"] == ("0", "1", "2")
    blocked = system.block_operator("shift")
    assert np.allclose(blocked.reconstruct(), system.operators["shift"])


def test_extension_discovery_returns_tuple() -> None:
    assert isinstance(discover_extensions(), tuple)


def test_installed_verification() -> None:
    report = run_core_verification()
    assert report.passed
    assert report.passed_count == report.total_count == 6


def test_cli_info(capsys) -> None:
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert "komorebi-math" in output
    assert "operator calculus" in output


def test_cli_features(capsys) -> None:
    assert main(["features"]) == 0
    output = capsys.readouterr().out
    assert "core" in output
    assert "refinement" in output


def test_cli_verify(capsys) -> None:
    assert main(["verify", "core"]) == 0
    output = capsys.readouterr().out
    assert "6/6" in output


def test_cli_describe(capsys) -> None:
    assert main(["describe", "komorebi_math.core.packets.PacketSystem"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["identifier"] == "PSC-I-T20"


def test_unified_validation_detects_proper_seed_hull() -> None:
    system = UnifiedSystem(
        seed_system=SeedSystem([np.eye(2)], [np.array([1.0, 0.0])]),
    )
    certificate = system.validate()
    assert not certificate.valid
