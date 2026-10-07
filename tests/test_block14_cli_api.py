from __future__ import annotations

import json
import os
import subprocess
import sys
from fractions import Fraction

from komorebi_math import CoordinateTransform, feature
from komorebi_math.cli.main import main
from komorebi_math.compatibility import (
    DEFAULT_PUBLIC_NAMESPACES,
    check_beta_api,
    check_public_api,
    create_public_api_manifest,
)


def test_block14_public_manifest_covers_coordinate_namespace() -> None:
    manifest = create_public_api_manifest()
    assert "komorebi_math.coordinates" in DEFAULT_PUBLIC_NAMESPACES
    assert manifest.namespaces == DEFAULT_PUBLIC_NAMESPACES
    assert len(manifest.symbols) >= 980
    assert any(
        symbol.namespace == "komorebi_math.coordinates"
        and symbol.name == "CoordinateTransform"
        for symbol in manifest.symbols
    )
    assert CoordinateTransform.__module__.startswith("komorebi_math.coordinates")
    check_public_api(strict_signatures=True).certificate.require()
    check_beta_api().certificate.require()


def test_coordinate_feature_is_implemented() -> None:
    specification = feature("coordinates")
    assert specification.status.value == "implemented"
    assert "General coordinate atlas" in specification.title
    assert "Arbitrary exact bases" in specification.summary
    assert "Euler-to-Fibonacci" in specification.summary


def test_coordinate_cli_workflows(capsys) -> None:
    assert main(["coordinates", "euler-fibonacci", "--count", "10"]) == 0
    euler = json.loads(capsys.readouterr().out)
    assert euler["initial"] == ["1", "0"]
    assert euler["defects"][:4] == ["-2", "1", "6", "-5"]
    assert euler["homogeneous"] is False

    assert main([
        "coordinates", "recurrence", "--values", "1,2,3,4,5,6",
        "--source", "1,1", "--target", "2,-1,3",
    ]) == 0
    recurrence = json.loads(capsys.readouterr().out)
    assert recurrence["verified"] is True
    assert recurrence["determinant"] == "1"
    assert recurrence["target_coordinates"] == ["1", "2", "3", "-5", "-8", "-11"]

    assert main(["coordinates", "polynomial", "--degree", "4"]) == 0
    polynomial = json.loads(capsys.readouterr().out)
    assert polynomial["verified"] is True
    assert polynomial["matrix"][0] == ["1", "-1/2", "-1", "7/4", "2"]

    assert main([
        "coordinates", "polynomial", "--source", "legendre",
        "--target", "chebyshev-t", "--degree", "4",
    ]) == 0
    classical = json.loads(capsys.readouterr().out)
    assert classical["verified"] is True
    assert classical["determinant"] == "525/2048"

    assert main([
        "coordinates", "basis",
        "--source-matrix", "1,0;0,1",
        "--target-matrix", "1,1;0,1",
        "--vector", "3,5",
    ]) == 0
    basis = json.loads(capsys.readouterr().out)
    assert basis["target_coordinates"] == ["-2", "5"]
    assert basis["verified"] is True

    assert main([
        "coordinates", "combinatorial", "--kind", "forward-difference",
        "--size", "6", "--values", "0,1,4,9,16,25",
    ]) == 0
    difference = json.loads(capsys.readouterr().out)
    assert difference["output"] == ["0", "1", "2", "0", "0", "0"]

    assert main(["coordinates", "spectral", "--kind", "haar", "--size", "4"]) == 0
    spectral = json.loads(capsys.readouterr().out)
    assert spectral["verified"] is True

    assert main(["coordinates", "packet", "--kind", "digit-reversal", "--length", "8"]) == 0
    packet = json.loads(capsys.readouterr().out)
    assert packet["verified"] is True

    assert main(["coordinates", "catalog"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert "arbitrary finite bases" in catalog["exact_linear"]
    assert "Legendre polynomials" in catalog["polynomial_families"]


def test_coordinate_and_block14_cli_verification() -> None:
    env = os.environ.copy()
    for suite in ('coordinates', 'block14'):
        completed = subprocess.run(
            [sys.executable, "-m", "komorebi_math", "verify", suite],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        assert "verification passed" in completed.stdout


def test_coordinate_transform_exact_roundtrip() -> None:
    transform = CoordinateTransform(
        "left",
        "right",
        ((1, 2), (3, 5)),
        "CC.TEST",
    )
    values = (Fraction(2, 3), Fraction(-5, 7))
    assert transform.inverse().apply(transform.apply(values)) == values
