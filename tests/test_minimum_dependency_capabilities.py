"""Exercise dependency capabilities absent from otherwise installable releases."""

from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.exceptions import BackendUnavailableError
from komorebi_math.quantum import MovingPacketSystem, moving_packet_leakage_bound
from komorebi_math.refinement import SmithCongruenceProblem, smith_decomposition


@pytest.mark.parametrize(("start", "stop", "expected"), [(0.0, 2.0, 6.0), (1.0, 1.0, 0.0)])
def test_moving_leakage_quadrature_without_numpy_2_api(
    monkeypatch: pytest.MonkeyPatch, start: float, stop: float, expected: float
) -> None:
    # NumPy 1.26 satisfies the base dependency but has no np.trapezoid.
    monkeypatch.delattr(np, "trapezoid", raising=False)
    projector = np.diag([1.0, 0.0])
    complement = np.eye(2) - projector
    system = MovingPacketSystem(
        lambda time: (projector, complement),
        lambda time: (np.zeros((2, 2)), np.zeros((2, 2))),
    )
    result = moving_packet_leakage_bound(
        lambda time: np.array([[0.0, 1.0 + 2.0 * time], [1.0 + 2.0 * time, 0.0]]),
        system,
        start,
        stop,
        0,
        samples=9,
    )
    # The nonnegative coupling norm is 1 + 2t, with known exact integral.
    assert result.require() == pytest.approx(expected)
    assert result.certificate.assumptions_unchecked == ("quadrature error between samples",)


@pytest.mark.parametrize("congruence", [False, True])
def test_missing_smith_decomposition_reports_dependency_capability(
    monkeypatch: pytest.MonkeyPatch, congruence: bool
) -> None:
    normalforms = pytest.importorskip("sympy.polys.matrices.normalforms")
    # SymPy 1.12 and 1.13 lack the decomposition with transformation matrices.
    monkeypatch.delattr(normalforms, "smith_normal_decomp", raising=False)
    with pytest.raises(BackendUnavailableError, match=r"SymPy.*1\.14"):
        if congruence:
            SmithCongruenceProblem([[2]], [2], prime=2, depth=3)
        else:
            smith_decomposition([[2]])
