from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.signals import (
    FiniteFilterBank,
    PolyphaseSystem,
    haar_analyze,
    haar_filter_bank,
    haar_synthesize,
    mixed_radix_address,
    periodic_five_three_lifting,
    polyphase_analyze,
    polyphase_convolution,
    refinement_children,
    split_mixed_radix_address,
)


def test_polyphase_roundtrip_one_and_two_dimensions() -> None:
    rng = np.random.default_rng(41)
    one = rng.normal(size=37)
    coordinates = polyphase_analyze(one, 5)
    assert np.allclose(coordinates.reconstruct(), one)
    coordinates.certificate(one).require()

    two = rng.normal(size=(4, 17))
    coordinates_two = polyphase_analyze(two, 3, axis=1)
    assert np.allclose(coordinates_two.reconstruct(), two)
    coordinates_two.certificate(two).require()


def test_polyphase_system_and_mixed_radix() -> None:
    system = PolyphaseSystem(4)
    x = np.arange(23)
    coordinates = system.analyze(x)
    assert np.array_equal(coordinates.reconstruct(), x)
    assert refinement_children(3, 12, 2) == (2, 5, 8, 11)
    address = mixed_radix_address((1, 2, 3), (2, 3, 5))
    assert split_mixed_radix_address(address, (2, 3, 5)) == (1, 2, 3)


def test_polyphase_convolution_matches_direct() -> None:
    rng = np.random.default_rng(42)
    for order in (2, 3, 5):
        for length in (1, 7, 16):
            x = rng.normal(size=length)
            h = rng.normal(size=6)
            result = polyphase_convolution(x, h, order)
            result.certificate.require()
            assert np.allclose(result.value, np.convolve(x, h))


def test_haar_and_finite_filter_bank() -> None:
    rng = np.random.default_rng(43)
    x = rng.normal(size=32)
    low, high = haar_analyze(x)
    assert np.allclose(haar_synthesize(low, high), x)
    bank = haar_filter_bank(32)
    report = bank.report()
    report.certificate.require()
    assert np.isclose(report.lower_frame_bound, 1.0)
    assert np.isclose(report.upper_frame_bound, 1.0)
    assert np.allclose(bank.synthesize(bank.analyze(x)), x)


def test_canonical_dual_and_perturbation_bound() -> None:
    rng = np.random.default_rng(44)
    analysis = rng.normal(size=(7, 4))
    bank = FiniteFilterBank(analysis, np.linalg.pinv(analysis))
    dual = bank.canonical_dual()
    dual.report().certificate.require()
    delta_e = 0.01 * rng.normal(size=analysis.shape)
    delta_r = 0.01 * rng.normal(size=dual.synthesis_matrix.shape)
    actual = np.linalg.norm((dual.synthesis_matrix + delta_r) @ (analysis + delta_e) - np.eye(4), 2)
    assert actual <= dual.perturbation_bound(delta_e, delta_r) + 1e-12


def test_periodic_lifting_roundtrip() -> None:
    rng = np.random.default_rng(45)
    x = rng.normal(size=64)
    scheme = periodic_five_three_lifting()
    smooth, detail = scheme.analyze(x)
    assert np.allclose(scheme.synthesize(smooth, detail), x)
    scheme.certificate(x).require()


def test_invalid_polyphase_and_haar_inputs() -> None:
    with pytest.raises(Exception):
        polyphase_analyze(np.arange(5), 0)
    with pytest.raises(Exception):
        haar_analyze(np.arange(5))
