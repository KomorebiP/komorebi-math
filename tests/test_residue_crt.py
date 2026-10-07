from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.refinement import (
    crt_many,
    crt_pair,
    dyadic_odd_factor,
    factor_residue,
    nested_residue_address,
    nested_residue_analysis,
    residue_analyze,
    residue_packet_system,
)


def test_residue_analysis_reconstructs() -> None:
    values = np.arange(17)
    coordinates = residue_analyze(values, 5)
    assert np.array_equal(coordinates.reconstruct(), values)
    assert tuple(channel.tolist() for channel in coordinates.channels) == (
        [0, 5, 10, 15],
        [1, 6, 11, 16],
        [2, 7, 12],
        [3, 8, 13],
        [4, 9, 14],
    )


def test_residue_packet_projectors() -> None:
    packets = residue_packet_system(9, 3)
    packets.certificate().require()
    vector = np.arange(9, dtype=float)
    assert np.allclose(packets.analyze(vector).reconstruct(), vector)


def test_nested_residue_address_formula() -> None:
    assert nested_residue_address(1, 4, 2, 3) == 9
    values = np.arange(24)
    nested = nested_residue_analysis(values, 4, 3)
    direct = residue_analyze(values, 12)
    assert all(np.array_equal(left, right) for left, right in zip(nested.channels, direct.channels, strict=True))


def test_crt_pair_and_many() -> None:
    pair = crt_pair(1, 4, 2, 3)
    assert pair.residue == 5
    assert pair.residue % 4 == 1
    assert pair.residue % 3 == 2
    many = crt_many((2, 3, 1), (3, 5, 7))
    assert tuple(many.residue % modulus for modulus in many.moduli) == many.residues


def test_factor_residue_and_dyadic_odd() -> None:
    factor = factor_residue(53, (8, 5))
    assert factor.residues == (5, 3)
    assert dyadic_odd_factor(53, 3, 5) == factor


def test_non_coprime_crt_rejected() -> None:
    with pytest.raises(ValueError, match="coprime"):
        crt_pair(1, 4, 3, 6)
