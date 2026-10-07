from __future__ import annotations

import numpy as np

from komorebi_math.signals import (
    HaarPacketTree,
    SingularPacket1D,
    TruncatedPowerTerm,
    best_basis,
    entropy_cost,
    overlap_matrix_error,
    retained_packet_reconstruction,
    scalar_region_error,
)


def test_haar_packet_tree_roundtrip_and_energy() -> None:
    rng = np.random.default_rng(51)
    x = rng.normal(size=64)
    tree = HaarPacketTree.analyze(x, 3)
    tree.certificate().require()
    pruning = tree.prune(("0", "10", "110", "111"))
    assert np.allclose(pruning.reconstruct(), x)
    assert np.isclose(pruning.energy(), np.vdot(x, x).real)


def test_best_basis_matches_simple_choice() -> None:
    x = np.arange(16, dtype=float)
    tree = HaarPacketTree.analyze(x, 2)
    costs = {path: 10.0 for path in tree.nodes}
    costs["0"] = 1.0
    costs["10"] = 1.0
    costs["11"] = 1.0
    costs[""] = 9.0
    result = best_basis(tree, costs)
    result.certificate.require()
    assert result.leaves == ("0", "10", "11")
    entropy_result = best_basis(tree, entropy_cost)
    entropy_result.certificate.require()


def test_singular_packet_trace_and_evaluation() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    packet = SingularPacket1D.from_jumps(
        [0.3, 0.6],
        [2.0, -0.5],
        orders=[0, 1],
    )
    values = packet.evaluate(grid)
    assert values.shape == grid.shape
    jumps = packet.trace_jumps()
    assert jumps[0.3][0] == 2.0
    assert jumps[0.6][1] == -0.5
    term = TruncatedPowerTerm(0.4, 3.0, 2)
    assert term.location_bound(0.45, p=2) > 0


def test_retained_packet_reconstruction_identity() -> None:
    rng = np.random.default_rng(52)
    smooth = rng.normal(size=20)
    packet = rng.normal(size=20)
    signal = smooth + packet
    q = rng.normal(size=(20, 20)) / 20
    estimate = packet + 0.01 * rng.normal(size=20)
    result = retained_packet_reconstruction(
        signal,
        estimate,
        q,
        true_smooth=smooth,
        true_packet=packet,
    )
    result.certificate.require()


def test_region_overlap_formulas() -> None:
    rng = np.random.default_rng(53)
    first = rng.random(500) < 0.4
    second = rng.random(500) < 0.3
    scalar_region_error(first, 1.2, second, -0.8, p=3).certificate.require()
    labels_a = rng.integers(0, 3, size=500)
    labels_b = rng.integers(0, 4, size=500)
    overlap_matrix_error(labels_a, [0.0, 1.0, -0.5], labels_b, [0.1, 0.7, -0.2, 1.2]).certificate.require()
