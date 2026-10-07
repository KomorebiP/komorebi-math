"""Episode 008 tests for support-function and labeled branch packets."""

from __future__ import annotations

import numpy as np
import pytest

from komorebi_math.nonlinear.convex_packets import polyhedral_support_packet


def diamond_packet():
    # Dual unit ball of ||x||_infinity, so h_C(x)=||x||_infinity.
    return polyhedral_support_packet(
        np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        )
    )


def test_linear_image_support_pullback_identity() -> None:
    packet = polyhedral_support_packet(np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -0.5]]))
    operator = np.array([[2.0, -0.3], [0.4, 0.8], [1.1, -0.2]])
    image = packet.linear_image(operator)
    direction = np.array([0.7, -1.2, 0.3])

    assert image.support(direction) == pytest.approx(packet.support(operator.T @ direction))


def test_bellman_envelope_equals_max_of_branch_supports() -> None:
    packet = diamond_packet()
    matrices = [
        np.array([[1.0, 0.2], [0.0, 0.7]]),
        np.array([[0.4, -0.3], [0.8, 0.1]]),
    ]
    envelope = packet.bellman_envelope(matrices)

    for direction in (
        np.array([1.0, 0.3]),
        np.array([-0.2, 1.0]),
        np.array([0.7, -1.3]),
    ):
        branch = packet.branch_query(matrices, direction)
        assert envelope.support(direction) == pytest.approx(branch.value)


def test_bellman_composition_preserves_value_and_execution_words() -> None:
    packet = diamond_packet()
    first_family = [
        np.array([[1.0, 0.0], [0.2, 0.8]]),
        np.array([[0.6, -0.4], [0.1, 0.9]]),
    ]
    second_family = [
        np.array([[0.9, 0.3], [0.0, 0.7]]),
        np.array([[0.5, 0.1], [-0.2, 0.8]]),
    ]

    sequential = packet.bellman_envelope(first_family).bellman_envelope(second_family)
    products = [a @ b for b in second_family for a in first_family]
    direct = packet.bellman_envelope(products)

    # Geometry/value composes exactly even though the direct one-step labels
    # intentionally differ from the two-step execution words.
    for direction in (
        np.array([0.4, 1.0]),
        np.array([1.0, -0.6]),
        np.array([-0.8, -0.2]),
    ):
        assert sequential.support(direction) == pytest.approx(direct.support(direction))

    assert set(sequential.provenance) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_ando_shih_support_packet_is_exact_fixed_point_with_branch_policy() -> None:
    packet = diamond_packet()
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 1.0], [0.0, -1.0]])
    matrices = [first, second]
    envelope = packet.bellman_envelope(matrices)

    for direction in (
        np.array([2.0, 1.0]),
        np.array([1.0, 2.0]),
        np.array([-3.0, 0.5]),
        np.array([0.25, -4.0]),
        np.array([1.0, 1.0]),
    ):
        assert envelope.support(direction) == pytest.approx(packet.support(direction))

    assert packet.branch_query(matrices, np.array([2.0, 1.0])).active_branches == (0,)
    assert packet.branch_query(matrices, np.array([1.0, 2.0])).active_branches == (1,)
    assert packet.branch_query(matrices, np.array([1.0, 1.0])).active_branches == (0, 1)


def test_same_value_packet_can_have_different_policy_provenance() -> None:
    packet = diamond_packet()
    identity = np.eye(2)
    negative_identity = -np.eye(2)

    one_branch = packet.bellman_envelope([identity])
    two_branches = packet.bellman_envelope([identity, negative_identity])

    # The norm/support value is identical for every direction because the body
    # is centrally symmetric, but the maximizing action set is not recoverable
    # from the unlabeled support function alone.
    for direction in (
        np.array([1.0, 0.2]),
        np.array([-0.4, 1.3]),
        np.array([2.0, -3.0]),
    ):
        assert one_branch.support(direction) == pytest.approx(two_branches.support(direction))
        assert packet.branch_query([identity], direction).active_branches == (0,)
        assert packet.branch_query([identity, negative_identity], direction).active_branches == (0, 1)


def test_redundant_atom_is_value_dead_under_future_linear_branching() -> None:
    minimal = diamond_packet()
    redundant = polyhedral_support_packet(
        np.vstack([minimal.atoms, np.array([[0.2, 0.2], [0.0, 0.0]])])
    )
    family = [
        np.array([[0.8, 0.3], [-0.1, 0.9]]),
        np.array([[0.4, -0.6], [0.7, 0.2]]),
    ]

    minimal_future = minimal.bellman_envelope(family).bellman_envelope(family)
    redundant_future = redundant.bellman_envelope(family).bellman_envelope(family)

    for angle in np.linspace(0.0, 2.0 * np.pi, 33, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        assert redundant.support(direction) == pytest.approx(minimal.support(direction))
        assert redundant_future.support(direction) == pytest.approx(minimal_future.support(direction))


def test_exposed_face_atoms_transport_through_linear_image() -> None:
    packet = polyhedral_support_packet(
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    )
    operator = np.array([[1.0, 2.0], [-0.5, 0.25]])
    direction = np.array([0.7, -0.2])
    image = packet.linear_image(operator)

    source_face = packet.exposed_atoms(operator.T @ direction)
    expected = source_face @ operator.T
    actual = image.exposed_atoms(direction)

    assert sorted(map(tuple, np.round(actual, 12))) == sorted(map(tuple, np.round(expected, 12)))


def test_branch_envelope_active_provenance_matches_active_branches() -> None:
    packet = diamond_packet()
    first = np.array([[1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 1.0], [0.0, -1.0]])
    envelope = packet.bellman_envelope([first, second])

    assert set(envelope.query(np.array([2.0, 1.0])).active_provenance) == {(0,)}
    assert set(envelope.query(np.array([1.0, 2.0])).active_provenance) == {(1,)}
    assert set(envelope.query(np.array([1.0, 1.0])).active_provenance) == {(0,), (1,)}
