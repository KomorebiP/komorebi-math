"""Episode 009 tests for continuation-stable provenance packets."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from komorebi_math.nonlinear.convex_packets import polyhedral_support_packet
from komorebi_math.nonlinear.provenance_packets import (
    compile_observer_convex_packet,
    compile_policy_packet,
    finite_provenance_observer,
    observer_convex_packet,
)


def parity_observer():
    # Four reachable implementation states, but only parity is observable.
    # States 0/2 are even; 1/3 are odd. Branch 0 toggles a hidden copy bit,
    # branch 1 toggles parity and sometimes the hidden copy bit.
    return finite_provenance_observer(
        transitions=(
            (2, 1),
            (3, 0),
            (0, 3),
            (1, 2),
        ),
        outputs=("even", "odd", "even", "odd"),
    )


def newest_branch_observer():
    # State 2 means no branch has occurred. Processing stored histories from
    # oldest to newest leaves the newest/prepended branch as the final state.
    return finite_provenance_observer(
        transitions=((0, 1), (0, 1), (0, 1)),
        outputs=("branch-0", "branch-1", "empty"),
        initial_state=2,
    )


def raw_active_outputs(packet, observer, direction):
    query = packet.query(direction)
    return {
        observer.output_for_history(word)
        for word in query.active_provenance
    }


def test_moore_observer_minimization_is_future_exact() -> None:
    observer = parity_observer()
    minimized = observer.minimize()

    assert observer.state_count == 4
    assert minimized.observer.state_count == 2
    assert set(minimized.blocks) == {(0, 2), (1, 3)}

    for old_state in observer.reachable_states():
        new_state = minimized.old_to_new[old_state]
        for length in range(5):
            for prefix in product(range(observer.alphabet_size), repeat=length):
                old_future = observer.advance_prefix(old_state, prefix)
                new_future = minimized.observer.advance_prefix(new_state, prefix)
                assert observer.outputs[old_future] == minimized.observer.outputs[new_future]


def test_history_update_convention_matches_prepend_semantics() -> None:
    observer = newest_branch_observer()
    history = (1, 0, 0)
    state = observer.state_for_history(history)
    assert observer.outputs[state] == "branch-1"

    # Prepending branch 0 should be one incremental observer transition.
    advanced = observer.advance(state, 0)
    direct = observer.state_for_history((0,) + history)
    assert advanced == direct
    assert observer.outputs[direct] == "branch-0"


def test_query_indexed_provenance_compresses_exponential_histories() -> None:
    identity = np.eye(2)
    raw = polyhedral_support_packet(np.array([[1.0, 0.0]]))
    for _ in range(10):
        raw = raw.bellman_envelope([identity, identity])

    assert raw.atom_count == 2**10

    compiled = compile_policy_packet(raw, parity_observer())
    assert compiled.observer.state_count == 2
    assert compiled.atom_count == 2
    assert set(compiled.query(np.array([1.0, 0.0])).active_outputs) == {"even", "odd"}

    # Raw-history semantics are far more informative than this query: all
    # 1024 histories are still separately present in the predecessor packet.
    assert len(set(raw.provenance)) == 2**10


def test_compiled_packet_tracks_raw_observer_semantics_under_future_branches() -> None:
    observer = parity_observer()
    raw = polyhedral_support_packet(
        np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        )
    )
    compiled = compile_policy_packet(raw, observer)

    families = [
        [
            np.array([[1.0, 0.2], [0.0, 0.8]]),
            np.array([[0.5, -0.3], [0.7, 0.4]]),
        ],
        [
            np.array([[0.8, 0.1], [-0.2, 0.9]]),
            np.array([[0.4, 0.5], [-0.6, 0.7]]),
        ],
    ]
    for family in families:
        raw = raw.bellman_envelope(family)
        compiled = compiled.bellman_envelope(family)

    for direction in (
        np.array([1.0, 0.3]),
        np.array([-0.4, 1.2]),
        np.array([0.7, -1.1]),
        np.array([-1.0, -0.2]),
    ):
        assert compiled.support(direction) == pytest.approx(raw.support(direction))
        assert set(compiled.query(direction).active_outputs) == raw_active_outputs(raw, observer, direction)


def test_singular_continuation_resurrects_interior_provenance() -> None:
    observer = newest_branch_observer()
    atoms = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],  # strict interior of the diamond
        ]
    )
    provenance = ((0,), (0,), (0,), (0,), (1,))
    raw = polyhedral_support_packet(atoms, provenance=provenance)
    compiled = compile_policy_packet(raw, observer)

    # Interior provenance is hidden for a nonzero ordinary support query.
    assert set(compiled.query(np.array([1.0, 0.2])).active_outputs) == {"branch-0"}

    # A singular continuation can collapse all geometry and make every label
    # active. Hence deleting the interior branch-1 provenance would be unsound
    # if such continuations are admissible.
    collapsed = compiled.linear_image(np.zeros((2, 2)))
    assert set(collapsed.query(np.array([1.0, 0.0])).active_outputs) == {"branch-0", "branch-1"}


def test_invertible_continuations_do_not_resurrect_strict_interior_atom() -> None:
    observer = newest_branch_observer()
    atoms = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
        ]
    )
    provenance = ((0,), (0,), (0,), (0,), (1,))
    packet = compile_policy_packet(polyhedral_support_packet(atoms, provenance=provenance), observer)

    invertible_maps = (
        np.array([[2.0, 0.3], [0.2, 0.8]]),
        np.array([[0.7, -0.4], [0.5, 1.3]]),
        np.array([[1.0, 2.0], [-0.3, 0.9]]),
    )
    directions = tuple(
        np.array([np.cos(theta), np.sin(theta)])
        for theta in np.linspace(0.0, 2.0 * np.pi, 41, endpoint=False)
    )

    for matrix in invertible_maps:
        assert abs(np.linalg.det(matrix)) > 1e-6
        transformed = packet.linear_image(matrix)
        for direction in directions:
            assert "branch-1" not in set(transformed.query(direction).active_outputs)


def test_exact_duplicate_geometry_and_observer_state_can_be_removed() -> None:
    observer = parity_observer().minimize().observer
    raw = polyhedral_support_packet(
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        provenance=((0,), (0,), (1,)),
    )
    compiled = compile_policy_packet(raw, parity_observer())

    # The two parity-equivalent branch-0 histories at identical geometry merge,
    # while the odd observer state must remain.
    assert compiled.atom_count == 2
    assert set(compiled.states) == set(range(observer.state_count))


def test_unreachable_observer_states_are_removed_by_minimization() -> None:
    observer = finite_provenance_observer(
        transitions=((0, 1), (0, 1), (2, 2)),
        outputs=("zero", "one", "unreachable"),
    )
    minimized = observer.minimize()

    assert minimized.old_to_new[2] == -1
    assert minimized.observer.state_count == 2



def test_observer_convex_packet_matches_flat_compiled_policy_semantics() -> None:
    observer = parity_observer()
    raw = polyhedral_support_packet(
        np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.2, 0.2]]),
        provenance=((), (1,), (0,), (1, 0), (1, 1)),
    )
    flat = compile_policy_packet(raw, observer)
    grouped = compile_observer_convex_packet(raw, observer)

    families = [
        [np.array([[1.0, 0.2], [0.0, 0.8]]), np.array([[0.5, -0.4], [0.3, 0.9]])],
        [np.array([[0.7, 0.1], [-0.2, 1.1]]), np.array([[0.6, 0.5], [-0.4, 0.8]])],
    ]
    for family in families:
        flat = flat.bellman_envelope(family)
        grouped = grouped.bellman_envelope(family)

    for angle in np.linspace(0.0, 2.0 * np.pi, 53, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        assert grouped.support(direction) == pytest.approx(flat.support(direction))
        assert set(grouped.query(direction).active_outputs) == set(flat.query(direction).active_outputs)


def test_statewise_convexification_is_exact_for_policy_outputs() -> None:
    observer = newest_branch_observer().minimize().observer
    # State branch-0 contains a redundant interior generator. Removing it keeps
    # that state's convex hull, and therefore every active-output query, exact.
    with_redundancy = observer_convex_packet(
        (
            np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 1.0], [0.0, -1.0]]),
            np.empty((0, 2)),
        ),
        observer,
    )
    compact = observer_convex_packet(
        (
            np.array([[1.0, 0.0], [-1.0, 0.0]]),
            np.array([[0.0, 1.0], [0.0, -1.0]]),
            np.empty((0, 2)),
        ),
        observer,
    )

    for angle in np.linspace(0.0, 2.0 * np.pi, 65, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        assert with_redundancy.support(direction) == pytest.approx(compact.support(direction))
        assert set(with_redundancy.query(direction).active_outputs) == set(compact.query(direction).active_outputs)


def test_state_conditioning_preserves_interior_provenance_through_zero_map() -> None:
    observer = newest_branch_observer()
    raw = polyhedral_support_packet(
        np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]]),
        provenance=((0,), (0,), (0,), (0,), (1,)),
    )
    grouped = compile_observer_convex_packet(raw, observer)
    collapsed = grouped.linear_image(np.zeros((2, 2)))

    assert set(collapsed.query(np.array([1.0, 0.0])).active_outputs) == {"branch-0", "branch-1"}
    # The globally interior branch-1 point survives because convexification was
    # performed inside its provenance class rather than across labels.
    branch1_state = grouped.observer.state_for_history((1,))
    assert grouped.state_atom_counts[branch1_state] == 1


def test_observer_convex_bellman_update_matches_state_transition_formula() -> None:
    observer = newest_branch_observer().minimize().observer
    packet = observer_convex_packet(
        (
            np.array([[1.0, 0.0], [-1.0, 0.0]]),
            np.array([[0.0, 1.0]]),
            np.empty((0, 2)),
        ),
        observer,
    )
    matrices = [np.eye(2), 2.0 * np.eye(2)]
    updated = packet.bellman_envelope(matrices)

    # A newly prepended branch sets the newest-branch observer state, so every
    # source state flows to state 0 through branch 0 and to state 1 through branch 1.
    assert updated.state_atom_counts[0] == 3
    assert updated.state_atom_counts[1] == 3
    assert updated.state_atom_counts[2] == 0
