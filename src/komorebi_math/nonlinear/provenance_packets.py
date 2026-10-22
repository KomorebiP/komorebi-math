"""Research-tier provenance quotients for labeled support packets.

This module separates three notions that are easy to conflate:

* geometric value semantics, carried by a support packet;
* continuation-stable provenance semantics, carried by a finite Moore observer;
* raw execution history, which is strictly more informative unless the observer
  intentionally identifies histories.

Stored support-packet histories are extended by *prepending* newly applied
Bellman branches.  To make a finite observer update incrementally, histories are
therefore read from oldest to newest (the reverse of the stored tuple).  If
``q(h)`` is the observer state for a stored history ``h`` and branch ``a`` is
prepended, then

    q((a,) + h) = delta(q(h), a).

The observer minimization implemented here is the finite Moore-machine analogue
of Myhill--Nerode minimization: two states are identified exactly when every
future branch continuation produces the same observer output.

The policy packet deliberately preserves *set-valued* observer outputs for
active support atoms.  It does not preserve raw history identity or
multiplicity.  Those are different query semantics and generally admit less (or
no) compression.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import DimensionError, ValidationError
from .convex_packets import PolyhedralSupportPacket


def _real_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) and np.any(np.abs(np.imag(array)) > 0.0):
        raise ValidationError(f"{name} must be real")
    array = np.asarray(np.real(array), dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise DimensionError(f"{name} must be a nonempty matrix")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values")
    return array


def _real_vector(value: Any, *, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array) and np.any(np.abs(np.imag(array)) > 0.0):
        raise ValidationError(f"{name} must be real")
    array = np.asarray(np.real(array), dtype=float)
    if array.ndim != 1 or array.shape[0] != dimension:
        raise DimensionError(f"{name} must have shape ({dimension},)")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite values")
    return array


@dataclass(frozen=True, slots=True)
class FiniteProvenanceObserver:
    """Finite continuation observer for branch histories.

    ``transitions[s][a]`` is the state after a newly prepended branch ``a`` is
    added to a history whose current summary state is ``s``.  ``outputs[s]`` is
    the policy/provenance answer associated with state ``s``.
    """

    alphabet_size: int
    transitions: tuple[tuple[int, ...], ...]
    outputs: tuple[Hashable, ...]
    initial_state: int = 0

    def __post_init__(self) -> None:
        alphabet_size = int(self.alphabet_size)
        if isinstance(self.alphabet_size, bool) or alphabet_size != self.alphabet_size or alphabet_size <= 0:
            raise ValidationError("alphabet_size must be a positive integer")
        if not self.transitions:
            raise ValidationError("observer must contain at least one state")
        state_count = len(self.transitions)
        if len(self.outputs) != state_count:
            raise DimensionError("observer outputs must contain one value per state")
        initial = int(self.initial_state)
        if isinstance(self.initial_state, bool) or initial != self.initial_state or not 0 <= initial < state_count:
            raise ValidationError("initial_state is out of range")
        for state, row in enumerate(self.transitions):
            if len(row) != alphabet_size:
                raise DimensionError(f"transition row {state} has the wrong alphabet size")
            for target in row:
                index = int(target)
                if isinstance(target, bool) or index != target or not 0 <= index < state_count:
                    raise ValidationError("observer transition target is out of range")
        for output in self.outputs:
            if not isinstance(output, Hashable):
                raise ValidationError("observer outputs must be hashable")
        object.__setattr__(self, "alphabet_size", alphabet_size)
        object.__setattr__(self, "initial_state", initial)
        object.__setattr__(self, "transitions", tuple(tuple(int(v) for v in row) for row in self.transitions))
        object.__setattr__(self, "outputs", tuple(self.outputs))

    @property
    def state_count(self) -> int:
        return len(self.transitions)

    def advance(self, state: int, branch: int) -> int:
        state_index = int(state)
        branch_index = int(branch)
        if isinstance(state, bool) or state_index != state or not 0 <= state_index < self.state_count:
            raise ValidationError("observer state is out of range")
        if isinstance(branch, bool) or branch_index != branch or not 0 <= branch_index < self.alphabet_size:
            raise ValidationError("branch symbol is out of range")
        return self.transitions[state_index][branch_index]

    def advance_prefix(self, state: int, prefix: Iterable[int]) -> int:
        """Advance by a stored-order prefix that is prepended to a history."""

        result = int(state)
        word = tuple(int(symbol) for symbol in prefix)
        for symbol in reversed(word):
            result = self.advance(result, symbol)
        return result

    def state_for_history(self, history: Iterable[int]) -> int:
        """Return the observer state for a stored provenance history."""

        result = self.initial_state
        word = tuple(int(symbol) for symbol in history)
        for symbol in reversed(word):
            result = self.advance(result, symbol)
        return result

    def output_for_history(self, history: Iterable[int]) -> Hashable:
        return self.outputs[self.state_for_history(history)]

    def reachable_states(self) -> tuple[int, ...]:
        seen = {self.initial_state}
        queue: deque[int] = deque([self.initial_state])
        while queue:
            state = queue.popleft()
            for target in self.transitions[state]:
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return tuple(sorted(seen))

    def minimize(self) -> "ObserverMinimization":
        """Return the minimal reachable Moore observer for all future prefixes."""

        reachable = self.reachable_states()
        # Initial partition: present outputs must already agree.
        groups: list[list[int]] = []
        for state in reachable:
            for group in groups:
                if self.outputs[group[0]] == self.outputs[state]:
                    group.append(state)
                    break
            else:
                groups.append([state])

        while True:
            block_of = {state: block for block, group in enumerate(groups) for state in group}
            refined: list[list[int]] = []
            for group in groups:
                buckets: dict[tuple[int, ...], list[int]] = {}
                order: list[tuple[int, ...]] = []
                for state in group:
                    signature = tuple(block_of[self.transitions[state][symbol]] for symbol in range(self.alphabet_size))
                    if signature not in buckets:
                        buckets[signature] = []
                        order.append(signature)
                    buckets[signature].append(state)
                refined.extend(buckets[signature] for signature in order)
            if refined == groups:
                break
            groups = refined

        block_of = {state: block for block, group in enumerate(groups) for state in group}
        minimized_transitions: list[tuple[int, ...]] = []
        minimized_outputs: list[Hashable] = []
        for group in groups:
            representative = group[0]
            minimized_outputs.append(self.outputs[representative])
            minimized_transitions.append(
                tuple(block_of[self.transitions[representative][symbol]] for symbol in range(self.alphabet_size))
            )

        minimized = FiniteProvenanceObserver(
            alphabet_size=self.alphabet_size,
            transitions=tuple(minimized_transitions),
            outputs=tuple(minimized_outputs),
            initial_state=block_of[self.initial_state],
        )
        old_to_new = tuple(block_of.get(state, -1) for state in range(self.state_count))
        return ObserverMinimization(observer=minimized, old_to_new=old_to_new, blocks=tuple(tuple(g) for g in groups))


@dataclass(frozen=True, slots=True)
class ObserverMinimization:
    observer: FiniteProvenanceObserver
    old_to_new: tuple[int, ...]
    blocks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class PolicySupportQuery:
    value: float
    active_indices: tuple[int, ...]
    active_states: tuple[int, ...]
    active_outputs: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class PolicySupportPacket:
    """Support geometry labeled by minimized continuation-observer states.

    This packet is exact for set-valued observer outputs on active atoms.  It
    intentionally forgets raw histories and multiplicity.
    """

    atoms: np.ndarray
    ambient_dimension: int
    observer: FiniteProvenanceObserver
    states: tuple[int, ...]

    @property
    def atom_count(self) -> int:
        return int(self.atoms.shape[0])

    def support(self, direction: Any) -> float:
        x = _real_vector(direction, dimension=self.ambient_dimension, name="direction")
        return float(np.max(self.atoms @ x))

    def query(self, direction: Any, *, tolerance: float = 1e-12) -> PolicySupportQuery:
        x = _real_vector(direction, dimension=self.ambient_dimension, name="direction")
        values = self.atoms @ x
        value = float(np.max(values))
        scale = max(1.0, abs(value))
        active_indices = tuple(int(i) for i in np.flatnonzero(values >= value - tolerance * scale))

        active_states: list[int] = []
        active_outputs: list[Hashable] = []
        seen_states: set[int] = set()
        seen_outputs: set[Hashable] = set()
        for index in active_indices:
            state = self.states[index]
            if state not in seen_states:
                seen_states.add(state)
                active_states.append(state)
            output = self.observer.outputs[state]
            if output not in seen_outputs:
                seen_outputs.add(output)
                active_outputs.append(output)
        return PolicySupportQuery(
            value=value,
            active_indices=active_indices,
            active_states=tuple(active_states),
            active_outputs=tuple(active_outputs),
        )

    def linear_image(self, matrix: Any) -> "PolicySupportPacket":
        operator = _real_matrix(matrix, name="operator")
        if operator.shape[1] != self.ambient_dimension:
            raise DimensionError("operator has incompatible source dimension")
        return policy_support_packet(self.atoms @ operator.T, self.observer, self.states)

    def bellman_envelope(self, matrices: Iterable[Any]) -> "PolicySupportPacket":
        matrices_list = list(matrices)
        if not matrices_list:
            raise ValidationError("at least one branch operator is required")
        if len(matrices_list) > self.observer.alphabet_size:
            raise ValidationError("observer alphabet is too small for the branch family")

        blocks: list[np.ndarray] = []
        states: list[int] = []
        for branch_index, matrix in enumerate(matrices_list):
            operator = _real_matrix(matrix, name="operator")
            if operator.shape != (self.ambient_dimension, self.ambient_dimension):
                raise DimensionError("branch operator has the wrong dimension")
            blocks.append(self.atoms @ operator)
            states.extend(self.observer.advance(state, branch_index) for state in self.states)
        return policy_support_packet(np.vstack(blocks), self.observer, tuple(states), deduplicate=True)

    def deduplicate_exact(self) -> "PolicySupportPacket":
        return policy_support_packet(self.atoms, self.observer, self.states, deduplicate=True)


def finite_provenance_observer(
    transitions: Iterable[Iterable[int]],
    outputs: Iterable[Hashable],
    *,
    initial_state: int = 0,
) -> FiniteProvenanceObserver:
    transition_rows = tuple(tuple(int(value) for value in row) for row in transitions)
    if not transition_rows:
        raise ValidationError("observer must contain at least one transition row")
    alphabet_size = len(transition_rows[0])
    return FiniteProvenanceObserver(
        alphabet_size=alphabet_size,
        transitions=transition_rows,
        outputs=tuple(outputs),
        initial_state=initial_state,
    )


def policy_support_packet(
    atoms: Any,
    observer: FiniteProvenanceObserver,
    states: Iterable[int],
    *,
    deduplicate: bool = False,
) -> PolicySupportPacket:
    points = _real_matrix(atoms, name="atoms")
    state_tuple = tuple(int(state) for state in states)
    if len(state_tuple) != points.shape[0]:
        raise DimensionError("states must contain one observer state per atom")
    for state in state_tuple:
        if not 0 <= state < observer.state_count:
            raise ValidationError("observer state is out of range")

    if deduplicate:
        keep: list[int] = []
        seen: set[tuple[tuple[float, ...], int]] = set()
        for index, (row, state) in enumerate(zip(points, state_tuple, strict=True)):
            key = (tuple(float(value) for value in row), state)
            if key in seen:
                continue
            seen.add(key)
            keep.append(index)
        points = points[keep]
        state_tuple = tuple(state_tuple[index] for index in keep)

    frozen = np.array(points, copy=True)
    frozen.setflags(write=False)
    return PolicySupportPacket(
        atoms=frozen,
        ambient_dimension=frozen.shape[1],
        observer=observer,
        states=state_tuple,
    )


def compile_policy_packet(
    packet: PolyhedralSupportPacket,
    observer: FiniteProvenanceObserver,
    *,
    minimize_observer: bool = True,
    deduplicate: bool = True,
) -> PolicySupportPacket:
    """Compile raw provenance words into continuation-stable observer states."""

    compiled_observer = observer.minimize().observer if minimize_observer else observer
    states = tuple(compiled_observer.state_for_history(word) for word in packet.provenance)
    return policy_support_packet(packet.atoms, compiled_observer, states, deduplicate=deduplicate)


__all__ = [
    "FiniteProvenanceObserver",
    "ObserverMinimization",
    "PolicySupportPacket",
    "PolicySupportQuery",
    "compile_policy_packet",
    "finite_provenance_observer",
    "policy_support_packet",
]

@dataclass(frozen=True, slots=True)
class ObserverConvexQuery:
    """Active minimized provenance states of a state-conditioned support packet."""

    value: float
    state_values: tuple[float, ...]
    active_states: tuple[int, ...]
    active_outputs: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class ObserverConvexPacket:
    """One convex support packet for each minimized provenance state.

    If ``C_s`` is the convex hull generated by ``state_atoms[s]``, then the
    overall value and active policy states are

        h(x) = max_s h_{C_s}(x),
        active(x) = {s : h_{C_s}(x) = h(x)}.

    A Bellman branch ``b`` with primal matrix ``A_b`` updates the state bodies by

        C'_t = conv union_{delta(s,b)=t} A_b^T C_s.

    The arrays store finite generators rather than an explicitly reduced convex
    hull.  Replacing any state array by another generator set with the same
    convex hull leaves all packet semantics unchanged.
    """

    state_atoms: tuple[np.ndarray, ...]
    ambient_dimension: int
    observer: FiniteProvenanceObserver

    @property
    def total_atom_count(self) -> int:
        return sum(int(block.shape[0]) for block in self.state_atoms)

    @property
    def state_atom_counts(self) -> tuple[int, ...]:
        return tuple(int(block.shape[0]) for block in self.state_atoms)

    def state_support(self, state: int, direction: Any) -> float:
        state_index = int(state)
        if not 0 <= state_index < self.observer.state_count:
            raise ValidationError("observer state is out of range")
        x = _real_vector(direction, dimension=self.ambient_dimension, name="direction")
        block = self.state_atoms[state_index]
        if block.shape[0] == 0:
            return float("-inf")
        return float(np.max(block @ x))

    def support(self, direction: Any) -> float:
        x = _real_vector(direction, dimension=self.ambient_dimension, name="direction")
        values = [float(np.max(block @ x)) for block in self.state_atoms if block.shape[0] > 0]
        if not values:
            raise ValidationError("observer-convex packet has no atoms")
        return max(values)

    def query(self, direction: Any, *, tolerance: float = 1e-12) -> ObserverConvexQuery:
        x = _real_vector(direction, dimension=self.ambient_dimension, name="direction")
        state_values = tuple(
            float(np.max(block @ x)) if block.shape[0] > 0 else float("-inf")
            for block in self.state_atoms
        )
        finite_values = [value for value in state_values if np.isfinite(value)]
        if not finite_values:
            raise ValidationError("observer-convex packet has no atoms")
        value = max(finite_values)
        scale = max(1.0, abs(value))
        active_states = tuple(
            state
            for state, state_value in enumerate(state_values)
            if np.isfinite(state_value) and state_value >= value - tolerance * scale
        )
        outputs: list[Hashable] = []
        seen: set[Hashable] = set()
        for state in active_states:
            output = self.observer.outputs[state]
            if output not in seen:
                seen.add(output)
                outputs.append(output)
        return ObserverConvexQuery(
            value=value,
            state_values=state_values,
            active_states=active_states,
            active_outputs=tuple(outputs),
        )

    def linear_image(self, matrix: Any) -> "ObserverConvexPacket":
        operator = _real_matrix(matrix, name="operator")
        if operator.shape[1] != self.ambient_dimension:
            raise DimensionError("operator has incompatible source dimension")
        blocks = tuple(
            np.asarray(block @ operator.T, dtype=float)
            if block.shape[0] > 0
            else np.empty((0, operator.shape[0]), dtype=float)
            for block in self.state_atoms
        )
        return observer_convex_packet(blocks, self.observer)

    def bellman_envelope(self, matrices: Iterable[Any]) -> "ObserverConvexPacket":
        matrices_list = list(matrices)
        if not matrices_list:
            raise ValidationError("at least one branch operator is required")
        if len(matrices_list) > self.observer.alphabet_size:
            raise ValidationError("observer alphabet is too small for the branch family")

        targets: list[list[np.ndarray]] = [[] for _ in range(self.observer.state_count)]
        for branch_index, matrix in enumerate(matrices_list):
            operator = _real_matrix(matrix, name="operator")
            if operator.shape != (self.ambient_dimension, self.ambient_dimension):
                raise DimensionError("branch operator has the wrong dimension")
            for state, block in enumerate(self.state_atoms):
                if block.shape[0] == 0:
                    continue
                target = self.observer.advance(state, branch_index)
                targets[target].append(block @ operator)

        blocks: list[np.ndarray] = []
        for pieces in targets:
            if not pieces:
                blocks.append(np.empty((0, self.ambient_dimension), dtype=float))
                continue
            joined = np.vstack(pieces)
            # Exact duplicate removal is always safe inside one provenance state.
            unique: list[np.ndarray] = []
            seen: set[tuple[float, ...]] = set()
            for row in joined:
                key = tuple(float(value) for value in row)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(row)
            blocks.append(np.vstack(unique))
        return observer_convex_packet(tuple(blocks), self.observer)


def observer_convex_packet(
    state_atoms: Iterable[Any],
    observer: FiniteProvenanceObserver,
) -> ObserverConvexPacket:
    blocks_raw = tuple(state_atoms)
    if len(blocks_raw) != observer.state_count:
        raise DimensionError("state_atoms must contain one block per observer state")

    dimension: int | None = None
    frozen_blocks: list[np.ndarray] = []
    for index, value in enumerate(blocks_raw):
        array = np.asarray(value)
        if np.iscomplexobj(array) and np.any(np.abs(np.imag(array)) > 0.0):
            raise ValidationError(f"state atom block {index} must be real")
        array = np.asarray(np.real(array), dtype=float)
        if array.ndim != 2:
            raise DimensionError("each state atom block must be a matrix")
        if not np.all(np.isfinite(array)):
            raise ValidationError("state atom blocks must contain finite values")
        if dimension is None:
            dimension = int(array.shape[1])
            if dimension <= 0:
                raise DimensionError("state atom blocks must have positive ambient dimension")
        elif array.shape[1] != dimension:
            raise DimensionError("all state atom blocks must have the same ambient dimension")
        frozen = np.array(array, copy=True)
        frozen.setflags(write=False)
        frozen_blocks.append(frozen)

    if dimension is None or not any(block.shape[0] > 0 for block in frozen_blocks):
        raise ValidationError("observer-convex packet must contain at least one atom")
    return ObserverConvexPacket(tuple(frozen_blocks), dimension, observer)


def compile_observer_convex_packet(
    packet: PolyhedralSupportPacket,
    observer: FiniteProvenanceObserver,
    *,
    minimize_observer: bool = True,
) -> ObserverConvexPacket:
    """Group labeled support geometry by continuation-equivalent provenance state."""

    compiled_observer = observer.minimize().observer if minimize_observer else observer
    buckets: list[list[np.ndarray]] = [[] for _ in range(compiled_observer.state_count)]
    for atom, word in zip(packet.atoms, packet.provenance, strict=True):
        state = compiled_observer.state_for_history(word)
        buckets[state].append(atom)

    blocks: list[np.ndarray] = []
    for atoms in buckets:
        if not atoms:
            blocks.append(np.empty((0, packet.ambient_dimension), dtype=float))
            continue
        unique: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for atom in atoms:
            key = tuple(float(value) for value in atom)
            if key in seen:
                continue
            seen.add(key)
            unique.append(atom)
        blocks.append(np.vstack(unique))
    return observer_convex_packet(tuple(blocks), compiled_observer)


__all__.extend(
    [
        "ObserverConvexPacket",
        "ObserverConvexQuery",
        "compile_observer_convex_packet",
        "observer_convex_packet",
    ]
)
