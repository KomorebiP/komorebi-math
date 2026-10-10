"""Bounded exact decision machinery from AVOC, publication 8 October 2026.

Finite table fibers implement Sections 2--3; guarded Bellman tables and frozen
two-artifact quote bounds implement Sections 9--10. These constructions use
declared mathematical inputs. They neither authenticate artifacts nor establish
provider, sampling, persistence, or physical execution contracts.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral

from ..exceptions import ResourceLimitError, ValidationError

_UNSET = object()
_ZERO = Fraction(0)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValidationError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _fraction(value: object, name: str, bits: int) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
        raise ValidationError(f"{name} must be an integer or Fraction, not a rounded value")
    result = (
        Fraction(int(value))
        if isinstance(value, Integral)
        else Fraction(int(value.numerator), int(value.denominator))
    )
    return _bounded(result, bits)


def _bounded(value: Fraction, bits: int) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > bits:
        raise ResourceLimitError("exact rational exceeds max_bits")
    return value


def _sum(values: Iterable[Fraction], bits: int) -> Fraction:
    total = _ZERO
    for value in values:
        total = _bounded(total + value, bits)
    return total


def _owned(values: Iterable, limit: int, name: str) -> tuple:
    result = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValidationError(f"{name} must be iterable") from exc
    for value in iterator:
        if len(result) >= limit:
            raise ResourceLimitError(f"{name} exceeds its item limit")
        result.append(value)
    return tuple(result)


def _label_equal(left: Hashable, right: Hashable) -> bool:
    try:
        equal = left == right
    except Exception as exc:
        raise ValidationError("labels must have scalar Boolean equality") from exc
    if not isinstance(equal, bool):
        raise ValidationError("labels must have scalar Boolean equality")
    return equal


def _label(value: Hashable, name: str) -> Hashable:
    try:
        hash(value)
    except Exception as exc:
        raise ValidationError(f"{name} entries must be hashable") from exc
    if not _label_equal(value, value):
        raise ValidationError(f"{name} entries must have reflexive equality")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class _LabelKey:
    """Keep validated scalar equality consistent across hashing and decoding."""

    value: Hashable

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _LabelKey) and _label_equal(self.value, other.value)


def _labels(values: Iterable, limit: int, name: str) -> tuple[Hashable, ...]:
    return tuple(_label(value, name) for value in _owned(values, limit, name))


@dataclass(frozen=True, slots=True)
class TaskFibers:
    """Finite fiber decision and a total decoder on the declared alphabet.

    ``collision`` indexes the supplied, exhaustive domain table. A positive
    result says nothing about raw states absent from that declared table.
    """

    sufficient: bool
    decoder: tuple[tuple[Hashable, Hashable], ...]
    collision: tuple[int, int] | None
    reached_states: int

    def decode(self, observation: Hashable) -> Hashable:
        if not self.sufficient:
            raise ValidationError("an insufficient representation has no certified decoder")
        observation = _label(observation, "observation")
        decoder = {_LabelKey(label): target for label, target in self.decoder}
        try:
            return decoder[_LabelKey(observation)]
        except KeyError as exc:
            raise ValidationError("observation is outside the declared alphabet") from exc


def task_fibers(
    representation: Iterable[Hashable],
    target: Iterable[Hashable],
    *,
    observation_alphabet: Iterable[Hashable] | None = None,
    default_target: Hashable = _UNSET,
    max_items: int = 100_000,
) -> TaskFibers:
    """Decide the exact fiber criterion and construct its finite decoder.

    Entries are aligned raw states. An explicit ``default_target`` declares a
    target-codomain member for unreachable observations. Without it, the first
    target entry supplies that member; an empty raw domain needs an explicit
    default. The declared observation alphabet must include all reached labels.
    """
    limit = _integer(max_items, "max_items")
    observations = _labels(representation, limit, "representation")
    targets = _labels(target, limit, "target")
    if len(observations) != len(targets):
        raise ValidationError("representation and target must index the same domain")
    if default_target is _UNSET:
        if not targets:
            raise ValidationError("an empty domain needs a nonempty target codomain via default_target")
        default_target = targets[0]
    default_target = _label(default_target, "default_target")
    seen: dict[_LabelKey, tuple[int, Hashable]] = {}
    collision = None
    for index, (label, answer) in enumerate(zip(observations, targets, strict=True)):
        key = _LabelKey(label)
        if key in seen and not _label_equal(seen[key][1], answer) and collision is None:
            collision = (seen[key][0], index)
        seen.setdefault(key, (index, answer))
    alphabet = (
        tuple(key.value for key in seen)
        if observation_alphabet is None
        else _labels(observation_alphabet, limit, "observation_alphabet")
    )
    alphabet_keys = tuple(_LabelKey(label) for label in alphabet)
    if len(set(alphabet_keys)) != len(alphabet):
        raise ValidationError("observation_alphabet must have distinct entries")
    if not set(seen).issubset(alphabet_keys):
        raise ValidationError("observation_alphabet omits a reached label")
    decoder = tuple(
        (label, seen[key][1] if key in seen else default_target)
        for label, key in zip(alphabet, alphabet_keys, strict=True)
    )
    return TaskFibers(collision is None, decoder if collision is None else (), collision, len(seen))


def joint_task(
    tasks: Iterable[Iterable[Hashable]], *, domain_size: int | None = None, max_entries: int = 100_000
) -> tuple[tuple[Hashable, ...], ...]:
    """Join a finite family of aligned tasks, including the empty query family."""
    limit = _integer(max_entries, "max_entries")
    columns = []
    size = None if domain_size is None else _integer(domain_size, "domain_size")
    entries = 0
    for task in _owned(tasks, limit, "query family"):
        column = _labels(task, limit - entries, "task")
        entries += len(column)
        if size is None:
            size = len(column)
        if len(column) != size:
            raise ValidationError("tasks must index the same declared domain")
        columns.append(column)
    if size is None:
        raise ValidationError("an empty query family needs domain_size")
    if size > limit:
        raise ResourceLimitError("joint target exceeds max_entries")
    return tuple(tuple(column[index] for column in columns) for index in range(size))


def information_equivalent(
    left: Iterable[Hashable], right: Iterable[Hashable], *, max_items: int = 100_000
) -> bool:
    """Decide mutual finite-table refinement, without comparing storage or cost."""
    limit = _integer(max_items, "max_items")
    left, right = _labels(left, limit, "left"), _labels(right, limit, "right")
    return (
        task_fibers(left, right, default_target=(), max_items=limit).sufficient
        and task_fibers(right, left, default_target=(), max_items=limit).sufficient
    )


def _matrix(rows: Iterable[Iterable], limit: int, bits: int, name: str) -> tuple[tuple[Fraction, ...], ...]:
    result = []
    width = None
    entries = 0
    for row in _owned(rows, limit, name):
        values = tuple(_fraction(value, name, bits) for value in _owned(row, limit - entries, name))
        entries += len(values)
        if not values or (width is not None and len(values) != width):
            raise ValidationError(f"{name} must be rectangular with nonempty rows")
        width = len(values)
        result.append(values)
    if not result:
        raise ValidationError(f"{name} must have at least one row")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ChannelDecision:
    """Unnormalized observation/action scores and an attaining deterministic policy."""

    scores: tuple[tuple[Fraction, ...], ...]
    policy: tuple[int, ...]
    value: Fraction


@dataclass(frozen=True, slots=True)
class FiniteChannel:
    """A rational row-stochastic table, indexed world first, observation second."""

    rows: Iterable[Iterable]
    max_entries: int = field(default=100_000, repr=False, compare=False)
    max_bits: int = field(default=4096, repr=False, compare=False)

    def __post_init__(self) -> None:
        limit, bits = _integer(self.max_entries, "max_entries", 1), _integer(self.max_bits, "max_bits", 1)
        rows = _matrix(self.rows, limit, bits, "channel")
        if any(any(value < 0 for value in row) or _sum(row, bits) != 1 for row in rows):
            raise ValidationError("channel rows must be nonnegative and sum exactly to one")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "max_entries", limit)
        object.__setattr__(self, "max_bits", bits)

    @property
    def worlds(self) -> int:
        return len(self.rows)

    @property
    def observations(self) -> int:
        return len(self.rows[0])

    @classmethod
    def from_observation(
        cls,
        observation: Iterable[Hashable],
        *,
        observation_alphabet: Iterable[Hashable] | None = None,
        max_entries: int = 100_000,
        max_bits: int = 4096,
    ) -> FiniteChannel:
        """Embed a nonempty deterministic observation table as a channel."""
        limit = _integer(max_entries, "max_entries", 1)
        values = _labels(observation, limit, "observation")
        alphabet = (
            tuple(key.value for key in dict.fromkeys(_LabelKey(value) for value in values))
            if observation_alphabet is None
            else _labels(observation_alphabet, limit, "observation_alphabet")
        )
        alphabet_keys = tuple(_LabelKey(label) for label in alphabet)
        if not values or not alphabet or len(set(alphabet_keys)) != len(alphabet):
            raise ValidationError("deterministic channel needs nonempty, distinct observation alphabet")
        if not set(_LabelKey(value) for value in values).issubset(alphabet_keys):
            raise ValidationError("observation_alphabet omits a reached label")
        if len(values) * len(alphabet) > limit:
            raise ResourceLimitError("deterministic channel exceeds max_entries")
        return cls(
            tuple(tuple(int(_label_equal(value, label)) for label in alphabet) for value in values),
            limit,
            max_bits,
        )

    def garble(self, following: FiniteChannel) -> FiniteChannel:
        """Compose ``E(o|world)`` with ``K(y|o)`` using exact rational sums."""
        if not isinstance(following, FiniteChannel) or following.worlds != self.observations:
            raise ValidationError("garbling must have one row per current observation")
        limit, bits = min(self.max_entries, following.max_entries), min(self.max_bits, following.max_bits)
        if self.worlds * following.observations * self.observations > limit:
            raise ResourceLimitError("garbling contraction exceeds max_entries")
        rows = []
        for row in self.rows:
            output = []
            for column in range(following.observations):
                value = _ZERO
                for index, weight in enumerate(row):
                    term = _bounded(weight * following.rows[index][column], bits)
                    value = _bounded(value + term, bits)
                output.append(value)
            rows.append(tuple(output))
        return FiniteChannel(tuple(rows), limit, bits)

    def optimal_policy(self, prior: Iterable, utilities: Iterable[Iterable]) -> ChannelDecision:
        """Maximize deterministic decision value without dividing by evidence mass."""
        prior = tuple(
            _fraction(value, "prior", self.max_bits) for value in _owned(prior, self.worlds, "prior")
        )
        if len(prior) != self.worlds or any(value < 0 for value in prior) or _sum(prior, self.max_bits) != 1:
            raise ValidationError("prior must be a probability vector on the declared worlds")
        utility = _matrix(utilities, self.max_entries, self.max_bits, "utilities")
        if len(utility) != self.worlds:
            raise ValidationError("utility must have one row per world")
        actions = len(utility[0])
        if self.observations * actions * self.worlds > self.max_entries:
            raise ResourceLimitError("decision score contraction exceeds max_entries")
        scores = []
        for observation in range(self.observations):
            row = []
            for action in range(actions):
                value = _ZERO
                for world in range(self.worlds):
                    term = _bounded(prior[world] * self.rows[world][observation], self.max_bits)
                    term = _bounded(term * utility[world][action], self.max_bits)
                    value = _bounded(value + term, self.max_bits)
                row.append(value)
            scores.append(tuple(row))
        policy = tuple(max(range(actions), key=row.__getitem__) for row in scores)
        value = _ZERO
        for row, action in zip(scores, policy, strict=True):
            value = _bounded(value + row[action], self.max_bits)
        return ChannelDecision(tuple(scores), policy, value)

    def policy_value(self, prior: Iterable, utilities: Iterable[Iterable], policy: Iterable[int]) -> Fraction:
        """Evaluate a declared deterministic policy, rejecting missing or illegal actions."""
        decision = self.optimal_policy(prior, utilities)
        choices = tuple(
            _integer(value, "policy action") for value in _owned(policy, self.observations, "policy")
        )
        if len(choices) != self.observations or any(value >= len(decision.scores[0]) for value in choices):
            raise ValidationError("policy must choose one available action per observation")
        value = _ZERO
        for row, action in zip(decision.scores, choices, strict=True):
            value = _bounded(value + row[action], self.max_bits)
        return value


@dataclass(frozen=True, slots=True)
class GuardedAction:
    """One admitted action: reward, common observation row, and successor indices."""

    name: Hashable
    reward: Fraction | int
    probabilities: Iterable
    successors: Iterable[int]
    max_entries: int = field(default=100_000, repr=False, compare=False)
    max_bits: int = field(default=4096, repr=False, compare=False)

    def __post_init__(self) -> None:
        limit, bits = _integer(self.max_entries, "max_entries", 1), _integer(self.max_bits, "max_bits", 1)
        _label(self.name, "action name")
        reward = _fraction(self.reward, "reward", bits)
        probabilities = tuple(
            _fraction(value, "probability", bits)
            for value in _owned(self.probabilities, limit, "probabilities")
        )
        successors = tuple(
            _integer(value, "successor") for value in _owned(self.successors, limit, "successors")
        )
        if not probabilities or len(probabilities) != len(successors):
            raise ValidationError("an action needs matching nonempty probability/successor rows")
        if any(value < 0 for value in probabilities) or _sum(probabilities, bits) != 1:
            raise ValidationError("action probabilities must be nonnegative and sum exactly to one")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "successors", successors)
        object.__setattr__(self, "max_entries", limit)
        object.__setattr__(self, "max_bits", bits)


@dataclass(frozen=True, slots=True)
class BellmanResult:
    """Value at total horizon h; Q at continuation horizon h; policy at total h+1."""

    values: tuple[tuple[Fraction, ...], ...]
    q_values: tuple[tuple[tuple[Fraction, ...], ...], ...]
    policy: tuple[tuple[int, ...], ...]

    @property
    def horizon(self) -> int:
        return len(self.policy)


@dataclass(frozen=True, slots=True)
class FiniteGuardedSystem:
    """An exhaustive finite-state model with a nonempty admitted action list per state.

    Inadmissible actions are absent. All actions share an observation alphabet
    by row position. Zero-probability branches still need valid successors.
    """

    actions_by_state: Iterable[Iterable[GuardedAction]]
    max_entries: int = field(default=100_000, repr=False, compare=False)
    max_bits: int = field(default=4096, repr=False, compare=False)

    def __post_init__(self) -> None:
        limit, bits = _integer(self.max_entries, "max_entries", 1), _integer(self.max_bits, "max_bits", 1)
        rows, count, observations = [], 0, None
        for row in _owned(self.actions_by_state, limit, "states"):
            actions = _owned(row, limit - count, "actions")
            if not actions or any(not isinstance(action, GuardedAction) for action in actions):
                raise ValidationError("each state needs a nonempty list of GuardedAction objects")
            if len({_LabelKey(action.name) for action in actions}) != len(actions):
                raise ValidationError("admitted action names must be distinct within a state")
            for action in actions:
                count += 1 + len(action.probabilities)
                if count > limit:
                    raise ResourceLimitError("guarded system exceeds max_entries")
                if observations is None:
                    observations = len(action.probabilities)
                if len(action.probabilities) != observations:
                    raise ValidationError("actions must share a finite observation alphabet")
                _fraction(action.reward, "reward", bits)
                for probability in action.probabilities:
                    _fraction(probability, "probability", bits)
            rows.append(actions)
        if not rows or any(
            successor >= len(rows) for row in rows for action in row for successor in action.successors
        ):
            raise ValidationError("guarded system needs states and in-range successors on every branch")
        object.__setattr__(self, "actions_by_state", tuple(rows))
        object.__setattr__(self, "max_entries", limit)
        object.__setattr__(self, "max_bits", bits)

    @property
    def states(self) -> int:
        return len(self.actions_by_state)

    def _work(self, horizon: int, max_updates: int, max_output_entries: int) -> int:
        horizon = _integer(horizon, "horizon")
        updates = _integer(max_updates, "max_updates")
        outputs = _integer(max_output_entries, "max_output_entries")
        action_count = sum(map(len, self.actions_by_state))
        branches = sum(len(action.probabilities) for row in self.actions_by_state for action in row)
        if horizon * (branches + action_count) > updates:
            raise ResourceLimitError("Bellman work exceeds max_updates")
        if (horizon + 1) * self.states + horizon * (self.states + action_count) > outputs:
            raise ResourceLimitError("Bellman result exceeds max_output_entries")
        return horizon

    def _backup(self, action: GuardedAction, continuation: tuple[Fraction, ...]) -> Fraction:
        score = action.reward
        for probability, successor in zip(action.probabilities, action.successors, strict=True):
            term = _bounded(probability * continuation[successor], self.max_bits)
            score = _bounded(score + term, self.max_bits)
        return score

    def bellman(
        self, horizon: int, *, max_updates: int = 1_000_000, max_output_entries: int = 1_000_000
    ) -> BellmanResult:
        """Build every exact table up to a bounded horizon; ties use input order."""
        horizon = self._work(horizon, max_updates, max_output_entries)
        values, all_q, policy = [(_ZERO,) * self.states], [], []
        for _ in range(horizon):
            scores = tuple(
                tuple(self._backup(action, values[-1]) for action in row) for row in self.actions_by_state
            )
            choices = tuple(max(range(len(row)), key=row.__getitem__) for row in scores)
            values.append(tuple(row[action] for row, action in zip(scores, choices, strict=True)))
            all_q.append(scores)
            policy.append(choices)
        return BellmanResult(tuple(values), tuple(all_q), tuple(policy))

    def policy_value(
        self,
        policy: Iterable[Iterable[int]],
        *,
        max_updates: int = 1_000_000,
        max_output_entries: int = 1_000_000,
    ) -> tuple[Fraction, ...]:
        """Evaluate a legal remaining-horizon policy on every initial state.

        ``policy[h-1][s]`` chooses the action with h total steps remaining.
        This supports Bellman-optimal and frozen lower-Q policies alike.
        """
        updates = _integer(max_updates, "max_updates")
        outputs = _integer(max_output_entries, "max_output_entries")
        # Bound ownership before consuming a potentially unending policy iterator.
        max_rows = min(updates, outputs) // self.states
        rows = []
        for row in _owned(policy, max_rows, "policy horizon"):
            choices = tuple(
                _integer(value, "policy action") for value in _owned(row, self.states, "policy row")
            )
            if len(choices) != self.states or any(
                choice >= len(actions) for choice, actions in zip(choices, self.actions_by_state, strict=True)
            ):
                raise ValidationError("policy must choose an admitted action in every state")
            rows.append(choices)
        self._work(len(rows), updates, outputs)
        values = (_ZERO,) * self.states
        for choices in rows:
            values = tuple(
                self._backup(actions[choice], values)
                for choice, actions in zip(choices, self.actions_by_state, strict=True)
            )
        return values


@dataclass(frozen=True, slots=True)
class QuoteBand:
    """Natural setup, nonnegative rational gross, and rational probability bands."""

    setup: tuple[int, int]
    gross: tuple[Fraction | int, Fraction | int]
    success: tuple[Fraction | int, Fraction | int]
    max_bits: int = field(default=4096, repr=False, compare=False)

    def __post_init__(self) -> None:
        bits = _integer(self.max_bits, "max_bits", 1)
        setup = tuple(_integer(value, "setup") for value in _owned(self.setup, 2, "setup band"))
        gross = tuple(_fraction(value, "gross", bits) for value in _owned(self.gross, 2, "gross band"))
        success = tuple(
            _fraction(value, "success", bits) for value in _owned(self.success, 2, "success band")
        )
        if len(setup) != 2 or len(gross) != 2 or len(success) != 2:
            raise ValidationError("bands need exactly two endpoints")
        for value in setup:
            _fraction(value, "setup", bits)
        if setup[0] > setup[1] or not 0 <= gross[0] <= gross[1] or not 0 <= success[0] <= success[1] <= 1:
            raise ValidationError("quote endpoints must be ordered and in their declared domains")
        object.__setattr__(self, "setup", setup)
        object.__setattr__(self, "gross", gross)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "max_bits", bits)


@dataclass(frozen=True, slots=True)
class QuoteRealization:
    setup: int
    gross: Fraction | int
    success: Fraction | int


@dataclass(frozen=True, slots=True)
class FrozenQuoteBounds:
    """Section 10 uniform and local error bounds for one frozen two-kind episode."""

    epsilon: Fraction
    delta: Fraction
    magnitude: Fraction
    setup_widths: tuple[Fraction, Fraction]
    gross_widths: tuple[Fraction, Fraction]
    probability_widths: tuple[Fraction, Fraction]
    max_bits: int = field(default=4096, repr=False, compare=False)

    def _horizon(self, horizon: int) -> int:
        h = _integer(horizon, "horizon")
        _fraction(h, "horizon", self.max_bits)
        return h

    def value_radius(self, horizon: int) -> Fraction:
        h = self._horizon(horizon)
        return _bounded(self.epsilon * h + self.delta * self.magnitude * (h * (h - 1) // 2), self.max_bits)

    def local_radius(self, continuation_horizon: int, action: str) -> Fraction:
        h = self._horizon(continuation_horizon)
        if action == "stop":
            reward_error = row_error = _ZERO
        elif action in ("inspectWarm", "generateCold"):
            k = 0 if action == "inspectWarm" else 1
            reward_error, row_error = self.setup_widths[k], 2 * self.probability_widths[k]
        elif action in ("deployWarm", "deployCold"):
            reward_error, row_error = self.gross_widths[0 if action == "deployWarm" else 1], _ZERO
        else:
            raise ValidationError("unknown staged artifact action")
        return _bounded(self.value_radius(h) + reward_error + row_error * h * self.magnitude, self.max_bits)

    def gamma(self, continuation_horizon: int) -> Fraction:
        h = self._horizon(continuation_horizon)
        return _bounded(self.value_radius(h) + self.epsilon + self.delta * h * self.magnitude, self.max_bits)

    def regret_bound(self, horizon: int) -> Fraction:
        h = self._horizon(horizon)
        return _bounded(
            self.epsilon * h * (h + 1) + self.delta * self.magnitude * Fraction(h * (h - 1) * (h + 1), 3),
            self.max_bits,
        )


@dataclass(frozen=True, slots=True)
class ArtifactState:
    """Statuses are 0=untried, 1=failed, 2=ready; budget uses frozen reservations."""

    warm_status: int
    cold_status: int
    remaining_credit: int
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class FrozenPolicyResult:
    nominal: BellmanResult
    lower_q: tuple[tuple[tuple[Fraction, ...], ...], ...]
    upper_q: tuple[tuple[tuple[Fraction, ...], ...], ...]
    policy: tuple[tuple[int, ...], ...]
    regret_bound: Fraction

    def dominates(self, horizon: int, state: int, action: int) -> bool:
        """Distinct-competitor certificate; excludes the chosen action itself."""
        horizon = _integer(horizon, "horizon", 1)
        state, action = _integer(state, "state"), _integer(action, "action")
        if horizon > len(self.policy) or state >= len(self.policy[0]):
            raise ValidationError("dominance query is outside the computed tables")
        lowers, uppers = self.lower_q[horizon - 1][state], self.upper_q[horizon - 1][state]
        if action >= len(lowers):
            raise ValidationError("dominance action is not admitted")
        return all(upper <= lowers[action] for index, upper in enumerate(uppers) if index != action)


@dataclass(frozen=True, slots=True)
class FrozenQuotePlanner:
    """Finite staged workflow and quote bounds with common upper-setup guards.

    The initial state has both kinds untried. Reachability includes every
    Boolean branch, even probability-zero branches. The state graph and guards
    are shared verbatim by every compatible realization. Quote identities and
    evidence admission are outside this mathematical adapter.
    """

    warm: QuoteBand
    cold: QuoteBand
    credit: int
    max_states: int = field(default=256, repr=False, compare=False)
    max_bits: int = field(default=4096, repr=False, compare=False)
    states: tuple[ArtifactState, ...] = field(init=False)
    nominal: FiniteGuardedSystem = field(init=False)
    bounds: FrozenQuoteBounds = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.warm, QuoteBand) or not isinstance(self.cold, QuoteBand):
            raise ValidationError("warm and cold must be QuoteBand objects")
        limit, bits = _integer(self.max_states, "max_states", 1), _integer(self.max_bits, "max_bits", 1)
        credit = _integer(self.credit, "credit")
        _fraction(credit, "credit", bits)
        for band in (self.warm, self.cold):
            for value in (*band.setup, *band.gross, *band.success):
                _fraction(value, "quote endpoint", bits)
        initial = ArtifactState(0, 0, credit)
        states, indices, raw_rows = [initial], {initial: 0}, []
        cursor = 0
        while cursor < len(states):
            state = states[cursor]
            branches = self._state_actions(state)
            row = []
            for name, successors in branches:
                next_indices = []
                for successor in successors:
                    if successor not in indices:
                        if len(states) >= limit:
                            raise ResourceLimitError("quote workflow exceeds max_states")
                        indices[successor] = len(states)
                        states.append(successor)
                    next_indices.append(indices[successor])
                row.append((name, tuple(next_indices)))
            raw_rows.append(tuple(row))
            cursor += 1
        object.__setattr__(self, "states", tuple(states))
        object.__setattr__(self, "credit", credit)
        object.__setattr__(self, "max_states", limit)
        object.__setattr__(self, "max_bits", bits)
        nominal_rows = tuple(
            tuple(self._action(name, successors) for name, successors in row) for row in raw_rows
        )
        object.__setattr__(self, "nominal", FiniteGuardedSystem(nominal_rows, max_bits=bits))
        setup = tuple(Fraction(band.setup[1] - band.setup[0]) for band in (self.warm, self.cold))
        gross = tuple(_bounded(band.gross[1] - band.gross[0], bits) for band in (self.warm, self.cold))
        probability = tuple(
            _bounded(band.success[1] - band.success[0], bits) for band in (self.warm, self.cold)
        )
        epsilon = _sum(setup + gross, bits)
        delta = _bounded(2 * _sum(probability, bits), bits)
        magnitude = _sum((Fraction(band.setup[1]) + band.gross[1] for band in (self.warm, self.cold)), bits)
        object.__setattr__(
            self, "bounds", FrozenQuoteBounds(epsilon, delta, magnitude, setup, gross, probability, bits)
        )

    def _state_actions(
        self, state: ArtifactState
    ) -> tuple[tuple[str, tuple[ArtifactState, ArtifactState]], ...]:
        if state.terminal:
            return (("stop", (state, state)),)
        terminal = ArtifactState(state.warm_status, state.cold_status, state.remaining_credit, True)
        actions = [("stop", (terminal, terminal))]
        for warm in (True, False):
            band = self.warm if warm else self.cold
            status = state.warm_status if warm else state.cold_status
            if status == 0 and state.remaining_credit >= band.setup[1]:
                budget = state.remaining_credit - band.setup[1]
                failed = ArtifactState(
                    1 if warm else state.warm_status, state.cold_status if warm else 1, budget
                )
                ready = ArtifactState(
                    2 if warm else state.warm_status, state.cold_status if warm else 2, budget
                )
                actions.append(("inspectWarm" if warm else "generateCold", (failed, ready)))
            if status == 2:
                actions.append(("deployWarm" if warm else "deployCold", (terminal, terminal)))
        return tuple(actions)

    def _action(
        self, name: str, successors: tuple[int, ...], realization: QuoteRealization | None = None
    ) -> GuardedAction:
        if name == "stop":
            return GuardedAction(name, 0, (1, 0), successors, max_bits=self.max_bits)
        band = self.warm if name in ("inspectWarm", "deployWarm") else self.cold
        setup = band.setup[1] if realization is None else realization.setup
        gross = band.gross[0] if realization is None else realization.gross
        success = band.success[0] if realization is None else realization.success
        if name in ("inspectWarm", "generateCold"):
            return GuardedAction(name, -setup, (1 - success, success), successors, max_bits=self.max_bits)
        return GuardedAction(name, gross, (1, 0), successors, max_bits=self.max_bits)

    def _compatible(self, band: QuoteBand, realization: QuoteRealization) -> QuoteRealization:
        if not isinstance(realization, QuoteRealization):
            raise ValidationError("realization must be a QuoteRealization")
        setup = _integer(realization.setup, "realized setup")
        gross = _fraction(realization.gross, "realized gross", self.max_bits)
        success = _fraction(realization.success, "realized success", self.max_bits)
        if (
            not band.setup[0] <= setup <= band.setup[1]
            or not band.gross[0] <= gross <= band.gross[1]
            or not (band.success[0] <= success <= band.success[1])
        ):
            raise ValidationError("realization is outside the frozen quote bands")
        return QuoteRealization(setup, gross, success)

    def realized_system(self, warm: QuoteRealization, cold: QuoteRealization) -> FiniteGuardedSystem:
        """Change rewards/probabilities inside bands, keeping guards and graph fixed."""
        warm, cold = self._compatible(self.warm, warm), self._compatible(self.cold, cold)
        rows = tuple(
            tuple(
                self._action(
                    action.name,
                    action.successors,
                    warm if action.name in ("inspectWarm", "deployWarm") else cold,
                )
                for action in row
            )
            for row in self.nominal.actions_by_state
        )
        return FiniteGuardedSystem(rows, max_bits=self.max_bits)

    def robust_policy(
        self, horizon: int, *, max_updates: int = 1_000_000, max_output_entries: int = 1_000_000
    ) -> FrozenPolicyResult:
        """Maximize certified lower-Q at every remaining horizon; not exact minimax."""
        horizon = self.nominal._work(horizon, max_updates, max_output_entries)
        # Both enclosures are owned in addition to the nominal tables/policy.
        entries = horizon * 2 * sum(map(len, self.nominal.actions_by_state))
        base_entries = (horizon + 1) * len(self.states) + horizon * (
            len(self.states) + sum(map(len, self.nominal.actions_by_state))
        )
        if base_entries + entries + horizon * len(self.states) > max_output_entries:
            raise ResourceLimitError("frozen policy result exceeds max_output_entries")
        nominal = self.nominal.bellman(
            horizon, max_updates=max_updates, max_output_entries=max_output_entries
        )
        lower, upper, policies = [], [], []
        for h, table in enumerate(nominal.q_values):
            lo_rows, hi_rows, choices = [], [], []
            for scores, actions in zip(table, self.nominal.actions_by_state, strict=True):
                radii = tuple(self.bounds.local_radius(h, action.name) for action in actions)
                lows = tuple(
                    _bounded(score - radius, self.max_bits)
                    for score, radius in zip(scores, radii, strict=True)
                )
                highs = tuple(
                    _bounded(score + radius, self.max_bits)
                    for score, radius in zip(scores, radii, strict=True)
                )
                lo_rows.append(lows)
                hi_rows.append(highs)
                choices.append(max(range(len(lows)), key=lows.__getitem__))
            lower.append(tuple(lo_rows))
            upper.append(tuple(hi_rows))
            policies.append(tuple(choices))
        return FrozenPolicyResult(
            nominal, tuple(lower), tuple(upper), tuple(policies), self.bounds.regret_bound(horizon)
        )
