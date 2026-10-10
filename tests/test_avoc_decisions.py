"""Independent exact finite controls for AVOC Sections 2, 3, 9 and 10."""

from fractions import Fraction as F
from itertools import product, repeat

import pytest

from komorebi_math.decision.avoc import (
    FiniteChannel,
    FiniteGuardedSystem,
    FrozenQuotePlanner,
    GuardedAction,
    QuoteBand,
    QuoteRealization,
    information_equivalent,
    joint_task,
    task_fibers,
)
from komorebi_math.exceptions import ResourceLimitError, ValidationError


def test_every_four_state_binary_fiber_table_against_exhaustive_decoders():
    checks = 0
    for representation, target in product(product(range(2), repeat=4), repeat=2):
        analysis = task_fibers(representation, target, observation_alphabet=(0, 1, 2), default_target=0)
        admissible = [
            decoder
            for decoder in product(range(2), repeat=3)
            if all(decoder[z] == answer for z, answer in zip(representation, target, strict=True))
        ]
        assert analysis.sufficient == bool(admissible)
        assert analysis.reached_states == len(set(representation))
        if analysis.sufficient:
            assert tuple(analysis.decode(z) for z in representation) == target
            assert analysis.decode(2) == 0
        else:
            first, second = analysis.collision
            assert representation[first] == representation[second] and target[first] != target[second]
            with pytest.raises(ValidationError, match="insufficient"):
                analysis.decode(0)
        checks += 1
    assert checks == 256


def test_joint_tasks_refinement_preorder_and_empty_domain_contract():
    tables = tuple(product(range(2), repeat=3))
    for representation, first, second in product(tables, repeat=3):
        joined = joint_task((first, second))
        assert task_fibers(representation, joined).sufficient == (
            task_fibers(representation, first).sufficient and task_fibers(representation, second).sufficient
        )
        if task_fibers(representation, first).sufficient and task_fibers(first, second).sufficient:
            assert task_fibers(representation, second).sufficient
        assert information_equivalent(representation, first) == (
            task_fibers(representation, first).sufficient and task_fibers(first, representation).sufficient
        )
        assert task_fibers(representation, representation).sufficient
        if task_fibers(representation, first).sufficient:
            assert len(set(first)) <= len(set(representation))
    assert joint_task((), domain_size=3) == ((), (), ())
    empty = task_fibers((), (), observation_alphabet=("unreachable",), default_target="target")
    assert empty.sufficient and empty.decode("unreachable") == "target"
    assert information_equivalent((), ())
    with pytest.raises(ValidationError, match="nonempty target"):
        task_fibers((), ())


def test_nonreflexive_or_ambiguous_label_equality_is_rejected():
    nan = float("nan")
    for operation in (
        lambda: task_fibers((nan,), (1,)),
        lambda: task_fibers((1,), (nan,)),
        lambda: task_fibers((1,), (1,), observation_alphabet=(1, nan)),
        lambda: task_fibers((), (), default_target=nan),
        lambda: joint_task(((nan,),)),
        lambda: information_equivalent((nan,), (nan,)),
        lambda: FiniteChannel.from_observation((nan,)),
        lambda: GuardedAction(nan, 0, (1,), (0,)),
    ):
        with pytest.raises(ValidationError, match="reflexive"):
            operation()

    class AmbiguousEquality:
        def __hash__(self):
            return 0

        def __eq__(self, other):
            return []

    with pytest.raises(ValidationError, match="scalar Boolean"):
        task_fibers((AmbiguousEquality(),), (1,))

    class AmbiguousOtherEquality:
        def __hash__(self):
            return 0

        def __eq__(self, other):
            return True if self is other else []

    first, second = AmbiguousOtherEquality(), AmbiguousOtherEquality()
    with pytest.raises(ValidationError, match="scalar Boolean"):
        task_fibers((first, second), (1, 1))
    with pytest.raises(ValidationError, match="scalar Boolean"):
        task_fibers((0, 0), (first, second))
    sufficient = task_fibers(("a", "b", "a"), ((1, 2), (3, 4), (1, 2)))
    assert sufficient.sufficient
    assert tuple(sufficient.decode(label) for label in ("a", "b", "a")) == ((1, 2), (3, 4), (1, 2))
    with pytest.raises(ValidationError, match="reflexive"):
        sufficient.decode(nan)


def brute_channel_value(channel, prior, utility, policy):
    return sum(
        (
            prior[world] * channel.rows[world][observation] * utility[world][policy[observation]]
            for world in range(len(prior))
            for observation in range(channel.observations)
        ),
        F(0),
    )


def test_rational_channels_garbling_and_values_against_all_policies():
    utility = ((F(-2), F(3), F(1, 3)), (F(5, 2), F(-1), F(2)))
    cases = 0
    for left, right, prior_mass in product((F(0), F(1, 4), F(1, 2), F(3, 4), F(1)), repeat=3):
        channel = FiniteChannel(((left, 1 - left, 0), (right, 1 - right, 0)))
        prior = (prior_mass, 1 - prior_mass)
        decision = channel.optimal_policy(prior, utility)
        brute = max(
            brute_channel_value(channel, prior, utility, policy) for policy in product(range(3), repeat=3)
        )
        assert decision.value == brute
        assert channel.policy_value(prior, utility, decision.policy) == brute
        assert decision.scores[2] == (0, 0, 0) and decision.policy[2] == 0
        for first, second in product((F(0), F(1, 2), F(1)), repeat=2):
            garbling = FiniteChannel(((first, 1 - first), (second, 1 - second), (F(1, 3), F(2, 3))))
            composed = channel.garble(garbling)
            for world, observation in product(range(2), repeat=2):
                assert composed.rows[world][observation] == sum(
                    channel.rows[world][o] * garbling.rows[o][observation] for o in range(3)
                )
            coarse = composed.optimal_policy(prior, utility)
            assert coarse.value <= brute
            assert coarse.value == max(
                brute_channel_value(composed, prior, utility, policy)
                for policy in product(range(3), repeat=2)
            )
            cases += 1
    assert cases == 1125


def test_information_price_and_impossible_observation_are_distinct():
    fine = FiniteChannel.from_observation(("left", "right"), observation_alphabet=("left", "right", "absent"))
    coarse = fine.garble(FiniteChannel(((1,), (1,), (1,))))
    prior, utility = (F(1, 2), F(1, 2)), ((1, 0), (0, 1))
    fine_value, coarse_value = (
        fine.optimal_policy(prior, utility).value,
        coarse.optimal_policy(prior, utility).value,
    )
    assert (fine_value, coarse_value) == (1, F(1, 2))
    assert fine_value - F(1, 8) > coarse_value
    assert fine_value - F(3, 4) < coarse_value


def all_adaptive_tree_values(system, state, horizon):
    """Enumerate complete trees, separately choosing a child after each observation."""
    if horizon == 0:
        return [F(0)]
    values = []
    for action in system.actions_by_state[state]:
        child_values = [
            all_adaptive_tree_values(system, successor, horizon - 1) for successor in action.successors
        ]
        for children in product(*child_values):
            values.append(
                action.reward
                + sum(
                    (weight * child for weight, child in zip(action.probabilities, children, strict=True)),
                    F(0),
                )
            )
    return values


def small_guarded_system():
    return FiniteGuardedSystem(
        (
            (
                GuardedAction("stay", -1, (1, 0), (0, 1)),
                GuardedAction("gamble", F(2, 3), (F(1, 3), F(2, 3)), (0, 1)),
            ),
            (GuardedAction("reset", -2, (1, 0), (0, 1)),),
        )
    )


def test_guarded_bellman_against_complete_adaptive_trees_and_all_markov_policies():
    system = small_guarded_system()
    result = system.bellman(4)
    assert result.values[0] == (0, 0) and result.horizon == 4
    for horizon in range(5):
        for state in range(2):
            brute = all_adaptive_tree_values(system, state, horizon)
            assert result.values[horizon][state] == max(brute)
        if horizon:
            assert system.policy_value(result.policy[:horizon]) == result.values[horizon]
        if horizon < 4:
            for state, row in enumerate(system.actions_by_state):
                for action_index, action in enumerate(row):
                    independent_q = action.reward + sum(
                        (
                            p * max(all_adaptive_tree_values(system, target, horizon))
                            for p, target in zip(action.probabilities, action.successors, strict=True)
                        ),
                        F(0),
                    )
                    assert result.q_values[horizon][state][action_index] == independent_q
    for sequence in product((0, 1), repeat=4):
        value = system.policy_value(tuple((action, 0) for action in sequence))
        assert all(observed <= optimal for observed, optimal in zip(value, result.values[4], strict=True))


def test_state_dependent_guards_and_tie_policy():
    system = small_guarded_system()
    with pytest.raises(ValidationError, match="admitted"):
        system.policy_value(((0, 1),))
    tied = FiniteGuardedSystem(
        ((GuardedAction("first", 0, (1,), (0,)), GuardedAction("second", 0, (1,), (0,))),)
    )
    assert tied.bellman(3).policy == ((0,),) * 3
    assert tied.policy_value(()) == (0,)
    with pytest.raises(ValidationError, match="in-range"):
        FiniteGuardedSystem(((GuardedAction("bad_zero_branch", 0, (1, 0), (0, 1)),),))


def test_exact_artifact_quote_fixture_and_horizon_indexing():
    planner = FrozenQuotePlanner(
        QuoteBand((133, 133), (15000, 15000), (1, 1)), QuoteBand((196, 196), (15000, 15000), (1, 1)), 196
    )
    result = planner.robust_policy(2)
    names = tuple(action.name for action in planner.nominal.actions_by_state[0])
    assert names == ("stop", "inspectWarm", "generateCold")
    assert result.nominal.q_values[0][0] == (0, -133, -196)
    assert result.nominal.q_values[1][0] == (0, 14867, 14804)
    assert result.policy[0][0] == 0 and result.policy[1][0] == 1
    assert result.lower_q == result.upper_q == result.nominal.q_values
    assert result.regret_bound == 0 and result.dominates(2, 0, 1)
    warm_action = planner.nominal.actions_by_state[0][1]
    failed, ready = warm_action.successors
    assert planner.states[failed].warm_status == 1 and planner.states[ready].warm_status == 2
    assert planner.states[failed].remaining_credit == planner.states[ready].remaining_credit == 63
    assert all(action.name != "deployWarm" for action in planner.nominal.actions_by_state[failed])


def test_source_conservative_quote_fixture_is_not_exact_minimax():
    planner = FrozenQuotePlanner(
        QuoteBand((0, 0), (200, 200), (F(3, 5), 1)), QuoteBand((0, 0), (60, 60), (1, 1)), 0
    )
    result = planner.robust_policy(2)
    assert (planner.bounds.epsilon, planner.bounds.delta, planner.bounds.magnitude) == (0, F(4, 5), 260)
    assert result.nominal.q_values[1][0] == (0, 120, 60)
    assert result.lower_q[1][0] == (0, -88, 60)
    assert result.upper_q[1][0] == (0, 328, 60)
    assert result.policy[1][0] == 2 and not result.dominates(2, 0, 2)
    assert planner.bounds.local_radius(1, "inspectWarm") == 208
    assert planner.bounds.local_radius(1, "generateCold") == 0
    true_scores = []
    for probability in (F(3, 5), F(4, 5), F(1)):
        true = planner.realized_system(QuoteRealization(0, 200, probability), QuoteRealization(0, 60, 1))
        true_scores.append(true.bellman(2).q_values[1][0])
    assert min(row[1] for row in true_scores) == 120 > 60


def test_frozen_conservative_guards_are_shared_by_smaller_true_setup():
    frozen = FrozenQuotePlanner(QuoteBand((1, 3), (200, 200), (1, 1)), QuoteBand((3, 3), (0, 0), (0, 0)), 2)
    true = frozen.realized_system(QuoteRealization(1, 200, 1), QuoteRealization(3, 0, 0))
    assert tuple(action.name for action in true.actions_by_state[0]) == ("stop",)
    assert true.bellman(2).values[2][0] == 0
    extra_guard = FrozenQuotePlanner(
        QuoteBand((1, 1), (200, 200), (1, 1)), QuoteBand((3, 3), (0, 0), (0, 0)), 2
    )
    assert extra_guard.nominal.bellman(2).values[2][0] == 199


def test_quote_enclosures_and_whole_policy_regret_across_exact_grid():
    warm = QuoteBand((1, 2), (3, 6), (F(1, 3), F(2, 3)))
    cold = QuoteBand((0, 1), (1, 4), (F(1, 4), F(3, 4)))
    planner = FrozenQuotePlanner(warm, cold, 3)
    result = planner.robust_policy(3)
    warm_grid = [
        QuoteRealization(c, g, p) for c, g, p in product((1, 2), (3, 6), (F(1, 3), F(1, 2), F(2, 3)))
    ]
    cold_grid = [
        QuoteRealization(c, g, p) for c, g, p in product((0, 1), (1, 4), (F(1, 4), F(1, 2), F(3, 4)))
    ]
    for warm_true, cold_true in product(warm_grid, cold_grid):
        true = planner.realized_system(warm_true, cold_true)
        exact = true.bellman(3)
        assert tuple(tuple(action.successors for action in row) for row in true.actions_by_state) == tuple(
            tuple(action.successors for action in row) for row in planner.nominal.actions_by_state
        )
        for horizon in range(4):
            for state, (nominal, actual) in enumerate(
                zip(result.nominal.values[horizon], exact.values[horizon], strict=True)
            ):
                assert abs(actual - nominal) <= planner.bounds.value_radius(horizon)
                if horizon:
                    for action, actual_q in enumerate(exact.q_values[horizon - 1][state]):
                        assert (
                            result.lower_q[horizon - 1][state][action]
                            <= actual_q
                            <= result.upper_q[horizon - 1][state][action]
                        )
                        if result.dominates(horizon, state, action):
                            assert actual_q == actual
            policy_value = true.policy_value(result.policy[:horizon])
            assert all(
                0 <= optimal - chosen <= planner.bounds.regret_bound(horizon)
                for optimal, chosen in zip(exact.values[horizon], policy_value, strict=True)
            )
    assert len(warm_grid) * len(cold_grid) == 144
    for horizon in range(7):
        assert planner.bounds.regret_bound(horizon) == sum(
            (2 * planner.bounds.gamma(h) for h in range(horizon)), F(0)
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda: task_fibers((0,), (0, 1)),
        lambda: task_fibers((0,), (0,), observation_alphabet=(1,)),
        lambda: task_fibers((0,), (0,), observation_alphabet=(0, 0)),
        lambda: task_fibers(([0],), (0,)),
        lambda: joint_task(((0,), (0, 1))),
        lambda: joint_task(()),
        lambda: FiniteChannel(((F(1, 2), F(1, 3)),)),
        lambda: FiniteChannel(((2, -1),)),
        lambda: FiniteChannel(((0.5, 0.5),)),
        lambda: FiniteChannel(((True,),)),
        lambda: FiniteChannel(((1,),)).optimal_policy((F(1, 2),), ((1,),)),
        lambda: FiniteChannel(((1,),)).optimal_policy((1,), ()),
        lambda: FiniteChannel(((1,),)).garble(FiniteChannel(((1,), (1,)))),
        lambda: FiniteChannel(((1,),)).policy_value((1,), ((1,),), (1,)),
        lambda: GuardedAction("bad", 0, (F(1, 2),), (0,)),
        lambda: GuardedAction("bad", 0, (1,), (-1,)),
        lambda: FiniteGuardedSystem(((),)),
        lambda: FiniteGuardedSystem(
            ((GuardedAction("a", 0, (1,), (0,)), GuardedAction("a", 0, (1,), (0,))),)
        ),
        lambda: QuoteBand((3, 1), (0, 1), (0, 1)),
        lambda: QuoteBand((0, 0), (-1, 1), (0, 1)),
        lambda: QuoteBand((0, 0), (0, 1), (0, F(3, 2))),
        lambda: QuoteBand((F(1, 2), 1), (0, 1), (0, 1)),
    ],
)
def test_malformed_exact_inputs_are_rejected(operation):
    with pytest.raises(ValidationError):
        operation()


def test_bounded_ownership_preflight_and_exact_arithmetic_growth():
    with pytest.raises(ResourceLimitError):
        task_fibers(repeat(0), repeat(0), max_items=3)
    with pytest.raises(ResourceLimitError):
        joint_task(repeat((0,)), max_entries=3)
    with pytest.raises(ResourceLimitError):
        FiniteChannel.from_observation(range(100), max_entries=100)
    with pytest.raises(ResourceLimitError):
        FiniteChannel(((F(1, 22), F(1, 26), F(10, 22), F(12, 26)),), max_bits=5)
    channel = FiniteChannel(((1, 0), (0, 1)), max_entries=4)
    with pytest.raises(ResourceLimitError):
        channel.garble(channel)
    with pytest.raises(ResourceLimitError):
        channel.optimal_policy((F(1, 2), F(1, 2)), ((1, 0), (0, 1)))
    with pytest.raises(ResourceLimitError):
        small_guarded_system().bellman(10**100, max_updates=100)
    with pytest.raises(ResourceLimitError):
        small_guarded_system().bellman(3, max_output_entries=1)
    with pytest.raises(ResourceLimitError):
        small_guarded_system().policy_value(repeat((0, 0)), max_updates=3)
    growth = FiniteGuardedSystem(((GuardedAction("grow", 2**20, (1,), (0,)),),), max_bits=21)
    with pytest.raises(ResourceLimitError):
        growth.bellman(2)
    with pytest.raises(ResourceLimitError):
        FrozenQuotePlanner(
            QuoteBand((0, 0), (1, 1), (0, 1)), QuoteBand((0, 0), (1, 1), (0, 1)), 0, max_states=1
        )
    planner = FrozenQuotePlanner(QuoteBand((0, 0), (1, 1), (0, 1)), QuoteBand((0, 0), (1, 1), (0, 1)), 0)
    with pytest.raises(ResourceLimitError):
        planner.robust_policy(2, max_output_entries=100)
    with pytest.raises(ValidationError, match="outside"):
        planner.realized_system(QuoteRealization(1, 1, 1), QuoteRealization(0, 1, 1))
    with pytest.raises(ValidationError):
        planner.bounds.local_radius(1, "unrecognized")
    with pytest.raises(ValidationError):
        planner.robust_policy(0).dominates(1, 0, 0)
    with pytest.raises(ResourceLimitError):
        planner.bounds.regret_bound(1 << 4096)


def test_fixed_width_fraction_parts_become_python_integers_before_arithmetic():
    import numpy as np

    large_integer = 2**62
    large = F(np.int64(large_integer))
    minimum = F(np.int64(-(2**63)))
    assert isinstance(large.numerator, np.integer)
    action = GuardedAction("grow", large, (1,), (0,))
    assert type(action.reward.numerator) is int
    exact = FiniteGuardedSystem(((action,),), max_bits=65).bellman(4)
    assert tuple(row[0] for row in exact.values) == tuple(F(h * large_integer) for h in range(5))
    negative = FiniteGuardedSystem(((GuardedAction("negative", minimum, (1,), (0,)),),))
    assert negative.bellman(3).values[3][0] == -3 * 2**63
    with pytest.raises(ResourceLimitError, match="max_bits"):
        GuardedAction("too_large", large, (1,), (0,), max_bits=62)
    capped = FiniteGuardedSystem(((action,),), max_bits=63)
    with pytest.raises(ResourceLimitError, match="max_bits"):
        capped.bellman(2)

    third = F(np.int64(1), np.int64(3))
    two_thirds = F(np.int64(2), np.int64(3))
    channel = FiniteChannel(((third, two_thirds), (two_thirds, third)))
    utility = ((large, minimum), (large, minimum))
    decision = channel.optimal_policy((third, two_thirds), utility)
    assert decision.value == large_integer and decision.policy == (0, 0)
    assert all(
        type(value.numerator) is int and type(value.denominator) is int
        for row in (*channel.rows, *decision.scores)
        for value in row
    )

    probability = F(np.int64(large_integer - 1), np.int64(large_integer))
    tiny = F(np.int64(1), np.int64(large_integer))
    process = FiniteGuardedSystem(
        (
            (GuardedAction("wait", large, (probability, tiny), (0, 1)),),
            (GuardedAction("stop", 0, (1, 0), (1, 1)),),
        )
    )
    reference_probability = F(large_integer - 1, large_integer)
    expected = large_integer * sum((reference_probability**h for h in range(4)), F(0))
    assert process.bellman(4).values[4][0] == expected

    half = F(np.int64(1), np.int64(2))
    planner = FrozenQuotePlanner(
        QuoteBand((np.int64(0), np.int64(0)), (F(np.int64(large_integer - 1)), large), (half, 1)),
        QuoteBand((0, 0), (large, large), (1, 1)),
        0,
    )
    assert (planner.bounds.epsilon, planner.bounds.delta, planner.bounds.magnitude) == (
        1,
        1,
        2 * large_integer,
    )
    assert planner.bounds.local_radius(1, "inspectWarm") == 1 + 2 * large_integer
    true = planner.realized_system(QuoteRealization(0, large, half), QuoteRealization(0, large, 1))
    assert true.bellman(2).q_values[1][0] == (0, F(large_integer, 2), large_integer)
