"""Independent finite controls for decision, inverse and arithmetic machinery.

Oracles here enumerate policies, synthesize costs by node count, evaluate
matrix words directly and form the defining q-product. They do not import
repository tests or turn the bounded ranges into universal paper approval.
"""

from fractions import Fraction as Q
from itertools import product
from math import comb, isqrt, prod

from ..certificates import Exactness
from .runner import CheckOutcome, SkipCheck, StressCheck, StressContext


def _decisions(context: StressContext) -> CheckOutcome:
    from ..decision.avoc import FiniteChannel, FiniteGuardedSystem, GuardedAction, task_fibers

    cases = []
    for _ in range(context.parameters["samples"]):
        numerators = [context.rng.randrange(5) for _ in range(3)]
        rows = tuple((Q(n, 4), 1 - Q(n, 4)) for n in numerators)
        utilities = tuple(tuple(Q(context.rng.randrange(-5, 6), 3) for _ in range(2)) for _ in range(3))
        prior = (Q(1, 6), Q(1, 3), Q(1, 2))
        channel = FiniteChannel(rows)
        decision = channel.optimal_policy(prior, utilities)
        policy_values = {
            policy: sum(prior[s] * rows[s][o] * utilities[s][policy[o]] for s in range(3) for o in range(2))
            for policy in product(range(2), repeat=2)
        }
        assert decision.value == max(policy_values.values()), "complete channel policy enumeration"
        assert policy_values[decision.policy] == decision.value, "attained channel value"
        actions = tuple(
            tuple(
                GuardedAction(a, Q(context.rng.randrange(-3, 4), 2), (Q(1, 3), Q(2, 3)), (0, 1))
                for a in range(1 + s)
            )
            for s in range(2)
        )
        system = FiniteGuardedSystem(actions)
        result = system.bellman(3)
        row_choices = tuple(product(range(1), range(2)))
        maxima = [None, None]
        for policy in product(row_choices, repeat=3):

            def value(state, remaining, actions=actions, policy=policy):
                if remaining == 0:
                    return Q(0)
                action = actions[state][policy[remaining - 1][state]]
                return action.reward + sum(
                    p * value(s, remaining - 1)
                    for p, s in zip(action.probabilities, action.successors, strict=True)
                )

            for state in range(2):
                candidate = value(state, 3)
                maxima[state] = candidate if maxima[state] is None else max(maxima[state], candidate)
        assert result.values[3] == tuple(maxima), "complete guarded policy enumeration"
        assert system.policy_value(result.policy) == tuple(maxima), "legal policy attainment"
        cases.append(
            {
                "channel": rows,
                "utilities": utilities,
                "channel_value": decision.value,
                "rewards": [[a.reward for a in row] for row in actions],
                "bellman_value": tuple(maxima),
            }
        )
    good = task_fibers((0, 0, 1, 1), (2, 2, 3, 3))
    bad = task_fibers((0, 0, 1, 1), (2, 3, 3, 3))
    assert good.sufficient and not bad.sufficient and bad.collision is not None, "fiber collision obstruction"
    return CheckOutcome(
        True, {"cases": cases, "fiber_collision_rejected": True, "guarded_policies_per_case": 8}
    )


def _tree_value(node, values):
    if node.operation == "literal":
        return bool(node.atom), 1
    if node.operation == "variable":
        return bool(values[node.atom - 1]), 1
    children = [_tree_value(child, values) for child in node.children]
    cost = 1 + sum(item[1] for item in children)
    if node.operation == "not":
        return not children[0][0], cost
    left, right = (item[0] for item in children)
    value = {"and": left and right, "or": left or right, "xor": left != right}[node.operation]
    return value, cost


def _boolean(context: StressContext) -> CheckOutcome:
    from ..algorithms.boolean_synthesis import audit_boolean_closure_bank, build_boolean_closure_bank

    bank = build_boolean_closure_bank()
    assert audit_boolean_closure_bank(bank).valid, "constructor-closed bank acceptance"
    # Independent node-count induction, rather than the production depth recurrence.
    costs = {signature: 1 for signature in (0, 255, 240, 204, 170)}
    by_cost = {1: set(costs)}
    for count in range(2, 11):
        attained = {signature ^ 255 for signature in by_cost.get(count - 1, ())}
        for left_cost in range(1, count - 1):
            right_cost = count - left_cost - 1
            for left in by_cost.get(left_cost, ()):
                for right in by_cost.get(right_cost, ()):
                    attained.update((left & right, left | right, left ^ right))
        by_cost[count] = attained - costs.keys()
        costs.update({signature: count for signature in by_cost[count]})
    assert len(costs) == 256, "complete node-count oracle"
    for entry in bank.entries:
        outputs = [_tree_value(entry.expression, inputs) for inputs in product((0, 1), repeat=3)]
        signature = sum(int(value) << i for i, (value, _) in enumerate(outputs))
        assert signature == entry.signature and all(count == entry.node_count for _, count in outputs), (
            "direct AST witness"
        )
        assert entry.node_count == costs[signature], "independent minimum node count"
    return CheckOutcome(
        True,
        {
            "signatures": 256,
            "valuation_witnesses": 2048,
            "maximum_minimum_nodes": max(costs.values()),
            "node_cost_histogram": {str(n): len(s) for n, s in by_cost.items()},
        },
    )


def _tikhonov(context: StressContext) -> CheckOutcome:
    try:
        import sympy  # noqa: F401
    except ImportError as exc:
        raise SkipCheck("Tikhonov controls require the SymPy exact extra") from exc
    from ..signals.tikhonov import (
        bracket_discrepancy,
        certify_squared_discrepancy,
        normal_residual_correction,
        tikhonov_system,
    )

    cases = []
    for _ in range(context.parameters["samples"]):
        diagonal = [Q(context.rng.randrange(-3, 4), 2) for _ in range(3)]
        data = [Q(context.rng.randrange(-4, 5), 3) for _ in range(3)]
        parameter = Q(context.rng.randrange(1, 6), 5)
        operator = [[diagonal[i] if i == j else 0 for j in range(3)] for i in range(3)]
        system = tikhonov_system(operator, parameter)
        solution = system.solve(data)
        expected = [a * v / (a * a + parameter) for a, v in zip(diagonal, data, strict=True)]
        actual = [Q(int(v.p), int(v.q)) for v in solution.source]
        assert actual == expected and solution.certificate.valid, "independent diagonal normal solve"
        candidate = [x + Q(i + 1, 7) for i, x in enumerate(expected)]
        increment = system.objective_increment(data, candidate)
        objective = sum(
            (a * x - v) ** 2 + parameter * x * x for a, x, v in zip(diagonal, candidate, data, strict=True)
        )
        assert increment.candidate_objective == objective and increment.certificate.valid, (
            "independent primal objective"
        )
        correction = normal_residual_correction(system, data, candidate)
        residual_sq = sum(
            ((a * a + parameter) * x - a * v) ** 2 for a, x, v in zip(diagonal, candidate, data, strict=True)
        )
        assert correction.normal_squared == residual_sq and correction.certificate.valid, (
            "actual residual recomputation"
        )
        cases.append({"diagonal": diagonal, "data": data, "parameter": parameter, "solution": expected})
    bracket = bracket_discrepancy([[1, 0], [0, 0]], [3, 4], Q(18), width=Q(1, 1024))

    def curve(parameter):
        return 16 + 9 * parameter**2 / (1 + parameter) ** 2

    assert bracket.converged and bracket.certificate.valid and bracket.upper - bracket.lower <= Q(1, 1024)
    assert curve(bracket.lower) <= 18 <= curve(bracket.upper), "independent discrepancy endpoint signs"
    system = tikhonov_system([[1]], 1)
    arguments = {"eta": Q(1, 4), "tau": 2, "rho": 0, "xi": 0, "kappa": 0, "measured": Q(1, 2), "omega": 0}
    valid = certify_squared_discrepancy(system, [1], [Q(1, 2)], **arguments)
    forged = certify_squared_discrepancy(system, [1], [Q(3, 4)], **arguments)
    assert valid.certificate.valid and not forged.certificate.valid, "false normal residual budget rejected"
    return CheckOutcome(
        True,
        {
            "cases": cases,
            "discrepancy_interval": [bracket.lower, bracket.upper],
            "attainable_floor_squared": 16,
            "forged_residual_rejected": True,
        },
    )


def _word_state(matrices, seed, word, modulus):
    state = seed
    for digit in word:
        matrix = matrices[digit]
        state = tuple(sum(a * x for a, x in zip(row, state, strict=True)) % modulus for row in matrix)
    return state


def _modules(context: StressContext) -> CheckOutcome:
    from ..generating.residual_modules import (
        RingLinearRealization,
        canonical_residual_module,
        finite_word_comparison,
    )

    cases = []
    source = RingLinearRealization(2, 2, (((1, 0), (2, 1)),), (1, 0), (2, 0))
    nonfree = canonical_residual_module(source)
    assert (nonfree.ambient_cardinality, len(nonfree.reachable_states), nonfree.carrier_cardinality) == (
        16,
        8,
        2,
    )
    assert nonfree.additive_p_exponents() == (1,), "nonfree quotient retained"
    words = [word for length in range(5) for word in product(range(2), repeat=length)]
    for _ in range(context.parameters["samples"]):
        matrices = tuple(
            tuple(tuple(context.rng.randrange(4) for _ in range(2)) for _ in range(2)) for _ in range(2)
        )
        seed, output = (1, 0), tuple(context.rng.randrange(4) for _ in range(2))
        realization = RingLinearRealization(2, 2, matrices, seed, output)
        quotient = canonical_residual_module(realization)
        # Full supplied orbit, then independent additive subgroup generation.
        orbit, pending = {seed}, [seed]
        while pending:
            for digit in range(2):
                state = _word_state(matrices, pending[0], (digit,), 4)
                if state not in orbit:
                    orbit.add(state)
                    pending.append(state)
            pending.pop(0)
        reachable = {(0, 0)}
        for generator in sorted(orbit):
            reachable = {
                tuple((x + c * y) % 4 for x, y in zip(state, generator, strict=True))
                for state in reachable
                for c in range(4)
            }
        assert set(quotient.reachable_states) == reachable, "independent additive reachable carrier"

        def signature(state, output=output, matrices=matrices):
            return tuple(
                sum(a * x for a, x in zip(output, _word_state(matrices, state, word, 4), strict=True)) % 4
                for word in words
            )

        signatures = {state: signature(state) for state in reachable}
        assert quotient.carrier_cardinality == len(set(signatures.values())), "all-short-word quotient oracle"
        assert set(quotient.unobservable_states) == {
            state for state, sig in signatures.items() if not any(sig)
        }
        assert quotient.verify_laws().valid and finite_word_comparison(realization, quotient).equivalent
        for word in words:
            expected = (
                sum(a * x for a, x in zip(output, _word_state(matrices, seed, word, 4), strict=True)) % 4
            )
            assert quotient.evaluate(word) == expected, "direct noncommuting matrix word"
        cases.append(
            {
                "matrices": matrices,
                "output": output,
                "reachable": len(reachable),
                "quotient": quotient.carrier_cardinality,
            }
        )
    return CheckOutcome(
        True, {"cases": cases, "words_per_case": len(words), "nonfree_carrier_cardinality": 2}
    )


def _lehmer(context: StressContext) -> CheckOutcome:
    from ..arithmetic.lehmer_shells import (
        enumerate_five_core_shell,
        primitive_residuals_from_tau,
        tau_from_primitive_residuals,
    )

    # Direct multiplication of (1-q^k)^24, deliberately independent of production transforms.
    coefficients = [1] + [0] * 31
    for k in range(1, 32):
        factor = [(j * k, (-1) ** j * comb(24, j)) for j in range(min(24, 31 // k) + 1)]
        coefficients = [
            sum(value * coefficients[n - degree] for degree, value in factor if degree <= n)
            for n in range(32)
        ]
    tau = (0, *coefficients)
    primitive = primitive_residuals_from_tau(tau)
    assert tau_from_primitive_residuals(primitive) == tau, "finite square-supported Mobius inversion"
    indices = sorted(
        {1, 2, 23, 32, *(context.rng.randrange(1, 33) for _ in range(context.parameters["samples"]))}
    )
    cases = []
    for n in indices:
        shell = enumerate_five_core_shell(n)
        bound = isqrt(10 * n)
        ranges = [range(-bound + (i + bound) % 5, bound + 1, 5) for i in range(4)]
        expected = {}
        for prefix in product(*ranges):
            coordinates = (*prefix, -sum(prefix))
            if coordinates[4] % 5 == 4 and sum(x * x for x in coordinates) == 10 * n:
                numerator = prod(coordinates[i] - coordinates[j] for i in range(5) for j in range(i + 1, 5))
                assert numerator % 288 == 0
                expected[coordinates] = numerator // 288
        assert {packet.coordinates: packet.weight for packet in shell.packets} == expected, (
            "complete independent shell and weights"
        )
        assert shell.weighted_sum == tau[n] and shell.primitive_residual == primitive[n], (
            "q-product and content transforms"
        )
        cases.append(
            {
                "index": n,
                "packet_count": len(expected),
                "signed_total": tau[n],
                "primitive_total": primitive[n],
            }
        )
    return CheckOutcome(True, {"cases": cases, "q_product_maximum_index": 32, "cancellation_preserved": True})


def continuation_checks(samples: int = 8) -> tuple[StressCheck, ...]:
    """Five finite exact checks; dependencies and ranges remain explicit."""
    if type(samples) is not int or not 1 <= samples <= 128:
        raise ValueError("samples must be an integer between 1 and 128")
    specifications = (
        (
            "paper.avoc.finite-decisions",
            _decisions,
            {
                "samples": samples,
                "channel_shape": [3, 2],
                "channel_policies": 4,
                "guarded_states": 2,
                "horizon": 3,
                "policies_per_case": 8,
            },
        ),
        (
            "paper.avoc.boolean-minimum",
            _boolean,
            {"signatures": 256, "valuations": 8, "independent_node_counts": [1, 10]},
        ),
        (
            "paper.signals.tikhonov-certificates",
            _tikhonov,
            {
                "samples": samples,
                "dimension": 3,
                "diagonal_numerators": [-3, 3],
                "discrepancy_width": "1/1024",
                "required_extra": "SymPy",
            },
        ),
        (
            "paper.generating.residual-modules",
            _modules,
            {
                "samples": samples,
                "prime": 2,
                "exponent": 2,
                "dimension": 2,
                "alphabet_size": 2,
                "word_lengths": [0, 4],
                "ambient_cardinality": 16,
            },
        ),
        (
            "paper.arithmetic.lehmer-shells",
            _lehmer,
            {
                "sampled_indices": samples,
                "indices": [1, 32],
                "q_product_cutoff": 32,
                "independent_shell_coordinate_loops": 4,
            },
        ),
    )
    return tuple(
        StressCheck(
            identifier,
            "paper_continuation",
            function,
            Exactness.EXACT,
            bounds,
            parameters={"samples": samples},
            assumptions=("supplied finite exact models; no infinite-limit or conjecture completion",),
        )
        for identifier, function, bounds in specifications
    )


__all__ = ["continuation_checks"]
