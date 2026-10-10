"""Independent bounded controls for the five continuation mathematical modules.

Run with PYTHONPATH=src and the exact dependency installed. This does not prove
unbounded conjectures, authenticate unknown data, or qualify optional backends.
The Boolean bank has a separate complete finite algebra checker; the independent
cost-indexed enumeration here checks all 256 states through node cost ten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from fractions import Fraction as F
from importlib.metadata import version
from itertools import combinations, product, repeat
from math import isqrt, prod
from pathlib import Path
from random import Random

import sympy as s

from komorebi_math.algorithms.boolean_synthesis import (
    audit_boolean_closure_bank,
    build_boolean_closure_bank,
)
from komorebi_math.arithmetic.lehmer_shells import (
    FiveCoreShell,
    auxiliary_shell_polynomial,
    enumerate_five_core_shell,
    primitive_residuals_from_tau,
    tau_from_primitive_residuals,
    tau_from_q_product,
)
from komorebi_math.decision.avoc import (
    FiniteChannel,
    FiniteGuardedSystem,
    FrozenQuotePlanner,
    GuardedAction,
    QuoteBand,
    QuoteRealization,
    task_fibers,
)
from komorebi_math.exceptions import ValidationError
from komorebi_math.generating.residual_modules import (
    RingLinearRealization,
    canonical_residual_module,
    finite_word_comparison,
)
from komorebi_math.signals.tikhonov import (
    bracket_discrepancy,
    candidate_noise_acceptance,
    candidate_norm_domination,
    candidate_relative_feasibility,
    certify_squared_discrepancy,
    normal_residual_correction,
    select_discrepancy,
    tikhonov_system,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 68309
MODULES = (
    "algorithms/boolean_synthesis.py",
    "generating/residual_modules.py",
    "arithmetic/lehmer_shells.py",
    "signals/tikhonov.py",
    "decision/avoc.py",
)


def hashes():
    paths = [ROOT / "src/komorebi_math" / module for module in MODULES]
    paths.append(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def rejects(call):
    try:
        call()
    except ValidationError:
        return
    raise AssertionError("expected validation refusal")


def ring_controls():
    rng = Random(SEED)
    regimes = ((2, 2, 2), (3, 1, 2), (2, 3, 1))
    cases = 0
    for prime, exponent, dimension in regimes:
        modulus = prime**exponent
        for _ in range(6):
            matrices = [
                [[rng.randrange(modulus) for _ in range(dimension)] for _ in range(dimension)]
                for _ in range(2)
            ]
            seed = tuple(rng.randrange(modulus) for _ in range(dimension))
            output = tuple(rng.randrange(modulus) for _ in range(dimension))
            model = RingLinearRealization(prime, exponent, matrices, seed, output)
            quotient = canonical_residual_module(model)

            def add(x, y, modulus=modulus):
                return tuple((a + b) % modulus for a, b in zip(x, y, strict=True))

            def scale(a, x, modulus=modulus):
                return tuple(a * b % modulus for b in x)

            def transition(a, x, matrices=matrices, dimension=dimension, modulus=modulus):
                return tuple(
                    sum(matrices[a][i][j] * x[j] for j in range(dimension)) % modulus
                    for i in range(dimension)
                )

            reachable = {(0,) * dimension, seed}
            while True:
                old = set(reachable)
                reachable |= {add(x, y) for x in old for y in old}
                reachable |= {scale(a, x) for a in range(modulus) for x in old}
                reachable |= {transition(a, x) for a in range(2) for x in old}
                if reachable == old:
                    break
            # Row-functional closure is independent of producer partition refinement.
            rows = {output}
            while True:
                old = set(rows)
                rows |= {
                    tuple(
                        sum(row[i] * matrices[a][i][j] for i in range(dimension)) % modulus
                        for j in range(dimension)
                    )
                    for row in old
                    for a in range(2)
                }
                if rows == old:
                    break

            closed_rows = tuple(sorted(rows))

            def key(x, dimension=dimension, modulus=modulus, rows=closed_rows):
                return tuple(sum(row[i] * x[i] for i in range(dimension)) % modulus for row in rows)

            expected = {frozenset(x for x in reachable if key(x) == key(y)) for y in reachable}
            assert frozenset(quotient.reachable_states) == frozenset(reachable)
            assert {frozenset(group) for group in quotient.classes} == expected
            assert quotient.verify_laws().valid
            assert finite_word_comparison(model, quotient).equivalent
            for word in product(range(2), repeat=5):
                assert model.evaluate(word) == quotient.evaluate(word)
            cases += 1
    return {
        "cases": cases,
        "prime_exponent_dimension": regimes,
        "alphabet_size": 2,
        "word_length": 5,
        "oracle": "additive closure and independently closed dual output functionals",
    }


def lehmer_controls():
    maximum = 30
    tau = [0, 1] + [0] * (maximum - 1)

    def sigma(n):
        return sum(d for d in range(1, n + 1) if n % d == 0)

    # Logarithmic derivative recurrence, not producer q-product multiplication.
    for n in range(2, maximum + 1):
        numerator = -24 * sum(sigma(k) * tau[n - k] for k in range(1, n))
        assert numerator % (n - 1) == 0
        tau[n] = numerator // (n - 1)
    assert tuple(tau) == tau_from_q_product(maximum)
    residuals = primitive_residuals_from_tau(tau)
    for n in range(1, maximum + 1):
        shell = enumerate_five_core_shell(n)
        assert shell.weighted_sum == tau[n]
        assert auxiliary_shell_polynomial(shell).evaluate(5) == 288 * tau[n]
        assert shell.primitive_residual == residuals[n]
        if n <= 12:
            bound = isqrt(10 * n)
            grids = tuple(tuple(x for x in range(-bound, bound + 1) if x % 5 == i) for i in range(5))
            independent = {v for v in product(*grids) if sum(v) == 0 and sum(x * x for x in v) == 10 * n}
            assert independent == {packet.coordinates for packet in shell.packets}
            assert (
                sum(prod(v[i] - v[j] for i in range(5) for j in range(i + 1, 5)) // 288 for v in independent)
                == tau[n]
            )
            for packet in shell.packets:
                assert packet.primitive().scale_content(packet.content) == packet
                assert packet.involution().involution() == packet
                assert packet.involution().weight == packet.weight
    assert tau_from_primitive_residuals(residuals) == tuple(tau)
    shell = enumerate_five_core_shell(1)
    rejects(lambda: FiveCoreShell(1, repeat(shell.packets[0]), "quadratic", 1))
    return {
        "indices": [1, 30],
        "independent_five_coordinate_indices": [1, 12],
        "oracle": "logarithmic derivative tau recurrence and full five-coordinate product",
        "infinite_constructor_input_rejected": True,
    }


def boolean_controls():
    inputs = tuple(product((False, True), repeat=3))
    levels = {1: {(False,) * 8, (True,) * 8, *{tuple(x[j] for x in inputs) for j in range(3)}}}
    minimum = {truth: 1 for truth in levels[1]}
    for size in range(2, 11):
        candidates = {tuple(not bit for bit in truth) for truth in levels.get(size - 1, set())}
        for left_size in range(1, size - 1):
            for x in levels.get(left_size, set()):
                for y in levels.get(size - 1 - left_size, set()):
                    candidates.update(
                        (
                            tuple(a and b for a, b in zip(x, y, strict=True)),
                            tuple(a or b for a, b in zip(x, y, strict=True)),
                            tuple(a != b for a, b in zip(x, y, strict=True)),
                        )
                    )
        levels[size] = candidates
        for truth in candidates:
            minimum.setdefault(truth, size)
    assert len(minimum) == 256
    bank = build_boolean_closure_bank()
    assert audit_boolean_closure_bank(bank).valid
    for entry in bank.entries:
        truth = tuple(bool((entry.signature >> i) & 1) for i in range(8))
        assert entry.node_count == minimum[truth]
    return {
        "truth_signatures": 256,
        "maximum_attained_node_cost": max(minimum.values()),
        "oracle": "node-cost indexed DP on ordinary eight-Boolean tuples",
    }


def avoc_controls():
    rows = (
        (
            GuardedAction("a", -1, (F(1, 3), F(2, 3)), (0, 1)),
            GuardedAction("b", 2, (F(1, 2), F(1, 2)), (1, 0)),
        ),
        (GuardedAction("a", 3, (F(1, 4), F(3, 4)), (0, 1)), GuardedAction("b", 0, (1, 0), (1, 0))),
    )
    system = FiniteGuardedSystem(rows)
    values = []
    for choices in product((0, 1), repeat=6):
        policy = tuple(choices[2 * h : 2 * h + 2] for h in range(3))
        value = (F(0), F(0))
        for row_choices in policy:
            value = tuple(
                row[a].reward
                + sum(
                    p * value[next_state]
                    for p, next_state in zip(row[a].probabilities, row[a].successors, strict=True)
                )
                for a, row in zip(row_choices, rows, strict=True)
            )
        assert system.policy_value(policy) == value
        values.append(value)
    bellman = system.bellman(3)
    assert tuple(max(value[state] for value in values) for state in range(2)) == bellman.values[3]
    assert system.policy_value(bellman.policy) == bellman.values[3]
    planner = FrozenQuotePlanner(
        QuoteBand((1, 2), (3, 7), (F(1, 4), F(3, 4))), QuoteBand((0, 1), (2, 5), (F(1, 3), F(2, 3))), 3
    )
    robust = planner.robust_policy(4)
    endpoints = 0
    for ws, wg, wp, cs, cg, cp in product(
        (1, 2), (3, 7), (F(1, 4), F(3, 4)), (0, 1), (2, 5), (F(1, 3), F(2, 3))
    ):
        true = planner.realized_system(QuoteRealization(ws, wg, wp), QuoteRealization(cs, cg, cp))
        exact = true.bellman(4)
        for horizon in range(5):
            assert all(
                abs(a - b) <= planner.bounds.value_radius(horizon)
                for a, b in zip(exact.values[horizon], robust.nominal.values[horizon], strict=True)
            )
            selected = true.policy_value(robust.policy[:horizon])
            assert all(
                0 <= a - b <= planner.bounds.regret_bound(horizon)
                for a, b in zip(exact.values[horizon], selected, strict=True)
            )
        for horizon, (actual_rows, nominal_rows) in enumerate(
            zip(exact.q_values, robust.nominal.q_values, strict=True)
        ):
            for state, (actual, nominal) in enumerate(zip(actual_rows, nominal_rows, strict=True)):
                for action, (a, b, admitted) in enumerate(
                    zip(actual, nominal, true.actions_by_state[state], strict=True)
                ):
                    assert abs(a - b) <= planner.bounds.local_radius(horizon, admitted.name)
                    if robust.dominates(horizon + 1, state, action):
                        assert a == max(actual)
        endpoints += 1
    channel = FiniteChannel(((F(2, 3), F(1, 3)), (F(1, 4), F(3, 4))))
    garbled = channel.garble(FiniteChannel(((1, 0), (F(1, 2), F(1, 2)))))
    prior, utility = (F(2, 5), F(3, 5)), ((3, -1), (0, 4))
    for model in (channel, garbled):
        oracle = max(
            sum(prior[w] * model.rows[w][o] * utility[w][a] for o, a in enumerate(policy) for w in range(2))
            for policy in product((0, 1), repeat=2)
        )
        assert model.optimal_policy(prior, utility).value == oracle
    assert channel.optimal_policy(prior, utility).value >= garbled.optimal_policy(prior, utility).value
    rejects(lambda: task_fibers([float("nan")], [1]))
    rejects(lambda: task_fibers([1], [float("nan")]))
    rejects(lambda: task_fibers([1], [1], default_target=float("nan")))
    fibers = task_fibers(("a", "b", "a"), (1, 2, 1))
    assert fibers.sufficient and fibers.decode("a") == 1 and fibers.decode("b") == 2
    return {
        "exhaustive_markov_policies": 64,
        "states": 2,
        "horizon": 3,
        "quote_endpoint_realizations": endpoints,
        "quote_graph_states": len(planner.states),
        "quote_horizons": [0, 4],
        "nan_label_defect_rejected": True,
        "oracle": "complete policies, frozen quote endpoint Q/value/regret checks and complete channel policies",
    }


def principal_minor_psd(matrix):
    assert matrix == matrix.T and matrix.rows <= 2
    return all(
        matrix.extract(indices, indices).det() >= 0
        for size in range(1, matrix.rows + 1)
        for indices in combinations(range(matrix.rows), size)
    )


def tikhonov_controls():
    cases = 0
    for diagonal in ((0, 0), (1, 0), (2, 3), (-1, 4)):
        for parameter in (F(1, 4), F(2, 3), F(3)):
            operator = s.diag(*diagonal)
            system = tikhonov_system(operator, parameter)
            data, source = s.ImmutableMatrix([F(2, 3), F(-3, 5)]), s.ImmutableMatrix([2, -1])
            independent = s.ImmutableMatrix(
                [
                    s.Rational(a) * data[i] / (s.Rational(a * a) + s.Rational(parameter))
                    for i, a in enumerate(diagonal)
                ]
            )
            solution = system.solve(data)
            assert solution.source == independent and solution.certificate.valid
            result = system.bias_noise(source, data - operator * source)
            assert result.error == independent - source and result.certificate.valid
            correction = normal_residual_correction(system, data, [1, 2])
            assert (
                correction.correction == s.ImmutableMatrix([1, 2]) - independent
                and correction.certificate.valid
            )
            assert system.objective_increment(data, [1, 2]).certificate.valid
            cases += 1
    operator = s.ImmutableMatrix([[1, 1], [2, 2]])
    system = tikhonov_system(operator, F(1, 2))
    solution = system.solve([3, 5])
    assert solution.source[0] == solution.source[1] and solution.certificate.valid
    assert system.bias_noise([2, -1], [F(1, 7), F(-2, 3)]).certificate.valid
    bracket = bracket_discrepancy([[1]], [1], F(1, 2), width=F(1, 100000))
    assert s.Rational(bracket.lower) <= 1 + s.sqrt(2) <= s.Rational(bracket.upper)
    assert bracket.certificate.valid and bracket.exact_parameter is None and bracket.converged
    assert bracket_discrepancy([[1]], [1], F(1, 4)).exact_parameter == 1
    assert bracket_discrepancy([[1], [0]], [1, F(1, 3)], F(1, 3)).certificate.valid
    selection = select_discrepancy([[1]], [1], F(1, 10), 2)
    assert selection.certificate.valid and F(1, 100) <= selection.fitting_squared <= F(1, 25)
    system = tikhonov_system([[1]], F(1, 4))
    source = system.solve([1]).source
    assert candidate_noise_acceptance(system, [1], source, eta=F(1, 10), delta=0, tau=2).accepted
    assert not candidate_noise_acceptance(system, [1], [F(3, 4)], eta=F(1, 10), delta=0, tau=3).accepted
    assert candidate_norm_domination(system, [1], [1], eta=F(1, 10), delta=0).certificate.valid
    for measured, accepted in ((F(1, 5), True), (F(0), False), (F(1, 2), False)):
        certificate = certify_squared_discrepancy(
            system, [1], source, eta=F(1, 10), tau=2, rho=0, xi=0, kappa=0, measured=measured, omega=0
        )
        assert certificate.certificate.valid == accepted
    assert not candidate_noise_acceptance(
        tikhonov_system([[0]], 1), [1], [0], eta=F(1, 10), delta=1, tau=2
    ).accepted
    assert not candidate_relative_feasibility([1], F(1, 10), 1, 2).valid
    whole_map_cases = 0
    for operator in (
        s.diag(0, 0),
        s.diag(1, 0),
        s.diag(2, 3),
        s.ImmutableMatrix([[1, 1], [0, 1]]),
        s.ImmutableMatrix([[1, 1], [2, 2]]),
    ):
        system = tikhonov_system(operator, F(1, 2))
        assert system.whole_map_bounds().valid
        regularized = system.regularized
        fitted = operator * regularized
        assert principal_minor_psd(
            s.eye(operator.cols) / (4 * system.parameter) - regularized * regularized.T
        )
        assert principal_minor_psd(s.eye(operator.rows) - fitted.T * fitted)
        assert principal_minor_psd(
            s.eye(operator.rows) - (fitted - s.eye(operator.rows)).T * (fitted - s.eye(operator.rows))
        )
        whole_map_cases += 1
    for operator, other, epsilon in (
        (s.diag(1, 2), s.diag(F(3, 2), F(5, 3)), F(1, 2)),
        (s.ImmutableMatrix([[1, 1], [0, 1]]), s.diag(1, 2), F(2)),
    ):
        system, following = tikhonov_system(operator, F(1, 2)), tikhonov_system(other, F(1, 2))
        bound = system.operator_sensitivity(
            following, operator_bound=2, other_operator_bound=2, difference_bound=epsilon
        )
        assert bound.valid
        allowance = bound.diagnostics["regularized_difference_norm_upper"]
        difference = system.regularized - following.regularized
        assert principal_minor_psd(allowance**2 * s.eye(operator.rows) - difference.T * difference)
        assert not system.operator_sensitivity(
            following, operator_bound=2, other_operator_bound=2, difference_bound=F(1, 100)
        ).valid
    system = tikhonov_system(s.diag(2, 3), F(1, 2))
    witness = s.ImmutableMatrix([1, 2])
    source = system.operator.T * system.operator * witness
    assert system.source_condition_bias(source, witness).valid
    assert not system.source_condition_bias(source + s.ImmutableMatrix([1, 0]), witness).valid
    system = tikhonov_system(s.diag(1, 0), F(1, 2))
    assert system.conditioned_bias_noise([2, 3], [F(-1, 3), F(2, 5)], 1).valid
    assert not system.conditioned_bias_noise([2, 3], [F(-1, 3), F(2, 5)], 2).valid
    return {
        "independent_diagonal_cases": cases,
        "nonorthogonal_rank_deficient_case": True,
        "whole_map_principal_minor_cases": whole_map_cases,
        "sensitivity_cases": 2,
        "rational_and_irrational_root_controls": True,
        "forged_measurement_operator_source_coercivity_controls": True,
        "oracle": "closed-form scalar inverses, exact radical roots and all principal minors in dimensions at most two",
    }


def main():
    if not __debug__:
        raise RuntimeError("optimized Python cannot run this assert-based qualification")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = hashes()
    checks = {}
    for name, check in (
        ("residual_modules", ring_controls),
        ("lehmer_shells", lehmer_controls),
        ("boolean_synthesis", boolean_controls),
        ("avoc_decisions", avoc_controls),
        ("tikhonov", tikhonov_controls),
    ):
        checks[name] = {"status": "passed", **check()}
    after = hashes()
    assert before == after, "review input sources changed during execution"
    report = {
        "schema": "komorebi-independent-continuation-review/v1",
        "status": "passed",
        "seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "sympy": version("sympy"),
            "platform": platform.platform(),
        },
        "reviewed_input_sha256_before": before,
        "reviewed_input_sha256_after": after,
        "inputs_unchanged": before == after,
        "checks": checks,
        "defect_corrections_rechecked": [
            "AVOC-reflexive-label-total-decoder",
            "Lehmer-finite-shell-input-ownership",
        ],
        "source_contract_notes": [
            "SR-47 small-data zero acceptance does not establish the separate true-noise or range-feasibility premise"
        ],
        "nonclaims": [
            "universal nonvanishing of tau",
            "correctness of arbitrary infinite-dimensional instances",
            "truth of supplied data-noise allowances",
            "exact robust minimax planning",
            "optional backend or cross-platform qualification",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks), "output": str(args.output)}))


if __name__ == "__main__":
    main()
