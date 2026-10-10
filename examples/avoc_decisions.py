"""Exact finite AVOC worked examples: fibers, information value, and quotes."""

from fractions import Fraction as F
from itertools import product

from komorebi_math.decision.avoc import (
    FiniteChannel,
    FrozenQuotePlanner,
    QuoteBand,
    QuoteRealization,
    task_fibers,
)


def main() -> None:
    raw = tuple(product(range(2), repeat=3))
    task = tuple(x[0] & x[1] for x in raw)
    parity = tuple(x[0] ^ x[1] for x in raw)
    collision = task_fibers(parity, task)
    assert not collision.sufficient
    assert task_fibers(task, task).sufficient
    print("parity does not retain conjunction; collision indices:", collision.collision)

    reveal = FiniteChannel(((1, 0), (0, 1)))
    forget = reveal.garble(FiniteChannel(((1,), (1,))))
    prior, utility = (F(1, 2), F(1, 2)), ((1, 0), (0, 1))
    assert reveal.optimal_policy(prior, utility).value == 1
    assert forget.optimal_policy(prior, utility).value == F(1, 2)
    print("revelation/constant decision values: 1 and 1/2")

    exact = FrozenQuotePlanner(
        QuoteBand((133, 133), (15000, 15000), (1, 1)), QuoteBand((196, 196), (15000, 15000), (1, 1)), 196
    )
    exact_result = exact.robust_policy(2)
    assert exact_result.nominal.q_values[1][0] == (0, 14867, 14804)
    assert exact_result.policy[1][0] == 1 and exact_result.regret_bound == 0
    print("two-action exact Q (stop/warm/cold):", exact_result.nominal.q_values[1][0])

    uncertain = FrozenQuotePlanner(
        QuoteBand((0, 0), (200, 200), (F(3, 5), 1)), QuoteBand((0, 0), (60, 60), (1, 1)), 0
    )
    result = uncertain.robust_policy(2)
    assert result.lower_q[1][0] == (0, -88, 60) and result.policy[1][0] == 2
    true = uncertain.realized_system(QuoteRealization(0, 200, 1), QuoteRealization(0, 60, 1))
    regret = true.bellman(2).values[2][0] - true.policy_value(result.policy)[0]
    assert regret == 140 and regret <= result.regret_bound
    print("two-action certified lower-Q (stop/warm/cold):", result.lower_q[1][0])
    print("at warm success one, expected policy regret/bound:", regret, result.regret_bound)


if __name__ == "__main__":
    main()
