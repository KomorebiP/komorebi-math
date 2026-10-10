# Exact finite AVOC decisions

`komorebi_math.decision.avoc` implements finite mathematical constructions
from *AVOC: Artifact value, guarded operations, and calibrated planning*,
traditional publication edition, 8 October 2026, based on AVOC Formal v0.41.0.
The module uses Python integers and `fractions.Fraction`; it has no optional
scientific dependency. Arithmetic inputs reject floats and Boolean values.
Accepted integers and `Fraction` numerator/denominator components are normalized
to Python integers before resource checks or arithmetic, including fractions
constructed from fixed-width integral scalars.
Labels for represented states, targets, and action names must be hashable and
have reflexive equality returning a scalar Python `bool`. Nonreflexive labels
such as NaN and ambiguous array-valued equality are rejected. Hashing and
decoder lookup use the same validated equality relation.

| Construction | API | Source contract |
|---|---|---|
| Task sufficiency and collision witnesses | `task_fibers`, `TaskFibers.decode` | Definition 2.1, Theorem 2.2, Proposition 2.3; complete aligned finite domain tables |
| Joint target and information equivalence | `joint_task`, `information_equivalent` | Section 2.1; finite query families and mutual decoder existence |
| Observation channels and decision value | `FiniteChannel.garble`, `optimal_policy`, `policy_value`, `from_observation` | Definitions 3.1 and 3.3, Theorems 3.2 and 3.4; nonnegative normalized rational prior, finite nonempty action menu |
| Guarded planning | `GuardedAction`, `FiniteGuardedSystem.bellman`, `policy_value` | Definition 9.1 and Theorem 9.2 restricted to an exhaustive finite state table |
| Frozen two-artifact quote planning | `QuoteBand`, `FrozenQuotePlanner`, `QuoteRealization` | Sections 9.1 and 10; upper-setup reservations and common action guards |
| Value/Q enclosures and policy regret | `FrozenQuoteBounds`, `FrozenPolicyResult.dominates` | Lemma 10.1, Theorem 10.2, Proposition 10.3, Theorem 10.5 |

An aligned table contains one entry for every raw state in the declared finite
domain. The fiber decision is exhaustive over that table. If the table is only
a sample from a larger domain, a positive result establishes sufficiency only
on the sample. A failed decision returns two raw-state indices with the same
representation and different targets.

The decoder covers the declared observation alphabet, including unreachable
labels. The first target entry supplies a default target-codomain member unless
`default_target` is explicit. An empty raw domain requires an explicit default;
an empty query family requires an explicit `domain_size`. Neither information
refinement nor equivalence compares byte length, acquisition cost, or execution
time.

## A worked exact example

The source's exact quote fixture reserves 133 for warm inspection and 196 for
cold generation. Both succeed with probability one and deploy for gross value
15000. Initial credit is 196.

```python
from komorebi_math.decision.avoc import FrozenQuotePlanner, QuoteBand

planner = FrozenQuotePlanner(
    QuoteBand((133, 133), (15000, 15000), (1, 1)),
    QuoteBand((196, 196), (15000, 15000), (1, 1)),
    credit=196,
)
result = planner.robust_policy(2)
assert result.nominal.q_values[0][0] == (0, -133, -196)
assert result.nominal.q_values[1][0] == (0, 14867, 14804)
assert result.policy[0][0] == 0  # stop with one total action
assert result.policy[1][0] == 1  # inspect warm with two total actions
assert result.regret_bound == 0
assert result.dominates(2, 0, 1)
```

`values[h][s]` uses **h total actions**. `q_values[h][s][a]` uses **h
continuation actions**, so acquisition followed by deployment is in
`q_values[1]`, and its choice is in `policy[1]`. State index zero is the initial
state of this planner. Admitted action order here is stop, warm inspection,
cold generation. The fee is paid after either acquisition outcome. Deployment
is available only after the corresponding status becomes ready.

The executable example `examples/avoc_decisions.py` also checks an insufficient
Boolean observation, decision value after garbling, and the conservative
uncertainty fixture below.

## Frozen quote bounds and their limits

Quote setup endpoints are natural numbers, gross endpoints are nonnegative
rationals, and success endpoints are rationals in `[0, 1]`. The nominal model
uses upper setup, lower gross, and lower success. Every compatible realization
uses the same successor indices and admitted action lists; only rewards and
report probabilities change. A lower true fee does not release the upper
reservation or introduce an additional action. The initial state has both
artifact kinds untried. A terminal state admits only stop.

Let setup, gross, and probability widths for the two kinds be `wc`, `wg`, and
`wp`. The implementation uses the source's uniform constants

\[
\epsilon=\sum_k(wc_k+wg_k),\qquad
\delta=2\sum_k wp_k,\qquad
M=\sum_k(c_k^++g_k^+).
\]

The value radius at total horizon h is

\[
D_h=h\epsilon+\frac{h(h-1)}2\delta M.
\]

The local Q radius is `D_h + reward_error + row_error*h*M`, with setup width
and twice the success width for acquisition, gross width and zero row error
for deployment, and zero primitive errors for stop. The uniform local radius
is `Gamma_h = D_h + epsilon + delta*h*M`. The expected whole-policy regret
bound at total horizon H is

\[
\epsilon H(H+1)+\frac{\delta M}{3}H(H-1)(H+1).
\]

`robust_policy` maximizes the certified lower Q at every state and remaining
horizon. It does not claim exact minimax optimization. For the source's abstract
fixture with zero setups, warm gross 200 and success band `[3/5, 1]`, and cold
gross 60 with success one, the two-action nominal scores are `(0, 120, 60)`.
The warm radius is 208, producing lower scores `(0, -88, 60)`. The lower-Q
policy chooses cold, although warm's exact compatible Q range is `[120, 200]`.
This is a conservative envelope, not an artifact-backed economic saving.

`dominates(H, s, a)` checks the chosen lower score against **distinct**
competitors' upper scores. It avoids the vacuous self-comparison that would
force the chosen radius to zero. This result is conditional on the declared
bands and common guards. The regret compares expected policy values; it does
not bound every individual realized outcome sequence.

## Validation and execution bounds

`tests/test_avoc_decisions.py` compares all 256 binary representation/target
pairs on a four-state domain against exhaustive decoder search, checks joint
query sufficiency and the refinement preorder, and checks 1125 rational
channel/garbling cases against every deterministic observation policy.
Complete legal adaptive observation trees are enumerated independently through
horizon four for a guarded two-state fixture. Quote enclosures, dominance, and
whole-policy regret are tested on all states through horizon three for 144
compatible exact realizations, including interior probabilities.

Negative controls reject malformed or rounded arithmetic inputs, missing
actions, invalid zero-probability successors, incompatible realizations, and
allocation or rational-bit overflow. A separate control demonstrates how using
the smaller true fee to enlarge the comparator's action guards changes the
optimization problem. This finite validation does not provide a universal
proof or rerun the source's Lean project.

Construction owns its supplied iterables under explicit entry/state limits.
Bellman calls check work and output limits before allocating horizon tables.
Exact intermediate rational results have a declared bit limit; an exceeded
limit raises `ResourceLimitError` and returns no partial qualification. The
caps describe this implementation's execution boundary. They are not physical
provider costs or a formal bit-complexity analysis.

The finite-state evaluator is narrower than the paper's potentially infinite
state space with finite actions/observations. Quote identity validation,
artifact admission, calibration, durable recovery, and source-generative
recorders are separate machinery. No such operational evidence is inferred
from these mathematical tables.

The checked local source snapshot is `AVOC_Traditional_Final.pdf`, SHA256
`0426cccf0d2170e79fb495e41bd6d08879a2f2dd0086377021b1408d310c6959`;
its extracted text SHA256 is
`359bbc188e750ccf1b18a46b03465157e8d10272e7ffd71c8ee9210f59a6e7f0`.
These identify the inspected publication, not a fresh formal compilation.
