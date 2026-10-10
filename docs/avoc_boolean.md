# Exact Boolean synthesis and a global minimum bank

`komorebi_math.algorithms.boolean_synthesis` implements AVOC §§6.1–6.4 in
the standardized publication edition of 8 October 2026 (the exposition of
AVOC Formal v0.41.0). The checked PDF has SHA-256
`0426cccf0d2170e79fb495e41bd6d08879a2f2dd0086377021b1408d310c6959`.
This implementation uses the paper's precise grammar and tariff; no fresh
Lean compilation is claimed.

The grammar has two Boolean literals, variables `x1`, `x2`, `x3`, negation,
conjunction, disjunction, and exclusive-or. Each AST node contributes one
unit. The instrumented interpreter visits both children of a binary operation
and counts repeated occurrences even when Python objects are shared.
Signatures are integers from 0 through 255: bit `i` is the output on input `i`
in the lexicographic order `000, 001, …, 111`. Thus the variable signatures
are 240, 204, and 170.

## The paper's worked example

```python
from komorebi_math.algorithms.boolean_synthesis import (
    BooleanExpr,
    bounded_boolean_search,
    build_boolean_closure_bank,
    audit_boolean_closure_bank,
    globally_minimal_boolean,
    boolean_node_count,
)

x1, x2 = BooleanExpr.variable(1), BooleanExpr.variable(2)
reference = BooleanExpr.negate(BooleanExpr.binary(
    "or", BooleanExpr.negate(x1), BooleanExpr.negate(x2)
))
search = bounded_boolean_search(reference, 1, budget=6890, trial_cap=85)
assert search.certificate.valid
assert (search.reference_nodes, search.selected_nodes) == (6, 3)
assert (search.generation_charge, search.comparison_tariff) == (90, 80)
assert search.modeled_charge == 6890 and search.complete_layer

bank = build_boolean_closure_bank()
certificate = audit_boolean_closure_bank(bank)
assert certificate.valid
minimum = globally_minimal_boolean(bank, reference)
assert boolean_node_count(minimum) == 3
```

The complete depth-one layer has 85 entries, including duplicates. Generation
reserves the sum of layer counts from zero through the requested depth, so
the reservation here is `5 + 85 = 90`. Each comparison is conservatively
charged `8 * (N(reference) + B(depth) + 1) = 80`. The complete search therefore
costs `90 + 85 * 80 = 6890` modeled units and saves three eager visits on each
subsequent evaluation. Promotion and repayment economics are separate from
semantic equivalence and this exploration budget.

## Exact search and compaction contracts

The duplicate-preserving layer recurrence is `M(0)=5` and
`M(d+1)=2*M(d)+3*M(d)**2`; maximum node count satisfies `B(0)=1` and
`B(d+1)=2*B(d)+1`. `boolean_grammar_layer` emits retained expressions,
negations, then AND/OR/XOR expressions for all ordered pairs.

`bounded_boolean_search` examines the prefix permitted by both `trial_cap`
and the remaining declared budget. It retains an equivalent incumbent and
replaces it only for a strict node reduction. A complete layer gives the
minimum among equivalent expressions of height at most that depth, with the
reference as fallback. An incomplete prefix has only prefix optimality.
An unaffordable generation reservation returns the reference with zero
modeled search charge. It does not return a global minimum certificate.

`compact_boolean_entries` retains the first cheapest expression for each
signature. `compacted_boolean_layer` recursively constructs minimum semantic
representatives, as in Proposition 6.3, without allocating every constructor
candidate. Its counters record emitted candidate entries, their total node
mass, and one declared semantic-index lookup per candidate. These counters
exclude allocation, serialization, and hardware timing. Compaction preserves
the full grammar layer's optimum because every constructor is a semantic
congruence and has an additive nonnegative node tariff.

## What the global certificate checks

`BooleanClosureBank` contains proposed rows; constructing that object does
not certify it. `audit_boolean_closure_bank` independently checks:

1. Exactly 256 rows with distinct signatures covering the whole domain.
2. Each AST's eight outputs and eager visit count against its stored row.
3. Five atom lower-bound inequalities and all 256 negation inequalities.
4. All 196,608 ordered AND/OR/XOR constructor inequalities.

For each signature `s`, let `c(s)` be its row's claimed cost. The inequalities
and verified witnesses are precisely the finite hypotheses of Theorem 6.4.
Structural induction then gives `c(sem(p)) <= N(p)` for every finite AST in
this grammar, and the corresponding witness attains equality. The accepted
bank consequently gives global node minima across all finite heights in
this declared three-input tree language.

The current construction reaches a complete, constructor-closed bank after
three compacted rounds. Its largest attained minimum has 10 nodes. The
distribution of minimum costs is:

| Nodes | Signatures |
| ---: | ---: |
| 1 | 5 |
| 2 | 3 |
| 3 | 9 |
| 4 | 21 |
| 5 | 21 |
| 6 | 81 |
| 7 | 43 |
| 8 | 43 |
| 9 | 18 |
| 10 | 12 |

This is a conclusion about complete finite semantics, this AST grammar, and
the eager tree-node tariff. It does not minimize arbitrary programs, shared
circuit DAGs, short-circuit visits, wall time, or physical execution cost.

## Limits and validation

Full-layer arithmetic depth is capped at eight, and `max_entries` defaults
to 100,000. In particular a default depth-three full layer is rejected before
allocation. Recursive compaction and bank construction cap expansion at
32 rounds, with a default per-layer limit of 250,000 emitted candidates.
At 256 representatives, a full compacted constructor layer has only 197,120
candidates. `max_witness_nodes` defaults to 10,000; AST traversal also enforces
a maximum depth of 256. An exhausted resource limit raises or rejects bank
acceptance. No partial bank is returned with a global minimum assertion.

`tests/test_avoc_boolean.py` checks full layers through depth two against an
independent recursive truth-table and node-count oracle. It also checks the
whole bank with tuple-valued Boolean semantics, verifies every constructor
inequality independently of the producer's bit-mask algebra, reproduces the
paper's 6,890-unit example, and rejects incomplete, duplicated, forged,
semantically valid but nonminimal, and resource-exhausted banks.
