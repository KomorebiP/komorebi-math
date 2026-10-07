# Episode 009 Mathematical Audit

## Question

What is the coarsest compositional provenance information that must remain after convex value geometry has already been compressed?

## 1. History-query quotient

Let `Sigma` be the finite branch alphabet and let `q : Sigma* -> Y` be the per-history information a downstream policy query asks for. New branches are prepended to the stored word. Define

`h ~_q h'`

iff

`q(u h) = q(u h')`

for every future prefix `u in Sigma*`.

Then `~_q` is a left congruence under prefixing. If it has finite index, its classes form a finite deterministic continuation summary. Conversely every finite exact deterministic continuation summary induces such a finite-index congruence. The quotient is therefore the coarsest exact deterministic per-history summary up to isomorphism.

This is the prepend-oriented Moore/Myhill–Nerode theorem, not claimed as new mathematics.

### Consequences

- parity of the number of one-branches has two classes;
- a finite regular property has finitely many classes;
- the query `q(h)=h` that demands the complete raw history has singleton classes and therefore infinitely many states over unbounded histories.

Thus exact provenance compression is impossible without restricting what downstream queries are allowed to ask.

## 2. Observer-conditioned convex packet theorem

Take a finite labeled packet

`P = {(v_i, h_i)}`

with `v_i in R^d`. Let `s(h_i)` be the minimized observer state of history `h_i`. For each state `s`, define

`C_s = conv{v_i : s(h_i)=s}`.

Empty states are allowed.

### Theorem

For every direction `x`,

`max_i <v_i,x> = max_s h_{C_s}(x)`.

Moreover the set of active observer states is exactly

`{s : h_{C_s}(x) = max_t h_{C_t}(x)}`.

### Proof

Partition the finite maximum by observer state:

`max_i <v_i,x> = max_s max_{i:s(h_i)=s}<v_i,x>`.

The inner maximum is the support function of the convex hull `C_s`. Equality of the inner maximum with the global maximum is equivalent to the existence of an active atom in state `s`.

No information about active observer states is lost by replacing the raw atoms in each state with any other generating set of the same convex hull.

**Status: PROVED.**

## 3. Exact Bellman update

Suppose branch `b` uses primal matrix `A_b` and updates observer state by `delta(s,b)`. A dual atom `v` transforms to `A_b^T v`.

Therefore the state-conditioned bodies obey

`C'_t = conv union_{(s,b): delta(s,b)=t} A_b^T C_s`.

Taking support functions gives

`h_{C'_t}(x) = max_{delta(s,b)=t} h_{C_s}(A_b x)`.

This is an exact finite-state Bellman recursion. Repeated application composes because matrix multiplication and observer transition both compose.

**Status: PROVED.**

## 4. Singular resurrection theorem

Global convex redundancy is not enough to decide whether provenance may be deleted.

Let `v` be a strictly interior atom carrying a provenance state not carried by the boundary atoms. At the current packet it cannot maximize any nonzero linear functional on a full-dimensional body.

If the allowed continuation class contains the zero map, however,

`A=0`

sends every atom to the same point. For every nonzero query direction all transformed atoms tie. The previously invisible provenance state is now active.

Hence any compaction rule that deletes provenance solely because its atom is currently globally interior is unsound when singular collapse is admissible.

**Status: PROVED.**

## 5. Invertible hiding theorem

Assume the overall convex body is full dimensional, support directions are nonzero, and every admissible branch matrix is invertible.

If `v` is strictly interior, then `A_b^T v` is strictly interior to `A_b^T C` for every branch. A nonzero linear functional cannot attain its maximum at a strict interior point of a full-dimensional compact convex body. Therefore the atom is below the maximum of its own branch and cannot be globally Bellman-active.

Thus strict-interior provenance remains permanently hidden under invertible-only continuation.

**Status: PROVED under the stated full-dimensional/nonzero-direction assumptions.**

This fails immediately once singular maps are admitted.

## 6. Three complexity scales

The Episode 009 parity example makes three packet sizes diverge:

- value geometry: one geometric point;
- parity-policy semantics: two continuation states;
- raw trace semantics after depth 12: 4,096 histories.

All three are exact for different downstream tasks. None is the universally correct notion of packet size.

## 7. Computation

The deterministic probe used seed `20260923` and reported:

- 400 random finite observers;
- 22,380 sampled future-prefix equivalence checks;
- 0 observer-minimization mismatches;
- 800 random support/provenance systems;
- 2,400 sampled support/policy queries after two Bellman updates;
- 0 flat compiled-policy mismatches;
- 0 observer-conditioned convex-packet mismatches;
- 500 invertible-map tests of strict-interior hiding;
- 0 hiding failures;
- exact zero-map resurrection of the hidden provenance class;
- depth-12 identity branching: 4,096 raw histories -> 2 parity states.

These computations audit the implementation. They are not the proofs above.

## 8. Literature boundary

The continuation quotient is standard automata theory: Myhill–Nerode/Moore minimization already characterizes states by indistinguishability under all future inputs.

Database and program provenance likewise has established hierarchies in which more informative provenance representations factor to coarser task-specific views, and provenance traces separate execution history from derived views.

Automaton-indexed families of invariant sets and path-complete/memory-based Lyapunov constructions also show that finite automata coupled to multiple geometric pieces are established structures in switched-systems theory.

Therefore Block 13 does **not** establish novelty of the component mathematics.

The potentially distinctive Komorebi claim is narrower: one common packet interface now separates value geometry, future-query provenance state, admissible continuation class, and raw trace information, while making the loss incurred by each quotient explicit.

## 9. Failed routes retained

1. **Global convex hull as a sufficient policy packet — REFUTED.** Singular collapse can expose provenance attached to a globally interior atom.
2. **Raw history as the default provenance primitive — REJECTED as unnecessarily strong for regular downstream queries.** A finite future-equivalence quotient can be exponentially smaller.
3. **Observer quotient alone as a globally minimal combined packet — NOT ESTABLISHED.** It is minimal per history, but global geometry can hide or couple states depending on the continuation class and query surface.

## 10. Frontier

The next hard question is whether state-conditioned convex packets admit a universal algebraic formulation (for example as an idempotent/semiring or enriched-automaton object) that covers polyphase routing, switched systems, and piecewise-linear policies without merely renaming weighted-automata or multi-set machinery.


## 11. Source qualification

The frozen Block 13 source tree contains 146 test files and 1,667 collected tests. All 1,667 assertions passed. Qualification used completed deterministic file ledgers rather than treating wrapper timeouts as success.

One inherited predecessor file, `test_generating_foundation.py`, exhibited process-exit behavior that prevented a clean whole-file completion inside the wrapper window. Its 68 collected pytest node IDs were subsequently executed in seven bounded groups; all 68 passed with clean exits. This distinguishes assertion qualification from the unresolved whole-file teardown behavior.

An intermediate publication-gate batch also failed correctly while the source identified itself as Block 13 but the bundled artifacts were still Block 12. After rebuilding with wheel tag `4b13`, the same batch passed; only the successful rebuilt-artifact ledger is counted in the final qualification total.
