# Episode 017 mathematical audit — finite deformation and Frobenius separation

## Question

Can a mature deformation/companion subsystem incorrectly promote finite local trace data into an exact separator certificate?

## Counterexample in the predecessor

`finite_frobenius_separator({2: (1,1), 3: (2,2), 5: (3,3)})` has trace-matrix rank 1 for two declared constituents.  The predecessor correctly returned `separating=False` and `recover(...)` refused to solve the nonunique system, but its certificate was hard-coded `valid=True`, so `certificate.require()` passed.

The same failure occurs for an underdetermined one-row/three-constituent table.

## Finite theorem

Let `A` be the finite trace matrix with one column for each declared constituent.  Observed traces have the form

\[
y=A m.
\]

The multiplicity vector `m` is uniquely recoverable for every `y` in the image exactly when

\[
\ker A=0,
\]

or equivalently when `A` has full column rank.  Therefore

\[
\text{finite separator valid}
\iff
\operatorname{rank}(A)=\text{constituent count}.
\]

This is exact finite linear algebra.  It is independent of any global density theorem.

## Repair

Block 20 sets certificate validity to the already-computed `separating` Boolean and records `rank_deficiency` explicitly.  A rank-deficient exact computation now returns an exact **invalid** certificate and `.require()` raises.

The pre-existing unchecked assumption `Chebotarev density beyond the supplied table` is retained.  No finite-rank certificate is promoted to a global Galois/automorphic identification theorem.

## Finite computation

A deterministic seed-20260924 probe tested 1,800 random integer/rational trace tables against independent exact SymPy rank.  It observed 1,283 full-rank and 517 rank-deficient tables, with zero rank/status mismatches, zero certificate-validity mismatches, zero `.require()` contract mismatches, and zero exact-recovery failures on full-rank cases.

## Residual API caution

`FrobeniusSpine.agrees(other)` is a pointwise comparison on the common supplied places; disjoint spines therefore agree vacuously on their empty intersection.  This method does not produce a theorem certificate and no audited exact path treats that Boolean as global identification, so Episode 017 does not change it.  Callers needing evidence must separately require a nonempty declared comparison set.
