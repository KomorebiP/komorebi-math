# R4 Block 16 — Hankel-Rank Falsification

Block 16 tests whether Episode 009's discrete continuation observer should be
extended as the universal provenance-compression primitive.

The answer is **no** for field-valued semantics.

For a weighted series `f : Sigma* -> K`, define its residual after prefix `u` by

`f_u(v) = f(uv)`.

A discrete future-equivalence observer identifies prefixes only when the full
residual functions are equal.  A linear realization instead stores a vector in
the span of the residual functions.  Over a field, the dimension of this span is
the rank of the Hankel matrix `H[u,v] = f(uv)` and equals the state dimension of
a minimal weighted finite automaton.  This is classical weighted-automata / rational-series theory, not a new Komorebi theorem.

## Polyphase falsification

Take the existing Block 14 route packet for the two-tap filter `h=(1,1)`,
polyphase order `N`, and evaluate the coarse-lag polynomial variable at `z=2`.
The resulting route matrix is

`M_N = I + S`, with `S^N = 2 I`.

For the scalar downstream query

`f_N(n) = e_0^T M_N^n e_0`,

the `N x N` Hankel block has full rank `N`.  Thus the existing `N`-phase linear
route state is already minimal for this query.

For `N=2`,

`M = [[1,2],[1,1]]`

and

`f(n) = 1,1,3,7,17,41,...`,

which satisfies

`f(n+2)=2 f(n+1)+f(n)`.

All discrete residual functions `k -> f(n+k)` are distinct, so a deterministic
Moore observer storing exact future behavior requires infinitely many states.
The linear/Hankel representation requires dimension two.

This separation is real but not novel.  It is precisely the classical reason
weighted automata and linear state-space models are stronger than deterministic
state equality for numerical series.

## Architectural consequence

The continuation quotient must be typed by the observation algebra:

* set/Boolean semantics: future-equivalence classes / Myhill--Nerode;
* field-linear semantics: residual vector-space quotient / Hankel rank;
* other semirings: the appropriate semimodule/congruence notion, when it is well behaved.

Block 16 does **not** add a generic weighted-automaton layer to the package.  The
existing polyphase route packet is already a linear realization, and the
existing recurrence subsystem already carries companion-state coordinates.
Adding another wrapper would duplicate established machinery.

## Research decision

The packet trajectory should be compressed here rather than expanded.  The
interesting question is no longer whether a universal "observer packet" exists;
classical algebraic automata theory already supplies carrier-dependent notions
of minimal realization.  A future episode should test whether the accumulated
typing discipline catches a substantive error or yields a theorem in an
independent Komorebi subsystem.
