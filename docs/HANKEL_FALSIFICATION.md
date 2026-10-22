# Hankel rank and minimal linear realizations

Field-valued semantics admit linear state representations that can be much
smaller than a discrete observer storing exact future behavior.

For a weighted series `f : Sigma* -> K`, define its residual after prefix `u` by

`f_u(v) = f(uv)`.

A discrete future-equivalence observer identifies prefixes only when the full
residual functions are equal.  A linear realization instead stores a vector in
the span of the residual functions.  Over a field, the dimension of this span is
the rank of the Hankel matrix `H[u,v] = f(uv)` and equals the state dimension of
a minimal weighted finite automaton. This is the classical weighted-automata
and rational-series correspondence.

## A polyphase example

Take the [polyphase route packet](POLYPHASE_ROUTE_PACKETS.md) for the two-tap filter `h=(1,1)`,
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

This example illustrates why weighted automata and linear state-space models
can compress numerical series more effectively than deterministic state equality.

## Architectural consequence

The continuation quotient must be typed by the observation algebra:

* set/Boolean semantics: future-equivalence classes / Myhill--Nerode;
* field-linear semantics: residual vector-space quotient / Hankel rank;
* other semirings: the appropriate semimodule/congruence notion, when it is well behaved.

## API boundary

The polyphase route packet supplies the finite linear realization used here.
The recurrence subsystem supplies companion-state coordinates. A generic
weighted-automaton constructor is outside these APIs. Minimality depends on the
observation algebra, as described in
[carrier-change obstructions](CARRIER_CHANGE_OBSTRUCTIONS.md).
