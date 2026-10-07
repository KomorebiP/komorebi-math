# Episode 012 Handoff

## Decision

Do not add a generic weighted-automaton/Hankel packet layer to the Komorebi
stable or research API at this checkpoint.

## What survived

* For field-valued future queries, discrete future-equivalence is generally the
  wrong minimality notion.
* The correct classical linear continuation object is the residual vector space;
  its dimension is the Hankel rank.
* Existing `PolyphaseRoutePacket` objects already provide linear route
  realizations.
* Existing `LinearRecurrence` / recurrence-coordinate machinery already carries
  the corresponding finite state in the unary recurrence case.
* Hankel analysis remains useful as a diagnostic for redundant/unobservable
  route coordinates.

## Strongest exact project theorem

For `h=(1,1)`, polyphase order `N`, and evaluated coarse-lag point `z != 0`, let
`M_N=I+S_z` with `S_z^N=zI`, and `f_N(n)=e0^T M_N^n e0`.
The `N x N` Hankel block `H[i,j]=f_N(i+j)` has determinant
`+/- z^(N-1)` and therefore rank `N`.

For positive `z`, all unary residuals `k -> f_N(n+k)` are pairwise distinct, so
the discrete exact continuation quotient is infinite while the weighted linear
dimension is `N`.

## Finite verification

* 23/23 focused route/recurrence/Hankel tests passed.
* Deterministic probe: pass.
* Orders 2..10: Hankel rank = route order.
* 65 sampled residuals distinct for each tested order.
* 120/120 random nondegenerate two-symbol route audits had sampled rank equal to
  phase dimension.
* Deliberate six-phase degenerate route: Hankel rank 1.
* Order-two route sequence matched `LinearRecurrence((1,2),(1,1))` exactly.

## Qualification boundary

The finite computation does not prove generic full rank and does not establish
the external minimal-WFA theorem.  The external theorem was checked separately
against modern weighted-automata literature.

## Recommended next inquiry

Leave the packet/automata trajectory for at least one episode.  Audit a mature
independent Komorebi subsystem using the accumulated rules:

1. identify the exact carrier/query/continuation semantics actually required;
2. search for an invalid compression or missing hypothesis;
3. only if the audit finds a nontrivial transferable result, reconnect it to the
   packet synthesis.

A mature arithmetic subsystem is preferable because it is structurally far from
weighted automata and will provide stronger novelty pressure.
