# R4 Block 13 — Continuation-Stable Provenance Packets

Block 13 is a research-tier continuation of Blocks 11–12. It does not widen the frozen aggregate stable API.

## Purpose

Block 12 separated convex value semantics from raw branch provenance. Block 13 asks which provenance distinctions must survive when future Bellman branches may still be appended to the computation.

The implementation adds a finite continuation observer and two compiled packet forms:

- `PolicySupportPacket`: flat atoms labeled by minimized observer states;
- `ObserverConvexPacket`: one convex generator family for each minimized observer state.

The observer is a finite Moore machine adapted to the package's history convention: a newly applied Bellman branch is prepended to the stored provenance tuple, so histories are summarized from oldest to newest and one prepend is one observer transition.

## Mathematical contracts

### Continuation quotient

For a history query `q` over branch words define

`h ~ h'` iff every future branch prefix produces the same query output.

For finite-state queries this is the standard Moore/Myhill–Nerode future-equivalence relation. The quotient is the coarsest deterministic continuation-stable summary for the specified per-history query.

### State-conditioned convexification

Let `s(h)` be the minimized observer state of history `h`, and let

`C_s = conv{v_i : s(h_i)=s}`.

Then

`h(x) = max_s h_{C_s}(x)`

and state `s` is active exactly when `h_{C_s}(x)=h(x)`.

For a Bellman family `A_b`, the exact update is

`C'_t = conv union_{delta(s,b)=t} A_b^T C_s`.

Thus convex compaction is safe *inside* each provenance class, even when global convex compaction would destroy policy information.

### Continuation-class observability

For a full-dimensional finite convex packet and nonzero support directions:

- invertible linear continuations cannot make a strictly interior atom support-active;
- a singular zero continuation sends every atom to the same point and makes every retained provenance class active.

Therefore whether hidden provenance may be discarded depends on the admissible continuation class. Global geometric redundancy alone is not a sufficient deletion criterion.

## Scope boundary

The new packet semantics preserve sets of observer outputs/states that attain a support maximum. They do **not** preserve raw history identity, history multiplicity, or arbitrary nonregular history queries unless the observer is chosen to retain those distinctions.

The implementation deliberately stays under `komorebi_math.nonlinear.provenance_packets` and is not re-exported through the stable aggregate namespace.


## Qualification

The final source checkpoint records **1,667/1,667 passing tests across 146 files**. The source qualification ledger preserves timeout/retry boundaries explicitly. Release artifact and installed-wheel checks are recorded under `release/` and `verification/`; the research-tier module remains outside the frozen aggregate API.
