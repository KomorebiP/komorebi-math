# Episode 009 Handoff

Authoritative predecessor: `Komorebi_Math_R4_Block12_Support_Packets_2026-09-22.zip`.

Current checkpoint: R4 Block 13 research-tier provenance-continuation quotient.

## Added

- `src/komorebi_math/nonlinear/provenance_packets.py`
- `tests/test_r4_block13_provenance_packets.py`
- `verification/episode009_provenance_probe.py`
- `verification/EPISODE009_PROVENANCE_PROBE.json`
- `verification/EPISODE009_COMPUTATION_MANIFEST.json`
- `docs/BLOCK_13_RELEASE.md`
- `EPISODE009_MATHEMATICAL_AUDIT.md`

## Core result

Raw histories should first be quotiented by future-query equivalence (finite Moore/Myhill–Nerode when applicable), then geometry should be convexified separately inside each minimized provenance state.

For minimized observer states `s`, store bodies `C_s`. The exact Bellman update is

`C'_t = conv union_{delta(s,b)=t} A_b^T C_s`.

This retains policy semantics under arbitrary linear branches, including singular maps.

## Critical boundary

A globally interior history may be safely hidden under invertible-only continuations but can be resurrected by a singular collapse. Therefore provenance deletion depends on the admissible continuation class, not only on current convex geometry.

## Final qualification

- Source suite: **1,667/1,667 PASS across 146 test files** using completed deterministic batch ledgers.
- `test_generating_foundation.py` did not exit cleanly as one monolithic invocation inside the wrapper window; its 68 collected nodes were therefore rerun in bounded explicit-node groups, all of which passed with clean exits. The whole-file timeout is retained as an execution qualification note rather than counted as a pass.
- Block 13 distributions use build tag `4b13`; source/distribution parity and byte-reproducibility are checked in the release gate.
- The light installed-artifact gate and a Block-13-specific installed-wheel probe are required and recorded separately.
- A full installed-runtime suite is **not** claimed unless explicitly recorded as completed.
