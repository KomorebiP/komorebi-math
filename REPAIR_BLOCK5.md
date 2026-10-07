# Komorebi Math R4 Block 5

Checkpoint: `1.0.0b4-repair-r4-block5`. Wheel build: `4b5`.

This scoped checkpoint repairs certificate-prerequisite composition in selected
nonlinear packet producers and moving-packet transport. It is not final R4.

Key rules:

- a locally vanishing residual does not certify a theorem over an invalid parent `PacketSystem`;
- selected nonlinear packet certificates freshly recompute and embed their parent packet certificates;
- unresolved assumptions of parent packet certificates remain visible after composition;
- sampled generator identities do not prove that supplied projector derivatives are actual time derivatives;
- Kato transport and refinement records carry that derivative-correctness premise explicitly;
- external premises such as Chebotarev remain unchecked unless separately discharged.

The 131-name stable API is not intentionally changed. The new prerequisite helper is private.
See the outer Block 5 repair report for finite evidence and remaining scope.
