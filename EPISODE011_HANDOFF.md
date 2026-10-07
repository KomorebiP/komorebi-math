# Episode 011 Handoff

Authoritative predecessor: `Komorebi_Math_R4_Block14_Polyphase_Route_Packets_2026-09-23.zip`

Current checkpoint: R4 Block 15 carrier-change obstructions (`wheel_build_tag=4b15`).

## Added

- `src/komorebi_math/core/carrier_change.py`
  - `finite_factorization_audit`
  - `SignedMassPacket`
  - `TropicalResidualPacket`
  - `boolean_support_nonnegative`
- `tests/test_r4_block15_carrier_change.py`
- `verification/episode011_carrier_change_probe.py`
- `verification/EPISODE011_CARRIER_CHANGE_PROBE.json`
- `verification/EPISODE011_COMPUTATION_MANIFEST.json`
- `docs/R4_BLOCK15_CARRIER_CHANGE_OBSTRUCTIONS.md`
- `EPISODE011_MATHEMATICAL_AUDIT.md`

The aggregate stable API was not widened.

## Core mathematical state

Exact semantic translation between two homomorphic carrier summaries exists iff
the target is constant on source fibers (`ker(source) subset ker(target)`). If
not, a source collision with different target values is a carrier-change
obstruction witness. The joint semantic quotient is the exact completion.

Two algebraic no-go mechanisms are now explicit:

1. cancellation barrier: nonzero rings do not map unital-homomorphically into
   zerosumfree carriers;
2. idempotence barrier: nontrivial additively idempotent carriers do not map
   unital-homomorphically into additively cancellative carriers.

The tropical residual packet gives an exact finite-temperature completion of
max-plus for scalar sum/max semantics, but does not preserve argmax provenance.

## Qualification boundary

- focused and affected source tests: clean;
- reproducible `4b15` wheel/sdist: clean;
- source/artifact integrity gate: clean after restoring executable bits stripped
  by ZIP extraction;
- light installed-artifact gate: clean;
- Block-15-specific installed-wheel probe: clean;
- full source suite: not rerun in Episode 011;
- full installed-runtime suite: not run;
- `ruff` and `mypy`: unavailable in the environment.

Do not upgrade this checkpoint to full-suite qualification without actually
running those stronger gates.
