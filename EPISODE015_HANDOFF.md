# Episode 015 Handoff — Arithmetic Semantic Repair

## Authoritative predecessor

`Komorebi_Math_R4_Block18_Lindblad_Cyclic_Repair_2026-09-23.zip`

SHA-256:

`9609e1d5b85bf78171869478998529166a0aeccf1bc4649cdfd602c20c3d9b91`

## Episode result

The arithmetic wild-card audit found two independent semantic-contract defects.

1. Truncated p-adic domains returned infinite valuation for every zero residue, despite their visible/capped precision contract. Block 19 returns the precision cap instead and keeps exact `padic_valuation(0,p)=+infinity` separate.
2. `UnitModeMahler.certificate(maximum_value, degree)` checked only through `min(maximum_value, degree)` but could return an exact valid certificate for a wrong undertruncated tail. Block 19 checks the entire requested finite range.

No Euler, standard-window, Hurwitz, recurrence, or affine-kernel formula was changed.

## Files changed

- `src/komorebi_math/arithmetic/domains.py`
- `src/komorebi_math/generating/domains.py`
- `src/komorebi_math/arithmetic/mahler.py`
- `tests/test_arithmetic_domains_valuation.py`
- `tests/test_generating_domains.py`
- `tests/test_r4_block19_arithmetic_semantic_repair.py`
- `verification/episode015_arithmetic_semantic_probe.py`
- `verification/EPISODE015_ARITHMETIC_SEMANTIC_PROBE.json`
- Episode 015 audit/evidence/release files

## Qualification boundary

This checkpoint is intended to be incrementally qualified. The mature Block 6 deterministic audit is rerun because it covers the repaired arithmetic layer. The complete modern source suite and full installed-runtime suite are not implied unless explicitly recorded in the final qualification ledger.

## Deferred issue

`MahlerSeries.reconstruction_certificate()` is numerically brittle for floating coefficients because it compares reconstructed tuples by exact equality. Do not patch this with an arbitrary epsilon. Any later repair should derive and expose a conditioning-aware tolerance or use a separate exact rationalized audit path.

## Recommended next inquiry

Do not stay in arithmetic solely because two defects were found. The stronger next test is either:

- inspect approximate numerical certificates where exactness changes to numerical semantics, beginning with the deferred Mahler reconstruction issue; or
- switch domains again and test whether semantic typing finds an independently reproducible problem in a non-quantum/non-arithmetic subsystem.
