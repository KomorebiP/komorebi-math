# Episode 013 Handoff — Quantum Kraus/Error-Span Gauge Repair

## Authoritative predecessor

`Komorebi_Math_R4_Block16_Hankel_Falsification_2026-09-23.zip`

SHA-256:

`1f62bfa5723d1f0c0a1e75c8f4cafbeaf8c581e721fa208e2bb072dcc04d03dd`

## Repair summary

Episode 013 left the packet/automata trajectory and audited the mature quantum subsystem. It found a real semantic bug in `QuantumCode.knill_laflamme_report()`:

- exact correctability is a property of the linear error span;
- Block 16 certified it using the largest residual in the arbitrary declared error basis;
- a default-tolerance Hadamard change of error labels flipped the certificate from invalid to valid without changing the error span or CP map.

Block 17 repairs certification by Hilbert–Schmidt orthonormalizing the declared error span and taking the spectral norm of the complete Knill–Laflamme residual block operator.

`recovery_channel()` is also constructed from the canonicalized error span, eliminating generator-scaling dependence.

The legacy `maximum_residual` remains available as a declared-basis diagnostic and must not be interpreted as gauge invariant.

## Key files

- `src/komorebi_math/quantum/error_correction.py`
- `tests/test_r4_block17_quantum_kraus_gauge.py`
- `verification/episode013_quantum_gauge_probe.py`
- `verification/EPISODE013_QUANTUM_GAUGE_PROBE.json`
- `EPISODE013_MATHEMATICAL_AUDIT.md`
- `EPISODE013_LITERATURE_AUDIT.md`

## Qualification target

This is an incremental semantic-repair checkpoint. Qualify:

1. all quantum source tests plus Block 17 adversarial tests;
2. installed quantum verification;
3. release/publication parity after rebuilding `4b17` distributions;
4. offline light installed-artifact gate;
5. a Block-17-specific installed-wheel probe.

Do not claim a fresh complete-source or full installed-runtime qualification unless those gates are actually run.

## Remaining boundary

Numerical error-span discovery uses SVD. Arbitrarily ill-conditioned generating families can cross the finite-precision rank threshold even though exact span semantics does not change. Certificate diagnostics retain the singular spectrum and threshold; this conditioning boundary should not be hidden.
