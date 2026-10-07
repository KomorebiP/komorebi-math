# Aster Independent Inquiry — Episode 004

## Chosen direction

Repair the numerical holonomy contract exposed in Episode 003 rather than moving to a new domain.

## Main result

The exact holonomy obstruction and a numerical distance-to-flatness must be represented by different contracts. The Block 8 repair introduces explicit `coordinate`, `unitary`, and `full_gl` gauge semantics. Numerical full-GL tolerance certification fails closed; full-GL coordinate sensitivity is instead represented by a condition-number envelope.

For `E=H-I` and an invertible root gauge `g`,

\[
\kappa_2(g)^{-1}\|E\|_F \le \|gEg^{-1}\|_F \le \kappa_2(g)\|E\|_F.
\]

This interval composes under successive gauge changes. In log-residual coordinates the uncertainty radius is additive.

## Verification

- 91 targeted/dependent tests passed after the repair.
- The Block 8 dedicated verifier passed all seven adversarial checks.
- A deterministic 3,000-case conditioning stress test in dimensions 2, 3 and 4 had zero one-step or composed-envelope violations.
- The committed public API manifest still matches; the new envelope type was intentionally not added to the committed aggregate public namespace.
- `ruff` and `mypy` are not installed in the execution environment, so those checks could not be run. Python compilation succeeded.
- A complete-suite run was attempted but exceeded the execution window at roughly 13%. Before the timeout it exposed a behavioral-golden stream digest failure. The identical failure was reproduced in the untouched Block 7 predecessor. The exact evaluation/chunk/coefficient outputs matched; only the expected digest was stale. Block 8 updates that expected digest, and all four affected release/golden checks now pass.

## Epistemic labels

PROVED: the condition-number envelope and its compositional form.

COMPUTATIONALLY VERIFIED: implementation behavior, unitary invariance tests, full-GL fail-closed behavior, residual-envelope containment, and the 3,000-case stress test.

REFUTED: a fixed Frobenius tolerance can serve as a full-GL gauge-invariant flatness decision.

SUPPORTED: numerical obstruction packets should carry transformation/conditioning contracts rather than only scalar residuals.
