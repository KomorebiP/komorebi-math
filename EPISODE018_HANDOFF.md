# Episode 018 Handoff

## Decision
Repair the demonstrated certificate inconsistencies and stop short of redesigning the certificate API.

## Changed source
- `signals/wavelets.py`: exact dyadic best-basis comparisons.
- `algorithms/compiler.py`: exact dyadic Lagrangian/budget comparisons and exact budget quantization.
- `algorithms/precision.py`: exact dyadic dynamic-programming costs.
- `langlands/finite_image/groups.py`: certificate independently revalidates all finite-group axioms.
- coordinate certificate docstrings: distinguish valid map from invertible isomorphism.
- `tests/test_r4_block21_certificate_consistency.py`: adversarial regressions.

## Evidence
- `verification/EPISODE018_PREDECESSOR_WITNESSES.json`
- `verification/EPISODE018_CERTIFICATE_CONSISTENCY_PROBE.json`
- `verification/EPISODE018_CERTIFICATE_INVENTORY.json`

## Remaining boundary
The audit found no evidence that the `Certificate` data structure itself must be redesigned. The next leverage point is automated producer-policy enforcement: detect exact certificates that combine trust bypasses, rounded float decisions, bounded-only evidence, or failure diagnostics disconnected from validity.
