# Remaining limitations after scoped R4 Block 3

This checkpoint does not close the whole-library audit. It repairs the named
finite tensor and numerical obstruction decision paths and their tested controls.

- Not all numerical norms, exact comparisons, derivative providers, GPU paths or
  backend adapters have been reviewed. See the enclosing source-review queue.
- Other nonlinear correction, elimination, radius, tail and perturbation estimates
  need the Block 4 analytic audit. Do not infer guarantees for untested paths.
- SVD rank selection is a numerical retained-model decision. No interval enclosure,
  perturbation-certified true rank or correctly rounded linear algebra is provided.
- Global multilinearity, derivative correctness, support correctness, completeness
  and analytic norm bounds remain supplied where stated. Probes are not global proofs.
- Already issued certificates are historical observations, not self-invalidating live
  checks. General mutation, cache, serialization and trust-boundary audits remain open.
- Symbolic entries can be retained by in-memory capture. No general SymPy stream
  codec, verified parser, native big-integer bit-cost proof or resource qualification
  for arbitrary large objects is introduced.
- Source and wheel tests use one shared Linux/Python dependency environment. Windows,
  native PowerShell, CUDA, additional combinations and dependency-isolated installation
  are not qualified by this checkpoint. No Lean or independent referee was run.
- No new observable-quotient, weighted-majorant or Polyphase feature is included.

Public stable names/signatures are not a guarantee of identical behavior: the
reported repairs intentionally reject invalid cases previously accepted.
