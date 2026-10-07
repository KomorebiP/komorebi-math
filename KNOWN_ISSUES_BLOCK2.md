# Remaining issues after R4 Block 2

This is not final R4 or public-release approval. The external Block 2 report
states the exact verification scope. Earlier known-issues files remain history.

- 175 inherited static-comparison rows remain unreviewed, including nonlinear
  execution/elimination/correction and additional refinement/sparse paths.
- Producer-wide parent-assumption validation is incomplete. The repaired value
  comparisons do not prove the validity of every supplied packet/derivative.
- Certificate lifecycle, mutation, cache identity, serialization and safe
  revalidation require a separate sweep. Old returned certificates remain old.
- Exactness follows the declared backend/provider contract, not physical real
  inputs before rounding. SymPy simplification and symbolic domains remain
  trusted implementation behavior, not a universal equality prover.
- Numerical backends and optional derivative adapters are not globally qualified.
  Taylor residuals in the declared provider remain diagnostics, not remainder
  theorems; arbitrary symbolic Taylor diagnostics are not repaired here.
- Prime-field exact object arrays can be slower; callers that require native
  NumPy machine integers must account for the changed dtype. Huge primes still
  use inherited trial division and are not resource-qualified.
- Windows, PowerShell, CUDA, dependency-isolated installation, strict docs,
  independent review and complete publication/licensing gates remain open.
- Observable quotient, weighted-row-majorant and Polyphase APIs are not added.
