# Remaining work after R4 Block 1

This file restricts the current development checkpoint, even when tests pass.

- The inherited static queue has 220 candidate comparisons. Five queue rows map
  to rewritten producers; 215 were not reviewed in this block. This is not a
  count of defects or a completeness measure.
- Other exact-certificate producers, derivative/jet/connection/holonomy paths,
  backend adapters and specialized certificate protocols need their own review.
- Old Certificate objects and decoded valid=True flags are records, not freshly
  authenticated proofs. Snapshot binding, aliasing, mutation and replay remain
  a separate workstream.
- Symbolic equality can be unknown. SymPy assumptions and symbolic domain
  restrictions are trusted, not proven; arbitrary expression work has no global
  security/performance guarantee.
- Numeric tolerance checks do not bound accumulated floating arithmetic error.
  GPU/CuPy/JAX/Torch underflow/precision behavior is not qualified by this block.
- Exact scalar guards do not verify user-supplied multilinear norm estimates or
  completeness. Their power-work cap is not a universal resource guarantee.
- Windows/PowerShell, other Python/dependency versions, CUDA, clean dependency
  isolation and strict documentation builds still need fresh qualification.
- The observable-quotient prototype and weighted-majorant proposals remain
  separate research material, not newly integrated production APIs.
- Public distribution terms need an explicit owner decision. LICENSE.md is
  unchanged and remains the retained private-evaluation licence.
