# Repair R3: first implementation-hardening pass

Author: Komorebi P.
Date: 6 September 2026
Public package version: 1.0.0b4
Delivery identity: 1.0.0b4-repair-r3
Wheel build tag: 3

## Reading precedence

The completed Block 47 manual describes the R2 implementation. Its claims about
R2 behavior and its recorded source hashes are historical for changed files.
This addendum supersedes the specific behaviors listed here. The manual PDF is
not silently relabeled or rebuilt as an R3 audit. Original archives are unchanged.

## Changes

* Certificate validity accepts real Python or NumPy booleans, normalizes NumPy
  scalars, and rejects strings, integers, collections and None. require() accepts
  only the canonical True singleton. Portable decoding no longer calls bool()
  on the stored validity flag.
* Hilbert certification recomputes packet axioms as well as self-adjointness,
  including after validate=False or changes to retained matrices. Its residuals
  now include idempotence, annihilation and completeness.
* Portable schema/codec versions, Fraction fields, and NumPy shape dimensions
  reject equality-compatible or coercible nonintegers. Producer metadata must be
  a nonempty string but is still not authenticated. Array cardinality and platform
  index representability are checked before nested decoding or array allocation.
  Structured-array encoding is explicitly rejected rather than writing a format
  the reader cannot reliably reconstruct.
* The JSON reader checks bracket nesting before its recursive parser. It reports
  interpreter recursion exhaustion as ValidationError. Semantic max_depth still
  applies. This is not a universal file-size or memory ceiling.
* FiniteNormalizedSeries checks each degree before coercion and each coefficient
  bit length immediately. An unsized iterable is consumed only through the first
  excess item. A user-defined iterator or coercer can still perform arbitrary work.
* Packet-jet planning checks a saturating Cartesian-product count before calling
  addresses(). It retains the existing maximum-address contract and a defensive
  check of the returned enumeration.
* Expression preflight enforces node, traversal-step and active-path depth limits
  while walking, before expanding the remaining graph. Final DAG depth/statistic
  checks remain in place. Existing inputs/callback allocations are outside this
  traversal guard.
* Leaf-count tree enumeration computes a capped counting table before invoking
  the tree generator or its persistent cache. The count handles zero-cardinality
  degrees without falsely rejecting them because a smaller degree was large.
  Counting-table work, recursive generator depth and cumulative caches still need
  separate budgets for extreme requests.
* CoordinateBudget.require_matrix rejects booleans and noninteger dimensions.

## Compatibility boundary

Normal typed inputs and finite positive controls retain their behavior. Malformed
payloads that formerly relied on coercion are now rejected. Hilbert certificates
can correctly turn negative for invalid packet systems and return more residual
fields. Strict input validation is not proof verification: a caller can still
construct a true record without proving a theorem. Conditional assumptions are
still explicit and require() does not prove them.

## Not completed by this pass

This is not an all-platform or hostile-input security qualification. Arbitrary
registered decoders, nested diagnostic mutation, full byte/allocation limits,
all specialized certificate protocols, cache quotas, extension-process isolation,
and every source-coverage gap remain separate work. Windows/PowerShell, CUDA,
other Python/dependency versions, strict MkDocs and a fully dependency-isolated
installation have not been established by this repair. No Lean proof audit was
performed. No license permission was changed.

## Installation

With the required dependencies already installed, from the extracted source root:

```powershell
python -m pip install --no-index --no-deps --force-reinstall release/dist/komorebi_math-1.0.0b4-3-py3-none-any.whl
```

The public version remains 1.0.0b4 to preserve the pinned API contract. The wheel
build number and RELEASE_ID.json distinguish this repair. Force replacement of
an older build with the same version. Keep original data before retrying any
previously malformed portable file; do not erase a refusal by changing its digest.

## Reproduction

The delivered evidence records current executions, environment, source and artifact
hashes, tests and expected negative controls. Inherited R1/R2 and Block 17-21 release
records are not new R3 qualification. Do not infer R3 publication readiness from
an old passing JSON record. Use the R3 manifest and replay evidence instead.
