# Episode 018 Mathematical Audit — Certificate Consistency

## Scope
Repository-wide adversarial audit of producers that emit `Certificate(..., exactness=Exactness.EXACT, ...)`. The audit asks whether `valid`, exactness, producer-side diagnostics, and construction assumptions agree.

## Result
The certificate framework is not globally broken, but two recurring producer failure modes were demonstrated.

### 1. Validation bypass can mint a false exact certificate
`FiniteGroup(..., validate=False)` is an explicit trust/performance construction path. In Block 20, `FiniteGroup.certificate()` nevertheless returned an unconditional exact valid certificate. The binary operation `a*b=0` on `{0,1}` with identity `0` therefore produced a passing `valid finite group` certificate even though the group axioms fail.

Block 21 makes `FiniteGroup.certificate()` an independent exact axiom audit: closure, two-sided identity, unique two-sided inverses, and associativity are rechecked regardless of constructor validation mode.

### 2. Rounded binary-float aggregation contradicted EXACT optimality
Let

`h = 0.5`, `l = nextafter(0.5, 0) = 0.49999999999999994`.

As exact dyadic rationals,

`h+l = 18014398509481983 / 18014398509481984 < 1`,

but binary floating addition rounds `h+l` to `1.0`.

This produced false exact optimization decisions in four predecessor paths: Haar best basis, tree Lagrangian compilation, tree budget compilation, and discrete precision assignment. Each returned `Exactness.EXACT` while choosing a plan that was not minimal over the exact binary values supplied by the API.

Block 21 preserves the public float result fields but performs all discrete comparisons and finite sums with `Fraction.from_float`, i.e. the exact dyadic rational represented by each supplied Python float. Thus the certificate can remain EXACT relative to the declared finite inputs.

## The certificate obligations are orthogonal
A sound exact certificate has at least two independent obligations:

1. **Truth obligation:** `valid=True` must mean the certified predicate actually holds, independent of any construction-time trust shortcut.
2. **Arithmetic obligation:** `Exactness.EXACT` requires the deciding computation to be exact under the input semantics; rounded floating aggregation is not exact merely because the search space is finite.

A producer may satisfy one and fail the other.

## Static inventory
The post-repair tree contains 159 explicit `Exactness.EXACT` certificate call sites across 89 source files. Twenty-four still use literal `valid=True`. Manual review did not classify those 24 as 24 defects: most are exact-by-construction/replayed-witness producers, previously audited integration paths, or certificates of a structurally valid map rather than of invertibility. The inventory is preserved in `verification/EPISODE018_CERTIFICATE_INVENTORY.json`.

The generic coordinate-transform certificate was documentation-hardened rather than behavior-changed: validity certifies a well-formed finite map; invertibility is separately reported in diagnostics. No downstream false theorem based on `.require()` was found.

## Finite verification
Seed `20260924`. Six hundred trials were run for each of: FiniteGroup certification, Haar best basis, tree Lagrangian, tree budget, and discrete precision assignment. Independent exact oracles used integer/rational arithmetic and `Fraction.from_float`. There were zero mismatches. Validation-bypass probes for FiniteAbelianGroup, PacketSystem, and PhaseTypeSurvival remained sound; FiniteGroup was the repaired outlier.

## Nonclaims
This audit does not prove every certificate producer correct. Static hard-coded-true detection is a risk heuristic, not a proof of unsoundness. Numerical certificates and producers without explicit `Exactness.EXACT` require different audits.
