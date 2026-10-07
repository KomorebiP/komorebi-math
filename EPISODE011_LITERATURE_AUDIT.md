# Episode 011 Literature Audit

Scope: bounded overlap check for the carrier-change results used in Episode 011.
This is not a global novelty search.

## Sources checked

1. G. L. Litvinov, *The Maslov dequantization, idempotent and tropical mathematics: A brief introduction*, arXiv:math/0507014 (v1, 2005).
   - Explicitly gives the deformation
     `u plus_h v = h log(exp(u/h)+exp(v/h))`
     and its `h -> 0` limit `max(u,v)`.
   - Classification for Block 15: **known verbatim / known after notation translation** for the finite-temperature sum-product to max-plus deformation.

2. Pierre Blanchard, Desmond J. Higham, Nicholas J. Higham, *Accurate Computation of the Log-Sum-Exp and Softmax Functions*, arXiv:1909.03469 (2019).
   - Confirms shifted log-sum-exp as standard numerical machinery and studies its floating-point behavior.
   - Classification: **known verbatim** for the shifted numerical form used by the implementation.

3. Marianne Akian, Stephane Gaubert, Alexander Guterman, *Linear independence over tropical semirings and beyond*, arXiv:0812.3496 (2008/2009).
   - Explicitly revisits symmetrization of max-plus algebra and signed tropical structures.
   - Classification: **adjacent / established machinery** for sign-retaining extensions; Block 15's `SignedMassPacket` is a simpler positive/negative mass lift and is not claimed as a new symmetrized tropical construction.

4. Checked semiring reference/thesis, Section 2.3, Morphisms of Semirings.
   - Gives the nonnegative-real support map to the Boolean semiring as a semiring homomorphism.
   - Gives the explicit `Z -> Boolean` impossibility via `1 + (-1) = 0`.
   - Classification: **known verbatim / known after translation** for the Boolean support and cancellation barrier examples.

## Overlap conclusion

The factorization criterion is standard quotient/factor-homomorphism mathematics.
The cancellation and idempotence barriers are elementary semiring consequences.
Maslov dequantization and log-sum-exp are established. Signed tropical / symmetrized
semiring constructions are established.

Accordingly, Episode 011 does **not** establish new mathematics at the level of
its component theorems.

The only remaining novelty question is architectural: whether explicitly typing
carrier semantics, legal pushdown maps, residual completions, continuation
contracts, and provenance queries in one executable interface yields useful new
cross-domain theorems or algorithms. That remains **unresolved**.
