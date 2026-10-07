# R4 Block 20 — finite Frobenius separator certificate repair

Block 20 is a narrow semantic repair discovered by the Episode 017 deformation audit.

`finite_frobenius_separator(...)` works only with the **supplied finite trace table**.  If its selected trace matrix has full column rank, then the declared constituent multiplicity vector is uniquely recoverable from those rows.  If the rank is smaller than the number of declared constituents, the finite data does not separate them.

The predecessor already computed the Boolean `separating` correctly, but its `Certificate` was hard-coded `valid=True`.  Consequently `certificate.require()` could pass even when `recover(...)` immediately refused because the trace matrix was rank deficient.

Block 20 makes the certificate validity coincide with the finite theorem:

\[
\text{valid}\iff\text{separating}\iff\operatorname{rank}(A)=r,
\]

where `A` is the supplied finite trace matrix and `r` is the constituent count.

The certificate remains `Exactness.EXACT` because matrix rank and recovery are exact finite algebra.  That exactness does **not** promote finite trace data to a global Galois or automorphic identification statement.  The pre-existing unchecked assumption

`Chebotarev density beyond the supplied table`

is retained explicitly.

No stable aggregate API names or signatures are changed.
