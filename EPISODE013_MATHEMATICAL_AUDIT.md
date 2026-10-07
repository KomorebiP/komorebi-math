# Episode 013 Mathematical Audit — Quantum Kraus/Error-Span Gauge Repair

## Research question

Does the semantic-typing discipline developed in Episodes 001–012 expose a real correctness defect in an independently developed mature Komorebi subsystem?

The selected target was finite quantum channels and quantum error correction, with particular attention to the distinction between:

1. a quantum channel / correctable error *space*;
2. one Kraus or error-operator generating family for that object;
3. numerical diagnostics attached to the generating labels.

## Classical semantic boundary

For a finite error family \(E_a\) and code projector \(P\) of rank \(k\), the Knill–Laflamme condition is

\[
P E_a^* E_b P = c_{ab} P.
\]

The condition is bilinear in the error operators. Consequently it holds for one spanning family iff it holds for every spanning family of the same error space.

For a completely positive map, equal-cardinality Kraus representations of the same map are related by a unitary change of Kraus labels. Thus any diagnostic advertised as a property of the channel or correctable error space must not depend on that harmless label gauge.

## Predecessor defect

Block 16 computed each residual

\[
R_{ab}=P E_a^*E_bP-c_{ab}P
\]

and certified the criterion using

\[
\max_{a,b}\|R_{ab}\|_2.
\]

This is not invariant under unitary changes of error labels.

A default-tolerance counterexample is:

\[
P=I_2,\qquad E=\operatorname{diag}(2\cdot10^{-5},0),
\]

with declared family

\[
(E,0,0,0).
\]

Block 16 reports

\[
\max\|R_{ab}\|_2=2\cdot10^{-10}
\]

and therefore rejects the family at tolerance \(10^{-10}\).

After a normalized \(4\times4\) Hadamard unitary mixes the error labels, every new generator is \(\pm E/2\). The error span and completely positive map are unchanged, but Block 16 reports

\[
\max\|R'_{ab}\|_2=5\cdot10^{-11}
\]

and accepts it.

The predecessor artifact therefore assigned different validity to the same semantic error space.

A second defect followed from the same coordinate dependence: uniformly shrinking an uncorrectable generating family could shrink the absolute residual below tolerance without changing its error span.

## Span-invariant obstruction

Let \(\mathcal E\) be the linear span of the declared errors. Choose any Hilbert–Schmidt orthonormal basis

\[
B_1,\ldots,B_r
\]

of \(\mathcal E\). Define

\[
c_{ab}=\frac1k\operatorname{Tr}(P B_a^*B_bP)
\]

and

\[
R_{ab}=P B_a^*B_bP-c_{ab}P.
\]

Assemble the full block operator

\[
\mathcal R=[R_{ab}]_{a,b=1}^r
\]

on \(\mathbb C^r\otimes\mathcal H\).

Define the numerical obstruction

\[
\boxed{\Omega(\mathcal E,P)=\|\mathcal R\|_2.}
\]

### Basis-invariance theorem

If \(B'_a=\sum_i U_{ai}B_i\) for a unitary matrix \(U\), then

\[
\mathcal R'
=(\overline U\otimes I)\mathcal R(\overline U\otimes I)^*.
\]

Hence

\[
\boxed{\|\mathcal R'\|_2=\|\mathcal R\|_2.}
\]

Any two Hilbert–Schmidt orthonormal bases of the same resolved error span differ by such a unitary, so \(\Omega\) is a property of the error span, not of a declared generating family.

The exact Knill–Laflamme condition holds iff

\[
\Omega=0.
\]

**Status: PROVED.**

## Numerical span contract

The implementation obtains the error span from the column space of vectorized declared errors using SVD. The rank threshold is

\[
\epsilon_{\rm mach}\,\max(d^2,m)\,\sigma_{\max}.
\]

This threshold scales with uniform rescaling of all declared errors and therefore does not create the predecessor's absolute-scale bug.

However, no floating implementation can reliably distinguish arbitrarily ill-conditioned generating families from a lower-dimensional span. Block 17 therefore records in the certificate diagnostics:

- the resolved error-span rank;
- the numerical rank threshold;
- all singular values of the declared generator matrix.

This is a finite-precision contract, not a claim of exact symbolic rank detection.

## Recovery-channel repair

The predecessor's `recovery_channel()` also diagonalized the coefficient matrix in the *declared error coordinates*. Uniform scaling of the correctable three-qubit bit-flip error family by \(10^{-8}\) caused the returned recovery superoperator to change by spectral norm approximately

\[
2.8284271247461903,
\]

because the scaled syndrome eigenvalues fell below the absolute activity threshold.

Block 17 now constructs the recovery from the Hilbert–Schmidt orthonormal basis of the resolved error span. Its syndrome eigenspace threshold is relative to machine precision and the largest canonical coefficient eigenvalue.

Under exact arithmetic, changing the orthonormal span basis only unitary-mixes the syndrome construction, leaving the resulting recovery channel unchanged. Numerically this was tested under unitary gauges, uniform scalings, and moderate invertible generator changes.

The distinct method `recovered_channel_certificate()` still constructs the explicitly declared equal-weight noise channel. That channel is invariant under unitary Kraus mixing, but not under arbitrary nonunitary changes of error generators. The API distinction is intentional.

## Finite computation

The deterministic probe uses seed `20260923`.

- 400 random CPTP channels under Kraus-unitary mixing: 0 semantic failures.
- 400 random error spans under moderate invertible generator changes: 0 invariant-obstruction failures.
- Largest absolute invariant-obstruction discrepancy: `1.2212453270876722e-15`.
- Largest ratio observed between decomposition-sensitive declared-basis maximum residuals: `40.82057756894647`.
- 80 bit-flip recovery tests under random unitary error-label gauges: 0 failures.
- 80 bit-flip recovery tests under moderate general-linear generator changes (maximum condition number about 49.47): 0 failures.
- Uniform generator scalings from `1e-12` to `1e4`: recovery superoperator remained invariant to about `7e-16` in the explicit scaling probe.
- The default-tolerance predecessor counterexample is rejected in both sparse and Hadamard-spread representations after repair, with invariant obstruction approximately `0.5` in each.

These computations verify the implementation in the stated range. The universal invariance statement is supplied by the block-operator proof, not by random sampling.

## Epistemic status

- **PROVED:** exact Knill–Laflamme validity depends only on the error span; the block spectral obstruction is invariant under orthonormal error-span basis changes; exact zero of the obstruction is equivalent to the Knill–Laflamme condition.
- **COMPUTATIONALLY VERIFIED:** the Block 17 implementation respects channel Kraus gauge, error-span generator changes in the tested condition range, and recovery gauge/scaling tests.
- **REFUTED:** the predecessor's largest declared-basis residual is a valid semantic numerical obstruction.
- **REFUTED:** absolute generator magnitude may legitimately determine correctability of a fixed error span.
- **SUPPORTED:** semantic typing is useful as an adversarial audit method; here it exposed and repaired a real mature-subsystem defect.

## Novelty boundary

The Knill–Laflamme criterion, unitary freedom of Kraus representations, error-span invariance, Hilbert–Schmidt orthonormalization, and unitary-invariant block norms are classical ingredients. No novelty claim is made for those pieces.

The contribution of Episode 013 is project-specific: it identified a decomposition-dependent numerical certificate where a semantic invariant was required, supplied a counterexample, repaired both certification and recovery construction, and froze adversarial regression tests.
