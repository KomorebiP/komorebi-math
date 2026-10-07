# Episode 014 Mathematical Audit — Lindblad Gauge and Cyclic-Closure Repair

## Research question

Does the semantic-versus-representation discipline that repaired quantum error correction in Episode 013 expose a second defect in the finite Lindblad/open-system subsystem?

The audit separated:

1. the Lindblad generator as a superoperator;
2. one Hamiltonian/jump-operator representation of that generator;
3. cyclic-symmetry data used to classify coherence sectors.

## Lindblad representation gauge

For

\[
\mathcal L(\rho)=-i[H,\rho]+\sum_j\left(L_j\rho L_j^\dagger-\frac12\{L_j^\dagger L_j,\rho\}\right),
\]

unitary mixing of the jumps,

\[
\widetilde L_a=\sum_jU_{aj}L_j,
\]

leaves the dissipator unchanged.

There is also the inhomogeneous gauge

\[
L'_a=\widetilde L_a+c_a I,
\]

with compensating Hamiltonian

\[
H'=H+\frac{1}{2i}\sum_a
\left(c_a^*\widetilde L_a-c_a\widetilde L_a^\dagger\right),
\]

up to an irrelevant real scalar multiple of the identity. Direct expansion shows

\[
\mathcal L_{H',L'}=\mathcal L_{H,L}.
\]

The current Komorebi `apply`, `superoperator`, `evolve`, trace-preservation certificate, and coherence-commutation calculation are generator-level operations and survived this audit.

The deterministic random probe used 800 gauge transformations in dimensions 2–4 with 1–4 jump operators. It recorded zero generator-semantic failures. Maximum relative discrepancies were about `8.72e-16` for the superoperator and `1.47e-15` for direct application.

A deliberately large identity shift changed the stored Hamiltonian by operator norm about `1.79` and made formerly traceless jumps have traces `(7.4+4.0j)` and `(-3.0+6.2j)`, while the superoperator changed only by about `1.76e-15`. Thus the stored Hamiltonian and individual jumps are representation coordinates, not semantic invariants.

**Status: PROVED for the gauge identity; COMPUTATIONALLY VERIFIED for the implementation in the stated finite range.**

## Predecessor cyclic-order defect

`LindbladGenerator.coherence_preservation_certificate(symmetry, order)` accepted an `order` argument but never checked

\[
U^{\text{order}}=I.
\]

The same omission existed in `charge_certificate`.

Take the zero Lindblad generator and

\[
U=\operatorname{diag}(1,e^{i\pi/3}).
\]

Declare `order=2`. The generator commutes with every conjugation, so the predecessor reports a zero coherence commutator and returns a valid certificate. But

\[
\|U^2-I\|_2=\sqrt3\approx1.73205,
\]

so this is not a representation of the declared cyclic action `Z_2`.

Likewise the zero operator has zero charge-relation residual for any phase convention, so the predecessor `charge_certificate(U,0,0,2)` also returns valid despite the false cyclic premise.

The untouched Block 17 predecessor counterexample is preserved in `EPISODE014_PREDECESSOR_COUNTEREXAMPLE.json`.

**Status: REFUTED predecessor validation contract.**

## Repair

Both certificates now compute

\[
r_{\rm cyc}=\|U^n-I\|_2
\]

and require cyclic closure in addition to their original conclusion residual.

For numerical robustness the closure tolerance is

\[
\varepsilon_{\rm cyc}=n\,\varepsilon
\]

for `n>=1`, reflecting accumulation through a finite matrix power. The original conclusion retains its own scale-sensitive tolerance. Certificate validity requires both tests independently; the public `tolerance` field records the larger display tolerance while the two individual thresholds are retained in diagnostics.

The repair intentionally requires only

\[
U^n=I,
\]

not that `n` be the minimal group-theoretic order of `U`. Non-faithful representations of `Z_n` are valid cyclic actions.

The error message in `cyclic_symmetry_projectors` was also clarified from “has the declared finite order” to the precise contract “satisfies U^order = I”. Its mathematical behavior was already correct.

## Finite computation

The deterministic Episode 014 probe uses seed `20260923`.

- 800 random Lindblad gauge transformations: 0 superoperator failures, 0 direct-application failures, 0 trace-certificate failures.
- 400 valid finite-cyclic samples: 0 correct-order failures.
- 397 sampled genuine wrong-order cases: 0 false passes in the repaired Lindblad coherence certificate and 0 false passes in the repaired charge certificate.
- Maximum correct-order closure residual: approximately `4.84e-15`.

These tests verify implementation behavior in the sampled range. The universal gauge statement follows from the algebraic expansion; the cyclic-closure requirement is definitional.

## Epistemic status

- **PROVED:** unitary jump mixing and identity-shift/Hamiltonian compensation preserve the Lindblad generator; a declared finite cyclic action must satisfy `U^n=I`.
- **COMPUTATIONALLY VERIFIED:** current generator-level methods are gauge invariant in the tested range; repaired cyclic certificates reject sampled false declarations.
- **REFUTED:** the Block 17 coherence and charge certificates fully checked their advertised finite-cyclic premise.
- **SUPPORTED:** the semantic-typing audit has predictive value inside the quantum subsystem, but the second defect is an assumption-validation error rather than a new quantum theorem.

## Novelty boundary

The non-uniqueness of Lindblad representations is classical. Albert, Bradlyn, Fraas, and Jiang explicitly give the identity-shift gauge in Appendix A of *Geometry and Response of Lindbladians*, Phys. Rev. X 6, 041031 (2016), DOI `10.1103/PhysRevX.6.041031`. Hayden and Sorce, arXiv:`2108.08316`, discuss the non-unique Hamiltonian/dissipator split and canonical traceless-jump gauge.

No novelty claim is made for the gauge theorem or the cyclic-closure condition. Episode 014 is a project-specific correctness and contract repair.
