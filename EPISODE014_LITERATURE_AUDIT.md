# Episode 014 Literature Audit — Lindblad Representation Gauge

## Source question

Verify the external fact used in Episode 014: the Hamiltonian/jump decomposition of a finite GKSL generator is non-unique, including identity shifts of jump operators with compensating Hamiltonian changes.

## Primary source

Victor V. Albert, Barry Bradlyn, Martin Fraas, Liang Jiang, *Geometry and Response of Lindbladians*, Physical Review X **6**, 041031 (2016), DOI `10.1103/PhysRevX.6.041031`, arXiv:`1512.08079`.

Appendix A, Eq. (A4), states the gauge transformation

\[
F_\ell\mapsto F_\ell+g_\ell I,
\]

with a compensating Hamiltonian term, while leaving the Lindbladian invariant. Their notation carries explicit rates `kappa_l`; Komorebi absorbs square-root rates into its stored jump matrices, so the Episode 014 formula is the corresponding rate-absorbed form.

Classification: **known after notation translation**.

## Secondary source

Patrick Hayden and Jonathan Sorce, *A canonical Hamiltonian for open quantum systems*, arXiv:`2108.08316`.

The paper's abstract explicitly identifies the non-uniqueness of the unitary/dissipative split and characterizes a canonical choice via traceless jump operators.

Classification: **adjacent/source-confirming result**.

## Application to Komorebi

The source result justifies treating `LindbladGenerator.hamiltonian` and `.jumps` as representation coordinates. Generator-level quantities such as the superoperator and generated semigroup are the semantic objects.

The cyclic-order repair itself does not depend on an external theorem: if an operator is supplied as the generator of a representation of `Z_n`, the defining relation is `U^n=I`. The predecessor failed to validate this declared hypothesis.

## Novelty conclusion

- Lindblad gauge freedom: **known verbatim / after notation translation**.
- Cyclic closure requirement: **definitional/classical**.
- Block 18 repair: **project-specific software-mathematics correction**.

No novelty claim is supported for the mathematical ingredients.
