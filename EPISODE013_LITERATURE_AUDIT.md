# Episode 013 Literature Audit

## Source question

The quantum audit depends on two external classical facts:

1. what the exact Knill–Laflamme error-correction condition is;
2. whether Kraus labels are representation data rather than channel semantics.

The project-specific block-norm repair is then derived directly.

## Source 1 — Knill and Laflamme

Emanuel Knill and Raymond Laflamme, **A Theory of Quantum Error-Correcting Codes**, arXiv:quant-ph/9604034 (1996).

Relevant statement: Theorem 3.2 gives the necessary and sufficient pairwise conditions on the products \(A_a^\dagger A_b\) restricted to the code. In projector form these are equivalent to

\[
P A_a^\dagger A_b P = c_{ab}P.
\]

Application to this project: because the condition is sesquilinear in \((A_a,A_b)\), validity for a spanning family is equivalent to validity for its entire linear span. This span-invariance is a formal consequence of the displayed condition; it is not being attributed as a separately named theorem in the source.

Classification: **KNOWN AFTER TRANSLATION OF NOTATION**.

## Source 2 — Watrous

John Watrous, **The Theory of Quantum Information**, Chapter 2, Corollary 2.23.

Corollary 2.23 states that if two equal-cardinality Kraus collections define the same completely positive map, then there exists a unitary matrix on the Kraus-index space relating the two collections:

\[
B_a=\sum_b U(a,b)A_b.
\]

Application to this project: a channel-level diagnostic cannot depend on which such Kraus basis is chosen. The predecessor Block 16 channel methods (`apply`, Choi matrix, superoperator, CPTP certificate) already satisfy this. The QEC `maximum_residual` did not.

Classification: **KNOWN VERBATIM** for unitary Kraus equivalence.

## Project derivation

For a Hilbert–Schmidt orthonormal error-span basis \(B_a\), define residual blocks

\[
R_{ab}=P B_a^*B_bP-c_{ab}P.
\]

Under another orthonormal basis \(B'_a=\sum_iU_{ai}B_i\),

\[
R'_{ab}=\sum_{ij}\overline{U}_{ai}U_{bj}R_{ij}.
\]

Thus the full block matrix transforms by unitary similarity:

\[
\mathcal R'=(\overline U\otimes I)\mathcal R(\overline U\otimes I)^*.
\]

Its spectral norm is therefore basis invariant.

This derivation is elementary linear algebra. No claim of novelty is made.

## Search boundary and conclusion

The literature check was targeted, not a global novelty search. It authenticated the classical semantic facts required for the audit and confirmed that the project should treat Kraus/error labels as representation data.

Episode 013 should therefore be described as a **semantic correctness repair grounded in classical quantum information**, not as a new quantum-error-correction theorem.
