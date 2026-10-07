# Episode 017 literature audit

The repaired statement is finite-dimensional linear algebra and does not require an external theorem: a finite trace matrix separates arbitrary constituent multiplicities exactly when it has full column rank.

The global step is deliberately different.  Standard Galois-representation arguments use density of Frobenius conjugacy classes together with semisimplicity/character theory to pass from Frobenius trace data on a sufficiently rich set of places to representation-level identification.  Episode 017 does not implement or certify that implication.

A checked reference is Bjorn Poonen, *A p-adic approach to rational points on curves*, where Corollary 4.6/Theorem 4.7 use finitely many Frobenius traces under explicit hypotheses and note that semisimple characteristic-zero representations are determined by trace.  The package's existing unchecked-assumption string — `Chebotarev density beyond the supplied table` — is therefore retained rather than silently discharged.

Novelty classification: the rank criterion is known elementary linear algebra; the global Frobenius/trace principle is established arithmetic geometry.  Episode 017 is a project-specific certificate repair, not a new deformation or Galois theorem.
