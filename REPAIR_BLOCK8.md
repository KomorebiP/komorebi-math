# Komorebi Math R4 Block 8 — Holonomy numerical gauge contract

Checkpoint: `1.0.0b4-repair-r4-block8`.

## Problem exposed

The exact holonomy criterion is gauge invariant: a fundamental holonomy is the identity if and only if any simultaneous conjugate is the identity. The previous NumPy certificate, however, used the Frobenius residual

\[
\|H-I\|_F
\]

against a fixed tolerance without recording that this quantity is coordinate dependent under unrestricted `GL_n` similarity. For

\[
H_\epsilon=I+\epsilon N,\qquad
D_t=\operatorname{diag}(t,1),
\]

with `N[0,1]=1`, one has

\[
D_t H_\epsilon D_t^{-1}=I+t\epsilon N.
\]

Thus a fixed numerical tolerance can change truth value inside one exact gauge orbit.

## Repair

`TransportGraph.gluing_certificate` now has an explicit `gauge_regime` contract.

- `coordinate` preserves the historical numerical behavior but labels the residual as coordinate dependent and makes no gauge-invariance claim.
- `unitary` uses the same Frobenius residual with the mathematically valid statement that it is invariant under unitary similarity. `gauge_transform(..., gauge_regime="unitary")` rejects nonunitary gauges.
- `full_gl` is accepted by exact backends. A numerical `full_gl` tolerance certificate fails closed with `ValidationError` because no continuous conjugacy-invariant positive distance can separate the identity from every nontrivial full-GL orbit.

The new `gluing_residual_envelope` provides a conditioning-aware alternative for numerical full-GL changes. For a root gauge `g` and residual `E=H-I`,

\[
\kappa_2(g)^{-1}\|E\|_F
\le
\|gEg^{-1}\|_F
\le
\kappa_2(g)\|E\|_F.
\]

The returned `GaugeResidualEnvelope` records the source residual, lower and upper bounds, root, metric, and condition number. It is explicitly a coordinate-sensitivity certificate, not a quotient metric.

## Structural obstruction retained

For the unipotent matrix

\[
H=I+N,\qquad N^2=0,\ N\ne0,
\]

`D_t H D_t^{-1}=I+tN` is conjugate to `H` for every positive `t` and converges to `I` as `t -> 0`. Therefore any continuous conjugacy-invariant scalar `F` satisfies

\[
F(H)=F(I).
\]

The block does not claim to construct a globally stable numerical distance on the full `GL_n` quotient. It makes that impossibility part of the API contract.

## Nonclaims

- A coordinate tolerance is not an exact proof of flatness.
- A unitary-gauge residual is invariant only under the declared unitary gauge class.
- The condition-number envelope does not account for all floating-point rounding error; it is the algebraic similarity bound evaluated with numerical condition estimates.
- This block does not solve canonicalization of arbitrary simultaneous-conjugacy orbits.

## Inherited predecessor regression repaired

The Block 7 predecessor already failed the committed behavioral-golden stream case: the mathematical payload, coefficient count, chunk count, and exact evaluation matched, but the committed `stream_sha256` was stale. The predecessor expected `5d8eb3...385e`; the current deterministic stream is `29469c...816b`. This block updates only that expected stream digest. The same failure was reproduced in the untouched Block 7 predecessor before the fixture was changed, so it is recorded as inherited debt rather than a Block 8 regression.

## Compositional conditioning packet

If two root gauges are applied successively, `g` and then `h`, then

\[
\kappa_2(hg)\le \kappa_2(h)\kappa_2(g).
\]

Therefore a residual envelope composes conservatively as

\[
r\mapsto [r/K,K r],\qquad K=\prod_j\kappa_2(g_j).
\]

For `r>0`, log-residual coordinates turn this into an additive uncertainty packet:

\[
|\log r' - \log r|\le \log K.
\]

A deterministic 3,000-case stress test in dimensions 2, 3, and 4 found zero violations of both the one-step and composed bounds. This computation checks the implementation-level arithmetic only; the inequalities themselves follow from submultiplicativity of the Frobenius and spectral norms.
