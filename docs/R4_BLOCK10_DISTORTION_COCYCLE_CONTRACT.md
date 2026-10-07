# R4 Block 10: distortion cocycles and metric-completion obstructions

This checkpoint records Episode 006 of the independent inquiry.  The block is
research-tier and does not add symbols to the frozen aggregate API.

## 1. Intrinsic anisotropy packet

For an invertible finite-dimensional map

\[
T:(V,Q_s)\longrightarrow(W,Q_t)
\]

with positive-definite Hermitian metrics, choose factors
\(Q_s=R_s^*R_s\) and \(Q_t=R_t^*R_t\).  Define

\[
B_Q(T)=R_tTR_s^{-1}
\]

and

\[
D_Q(T)=\kappa_2(B_Q(T))
       =\frac{\sigma_{\max}(B_Q(T))}{\sigma_{\min}(B_Q(T))}\ge 1.
\]

The singular values of `B_Q(T)` are independent of the chosen metric factors.
If coordinates change by `G_s,G_t` and the metrics are transported
covariantly, the new representative differs by unitary factors.  Therefore
`D_Q(T)` and the geometric-mean volume scale are coordinate invariants of the
metric-decorated map.

The implementation is in `komorebi_math.nonlinear.distortion`.  It is not yet
re-exported by `komorebi_math.nonlinear`.

## 2. Distortion cocycle law

For composable metric maps,

\[
D(T_2T_1)\le D(T_2)D(T_1),
\]

hence

\[
\delta(T)=\log D(T)
\]

is subadditive:

\[
\delta(T_2T_1)\le\delta(T_2)+\delta(T_1).
\]

For a transport path, intermediate metric factors cancel exactly, so the same
law applies edge by edge.

For a closed word with holonomy `H`, every metric satisfies

\[
D_Q(H)\ge
\frac{\max_i |\lambda_i(H)|}{\min_i |\lambda_i(H)|}.
\]

Thus, with

\[
\omega(H)=\log\frac{\max_i|\lambda_i(H)|}{\min_i|\lambda_i(H)|},
\]

any edge distortion budget obeys

\[
\sum_{e\in\gamma} \delta_e \ge \omega(H_\gamma),
\]

where traversal multiplicity is counted without orientation cancellation.
This is a necessary cycle/word obstruction.

## 3. Fundamental cycles alone are insufficient

Let

\[
A=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\qquad
B_t=\begin{pmatrix}0&-t\\1/t&0\end{pmatrix},\quad t>1.
\]

Both `A` and `B_t` have eigenvalues `+i,-i`; each is individually
unitarizable and has spectral distortion lower bound one.  But

\[
AB_t=\operatorname{diag}(-1/t,-t),
\]

so

\[
\omega(AB_t)=2\log t.
\]

Consequently, for every common metric `Q`,

\[
D_Q(A)D_Q(B_t)\ge t^2,
\]

and therefore

\[
\max(D_Q(A),D_Q(B_t))\ge t.
\]

The bound is attained by a metric with coordinate ratio `1/t`, so

\[
\inf_Q\max(D_Q(A),D_Q(B_t))=t.
\]

For `t=9`, the numerical grid search returned `9.005026990980893`; the exact
value is `9`.  This shows that a generating-cycle test can miss an obstruction
that appears only after word composition.

## 4. Conformal metric completion

A conformal metric completion of an invertible transport graph consists of
positive metrics `Q_v` and positive edge length scales `lambda_e` satisfying

\[
T_e^*Q_{t(e)}T_e=\lambda_e^2Q_{s(e)}.
\]

Rescale every vertex metric to determinant one and define

\[
\widehat T_e=|\det T_e|^{-1/n}T_e.
\]

Then the conformal equations are equivalent to

\[
\widehat T_e^*Q_{t(e)}\widehat T_e=Q_{s(e)}.
\]

After spanning-tree propagation this reduces to one root metric `Q` satisfying

\[
\widehat H_\gamma^*Q\widehat H_\gamma=Q
\]

for every closed word.  Therefore:

> **Conformal metric-completion criterion.** A finite invertible transport graph
> admits a conformal positive-definite metric completion if and only if its
> determinant-normalized holonomy representation is unitarizable. Equivalently,
> its normalized holonomy image is conjugate into a unitary group; in finite
> dimensions this is equivalent to relative compactness of that image.

The equivalence with relative compactness uses the classical Haar-averaging
construction for a compact group.  This theorem is a finite-graph organization
of established representation-theoretic machinery, not a novelty claim.

## 5. Metric-completion trichotomy

Normalize candidate root metrics by `det Q = 1`, and for a finite set of
normalized holonomy generators define

\[
F(Q)=\max_j\log D_Q(H_j),\qquad
\Delta_*=\inf_{Q>0,\det Q=1}F(Q).
\]

There are three operational cases.

1. **Completion:** `Delta_* = 0` and the infimum is attained.  Then the
   normalized representation preserves a positive metric.
2. **Gap obstruction:** `Delta_* > 0`.  Some anisotropy remains under every
   positive metric.
3. **Escape obstruction:** `Delta_* = 0` but the infimum is not attained.  Every
   sequence `Q_k` with `F(Q_k) -> 0` must have `cond(Q_k) -> infinity` along a
   subsequence, and in fact no such near-minimizing sequence can have uniformly
   bounded condition number.

The third claim is a compactness theorem.  If `det Q_k=1` and `cond(Q_k)` were
uniformly bounded, all eigenvalues would be bounded above and away from zero.
A convergent subsequence would then limit to a positive-definite `Q_*`; by
continuity `F(Q_*)=0`, contradicting nonattainment.

This turns numerical conditioning into an obstruction certificate: a system may
be arbitrarily close to metric completion only by sending its metric packet to
the boundary of the positive cone.

### Unipotent model

For

\[
H=\begin{pmatrix}1&1\\0&1\end{pmatrix},
\]

all eigenvalues have modulus one, so the spectral lower bound is exactly one.
There is nevertheless no positive-definite invariant metric.  Taking

\[
Q_s=\operatorname{diag}(s^2,1)
\]

gives

\[
D_{Q_s}(H)\downarrow1\quad(s\downarrow0),
\qquad
\kappa(Q_s)=s^{-2}\to\infty.
\]

The Episode 006 probe numerically observed

\[
(D_{Q_s}(H)-1)\sqrt{\kappa(Q_s)}\to1.
\]

Thus the unipotent obstruction is a zero-gap/nonattainment obstruction rather
than a positive spectral gap.

## 6. First nonlinear pressure test: curvature is not the obstruction

The tentative nonlinear hypothesis "curvature is precisely the obstruction to
metric completion" is false.

On a trivial real rank-three bundle over a planar base, use the constant
connection matrices

\[
A_x=\begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix},\qquad
A_y=\begin{pmatrix}0&0&0\\0&0&-1\\0&1&0\end{pmatrix}.
\]

Both are skew-symmetric, so the Euclidean metric is parallel.  Yet

\[
F_{xy}=[A_x,A_y]
=\begin{pmatrix}0&0&1\\0&0&0\\-1&0&0\end{pmatrix}\ne0.
\]

A finite rectangular loop at side length `0.2` produced nontrivial holonomy
with Frobenius distance `0.05637743157883952` from the identity, while its
orthogonality residual was only `4.624480671433126e-17`.

So nonzero curvature and nontrivial holonomy are compatible with exact metric
transport.  For a fixed connection the relevant question is instead whether
its holonomy preserves a positive metric.  Curvature contributes to the
holonomy algebra, but flatness is much stronger than metric compatibility.

## 7. Verification

`tests/test_r4_block10_distortion_cocycle.py` freezes five deterministic laws:

- coordinate covariance of metric distortion;
- submultiplicativity under composition;
- the exact two-generator mixed-word minimax example;
- the unipotent zero-gap/nonattainment model;
- determinant normalization preserving anisotropy.

`verification/episode006_distortion_probe.py` adds a deterministic 2,000-case
stress run.  It recorded zero covariance failures and zero composition failures.
The maximum observed composition ratio was `0.9999947164410563`, and the maximum
relative coordinate-covariance error was `8.588808363432907e-10`.

## 8. Boundary and nonclaims

- The finite graph criterion is not claimed as new representation theory.
- Spectral word inequalities are necessary, not sufficient, for completion.
- Checking only fundamental holonomy generators is not sufficient; mixed words
  can expose obstructions.
- `Delta_* = 0` does not imply an invariant metric unless the infimum is attained.
- The research module does not yet solve the global optimization problem for
  `Delta_*`.
- The nonlinear extension does not identify curvature with the obstruction;
  its next target is holonomy-compatible parallel metric completion.
