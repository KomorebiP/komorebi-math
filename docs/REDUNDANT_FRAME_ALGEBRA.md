# Redundant frame algebra

Block 18 closes the finite algebra of a general analysis--synthesis system.
It does not assume that analysis coordinates form an orthonormal basis.

Let a finite presentation be a pair

\[
A_X:X\longrightarrow K_X,
\qquad
S_X:K_X\longrightarrow X,
\qquad
S_XA_X=I_X.
\]

The coefficient-space identity is the generally oblique projection

\[
P_X=A_XS_X,
\qquad P_X^2=P_X.
\]

Every coefficient vector has the canonical gauge decomposition

\[
c=P_Xc+(I-P_X)c,
\qquad (I-P_X)c\in\ker S_X.
\]

Two coefficient vectors represent the same physical vector exactly when their
difference lies in \(\ker S_X\).  The positive semidefinite coefficient metric

\[
H_X=S_X^*S_X
\]

measures this quotient faithfully: \(\langle c,H_Xd\rangle=
\langle S_Xc,S_Xd\rangle\), and gauge vectors have zero seminorm.

For another presentation \((A_Y,S_Y)\) of the same physical space, the
canonical bridge \(R=A_YS_X\) is an exact quotient isometry:

\[
R^*H_YR=H_X.
\]

Moreover the reverse bridge composes with \(R\) to the corresponding oblique
coefficient projection.  Thus the bridges become genuine mutually inverse
isometries after quotienting by the synthesis kernels, even though raw
Euclidean coefficient norms can change drastically between presentations.

## Maps and closure

For a physical operator \(T:X\to Y\), its canonical coefficient lift is

\[
\widehat T=A_YTS_X.
\]

It satisfies

\[
P_Y\widehat T P_X=\widehat T,
\qquad
S_Y\widehat T A_X=T.
\]

Consequently, physical maps are in bijection with the sandwiched coefficient
matrices \(P_Y\operatorname{Mat}(K_Y,K_X)P_X\).  The inverse map is
\(L\mapsto S_YLA_X\).

If two maps use the same middle presentation, their lifts compose by ordinary
matrix multiplication.  If the output presentation of the first map and the
input presentation of the second map are different presentations of the same
physical space, the bridge

\[
R=A_{Y_\mathrm{in}}S_{Y_\mathrm{out}}
\]

is required:

\[
\widehat{U\circ T}=\widehat U\,R\,\widehat T.
\]

The lifted physical identity is \(P_X\), not the full identity on redundant
coefficient space.

## Coefficient blocks versus physical components

For a coefficient partition with selectors \(E_\alpha\), the small blocks

\[
E_\beta\widehat T E_\alpha
\]

assemble to \(\widehat T\) and obey ordinary block-matrix multiplication when
the middle partition is aligned.  These are the Block 18 compositional blocks.

The older physical components

\[
C_\beta T C_\alpha,
\qquad C_\alpha=S E_\alpha A,
\]

still reconstruct \(T\), but in a redundant or oblique system the
\(C_\alpha\) need not be orthogonal projectors.  Their tempting one-index
multiplication rule is therefore false in general.  The verifier includes an
explicit oblique counterexample so that the two block notions cannot be
silently conflated.

## Adjoints

The lift of the physical adjoint is

\[
\widehat{T^*}=A_XT^*S_Y.
\]

It is related to, but generally different from, the raw conjugate transpose of
the coefficient lift:

\[
\widehat T^*H_Y=H_X\widehat{T^*}.
\]

Raw conjugate transpose becomes the correct lift after passing to the dual
presentations \((S^*,A^*)\).  In Parseval presentations the distinction
disappears.

## Exact and numerical semantics

The numerical layer carries tolerance-aware certificates and immutable NumPy
matrices.  The exact layer uses immutable `Fraction` matrices and checks every
identity by equality, with `Exactness.EXACT` certificates.  Floating-point
inputs are intentionally not treated as exact rational data.

This block is finite-dimensional.  It does not claim infinite frame
completion, measurable direct integrals, or analytic convergence of an
infinite expansion.
