# Metric packets and coordinate covariance

A metric packet distinguishes a bare quotient, a quotient equipped with
geometric metadata, and a numerical compression of that metadata. This guide
describes finite redundant frames and metric-decorated holonomy.

## 1. Redundant analysis--synthesis presentations

Let

\[
A_i:X\to K_i,\qquad S_i:K_i\to X,\qquad S_iA_i=I_X.
\]

The coefficient gauge is \(\ker S_i\), and the quotient
\(K_i/\ker S_i\) represents the physical source space.  The positive
semidefinite coefficient metric

\[
M_i=S_i^*S_i
\]

is exactly the physical metric pulled back to coefficient space:

\[
\langle c,M_i d\rangle=\langle S_i c,S_i d\rangle.
\]

For two presentations of the same physical space, the canonical transition is

\[
T_{ji}=A_jS_i.
\]

It obeys

\[
S_jT_{ji}=S_i,
\qquad
T_{ji}^*M_jT_{ji}=M_i.
\]

Hence the induced quotient map is an exact isometry.  If
\(P_i=A_iS_i\), then

\[
T_{ij}T_{ji}=P_i,
\qquad
T_{ji}T_{ij}=P_j,
\]

so the transitions become genuine inverses after quotienting the coefficient
gauge directions.  They also compose transitively:

\[
T_{kj}T_{ji}=T_{ki}.
\]

These identities require only the left-inverse laws \(S_iA_i=I\).  They do
not require Parseval, orthogonal, or canonical-dual presentations.

## 2. Why Euclidean coefficient conditioning is representation dependent

If the quotient metric is discarded and raw Euclidean coefficient norm is
used instead, the same physical vector can acquire arbitrarily different
coefficient magnitudes under a nonunitary coordinate change.

For an invertible coefficient reparameterization \(G\),

\[
A'=GA,\qquad S'=SG^{-1}
\]

still gives \(S'A'=I\), while

\[
M'=G^{-*}MG^{-1}.
\]

Thus the metric-aware seminorm is unchanged,

\[
\|Gc\|_{M'}=\|c\|_M,
\]

while \(\|Gc\|_2/\|c\|_2\) can be arbitrarily large or small.  Scalar
condition-number envelopes are therefore a compression of richer covariant
metric information, not always the most fundamental packet.

## 3. Full-linear holonomy with transported metrics

Bare full-linear conjugacy data does not generally admit a continuous scalar
that distinguishes every nontrivial holonomy from the identity. Metric metadata
supplies a separate coordinate-covariant regime.  If a positive-definite
Hermitian root metric \(Q\) is part of the represented object and a full-GL
coordinate change acts by

\[
H'=gHg^{-1},
\qquad
Q'=g^{-*}Qg^{-1},
\]

then the metric Hilbert--Schmidt residual

\[
\|H-I\|_{HS,Q}^2
=
\operatorname{tr}\!\left(Q^{-1}(H-I)^*Q(H-I)\right)
\]

is invariant in exact arithmetic:

\[
\|H'-I\|_{HS,Q'}=\|H-I\|_{HS,Q}.
\]

The invariant is a
function of the pair \((H,Q)\), not of the bare conjugacy class \([H]\).
The semantics must therefore distinguish:

1. an active gauge symmetry with a fixed metric;
2. a coordinate reparameterization that transports the metric.

The bare `full_gl` numerical certificate fails closed.
`metric_gluing_certificate` handles the transported-metric regime, and
`transport_root_metric` supplies the required metric transformation.

## 4. Conditioning of the metric packet itself

Metric covariance does not eliminate floating-point conditioning.  From

\[
Q'=g^{-*}Qg^{-1}
\]

one obtains the standard conservative bound

\[
\kappa_2(Q')\le \kappa_2(Q)\,\kappa_2(g)^2.
\]

Therefore a coordinate change can make a previously well-resolved metric
numerically singular.  The implementation rejects positive-definite metrics
whose smallest eigenvalue is not resolved relative to the declared graph
tolerance.

This gives a hierarchy of represented information:

\[
\text{object}
\to
\text{quotient}
\to
\text{task quotient}
\to
\text{covariant geometric metadata}
\to
\text{conditioning envelope}.
\]

The fourth layer can remove coordinate distortion exactly.  The fifth layer
records when the fourth layer itself can no longer be represented reliably at
finite precision.

## 5. Scope boundary

These contracts use finite-dimensional linear algebra and frame geometry:
analysis and synthesis operators, dual frames, frame bounds, quotient spaces,
congruence transport of metrics, and matrix condition numbers. Exact covariance
requires the stated left-inverse identities and metric transport. Numerical
certificates additionally require the metric to be resolved at the declared
tolerance.

See [redundant-frame algebra](REDUNDANT_FRAME_ALGEBRA.md),
[distortion cocycles](DISTORTION_COCYCLES.md), and
[convex packets](CONVEX_PACKETS.md) for related representations.
