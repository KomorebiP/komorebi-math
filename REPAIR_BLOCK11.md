# Komorebi Math R4 Block 11 — convex packets and ellipsoidal compression

Checkpoint: `1.0.0b4-repair-r4-block11`.

## Scope

This research block tests the Episode 006 conjecture that quadratic metric
packets may be too restrictive.  The answer is task-dependent.

## Added

- research-tier `komorebi_math.nonlinear.convex_packets`;
- parallelotope norm packets `||Fx||_inf` with coordinate-covariant transport;
- one-sided quadratic operator norms for singular/projector maps;
- explicit John ellipsoid metric and `sqrt(n)` compression certificate;
- the sharp Ando--Shih convex-versus-quadratic separation;
- unipotent convex escape tests;
- deterministic 2,000-case convex packet stress probe;
- documentation distinguishing exact group isometry from forward-semigroup
  nonexpansivity.

## Main correction

Arbitrary Finsler norms do **not** enlarge exact invertible isometric/conformal
holonomy completion: an invariant norm has a compact linear isometry group and
therefore an averaged invariant inner product.

They **do** enlarge the exact packet class for one-sided semigroup problems.
The Block 11 two-matrix example is exactly nonexpansive in `l_infinity` but the
best common quadratic bound is `sqrt(2)`.  This saturates John's two-dimensional
ellipsoid approximation factor.

A quadratic positive gap must therefore be described as relative to the
quadratic packet class, not automatically as an absolute obstruction.
