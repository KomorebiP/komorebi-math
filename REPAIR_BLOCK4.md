# Komorebi Math R4 Block 4

Checkpoint: `1.0.0b4-repair-r4-block4`. Wheel build: `4b4`.

This scoped checkpoint repairs analytic-guarantee boundaries in right-inverse correction,
Lyapunov--Schmidt reduction, nonlinear Schur elimination, and fixed-point majorant seed
norm handling. It is not final R4.

Key rules:

- an invariant ball is not enlarged by numerical tolerance;
- tiny nonzero residuals use stable hypot reduction rather than squared norms;
- nonfinite linear/Jacobian data are rejected before factorization;
- tolerance scaling that overflows is rejected;
- a finite Newton/Picard residual is recorded as numerical evidence, not an exact
  existence or branch theorem;
- numerical Lyapunov--Schmidt and Schur records keep exact-root/local-branch premises
  explicit as unchecked assumptions;
- an explicitly supplied seed norm must dominate the norm of the stored seed.

The 131-name stable API is not intentionally changed. See the outer Block 4 repair report
for evidence and remaining scope.
