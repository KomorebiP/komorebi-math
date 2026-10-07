# Komorebi Math R4 Block 10 — distortion cocycle and metric-completion trichotomy

Checkpoint: `1.0.0b4-repair-r4-block10`.

## Scope

This repair/research block extends the Episode 005 metric-packet model without
changing the frozen stable API.  The new module
`komorebi_math.nonlinear.distortion` remains research-tier and is not aggregate
re-exported.

## Added

- intrinsic metric anisotropy packets for invertible finite maps;
- coordinate-covariant metric representatives;
- determinant normalization separating volume scale from anisotropy;
- conjugacy-invariant spectral lower bounds for closed-word distortion;
- conservative multiplicative distortion composition;
- tests for mixed-word obstruction and unipotent nonattainment;
- an Episode 006 deterministic 2,000-case stress probe;
- documentation of the conformal metric-completion criterion and the
  completion/gap/escape trichotomy.

## Main correction

Episode 005 asked whether scalar distortion could lift to exact geometry.  The
answer has two qualitatively different negative modes.  Some systems have a
strict positive distortion gap.  Others, such as a nontrivial unipotent
holonomy, have infimum distortion one but achieve it only at the boundary of
the positive-metric cone; any near-completing metric becomes arbitrarily ill
conditioned.

## Nonlinear pressure test

Curvature was tested as a candidate obstruction and rejected.  A metric
connection can have nonzero curvature and nontrivial holonomy while preserving
the metric exactly.  The next nonlinear target is therefore parallel-metric /
holonomy compatibility rather than curvature vanishing.
