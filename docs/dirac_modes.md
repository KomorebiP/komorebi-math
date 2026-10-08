# Fixed-mode Dirac evolution

The explicit research API is `komorebi_math.quantum.dirac`. It implements the
four-spinor Clifford coordinate actions and finite free Dirac evolution used in
the Physics publication set of 6 October 2026.

```python
from komorebi_math.quantum.dirac import (
    central_mode_momentum, dirac_symbol, dirac_propagator,
)

q = central_mode_momentum([1, 0, 0], mesh=1 / 12)
H = dirac_symbol(q, mass=1.0)
U = dirac_propagator(q, time=0.4, mass=1.0)
```

The units are `c = hbar = 1`. The torus is `(R/Z)^3` with Fourier phase
`exp(2*pi*i*n.x)`, so continuum physical momentum is `2*pi*n`. The spinor
coordinates have the source order `e0,e1,e2,e3`.

`dirac_matrices()` returns independent read-only snapshots of
`alpha_x,alpha_y,alpha_z,beta`. Their coordinate actions match the original
`Relativistic3DCliffordTransportU1.lean`, read directly from the supplied source
ZIP. Its SHA256 is
`52288e2f60a23a3315ceaa2e1c13ef52d15a25bb64ca07c29dbedb73a415d403`.
The related Block 44 Hilbert stability source SHA256 is
`62e4115ce90bb381a66237201754b75fc36d79684e8bdaea9e5a6cdb580f41b9`.
No new Lean compilation or Python-to-Lean certification is claimed.

`central_mode_momentum(mode,mesh)` accepts a strict integer triple. It computes
`p*sinc(mesh*p/pi)` for `p=2*pi*mode`, with the continuum value at zero mesh.
This form handles the removable zero-mesh limit and small meshes. The central
symbol is periodic in momentum; distinct Fourier labels can have the same
central symbol. There is no Nyquist restriction or claim of recovering labels
from that symbol. No sampled finite-grid Hilbert space is constructed.

`dirac_symbol(momentum,mass)` takes a real physical momentum triple and returns
`H = sum q_j*alpha_j + mass*beta`.

`dirac_propagator(momentum,time,mass)` uses the scalar-square identity
`H^2 = E^2 I`, with `E = hypot(mass,q_x,q_y,q_z)`, to evaluate
`cos(time*E) I - i*sin(time*E) H/E`. At zero energy it returns the identity.
Normalizing the real coefficients before constructing `H/E` handles tiny
positive energies without unstable complex division. Real negative mass, mesh
and time are allowed. Shapes, scalar reality, strict mode labels and finite
values are validated. Unrepresentable energy or phase is rejected.

These are numerical finite-matrix results. They do not certify infinite Fourier
support, an unbounded Dirac domain, growing cutoffs, sampled-grid norms, noisy
parameters or a uniform continuum-limit theorem. The source publication makes
those same distinctions between finite mode stability, ordinary manuscript
proofs and its named checked Lean coverage.

The dedicated tests verify all four Clifford involutions and their
anticommutators, coordinate actions, Hermiticity, scalar-square identities and
unitarity. Seventy-two varied momentum/mass/time cases compare the closed
formula to an independent eigensystem exponential. Further controls cover zero
and tiny energy, zero and tiny mesh, group composition, caller mutation, strict
integer/boolean guards, invalid shapes and finite-range rejection.
