# Quantum memory and numerical span contracts

The deep audit uses independent trajectories, finite integral references and
changes of generators to check behavior beyond the existing verification suite.

## Schrödinger memory convention

`QuantumFeshbachReduction.self_energy(z)` retains the Hamiltonian convention

\[
\Sigma(z)=H_{PQ}(zI-H_{QQ})^{-1}H_{QP}.
\]

`memory_kernel()` returns the kernel for the derivative equation obtained from
`psi' = -i H psi`. With zero hidden initial state,

\[
\psi_P'(t)=-iH_{PP}\psi_P(t)+\int_0^t K(t-s)\psi_P(s)\,ds,
\qquad K(t)=-H_{PQ}e^{-itH_{QQ}}H_{QP}.
\]

Both coupling maps in the returned `MemoryKernel` carry `-i`. The kernel's
`RationalMemoryRealization` therefore reconstructs the full Schrödinger
trajectory after the retained and hidden coordinates are ordered consistently.
In the open upper half plane, the convention is
`kernel.laplace(-1j*z) = -1j*self_energy(z)`.

For `H=[[0,1],[1,0]]`, the retained amplitude from initial state `[1,0]` is
`cos(t)`. A positive derivative kernel would instead produce `cosh(t)`.
The regression checks the direct matrix exponential and the analytic cosine.

## Infinite and finite Laplace transforms

`MemoryKernel.laplace(z)` requires the sufficient half plane

\[
\operatorname{Re}z>\max\operatorname{Re}\operatorname{spec}(D).
\]

This domain checks the full hidden realization. It is conservative when unstable
hidden modes cancel from `B exp(tD) C`. The eigenvalue and residual checks use
floating point arithmetic and retain numerical certificate status.

`rational_transform(z)` separately returns `B solve(zI-D,C)` whenever the
resolvent exists. It makes no claim that the infinite Laplace integral converges.
For the kernel `exp(t)`, its value at zero is `-1` as an algebraic continuation,
while `laplace(0)` rejects the divergent integral and `certificate(0)` is invalid.

`finite_laplace(z,horizon)` evaluates the finite integral by the upper right
block of an augmented matrix exponential. It does not invert `zI-D`, so it also
works at a resolvent pole. For `K(t)=exp(t)`, `z=1` and horizon `2`, the integral
is `2`. For the corresponding size two Jordan kernel `K(t)=t exp(t)`, it is `2`
as well. The finite integral remains a numerical result, with no rigorous
floating point enclosure implied.

## Constraint span closure

`constraint_closure` preserves its coefficients, residual matrix and maximum
residual in the declared generator coordinates. Its validity decision now uses
the spectral norm of the quotient commutator map in a Hilbert Schmidt
orthonormal basis of the resolved span. A unitary change of orthonormal basis
acts unitarily on the input tensor coordinates, so this norm is invariant.

Multiplying the two Pauli generators `X,Z` by `1e-6` does not close their span.
The previous coordinate threshold accepted that span because its commutators
scaled quadratically. The current span obstruction rejects it at every tested
uniform scale from `1e-100` through `1e100`.

The SVD rank threshold and singular spectrum are certificate diagnostics.
Extremely ill conditioned generating families remain a finite precision rank
boundary. Numerical span resolution does not certify an exact symbolic rank.

## Spectral dimension

`spectral_dimension` cancels the common exponential factor associated with the
smallest eigenvalue before computing the heat trace ratio. It therefore returns
`2000` for the scalar identity at time `1000`, even though the heat trace itself
underflows.

For a finite nonnegative operator the limit at time zero is zero. The large time
limit is zero when a zero eigenvalue is present. A strictly positive operator
instead has asymptotic spectral dimension `2*t*lambda_min`.

## Mutable inputs and finite validation

Lindblad generators and memory kernels snapshot their constructor matrices.
Caller mutation of the original arrays no longer changes the represented model.
The snapshots are read only. Lindblad operations also recheck the declared form
and reject reassigned nonfinite Hamiltonians, jumps or computed outputs before
they can produce a positive certificate.

Shared quantum Hermitian, unitary, projector and density validation rejects
nonfinite or negative tolerances. These are numerical tolerance decisions,
without an exact theorem claim about unrounded real inputs.
