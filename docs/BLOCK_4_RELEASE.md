# Block 4 release

## Release identity

```text
Distribution: komorebi-math
Version:      0.4.0a1
```

Block 4 is the first domain realization layer. It implements the finite signal theory of Paper V and the principal finite dynamics theory of Paper VI on top of the operator, refinement, obstruction, and structured computation foundations from Blocks 1 through 3.

## Executable path

```text
seed
  -> address
  -> operate
  -> project
  -> transfer
  -> refine
  -> constrain
  -> choose representation
  -> evolve or reconstruct
  -> certify
```

## Signal implementation

### Polyphase coordinates

The package supports finite one dimensional residue channels, rectangular lattice channels, nested mixed radix addresses, exact analysis and synthesis, full shaped residue projections, causal infinite polyphase kernels, and exact finite convolution block matrices.

The infinite kernel object and the finite block matrix object are deliberately separate.

```python
from komorebi_math.signals import (
    polyphase_filter_kernels,
    finite_polyphase_blocks,
)
```

`polyphase_filter_kernels` represents the causal zero extended law

```text
h[N m + target - source]
```

while `finite_polyphase_blocks` includes the declared finite boundary convention.

### Finite convolution conventions

The package treats the first argument as the signal and the second as the kernel.

* `full` returns length `n + k - 1`
* `same` returns exactly `n` samples using the centered slice beginning at `(k - 1) // 2`
* `valid` returns `max(n - k + 1, 0)` samples
* `circular` returns exactly `n` samples

This differs from NumPy's symmetric argument convention when the kernel is longer than the signal. The declared convention is stable under the packet transfer API and is tested directly.

### Filter banks and packet trees

Implemented objects include periodic analysis and synthesis banks, finite bank matrices, frame bounds, canonical dual synthesis, Haar analysis and synthesis, periodic five three lifting, complete Haar packet trees, arbitrary valid prunings, energy identities, and exact additive best basis selection.

### Singular packets

The package implements truncated power terms, finite singular packet sums, exact retained packet reconstruction identities, packet estimation error decompositions, overlap matrix errors, scalar region errors, and exact order zero location errors on finite intervals.

### Stable jump recovery

The deterministic theorem is available for scalar and vector samples. The result records detected cells, midpoint locations, amplitude estimates, location and amplitude bounds, and a conditional certificate.

The program checks the finite threshold inequalities. It cannot infer the unknown Lipschitz bound, noise bound, minimum jump magnitude, or one jump per cell hypothesis from the samples alone.

The sub Gaussian interface uses a coordinatewise simultaneous bound and converts it to a Euclidean vector bound. Independence and centered noise remain explicit unchecked assumptions.

### Nonlinear and Smith tools

The signal namespace includes exact residue packet products for flat coefficient sequences and compact packet channels. Compact channel multiplication propagates the mixed radix carry. It also includes shifted polynomial signal interactions, finite difference break bands, Fourier low pass approximations, a heuristic positive contrast estimator, and cyclic Smith and Cayley coordinates.

## Dynamics implementation

### Matrix and packet semigroups

`MatrixSemigroup` uses an optional SciPy exponential when available. The base installation contains a NumPy scaling and Taylor fallback that can be forced with

```python
matrix_exponential(matrix, method="taylor")
```

Negative evolution time is rejected. Packet semigroups expose time dependent transfer blocks and finite path support.

### Refinement defects

The package implements direct semigroup defects, Duhamel propagation, resolvent defects, norm bounds, finite multilevel towers, and bounded projective tower diagnostics.

### Markov lumping

The column probability convention is used throughout. Continuous generators and discrete transition matrices are declared separately. Exact lumpability is tested by the kernel obstruction and by coordinate rate equality. Exact coarse operators, approximate coarse operators, stationary distributions, and evolution comparison are available.

### Conditional expectation and martingales

Finite weighted conditional expectation is implemented as an orthogonal projector. Nested partitions give martingale difference packets, orthogonality certificates, and energy decompositions.

### Bayesian models

The package implements normalized updates, evidence, perturbation bounds, effective block likelihoods, exact two stage coarsening, fixed sufficiency, relative entropy decomposition, replicator flows, and finite hidden Markov filters.

`coarsest_sufficient_partition` is exact and requires identical likelihood profiles. A positive numerical tolerance is intentionally rejected because closeness is not an equivalence relation in general.

`approximate_likelihood_partition` is a separate numerical clustering utility. It does not certify sufficiency.

### Fractional and Green operators

Positive Hermitian fractional powers, fractional semigroups, exact naturality, positive gap defect bounds, and exact finite Dirichlet Green matrices are implemented.

### Feynman Kac transfer

Continuous finite state killing uses the backward generator

```text
Q.T - diag(V)
```

Discrete transfer uses the declared current state killing convention

```text
diag(exp(-V)) @ P.T
```

Exact coarse models require both exact Markov lumpability and a block constant potential.

### Feshbach reduction and memory

The package implements finite Schur Feshbach effective operators, self energies, projected resolvent certificates, exact hidden state memory kernels, rational auxiliary realizations, finite horizon kernel error bounds, and Volterra trajectory bounds.

### Survival

`PhaseTypeSurvival` requires a strictly stable transient subgenerator. It provides survival, density, hazard, Laplace transform, moments, residual life models, and competing risk packets.

## Execution discipline

Every substantial result is labeled as exact, numerical, conditional, or heuristic.

* Packet and finite matrix identities are checked by residuals unless an exact backend is used
* Recovery theorems retain unknown model assumptions in their certificates
* Approximate likelihood clustering is not promoted to exact sufficiency
* Finite Feynman Kac and semigroup objects do not claim a continuum field theory
* Memory quadrature for arbitrary callables remains conditional
* No GPU execution claim is made in Block 4

## Validation

The Block 4 release requires all of the following before packaging.

* the complete inherited unit suite
* installed package verification for all four blocks
* deterministic randomized signal and dynamics checks
* public API import checks
* full public type and docstring coverage for signals and dynamics
* clean wheel installation outside the source tree
* clean source archive installation
* executable example checks
* deterministic wheel and source archive reproduction
* package payload audit
* Git repository and manifest validation

## Remaining boundary

The following are not part of Block 4.

* automatic universal edge discovery
* general two dimensional geometry recovery theorems
* unbounded semigroup generator limits
* continuum stochastic or field theoretic constructions
* packet jets and nonlinear rooted tree expansions
* quantum packet calculus
* analytic reciprocal families
* arithmetic, deformation, and Langlands realizations
