# Block 9 release

Version `0.9.0a1` completes the finite-dimensional nonlinear runtime and integrates it with the existing packet, refinement, dynamics, quantum, verification, and export layers.

## Included subsystems

### Derivative providers

The package now exposes one directional derivative protocol for:

* user-declared analytic derivatives
* finite symmetric polynomial tensors
* numerical centered finite differences
* optional exact SymPy expressions
* composed maps evaluated through Faà di Bruno

A provider reports source dimension, target dimension, available orders, exactness, and a value and derivative interface. Declared providers can check permutation symmetry and selected Taylor identities. Numerical finite differences remain numerical evidence rather than proof of differentiability.

### Multilinear packet jets

`PacketJetFamily` resolves an order-`r` derivative into one output packet and `r` input packets. It provides:

* addressable packet blocks
* exact or numerical reconstruction
* symmetry checks
* support and zero-block records
* guarded support enumeration
* finite packet Taylor expansion

Support is complete only when every address was evaluated. Truncated support scans are marked conditional and retain the untested addresses.

### Higher composition

`faa_di_bruno` evaluates the finite directional Faà di Bruno formula by exact set partitions. `packet_faa_di_bruno` additionally resolves intermediate and endpoint packet labels. The result records raw, evaluated, and pruned term counts and can be compared with an independent direct derivative provider.

### Rooted nonlinear trees

The tree runtime provides:

* exact ordered rooted trees with explicit arity constraints
* height and leaf-count enumeration
* Catalan binary families
* finite polynomial fixed-point tree expansions
* a certificate that the finite tree sum equals the corresponding Picard iterate
* packet-labeled tree reconstruction with certified zero-block pruning
* scalar majorant conditions for ball invariance and contraction

Tree enumeration requires an explicit maximum count. Analytic convergence requires declared operation norm bounds and ambient completeness. Those assumptions remain visible.

### Graded nonlinear operations

`FiniteAbelianGroup` and `GradedPacketOperation` implement exact finite group labels, grading checks, packet convolution, Maurer-Cartan packet equations, finite higher-operation truncations, and divided-square identities.

The runtime certifies the finite algebra supplied to it. It does not infer an infinite differential graded or `L_infinity` completion from finite data.

### Correction and elimination

Block 9 adds:

* Hilbert-space obstruction projection through the Moore-Penrose decomposition
* least-norm corrections
* right-inverse contraction correction with visible radius and Lipschitz hypotheses
* target perturbation stability bounds
* Lyapunov-Schmidt kernel, range, and obstruction decomposition
* nonlinear Schur elimination through first and second order
* exact SymPy formulas when both providers are exact
* damped numerical hidden-branch solves otherwise

Local invertibility is checked at the supplied base point. Global continuation is not claimed.

### Filtered lifting and spectral sequences

The finite filtered runtime works over explicit prime fields and includes:

* decreasing finite filtrations
* associated-graded bases and coordinates
* one-step obstruction classes and correction cosets
* compatible finite branch verification
* conditional projective-limit completion records
* finite filtered cochain complexes
* exact `Z_r/B_r` page construction
* induced differential ranks
* page-to-page cohomology checks
* dimensional convergence to total cohomology

Infinite inverse-limit completion is represented by `ConditionalResult`. Completeness, separatedness, continuity, and existence of a compatible infinite branch must be supplied explicitly.

### Shared smooth transport

Moving orthogonal packet transport now lives in `komorebi_math.nonlinear.transport` and is re-exported from `komorebi_math.quantum` for compatibility. Both namespaces expose the same classes.

The shared layer includes:

* direct polar rotations
* derivative validation against centered finite differences
* tangent and differentiated partition identities
* the canonical skew generator
* midpoint exponential transport
* products of direct rotations
* nested-mesh refinement comparisons

Finite endpoint intertwining and unitarity are checked numerically. A two-mesh comparison does not prove an unsampled continuum limit.

## Compatibility

Existing imports remain valid:

```python
from komorebi_math.quantum import KatoTransport, MovingPacketSystem
```

The same objects are now available from:

```python
from komorebi_math.nonlinear import KatoTransport, MovingPacketSystem
```

The quantum re-export is intentional and tested by identity, not implemented as a second copy.

## Verification

The release validation contains:

* 456 unit tests
* 106 verification checks in the full validation environment
* 104 core-only clean-wheel checks when optional SymPy is absent
* 21,890 independent randomized and exhaustive Block 9 audit checks
* exact polynomial composition comparisons through fourth order
* packet-jet reconstruction tests
* rooted-tree and Picard comparisons
* random first- and second-order Schur formula comparisons
* 1,600 random filtered complexes over four prime fields
* smooth transport comparisons against exact planar rotations

The optional GPU validation was not executed in the release container because neither a CUDA device nor CuPy was available. This does not affect the NumPy runtime or the Block 9 nonlinear implementation.

Ruff, mypy, MkDocs, and PowerShell were not installed in the release container. Their commands were therefore not executed. Equivalent import, annotation, docstring, byte-compilation, Markdown-link, and source-integrity audits were run locally; the PowerShell script is included for Windows execution.

## Boundaries

Version `0.9.0a1` implements finite explicit models. It does not claim:

* automatic global analytic continuation
* automatic Banach or Fréchet completeness
* a universal infinite rooted-tree sum
* a general infinite spectral-sequence convergence theorem without hypotheses
* existence of an infinite compatible lift branch
* continuum quantum field transport
* GPU correctness on hardware that was not present during validation
