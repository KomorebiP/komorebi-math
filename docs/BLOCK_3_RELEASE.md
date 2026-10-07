# Block 3 release record

Version `0.3.0a1` implements the structured computational layer associated with Paper VIII.

The executable path is now:

```text
seed
  -> address
  -> operate
  -> project
  -> transfer
  -> refine
  -> constrain
  -> choose representation
  -> schedule
  -> finite reconstruct
```

The new choice and scheduling stages are finite computational realizations. They do not replace the mathematical objects from Blocks 1 and 2.

## Packet intermediate representation

`PacketIR` stores an operator as a finite mapping from source packet addresses to target packet addresses. Every block is a `StructuredOperator`.

```python
layout = PacketLayout((2, 1))
ir = PacketIR.square(layout, {(0, 0): A00, (1, 0): A10})
```

Addresses use `(target, source)` order. The representation supports application, materialization, adjoints, composition, powers, tensor products, packet diagonal extraction, storage counts, and modeled operation counts.

## Structured primitives

The release includes:

* explicit dense matrices
* scalar identities
* diagonal matrices
* block diagonal operators
* circulant operators
* rectangular Toeplitz operators
* two factor Kronecker products
* normalized and unnormalized Walsh Hadamard transforms

Every class implements one common interface for shape, dtype, application, materialization, adjoint, storage, and operation count.

## Low rank algebra

`LowRankOperator` represents

```text
left @ right.conj().T
```

without forming the dense matrix.

`BlockDiagonalLowRankOperator` represents a square block diagonal operator plus one low rank correction. It provides:

* direct application
* Woodbury solves
* exact products before recompression
* optimal recompression of the correction
* explicit storage and operation models

`recompress_core` handles rectangular small cores and rank deficient factors. Its certificate compares the actual residual with the singular value tail predicted by the rank constrained approximation theorem.

## Support algebra

The support module treats a packet block pattern as a directed relation. It computes:

* support products
* support powers
* polynomial support inclusions
* transitive closure
* inverse support for acyclic unit diagonal patterns
* Schur complement fill through hidden packets
* topological levels
* longest path depth

These are structural inclusions. Exact matrix blocks can cancel even when a support path exists.

## Krylov methods

The package implements:

* modified Gram Schmidt Krylov bases
* preconditioned conjugate gradient
* fully reorthogonalized Lanczos tridiagonalization
* packet block Jacobi construction
* normalized coupling and condition bounds
* power perturbation bounds

Conjugate gradient checks curvature on generated directions. A successful run does not by itself prove global positive definiteness of a callable operator.

## Precision planning

The continuous precision model allocates tolerances under a weighted error budget. The clipped model enforces packetwise maximum tolerances.

The discrete model accepts a finite option list for every packet:

```python
DTypeChoice(name, cost, error_units, dtype)
```

Dynamic programming returns the exact minimum cost choice whose additive integer error does not exceed the budget.

## Exact tree compiler

A `TreeNode` has one keep option and, when it has children, one split option. Costs and errors are additive over selected leaves.

The release provides:

* exact Lagrangian compilation
* exact additive integer error budget compilation
* the nondominated cost and error frontier
* weighted shared objectives for several operators
* local decision margins
* immutable local tree updates

Every node name must be unique throughout the tree. Terminal nodes cannot be split.

The budget compiler converts each leaf error independently to an integer weight before adding it. This is the model proved by the finite dynamic program.

## Scheduling and amortization

`Task` and `schedule_unlimited` implement earliest start scheduling for a finite acyclic task graph under unlimited concurrency. The reported makespan is the weighted critical path length.

Shape classes give an ideal batched launch count only when the backend can actually execute the declared batch representation.

The amortization model returns the smallest positive reuse count that strictly repays setup cost.

## Cost and benchmark contracts

The release keeps separate:

* arithmetic work
* stored scalar count
* one pass data movement
* arithmetic intensity
* roofline lower bound
* setup time
* kernel time
* device resident time
* end to end time
* output contract

A benchmark comparison is rejected when the case signature, output contract, or timing scope differs.

## Restricted discovery

`discover_packet_support` thresholds blocks in a fixed layout. It returns a heuristic certificate and the norm of the discarded matrix.

`discover_block_diagonal_low_rank` fixes the packet diagonal exactly and computes an optimal singular value truncation of the remaining matrix.

The release does not claim a universal structure discovery algorithm.

## Device execution

The NumPy packet executor is always available.

The CuPy executor is optional and imports CuPy only when requested. It transfers every block to the device when the executor is constructed. Its application method can accept device input and can leave output on the device.

No CuPy device was available during this build. The release therefore verifies the availability boundary and CPU semantics, but does not claim a hardware GPU benchmark.

## Verification

Block 3 is accepted only after:

* all previous Block 1 and Block 2 tests remain green
* every structured operator agrees with dense application on deterministic and randomized cases
* packet IR composition agrees with dense composition
* recompression errors agree with singular value tails
* Woodbury solves agree with dense solves
* all strict directed graphs on four vertices agree with an independent closure calculation
* random Schur fill patterns agree with explicit hidden path enumeration
* conjugate gradient solves random positive definite systems
* discrete precision assignments agree with exhaustive choice enumeration
* tree compilers agree with exhaustive pruning enumeration
* task schedules agree with independent critical path calculations
* discovery residuals agree with dense residual norms
* a clean wheel installs outside the source tree
* the source distribution rebuilds to the same installable package
