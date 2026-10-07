# Unified executable language

Komorebi Math uses one sequence of operations across its mathematical domains:

```text
seed
  -> address
  -> operate
  -> project
  -> transfer
  -> refine
  -> constrain
  -> complete
  -> reconstruct
```

This is an execution language, not an assertion that every domain contains the same objects. A signal packet, a module summand, a symmetry sector, and a residual deformation class have different meanings. They share an interface only when the corresponding operations and identities are preserved.

## 1. Seed

A seed is finite generating data for an operator action.

```python
SeedSystem(operators, seeds)
```

Implemented operations include orbit hulls, generation certificates, finite free presentations, semisimple seed rank, rational single operator seed rank, and finite seed searches.

## 2. Address

An address names a legal operator path or packet coordinate.

```python
OperatorGrammar
residue_analyze(values, modulus)
crt_many(residues, moduli)
polyphase_analyze(signal, order)
```

Typed paths, residue addresses, Chinese remainder addresses, mixed radix signal addresses, and packet tree paths are implemented.

## 3. Operate

```python
OperatorRealization(grammar, spaces, operators)
OperatorAlgebra(generators)
StructuredOperator
MatrixSemigroup(generator)
```

Relations are checked exactly with an exact backend or numerically with a residual and tolerance. Structured operators and finite semigroups provide executable domain realizations.

## 4. Project

```python
PacketSystem(projectors)
PacketIR.from_dense(matrix, layout)
ConditionalExpectation(weights, partition)
```

Projection covers algebraic packets, structured computational blocks, signal phases, and finite conditional expectation packets.

## 5. Transfer

```python
PacketOperator
TransferGraph
PolyphaseConvolution
PacketSemigroup
```

Transfer blocks reconstruct operators, compose by packet matrix multiplication, propagate through paths, and specialize to convolution, semigroup, Markov, and memory systems.

## Computational choice

Blocks 1 through 3 implement packet intermediate representations, structured operators, low rank recompression, support propagation, Krylov methods, precision allocation, exact finite tree compilation, scheduling, and optional CuPy execution.

The compiler is exact for the supplied finite additive tree. A precision assignment is exact for the supplied integer budget. Threshold structure discovery remains heuristic. Cost and roofline results are declared models rather than hardware guarantees.


## Signal realization

Status: implemented.

Signals realize the address, project, transfer, refine, and reconstruct stages through finite polyphase coordinates and packet trees.

```python
coordinates = polyphase_analyze(signal, order)
filtered = polyphase_convolution(signal, kernel, order)
tree = HaarPacketTree.analyze(signal, depth)
reconstruction = retained_packet_reconstruct(signal, packet, approximation)
```

The signal implementation includes exact finite coordinate identities, filter bank reconstruction, additive best basis selection, truncated power singular packets, stable separated jump recovery under declared assumptions, nonlinear residue products, and Smith chart coordinates.

A recovery certificate does not infer unknown smoothness or noise assumptions from the data. It records which finite threshold conditions were checked and which model hypotheses remain external.

## Dynamics realization

Status: implemented.

Dynamics realizes operate, transfer, refine, constrain, complete, and reconstruct through finite matrix evolutions and finite coarse models.

```python
semigroup = MatrixSemigroup(generator)
lumping = MarkovLumping(generator, partition)
posterior = BayesianUpdate(prior, likelihood)
fractional = fractional_power(operator, exponent)
reduction = FeshbachReduction(operator, retained)
kernel = reduction.memory_kernel()
```

The implementation includes semigroup and resolvent defects, compatible finite towers, exact Markov lumping, conditional expectation packets, Bayesian sufficiency, fractional spectral calculus, Green operators, finite Feynman Kac transfer, Schur Feshbach elimination, rational memory kernels, and stable phase type survival models.

Finite matrix theorems do not claim an unbounded continuum construction. Completion certificates state the topology, compatibility, and stability assumptions that were actually checked.


## Analytic realization

Status: implemented.

The analytic realization uses residue projection to turn one entire function into a finite cyclic differential system. Reciprocal, logarithmic, Appell, and theta families then operate on the resulting coefficient coordinates.

```python
packets = cyclic_packets(coefficients, order)
section = signed_section(order, residue, z)
coefficients = reciprocal_coefficients(order, parameter, degree)
zeros = positive_zero_string(order, count)
```

Exact coefficient operations and numerical zero operations remain distinct. A zero certificate checks only the returned residuals, ordering, and requested finite computation.

## Quantum realization

Status: implemented in finite dimension.

Quantum systems realize seed, project, transfer, constrain, and reconstruct through observable algebras, symmetry sectors, circuit paths, moving packets, channels, error correction, and effective dynamics.

```python
packets = cyclic_symmetry_projectors(symmetry, order)
circuit = CircuitPacketExpansion(gates, packet_systems)
channel = QuantumChannel(kraus_operators)
code = QuantumCode(projector, errors)
transport = KatoTransport(moving_packet_system)
```

Every operator acts on a declared finite Hilbert space. Continuum limits and gravitational endpoints require additional mathematics and are not represented as executable conclusions.

## 6. Refine

```python
PacketRefinement
RefinementTower
WeightedTransferGraph
LatticePacketData
SmithCongruenceProblem
SemigroupRefinement
```

Implemented refinement includes coarse and fine packet relations, residue and CRT refinement, weighted transfer paths, filtered inverses, integral separation, Smith congruences, operator defects, semigroup Duhamel defects, and finite compatible towers.

## 7. Constrain

```python
LinearObstructionProblem
LinearLiftProblem
PolynomialLiftProblem
TransportGraph
recover_separated_jumps
MarkovLumping
BayesianPartition
```

The finite constraint layer contains cokernel obstructions, prime power correction spaces, lift trees, finite holonomy, signal recovery under declared threshold hypotheses, exact Markov lumping, and fixed Bayesian sufficiency.

Block 9 implements packet jets, Faà di Bruno composition, rooted nonlinear operator trees, graded operations, nonlinear Schur and Lyapunov-Schmidt elimination, finite filtered lifting, and obstruction spectral sequences. Smooth Kato transport is shared by the nonlinear and quantum namespaces through one implementation.

## 8. Complete

There is no single universal completion object. A Banach inverse limit, a finite lift forest, a compatible semigroup tower, and a global arithmetic system require different runtime representations.

Blocks 4 and 5 implement finite semigroup towers, finite hidden state realizations, finite survival systems, finite Kato transport, Euclidean transfer, and bounded Hamiltonian tower diagnostics. It does not claim a general continuum semigroup constructor.

## 9. Reconstruct

Implemented reconstruction includes:

```python
PacketCoordinates.reconstruct()
PacketOperator.reconstruct()
ResidueCoordinates.reconstruct()
PolyphaseCoordinates.reconstruct()
PeriodicFilterBank.synthesize()
PrunedPacketCoefficients.reconstruct()
retained_packet_reconstruct(...)
FeshbachReduction.memory_kernel()
```

Signal reconstruction separates smooth approximation error from packet estimation error. Memory reconstruction separates retained dynamics from an auxiliary hidden realization.

## Execution classes

| Class | Meaning |
|---|---|
| exact | exact algebra or arithmetic under checked assumptions |
| numerical | floating point computation with residual and tolerance |
| conditional | valid when listed external assumptions hold |
| heuristic | discovery or estimation without a complete guarantee |
| planned | reserved namespace without a production implementation |

Examples:

* a SymPy Smith solve is exact
* a NumPy cokernel solve is numerical
* a jump recovery threshold can be checked numerically while regularity and noise hypotheses remain conditional
* exact finite Markov lumping is certified from the supplied matrix and partition
* a Feshbach resolvent identity is numerical unless exact matrix arithmetic is supplied
* future automatic packet discovery remains heuristic

## Arithmetic and sequence realization

In Block 6 a seed is an initial coefficient family, address means a residue or Mahler coordinate, operate means convolution or recurrence evolution, project means local or residue selection, transfer means triangular or affine coordinate transport, refine means prime power or polyphase refinement, constrain means a congruence or recurrence relation, complete means a certified finite model, and reconstruct means exact coefficient or term recovery.

## Deformation and Langlands realization

In Block 7 the unified stages acquire several new finite meanings. A residual class is a seed, a lift level is an address, reduction and transport are operators, packet cohomology and finite Hecke idempotents are projections, stable cores and support impose constraints, and a completion report records whether every required gate has been supplied.

```python
forest = residue_forest(levels, prime=p)
report = forest.report()

packet = TameWeilPacket(FrobeniusOrbit(q, degree, exponent))
packet.relation_certificate().require()

completion = AutomorphicCompletionReport(gates)
result = completion.result()
```

The same words do not identify deformation support with a finite spectral packet. Fitting support is computed from a module presentation. Central idempotent packets are used only where a finite algebra actually decomposes as a product.

Finite field, tame, Heisenberg, orbit, Mackey, and Clifford computations may be exact. Ramification comparisons, automorphic induction plans, patching support, compatible propagation, and final completion remain conditional whenever they rely on external arithmetic existence or comparison theorems.


## Portable output realization

Block 8 adds an output stage after reconstruction. A mathematical value may be normalized, rendered, written, and compiled without changing its execution classification.

```text
reconstruct
  -> normalize
  -> render
  -> write
  -> optionally compile LaTeX
  -> certify the artifact
```

`MathDocument` is renderer independent. Ordinary prose is escaped. Equations contain trusted mathematical LaTeX. `RawLatex` is an explicit trusted escape hatch. `export_data` supports data formats and report formats. `compile_latex` is a local tool with shell escape disabled by default.

Export success certifies that the requested artifact was written and, for PDF, that the local compiler returned success and produced a nonempty PDF. It does not add a mathematical proof to the underlying value.


## Advanced nonlinear realization

Block 9 makes the constrain and complete stages higher order. A derivative provider supplies the local multilinear data. Packet jets project each derivative slot. Faà di Bruno composes those jets. Rooted trees encode finite nonlinear iteration. Schur and Lyapunov-Schmidt reductions eliminate hidden or range variables. Filtered correction spaces organize successive obstructions. Finite spectral-sequence pages record which classes survive each filtration depth.

```text
derivative data
  -> packet jets
  -> higher composition or rooted trees
  -> obstruction projection
  -> correction or elimination
  -> filtered lift
  -> finite spectral sequence
  -> conditional completion record
```

The final arrow is conditional whenever it refers to an infinite inverse limit. The runtime certifies finite compatibility but does not infer completeness or the existence of an infinite branch.
