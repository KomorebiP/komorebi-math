> **Block 10 note:** This Block 9 review is retained as historical evidence. The current machine-readable contract is documented in [Public API compatibility contract for 0.10](API_COMPATIBILITY_0.10.md).

# API stability review for 0.9.0a1

## Scope

This review covers the public Block 9 nonlinear API. It is a pre-beta stability review, not a promise of semantic version stability before `1.0`.

## Stable design decisions

### One derivative protocol

All higher nonlinear components accept a `DerivativeProvider`. New derivative backends should implement the protocol rather than introduce separate jet APIs.

### Directional derivatives

The canonical higher derivative interface is:

```python
derivative(order, point, directions)
```

`directions` contains exactly `order` source vectors. Tensor storage is an implementation detail of specific providers.

### Packet addresses are explicit

`PacketJetAddress` stores one output label and an ordered tuple of input labels. No implicit global packet registry is used.

### Result objects carry certificates

Higher composition, tree, correction, filtered, and transport routines return result records rather than unstructured tuples. Certificates remain the compatibility mechanism for exactness, residuals, assumptions, and theorem identifiers.

### Infinite conclusions remain conditional

The finite runtime will not infer completeness, convergence, or inverse-limit existence. Future infinite APIs must continue to expose those assumptions.

### Shared transport has one implementation

`komorebi_math.quantum.kato` is a compatibility re-export from `komorebi_math.nonlinear.transport`. A second implementation would risk divergence and is not part of the supported design.

## Curated top-level API

The package root exposes common constructors and operations, including:

```python
CallableDerivativeProvider
FiniteDifferenceDerivativeProvider
PolynomialDerivativeProvider
SympyDerivativeProvider
PacketJet
PacketJetFamily
faa_di_bruno
packet_faa_di_bruno
PolynomialFixedPointProblem
FiniteAbelianGroup
GradedPacketOperation
HilbertObstructionProjector
RightInverseCorrectionProblem
LyapunovSchmidtReduction
NonlinearSchurElimination
FilteredVectorSpace
FilteredCorrectionProblem
FiniteFilteredCochainComplex
KatoTransport
MovingPacketSystem
```

Detailed result types and lower-level combinatorial utilities remain under `komorebi_math.nonlinear`.

## Compatibility review

* Existing Block 2 nonlinear imports remain unchanged.
* Existing quantum Kato imports remain unchanged.
* No old public symbol was removed.
* The nonlinear feature status changed from `partial` to `implemented` for finite explicit data.
* `komorebi verify all` now includes Block 9.
* `komorebi verify block8` remains available with its prior scope.

## Deferred decisions

The following are intentionally deferred until a beta release:

* a general sparse tensor storage format for very high-order jets
* automatic differentiation adapters for JAX, PyTorch, or other frameworks
* a GPU-specific higher-jet executor
* infinite rooted-tree containers
* persistent spectral-sequence serialization schemas
* a generic Banach-space abstraction

These additions should extend existing protocols rather than replace them.
