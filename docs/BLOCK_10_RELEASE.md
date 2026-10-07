# Block 10 release: scalable nonlinear execution and pre-beta hardening

Version `0.10.0a1` extends the finite nonlinear calculus without changing the directional derivative protocol introduced in Block 9. The release adds sparse symmetric derivative tensors, lazy automatic-differentiation adapters, projection-cached packet-jet execution, provider-independent packet-jet snapshots, versioned persistence, and a machine-readable public API contract.

## Implemented scope

### Sparse symmetric derivatives

`SparseSymmetricTensor` stores one value for each nonzero orbit of symmetric input indices. The output axis remains explicit. Exact values such as `fractions.Fraction` are preserved, and floating-point compression requires a declared tolerance. `SparsePolynomialDerivativeProvider` evaluates the same finite polynomial model as `PolynomialDerivativeProvider` while avoiding storage of repeated symmetric coefficients and structural zeros.

The sparse certificate checks canonical index ordering, dimensions, dense reconstruction, representative permutations, storage counts, and selected contractions. An order-zero tensor requires an explicit source dimension because its data contains no source axis from which that dimension could be inferred.

### Automatic differentiation adapters

`AutomaticDifferentiationProvider` accepts a framework-neutral Jacobian-vector-product function. `JaxDerivativeProvider` and `TorchDerivativeProvider` supply lazy adapters whose optional frameworks are imported only when evaluation begins. Higher directional derivatives are formed by nested JVPs, preserving the existing `DerivativeProvider` interface.

The package does not install JAX or PyTorch by default. Importing `komorebi_math` remains NumPy-only. Adapter availability can be inspected with `available_ad_frameworks()`.

### Projection-cached execution

`compile_packet_jet_plan` enumerates finite packet-jet addresses, caches ordered input projections, groups addresses that share those inputs, and records all pruning decisions. A support object may prune addresses only when its certificate is complete. Without complete support, the compiler keeps every address active.

`execute_packet_jet_plan` performs one provider derivative call per input group, applies the output packet projections, reconstructs the full derivative, and returns a certificate comparing the reconstruction with a direct derivative call. The reported call reduction is a deterministic structural count, not a wall-clock performance claim.

### Portable packet-jet snapshots

`PacketJetSnapshot` freezes one finite multilinear packet block by evaluating it on the complete standard basis. It can later evaluate directions independently of the original provider, point, or packet system. The capture certificate records the complete finite basis enumeration and the number of retained nonzero coefficients.

Snapshot size grows as `target_dimension * source_dimension**order`. `snapshot_packet_jet` therefore requires a finite evaluation guard. It is a serialization boundary for moderate finite blocks, not an implicit compression theorem for arbitrary high-order tensors.

### Versioned persistence

The portable persistence layer uses:

* an explicit codec registry
* per-type codec versions
* canonical JSON encoding
* a schema-versioned envelope
* a SHA-256 digest of the canonical payload
* atomic file replacement
* an overwrite guard
* a recursion-depth guard

Built-in codecs cover certificates, rooted trees, packet-jet snapshots, sparse symmetric tensors, lift branches and trees, and finite spectral-sequence records. Unknown codecs and unsupported versions are rejected. The decoder does not import or instantiate arbitrary classes named by the payload.

### API compatibility contract

The committed manifest covers the explicit `__all__` exports of:

* `komorebi_math`
* `komorebi_math.nonlinear`
* `komorebi_math.export`

The policy is `pre-beta-additive`. Added names pass. Removed names and broad kind changes fail. Signature changes are always reported and fail when strict signature checking is requested. The certificate explicitly leaves semantic and mathematical equivalence unchecked.

The deprecation helpers attach structured metadata and emit `KomorebiDeprecationWarning`. No Block 10 public name is scheduled for removal.

## Optional GPU boundary

Block 10 defines CuPy execution interfaces and runtime diagnostics. The runtime status separates package installation, runtime availability, and device count. Constructors fail with `BackendUnavailableError` when no usable runtime exists.

No CUDA device or CuPy installation was available in the release container. Consequently, this release makes no hardware-backed GPU correctness, compatibility, or performance claim. The NumPy execution path and the absence-of-runtime boundary are tested.

## Installed checks

Run the focused Block 10 suite:

```powershell
komorebi verify hardening
```

Run every installed suite through Block 10:

```powershell
komorebi verify block10
```

Check the committed API manifest:

```powershell
komorebi api check
komorebi api check --strict-signatures
```

Create a candidate manifest without overwriting the committed file:

```powershell
komorebi api snapshot --output .\public_api_candidate.json
```

## Release validation

The local release evidence consists of:

* 523 passing package tests
* 7 of 7 focused Block 10 verification checks
* 113 of 113 aggregate checks with the local optional exact dependency available
* 111 of 111 installed-wheel checks in the isolated environment using the declared NumPy dependency
* 15,924 independent randomized and exhaustive audit checks with zero failures
* 464 structurally frozen public symbols with no strict-signature differences
* 193 imported package modules with no import failures
* no missing public docstrings or annotations on the audited Block 10 surface
* three passing Block 10 Python examples
* byte-identical successive wheel and normalized source-distribution builds
* 523 passing tests from the extracted source distribution
* a wheel rebuilt from the source distribution that is byte-identical to the release wheel
* passing ZIP CRC validation for both release and rebuilt wheels

JAX and PyTorch adapter tests ran locally. CuPy, CUDA hardware, PowerShell, Ruff, mypy, and MkDocs were not available in the release container. Their configured workflow coverage is not presented as an executed local or hosted result. The built wheel was installed and verified in an isolated virtual environment using the container's system dependency set.

## Boundaries

Block 10 does not claim:

* sparse asymptotic improvement for dense derivative families
* automatic differentiation through unsupported Python or foreign operations
* global nonlinear convergence from finite jets
* infinite-dimensional packet-jet snapshots
* secure authenticity or authorship from an unkeyed SHA-256 digest
* semantic API equivalence from structural reflection
* CUDA correctness or speed without device-backed tests

These limits are part of the API and release documentation rather than implicit caveats.
