# Tenth major block report

## Executive result

Block 10 converts the finite nonlinear layer from a primarily dense, direct-evaluation implementation into a reusable execution and persistence substrate. It does so without changing the central directional derivative abstraction. Dense exact tensors, sparse symmetric tensors, custom JVP engines, JAX, PyTorch, finite differences, and declared derivatives can all feed the same packet-jet interface.

The block also establishes the first machine-readable compatibility boundary for the package. The result is still an alpha release. The contract is structural, persistence schemas are versioned, and unsupported global or hardware conclusions remain explicit.

## 1. Design constraints

The implementation was required to preserve five invariants:

1. NumPy-only installation and import must remain valid.
2. Optional frameworks must be imported lazily.
3. A finite support probe must not justify exact pruning unless completeness is certified.
4. Persistence must not instantiate arbitrary classes from input-controlled names.
5. Hardware performance or correctness must not be claimed without the hardware.

These constraints determined the split between sparse storage, framework adapters, execution plans, snapshots, persistence codecs, and runtime diagnostics.

## 2. Sparse derivative representation

An order-`r` symmetric multilinear map repeats coefficients across permutations of the input axes. The sparse representation stores only canonical nondecreasing tuples. Contraction enumerates the distinct orbit, so repeated indices receive the correct combinatorial multiplicity.

The implementation supports exact object arithmetic and numerical arithmetic. Dense conversion validates symmetry before compression. Mapping construction requires canonical indices and rejects malformed entries, dimension mismatches, duplicate keys, and non-finite numerical values.

The polynomial provider performs value evaluation by the finite Taylor formula encoded by its derivative tensors. It can be constructed from dense providers, mixed sparse/dense mappings, or explicit sparse tensors. The order-zero source dimension is inherited from positive-order tensors or supplied explicitly.

## 3. Directional automatic differentiation

The generic adapter treats a JVP as the primitive operation. Repeated nesting computes ordered higher directional derivatives without materializing a dense derivative tensor. The JAX adapter uses `jax.jvp`. The PyTorch adapter uses the framework functional JVP machinery while preserving the input dtype selected by the provider.

Validation covers values and directional derivatives through third order for explicit polynomials. Permutation checks test the symmetry expected of smooth derivatives. These finite checks verify adapter composition on the tested functions. They are not a universal proof that an arbitrary user function is smooth or traceable.

## 4. Certified packet execution plans

The compiler enumerates output and input packet addresses under a hard address guard. It validates packet counts, dimensions, order, and support provenance. Active addresses are grouped by ordered input packet labels. Each group requires one provider derivative evaluation, followed by the output projections associated with that group.

The plan certificate records full, active, pruned, grouped, and naive call counts. Execution verifies that the plan fingerprint matches the family. This prevents accidental reuse with different packets, dimensions, order, point, or provider metadata.

The reconstruction certificate compares the packet sum with the direct derivative. Exact providers require exact equality. Numerical providers use a declared tolerance and report the residual.

## 5. Frozen packet-jet snapshots

A live `PacketJet` refers to a provider and packet systems. A snapshot stores the finite multilinear coefficient table instead. Capture evaluates the jet on every standard-basis input tuple and retains nonzero output coefficients. Evaluation performs the multilinear contraction directly from the stored coefficients.

The representation is deliberately transparent. Its finite basis completeness is easy to audit, and persistence does not require serializing a function closure or framework graph. The cost is exponential growth in order, controlled by `max_evaluations`.

## 6. Persistence and integrity

The persistence registry maps trusted Python classes to explicit tags, versions, encoders, and decoders. The payload is recursively reduced to a deterministic data language and wrapped with a digest. Loading validates the envelope, digest, codec registration, version, dimensions, and depth before constructing the object.

This design separates two concerns:

* report serialization, which may preserve descriptive type markers without reconstruction
* executable persistence, which reconstructs only registered finite mathematical records

The digest provides integrity against accidental modification. Authenticity requires an external signature or authenticated storage system.

## 7. Compatibility and deprecation

The API manifest records structural properties of three namespaces. Additions are accepted in the alpha policy. Removals and kind changes are release failures. Signature changes are visible under every check and become failures in strict mode.

The deprecation decorator preserves wrapped metadata and adds a structured record. The warning category is silent by Python default, as expected for `DeprecationWarning`, but can be enabled by tests and downstream tooling.

## 8. Optional CuPy execution

There are two CuPy-facing layers:

* execution of live packet-jet families through a device derivative provider
* evaluation of frozen packet-jet snapshots

Both are runtime-gated. Status inspection distinguishes importability from a usable CUDA runtime. No transparent CPU fallback is performed inside the CuPy classes because that would make timing and backend claims ambiguous.

The release environment did not provide CuPy or a CUDA device. The code path is import-audited and the unavailable-runtime behavior is tested. Device-backed validation remains open.

## 9. Cross-platform CI definition

The repository workflow now defines Linux and Windows tests for Python 3.11, 3.12, and 3.13. The matrix installs the package, runs the unit suite, executes installed Block 10 verification, and checks the public API manifest. A quality job builds documentation, runs static analysis, and executes the deep audit. The dependent package job builds twice, compares the wheel and normalized source distribution byte for byte, tests the extracted source archive, rebuilds its wheel, and compares that wheel with the release build. A Windows PowerShell smoke step exercises the release script.

Defining the workflow is not the same as observing a hosted CI run. Local release evidence is reported separately from workflow coverage.

## 10. Validation architecture

The release uses multiple independent layers:

* focused unit tests
* preexisting regression tests
* installed-wheel verification with the required NumPy dependency
* optional JAX and PyTorch adapter tests when those packages are present
* randomized deep audit
* public API manifest comparison
* import-all and public annotation audits
* wheel and source-distribution payload inspection
* isolated virtual-environment installation of the built wheel
* source-distribution extraction and rebuild
* deterministic artifact comparison
* internal release checksum verification

Each layer has a different failure model. Passing one is not substituted for passing another.

## 11. Local release evidence

The final local release audit was run with Python 3.13.5 and produced the following evidence:

| Layer | Result |
|---|---:|
| Package unit suite | 523/523 passed |
| Focused Block 10 verification | 7/7 passed |
| Aggregate verification through Block 10 with local optional exact dependencies | 113/113 passed |
| Isolated installed-wheel verification with declared NumPy dependency | 111/111 passed |
| Independent randomized and exhaustive Block 10 audit | 15,924 checks, 0 failures |
| Public API compatibility | 464 symbols, no additions, removals, kind changes, or signature changes |
| Public API manifest digest | `b876814142b6f0b98edac38b22ccbab86ec4a627a119d743b016d4f20739c39b` |
| Production import audit | 193 package modules imported |
| Public Block 10 documentation audit | 0 missing docstrings and 0 missing annotations |
| Block 10 Python examples | 3/3 passed |
| Successive distribution builds | wheel and normalized source distribution byte-identical |
| Extracted source distribution | 523/523 tests passed |
| Wheel rebuilt from source distribution | byte-identical to the release wheel |
| Wheel ZIP integrity | all members passed CRC validation |

The JAX and PyTorch packages were present locally, and the adapter tests exercised directional derivatives through third order. The release container did not contain CuPy, a CUDA device, PowerShell, Ruff, mypy, or MkDocs. The unavailable-runtime CuPy boundary was tested, but device execution and the PowerShell smoke script were not executed. Ruff, mypy, MkDocs, Windows, and the full operating-system and Python-version matrix remain configured in the GitHub Actions workflow rather than locally observed release evidence. No hosted workflow run is claimed.

The wheel was also installed into an isolated virtual environment that reused the container's system dependency set. Import, version, aggregate verification, API compatibility, and resolution of the declared NumPy dependency were tested from outside the source tree. This is an installed-wheel test, not a claim that a dependency-free or strictly NumPy-exclusive environment was constructed.

## 12. Open boundaries

The largest remaining engineering block is actual accelerator validation and broader array-backend abstraction. The largest mathematical scaling boundary is the combinatorial growth of high-order finite jets and rooted expansions. The package can exploit sparsity and packet structure, but no general theorem eliminates that growth.

A subsequent block should prioritize device-backed tests when hardware exists, chunked or streamed snapshot storage, migration support for future codec versions, and a beta decision on the public namespaces. Those tasks should not weaken the explicit finite and conditional status discipline established here.
