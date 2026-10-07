# Eleventh major block report

## Executive result

Block 11 turns frozen nonlinear state into a resumable finite data pipeline and separates three compatibility questions that had previously been conflated:

1. whether an exported Python name and signature remain present
2. whether persisted records remain readable across schema revisions
3. whether selected mathematical behaviors remain unchanged

The implementation answers these with separate mechanisms: API manifests, explicit codec migrations, and closed behavioral golden fixtures. It also adds exact structural operation counts and local runtime observations without presenting timings as portable complexity claims.

## 1. Design invariants

The block preserves the following constraints:

* NumPy-only import remains valid.
* Optional frameworks remain lazy.
* Incomplete captures are never promoted to complete destinations.
* Every committed stream chunk is independently verifiable.
* Old persistence data is migrated explicitly rather than reinterpreted silently.
* Persisted data cannot select a Python class or evaluator by import path.
* Partial coefficient ranges are identified as partial sums.
* Structural counts and empirical timings remain distinct.
* The full package remains alpha even though one namespace receives beta-level signature protection.
* GPU and cross-platform claims require actual execution evidence.

## 2. Stream architecture

Direct capture enumerates the ambient standard-basis tuples in lexicographic ordinal order. For source dimension `d` and derivative order `r`, ordinal `k` is converted to its base-`d` index tuple. Each nonzero output coefficient is stored with that tuple.

Progress is represented by contiguous basis and coefficient ranges. A chunk file is atomically replaced and re-read on resume before the manifest can be extended. Resume starts from `evaluations_completed` or `coefficients_completed`, validates the recorded source metadata, re-verifies every committed prefix chunk, removes uncommitted orphan chunks, and continues with the original chunk size.

Snapshot mode does not reevaluate the derivative. It chunks the already retained coefficient sequence and records complete dense-basis coverage in the final manifest.

## 3. Reader semantics

The reader validates the manifest envelope, metadata digest, stream-content digest, chunk records, and optional chunk payloads. Iteration decodes one chunk at a time. Materializing a complete snapshot requires a coefficient guard.

Evaluation over the full stored range reconstructs the complete frozen multilinear block. Evaluation over `[start, stop)` returns a selected coefficient partial sum and a certificate with the omitted range listed as unchecked.

Exact streams use object arithmetic. Numerical streams accumulate complex-compatible values under the recorded tolerance.

## 4. Integrity model

The stream has four related fingerprints:

* a source fingerprint
* a metadata digest
* one payload digest per chunk
* a stream-content digest joining progress, chunk digests, and the final certificate

This model detects accidental modification and inconsistent partial replacement. It does not provide a secret key, signature, authorization, or confidentiality.

For a live arbitrary derivative provider, metadata cannot prove that the Python implementation is behaviorally identical after a process restart. The current direct-resume certificate leaves provider implementation identity unchecked. Existing snapshots do not have that limitation because their complete coefficient content is fingerprinted.

## 5. Schema migration

The portable codec registry now has a second registry for adjacent migrations. A migration must advance exactly from version `n` to `n + 1`. Loading an old payload applies the complete registered chain. Missing steps and future versions fail closed.

The packet-jet snapshot codec moved from version 1 to version 2. The provider field was normalized, and coefficient ordering became explicit. A version 1 payload is decoded into safe data, migrated, then passed to the current decoder. The persisted payload never chooses migration code.

## 6. Semantic regression boundary

The behavioral fixture contains five cases chosen to cross the principal Block 10 and Block 11 boundaries: exact contraction, sparse orbit multiplicity, grouped packet execution, codec migration, and stream evaluation.

Case selection is closed in source code. The JSON fixture contains data only. Unknown kinds are rejected. This gives the project a reviewable semantic boundary without allowing a fixture to import or execute arbitrary code.

The fixture is intentionally small. It catches selected semantic drift but does not replace the unit suite or randomized audit.

## 7. Complexity and profiling

The complexity estimator derives exact finite counts from packet dimensions, order, addresses, support, and execution grouping. It exposes how support pruning and shared input groups reduce provider calls while leaving the dense capture count visible.

The profiler records stages separately, including enumeration, compilation, execution, snapshot capture, portable serialization, and stream writing. Timings use local monotonic clocks. Traced memory uses Python allocation instrumentation. The report contains the process environment and explicitly rejects portable speed conclusions.

## 8. Beta namespace decision

Reviewing the complete research surface showed that promoting every export would freeze too many experimental names. Block 11 instead introduces `komorebi_math.beta` as an explicit compatibility façade.

Its 83 names cover certificates, packet systems, derivative providers, packet jets, sparse tensors, execution plans, snapshots, streaming, persistence, migrations, golden fixtures, profiling, and compatibility checks. Removal, broad kind changes, and signature changes fail the beta check. Additions remain allowed.

This is a namespace contract, not a declaration that every research module is beta-stable.

## 9. Verification architecture

The installed Block 11 suite adds eight checks to the 113 retained through Block 10. It validates direct resume, snapshot ranges, tamper rejection, migration, golden semantics, exact operation counts, observational profile structure, and both API contracts.

The independent randomized audit exercises exact and numerical stream capture, pause and resume, partial-sum reconstruction, tamper guards, migration, complexity counts, profiling records, and optional-device status.

A dedicated device script performs matched CPU and CuPy evaluations only when a CUDA runtime is usable. A self-hosted workflow requires that execution. The local container had no CuPy or CUDA runtime, so the device result is recorded as not executed.

## 10. Release validation

The final source and distribution audits produced the following evidence:

| Validation layer | Result |
|---|---:|
| Source unit tests | **545/545 passed** |
| Focused installed Block 11 checks | **8/8 passed** |
| Full installed verification with available optional dependencies | **121/121 passed** |
| Required-dependency-only installed verification | **119/119 passed** |
| Behavioral golden cases | **5/5 passed** |
| Independent randomized audit | **11,738/11,738 passed** |
| Research API manifest | **528 symbols, 0 differences** |
| Beta API manifest | **83 symbols, 0 differences** |
| Production Python files | **199 audited** |
| Imported package modules | **198/198 imported** |
| Missing audited docstrings | **0** |
| Missing audited annotations | **0** |
| Wheel payload | **211 files, 0 forbidden entries** |
| Source distribution | **427 members** |
| Extracted-source unit tests | **545/545 passed** |
| Rebuilt source-distribution wheel | **Byte-identical** |
| Successive wheel builds | **Byte-identical** |
| Successive normalized source distributions | **Byte-identical** |

The required-dependency-only environment intentionally excluded SymPy, SciPy, JAX, PyTorch, CuPy, PyYAML, and Matplotlib. Its two-check difference from the full installed suite consists of optional SymPy exact-arithmetic checks, not failures.

The local release host did not execute Windows, hosted CI, PowerShell, Ruff, mypy, MkDocs, CuPy, or CUDA. Workflow definitions and device harnesses were structurally inspected, but those unexecuted environments are not counted as passing evidence.

## 11. Remaining boundaries

Block 11 does not provide:

* subexponential high-order snapshot capture
* a constant-memory manifest for arbitrarily many chunks
* cryptographic signatures or authenticated provenance
* a universal semantic fingerprint for arbitrary Python derivative providers
* portable runtime or memory predictions
* executed Windows, hosted-CI, or CUDA evidence from the local release environment
* a stable 1.0 contract for the complete research API

These are explicit continuation targets rather than hidden assumptions.
