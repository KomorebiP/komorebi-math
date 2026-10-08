# Changelog

## GitHub readiness repairs, 2026-10-08

- Require SymPy 1.14 for the exact extra, matching the Smith decomposition API in use. Older manual installations receive explicit dependency guidance.
- Preserve NumPy 1.26 support in moving leakage quadrature and both canonical filter-bank pseudoinverses.
- Keep NumPy scalar defaults in API snapshots consistent across NumPy 1.26 and 2 without changing callable signatures or committed manifests.
- Check every claimed Smith certificate obligation, including integral entries, compatible shapes, unimodular transforms, diagonal divisibility, and invariant factors.
- Add adversarial certificate and dependency regressions to source and installed-artifact qualification, plus a minimum CPU dependency CI job.
- Correct the release instructions and distinguish current successful CI from historical preparation failures. The package version, research beta status, stable exports, and existing license are preserved.

## R4 Block 17 quantum Kraus/error-span gauge repair

- Audit Kraus/channel semantics against decomposition-sensitive quantum diagnostics.
- Repair the numerical Knill--Laflamme certificate: validity now uses a spectral norm of the full residual block operator after Hilbert--Schmidt orthonormalization of the declared error span.
- Retain the old maximum declared-basis residual only as a diagnostic because it is not invariant under harmless error-label mixing.
- Add adversarial default-tolerance, uniform-rescaling, random error-span, channel-gauge, and recovery-gauge tests.
- Preserve the frozen aggregate API; this is a semantic correctness repair inside the research-tier quantum subsystem.

## R4 Block 16 development checkpoint

- Falsify discrete future-equivalence as a universal compression primitive for field-valued semantics.
- Audit existing polyphase route packets through exact Hankel blocks instead of adding a duplicate weighted-automaton layer.
- Prove for the two-tap route family that the evaluated N-phase realization has exact Hankel rank N while the discrete exact future quotient is infinite.
- Cross-check the order-two route series against the existing `LinearRecurrence((1, 2), (1, 1))` state.
- Preserve the frozen aggregate API; Block 16 adds verification/documentation only, not a new package abstraction.

## R4 Block 14 development checkpoint

- Cross-domain falsification: convex/max carrier semantics do not preserve additive perfect-reconstruction filter-bank routing.
- Add research-tier `PolyphaseRoutePacket` with exact matrix-convolution composition of causal polyphase lag kernels.
- Prove the carrier boundary: identical route histories may be discarded under idempotent max/convex aggregation but not under ordinary additive reconstruction unless their weights are aggregated.
- Interpret raw tap/phase histories through the standard polyphase matrix algebra rather than claiming a new signal-processing theorem.
- Preserve the frozen aggregate API; route packets remain research-tier.

## R4 Block 12 development checkpoint

- Add research-tier finite support-function packets with exact linear-image pullback and Bellman branch transport.
- Separate convex value semantics from branch provenance: identical support packets can encode different maximizing action sets.
- Classify normal-fan / active-face geometry as derivable from the exact support function rather than an independent semantic packet.
- Establish future-complete convex compaction for value tasks: nonextreme atoms can be removed without changing any future linear/Bellman support query.
- Preserve branch words when policy or provenance is part of the downstream task.
- Preserve the frozen aggregate API; support packets remain research-tier.

## R4 Block 10 development checkpoint

- Add a research-tier metric-distortion cocycle with coordinate-covariant scale/anisotropy packets.
- Prove and test submultiplicative log distortion and closed-word spectral lower bounds.
- Distinguish positive-gap metric-completion obstructions from zero-gap/nonattained obstructions that force metric condition numbers to diverge.
- Record the conformal metric-completion criterion through determinant-normalized unitarizable holonomy.
- Reject curvature-vanishing as the nonlinear completion criterion: nonzero-curvature metric connections can preserve a positive metric exactly.
- Preserve the frozen aggregate API; the new distortion module remains research-tier.

## R4 Block 7 development checkpoint

- Advance the built-in `core.certificate` portable codec from version 1 to 2
  with an explicit adjacent migration.
- Treat loaded certificates as historical evidence: preserve `recorded_valid`,
  require producer-specific revalidation, and prevent `.require()` from
  promoting a persisted flag into a fresh proof.
- Reject unknown certificate fields, nonfinite/nonnumeric residuals, invalid
  tolerances, non-string assumptions and malformed theorem identifiers.
- Preserve stable constructor signatures and the 131-symbol stable export set.
- Keep SHA-256 persistence claims limited to integrity, not authentication.

## 1.0.0b4

- Add the Block 20 typed finite expression calculus at the research tier.
- Identify polynomial coordinate spaces by complete exact rational basis
  witnesses rather than display names.
- Add immutable typed values, linear maps, lexical bindings, polynomial
  primitives, exact matrix/vector normal forms, and decidable closed-language
  equality.
- Add conservative denotation-replayed rewrites, including coordinate
  naturality, `D J_a = I`, the correct `J_a D = I - u e_a` defect, and the
  density-weighted finite fundamental theorem row.
- Add a strict dedicated portable expression schema with alpha-invariant
  lexical digests and no change to the frozen 51-codec contract.
- Add exact dense matrix-chain dynamic programming, self-contained execution
  plans, plan-tamper rejection, and independent direct/planned comparison.
- Memoize shared DAG evaluation, budget exact intermediate outputs and planner
  work, replay generated transports through their strict decoder, and reject
  subclass or post-construction invariant spoofing.
- Add a worked Euler/Fibonacci pipeline with exact result `5/4`, compiled map
  `I4`, and retained affine/weighted regression `106/5`.
- Preserve the frozen 131-symbol stable API, 83-symbol beta API, 51 codecs,
  and 10 stable semantic fixtures.
- Do not claim arbitrary symbolic algebra, nonlinear or infinite-dimensional
  theorem proving, floating-point stability, or a universal optimizer.

## 1.0.0b3

- Add Block 19 finite measure and integration calculus at the research tier.
- Add exact sparse multivariate polynomials, positive atomic and bounded-interval
  measures, product measures, affine pushforwards, and exact polynomial moments.
- Keep signed quadrature functionals distinct from positive measures; certify
  polynomial exactness only after exact moment comparison with the target measure.
- Add composite Simpson evaluation with explicit rigorous, conditional,
  estimated, or unknown error status. Rigorous certification is restricted to
  a provenance-matched `ExactPolynomial` and exact rational interval, with
  exact rational nodes, weights, evaluation, summation, and error bounds;
  arbitrary floating callables remain estimated or conditional.
- Add exact finite FTC, integration-by-parts, Fubini, affine change-of-variables,
  pushforward, and coordinate-covariance identities.
- Add adapters for polynomial families, normalized formal series, derivative
  providers, and finite analysis--synthesis systems.
- Add installed integration and Block 19 verification, a 47-case committed
  adversarial runner, a 16-check independent mathematical verifier, and a
  worked example whose exact transported integral is `106/5`.
- Preserve the frozen 131-symbol stable API, 83-symbol beta API, 51 codecs,
  and 10 stable semantic fixtures.
- Keep the new surface research-only; this release does not claim general
  Lebesgue, improper, stochastic, manifold, or infinite-space integration.

## 1.0.0b2

- Add Block 18 redundant-frame coefficient algebra at the research tier.
- Add numerical gauge projections, quotient metrics, cross-presentation bridges,
  coefficient lifts, compositional small blocks, and distinct physical/dual adjoints.
- Add exact rational analysis--synthesis systems and maps with exact certificates.
- Fix canonical-dual rank and pseudoinverse threshold inconsistency.
- Preserve the frozen 131-symbol stable API, 83-symbol beta API, 51 codecs,
  and 10 stable semantic fixtures.

## 1.0.0b1

- Froze the 131-symbol stable API without changes from `0.16.0a2`.
- Added executable upgrade validation from `0.15.0a2` and `0.16.0a2`.
- Added seven digest-checked predecessor portable records.
- Added exact stable documentation coverage and platform-evidence contracts.
- Added separate public-beta and release-candidate readiness gates.
- Added Block 17 source, wheel, GPU, upgrade, and distribution runners.


## 0.16.0a2

Windows qualification hotfix for the Block 16 freeze candidate.

### Corrected

- normalized public API records for path-valued constants to the portable `pathlib.Path` identity rather than platform-specific `PosixPath` or `WindowsPath` identities
- made the POSIX executable-mode audit explicitly inapplicable on Windows, where ZIP extraction and `stat` do not provide POSIX execute bits
- added cross-platform regression tests for both defects
- regenerated the research, beta, and stable API manifests and the complete release contract
- advanced all package, distribution, validation, and documentation records to `0.16.0a2`

The mathematical APIs, semantic fixtures, persistence codecs, and Block 16 analysis--synthesis laws are unchanged.

## 0.16.0a1

Block 16 establishes the first complete 1.0 freeze-candidate contract.

### Added

- `AnalysisSynthesisSystem` and `AnalysisSynthesisBlocks` for finite left-inverse analysis and synthesis systems
- canonical reconstruction-component terminology for redundant frames
- portable persistence for analysis--synthesis systems
- versioned aliases `komorebi_math.v1` and `komorebi_math.api.v1`
- strict stable API manifest for `komorebi_math.stable`
- ten-case closed semantic fixture spanning analysis--synthesis, polyphase refinement, exact arithmetic, lattice refinement, operator realization, coordinates, generating functions, document rendering, scoped settings, and persistence
- package-owned portable codec inventory with adjacent migration-path checks
- aggregate stable compatibility reports
- machine-readable platform and optional-backend support policy
- crash-isolated CUDA qualification for PyTorch, CuPy, and the Komorebi PacketIR executor
- `komorebi verify stable`, `komorebi verify release`, `komorebi verify gpu`, and `komorebi verify block16`
- `komorebi api check --stable` and stable API snapshots
- `komorebi release qualify`, with optional and required GPU modes
- Block 16 source, installed-wheel, and GPU validation scripts
- release constraints, Windows CUDA environment specification, and hosted CPU CI matrix

### Changed

- API manifest factory signatures no longer embed a release version in their default argument
- the current public, beta, and stable manifests target `0.16.0a1`
- the current portable codec contract targets `0.16.0a1`
- cumulative installed verification now terminates at Block 16
- malformed contract and support records are rejected instead of silently dropping invalid entries

### Compatibility boundary

The broad research API remains available but is not frozen as a 1.x contract. The curated stable surface is frozen only after the `1.0.0b1` gate. Version `0.16.0a1` is therefore still an alpha release and may receive final candidate-surface corrections before beta.

## 0.15.0a2

- Isolated optional JAX and PyTorch discovery in child interpreters so a native import abort cannot terminate the main validation process.
- Replaced in-process optional automatic-differentiation integration tests with crash-contained worker processes.
- Added backend health diagnostics that distinguish package discovery from successful import and required JVP capability.
- Hardened Windows and POSIX source-validation scripts with interpreter, NumPy, pytest, and optional-backend preflight reporting.
- Added regression tests for successful probes, native abort return codes, timeouts, and guarded provider imports.

## 0.15.0a1

Established the first 1.0-candidate stabilization boundary and consolidated finite polyphase refinement.

* Added the canonical address law `a = r + M s` for divisible packet orders
* Added refinement and coarsening for signal, residue, exact arithmetic, and rectangular lattice coordinates
* Added multilevel polyphase refinement towers with direct-versus-sequential coherence certificates
* Added versioned JSON records and portable persistence codecs for refinement plans and coordinate records
* Added the curated `komorebi_math.stable`, `komorebi_math.v1`, and `komorebi_math.api.v1` namespaces
* Added a strict-signature committed API manifest for the 1.0-candidate surface
* Added Block 15 semantic, persistence, compatibility, and installation tests

## 0.14.0a2

Generalized Block 14 from a narrow companion-coordinate presentation into a complete coordinate atlas.

* Added arbitrary exact finite bases, subspace coordinates, coordinate graphs, path composition, and commuting-diagram certificates
* Added classical and custom polynomial-family catalogs, combinatorial transforms, interpolation coordinates, and polynomial operator calculus
* Added general recurrence-model coordinates, state-space canonical coordinates, controllability, observability, and transfer-equivalent state changes
* Added affine, barycentric, formal, finite-field, tensor, spectral, modal, residue, digit-reversal, Chinese-remainder, and packet coordinates
* Added generic trusted persistence for exact and numerical coordinate records
* Expanded the coordinate CLI, verification suite, golden fixtures, examples, and documentation
* Retained Euler-to-Fibonacci conversion as one named bridge rather than the scope-defining feature

## 0.14.0a1

Completed the exact companion coordinate calculus and imposed the pre-1.0 mathematical feature freeze.

* Added composable exact coordinate transformations with inverse, composition, determinant, and certificates
* Reused normalized generating-function basis descriptors for exact diagonal basis changes and lowering transport
* Added recurrence analysis and synthesis, unique defect streams, companion matrices, Green kernels, and exact finite changes between recurrence systems
* Added Euler numbers in forced Fibonacci companion coordinates and Euler polynomial connections in the Fibonacci polynomial basis
* Added finite realizations, observability, state recovery, intertwiner spaces, cyclic similarity, and defect realizations
* Added graded polynomial families, exact connection matrices, derivative transport, and operator conjugation
* Added shift polynomial arithmetic, extended gcd, and Bézout defect identities
* Added coordinate resource budgets, trusted portable codecs, CLI workflows, semantic golden fixtures, and installed verification
* Added two independent coordinate audits with 364,882 exact comparisons and no recorded failures
* Added a comprehensive Windows PowerShell validation harness for source, wheel, source distribution, API, persistence, examples, audits, and reproducibility

## 0.13.0a1

Completed the normalized generating-function and arithmetic-compiler integration.

Added:

1. exact normalized monomial bases with declarative and finite-table descriptors
2. explicit integer, rational, residue-ring, prime-field, and truncated p-adic coefficient domains
3. domain-aware finite normalized series with exact products, reciprocal, composition, exponential, logarithm, basis transfer, projectors, and lowering
4. declarative custom sources for recurrences, rational functions, hypergeometric ratios, algebraic branches, Euler products, compositions, and factorial-power Bell families
5. Mahler mask construction, exact inversion, analytic finite evaluation with tail bounds, Frobenius defects, and coboundary solvers
6. finite-mask carry matrices, ordinary-product carry closure, rational Cartier linear representations, reachable automata, and state minimization
7. Kummer-layer distributions, exact prime-power binomial and normalized-kernel residues, and carry-layer convolutions
8. Bell digit-sum reductions, the exact parity automaton, and Touchard-generated residue automata
9. all-prime Gaussian q-Kummer valuation decomposition and prime-power support decisions
10. Euler and cyclotomic collapse certificates, trusted finite persistence, finite signal adapters, CLI commands, installed verification, compatibility manifests, and independent deep auditing

The base installation remains NumPy-only. Symbolic recognition uses lazy optional SymPy imports. Persisted records contain validated finite data and never reconstruct arbitrary callbacks.

## 0.12.0a1

Completed the production finite signal-processing expansion.

Added:

1. deterministic boxcar, Hann, Hamming, Blackman, Bartlett, Kaiser, and Tukey windows with coherent-gain, power-gain, ENBW, overlap, COLA/NOLA, and canonical-dual diagnostics
2. finite framing, STFT, inverse STFT, spectrograms, normalized overlap-add reconstruction, and explicit noninvertible detrending boundaries
3. periodograms, Welch power spectra, cross spectra, coherence, H1 transfer estimates, and Parseval certificates
4. odd-length linear-phase lowpass, highpass, bandpass, and bandstop FIR design with direct, causal, same, valid, full, and stateful streaming execution
5. explicit upsampling, downsampling, rational polyphase resampling plans, length conventions, and direct-reference equivalence certificates
6. finite LMS, normalized LMS, and recursive least-squares system identification with complete weight histories and convergence assumptions left visible
7. multilevel Haar decomposition, hard, soft, and garrote shrinkage, universal and SURE thresholds, and observed finite-risk certificates
8. chunk-invariant streaming STFT analysis with exact batch framing agreement
9. finite direct-form II transposed FIR/IIR recursion, continuation-state transport, stable lowpass, highpass, bandpass, notch, and allpass biquads, and pole-radius certificates
10. zero-padded FFT convolution and bounded-buffer stateful overlap-add execution for real and complex signals
11. a dedicated installed DSP verifier, Block 12 CLI suite, examples, tests, API manifest coverage for `komorebi_math.signals`, an independent randomized deep audit, and an optional SciPy reference audit

The block is NumPy-only at import time. It does not claim real-time deadlines, stochastic convergence outside declared assumptions, universal denoising optimality, or GPU acceleration.

## 0.11.0a1

Resumable nonlinear persistence, semantic regression, finite profiling, and a curated beta contract.

* Added chunked directory-backed packet-jet streams with independently digested manifests and chunks
* Added budgeted direct capture, pause and resume, bounded snapshot writes, and atomic completion promotion
* Hardened resume so every committed prefix chunk is re-read and verified before new work can be promoted
* Added one-chunk-at-a-time iteration, guarded selection and materialization, and explicitly certified partial-range evaluation
* Added adjacent portable-codec migrations and advanced the packet-jet snapshot codec from version 1 to version 2
* Added a closed five-case behavioral golden fixture for exact contraction, sparse symmetry, cached execution, migration, and streaming
* Added exact finite structural complexity counts and environment-scoped runtime and traced-memory observations
* Added the curated `komorebi_math.beta` namespace with an additive strict-signature contract
* Added version 0.11 research and beta API manifests with zero compatibility differences at release
* Added Block 11 CLI commands, examples, installed verification, randomized deep auditing, and release documentation
* Added an executable CPU/CuPy device-validation program and a self-hosted CUDA workflow definition without claiming unexecuted hardware results
* Documented that digest chains provide integrity rather than authenticity and that arbitrary live providers lack a universal semantic fingerprint

## 0.10.0a1

Scalable nonlinear execution, portable state, and pre-beta hardening.

* Added sparse symmetric derivative tensors and sparse polynomial derivative providers with exact orbit contraction
* Added framework-neutral nested JVP execution plus lazy JAX and PyTorch directional derivative adapters
* Added projection-cached packet-jet plans with certified support pruning and reconstruction checks
* Added provider-independent finite packet-jet snapshots with guarded basis capture
* Added schema-versioned canonical persistence with explicit codecs, SHA-256 payload integrity, atomic writes, and tamper rejection
* Added guarded CuPy runtime diagnostics and live or frozen packet-jet execution interfaces without an implicit CPU fallback
* Added a committed structural public API manifest, compatibility checks, deprecation metadata, and command line snapshot tooling
* Added Linux and Windows CI definitions for Python 3.11 through 3.13, optional AD jobs, static analysis, documentation, and package audits
* Added focused Block 10 verification, randomized deep audits, examples, persistence documentation, and release records
* Fixed exact polynomial evaluation so structurally zero floating tensors cannot coerce rational results to floating point

## 0.9.0a1

Advanced finite nonlinear calculus and Block 9 integration.

* Added declared, polynomial, finite-difference, and optional SymPy derivative providers under one directional interface
* Added multilinear packet jets, support enumeration, reconstruction certificates, and finite packet Taylor expansion
* Added exact set-partition Faà di Bruno composition and packet-resolved higher chain rules
* Added guarded ordered rooted-tree enumeration, finite tree-Picard identities, packet-labeled reconstruction, and scalar majorant convergence certificates
* Added finite abelian grading, graded convolution, Maurer-Cartan packet equations, higher finite operations, and divided-square identities
* Added Hilbert obstruction projection, right-inverse contraction correction, target stability bounds, and Lyapunov-Schmidt reduction
* Added first- and second-order nonlinear Schur elimination with exact symbolic and numerical branches
* Added finite prime-field filtrations, associated-graded correction problems, compatible branch verification, conditional projective-limit records, and finite filtered spectral sequences
* Moved smooth packet transport into the nonlinear core and preserved the quantum namespace as a backward-compatible re-export
* Added derivative, generator, direct-rotation, midpoint transport, and mesh-refinement certificates
* Added nine installed advanced nonlinear verification checks and the Block 9 aggregate verification suite
* Added 35 focused unit tests, bringing the package suite to 456 passing tests
* Added a 21,890-check randomized and exhaustive Block 9 deep audit
* Added Block 9 examples, nonlinear calculus documentation, API stability review, and release records

## 0.8.0a1

Portable exports, mathematical reports, LaTeX generation, and PDF compilation.

* Added JSON, optional YAML, CSV, TSV, text, Markdown, HTML, LaTeX, PDF, NPY, and NPZ export
* Added exact normalization for fractions, complex values, NumPy arrays, dataclasses, certificates, and computation results
* Added renderer independent mathematical documents with prose, equations, tables, code, lists, and figures
* Added safe LaTeX escaping and an explicit trusted raw LaTeX escape hatch
* Added local LaTeX engine discovery and compilation through LuaLaTeX, XeLaTeX, PDFLaTeX, or latexmk
* Disabled shell escape by default and added timeouts, compiler logs, repeated passes, and auxiliary cleanup
* Added multi format ZIP bundles with normalized metadata and SHA256 manifests
* Added command line sequence, arithmetic, JSON, bundle, and LaTeX compilation exports
* Added PowerShell smoke tests and a complete export demonstration
* Added installed export verification and a Block 8 verification suite

## 0.7.0a1

Residual deformation and finite Langlands execution.

* Added finite lift forests, stable cores, visibility trees, and exact local counting charts
* Added finite cochain complexes, packet cohomology, Fitting generators, fiber support, and nested occupancy
* Added Frobenius spines, finite trace separation, companion forests, and compatible carrier selection
* Added exact prime power finite fields, tame Frobenius packets, Gauss sums, and monodromy blocks
* Added finite Heisenberg representations, alternating forms, quadratic refinements, projective envelopes, and bridge diagnostics
* Added explicit finite groups, induced representations, Mackey packets, symmetric and exterior orbit packets, and Clifford towers
* Added ramification adapted Clifford towers with dyadic phase diagnostics
* Added finite Hecke packets, patching support reports, dyadic sign diagnostics, and support comparisons
* Added five gate automorphic completion reports and compatible system propagation reports
* Added five executable Block 7 examples and installed verification suites

## 0.6.0a1

Local arithmetic, Mahler transfer, recurrence packets, and exact sequence execution.

* Added modular rings, prime fields, truncated local domains, and exact valuation utilities
* Added Hurwitz series, exponential modes, subgroup selectors, and finite inverses
* Added standard window periods, address independence checks, and rational periodic tails
* Added one Euler coordinate API with separate odd and dyadic integral charts
* Added sharp dyadic kernel valuations and local truncation certificates
* Added finite Mahler analysis, affine subsequence transfer, prime power sparsity, and matrix modes
* Added local unit difference gains and simultaneous congruence certificates
* Added exact constant coefficient recurrences, residue packets, modular terms, and period detection
* Added lazy exact sequences, a classical sequence catalog, and exact transforms
* Added finite automata for automatic sequences and polynomial coefficient recurrences for holonomic sequences
* Added a sequence compiler that separates declared models, detected finite patterns, and direct evaluation
* Added installed arithmetic and sequence verification suites and five executable Block 6 examples

## 0.5.0a1

Cyclic analytic functions and finite quantum packet calculus.

* Added cyclic coefficient projectors, residue reconstruction, and cyclic product convolution
* Added transfer and signed section families with finite companion matrix realizations
* Added reciprocal, logarithmic, congruential, Appell, and theta coefficient families
* Added exponentially scaled positive zero computation with residual and ordering certificates
* Added finite Hadamard products, coefficient diagnostics, zero count asymptotics, and truncated zero zeta sums
* Added quantum seed orbits, Schmidt rank, observable algebra rank, and computational basis addresses
* Added finite group and cyclic symmetry packets, charge selection, and tensor charge refinement
* Added circuit packet path expansions, interference identities, and block unitarity checks
* Added fixed leakage bounds, moving packet geometry, direct rotations, and finite Kato transport
* Added Kraus channels, Choi matrices, covariance checks, operator packets, and dephasing
* Added Lindblad generators and coherence sector preservation checks
* Added Knill Laflamme diagnostics, syndrome projectors, and explicit finite recovery channels
* Added Hermitian Feshbach reduction, hidden quantum memory, and upper half plane sign diagnostics
* Added finite Euclidean reflection positivity, periodic lattice spectra, constraint closure, and bounded Hamiltonian towers
* Added installed analytic and quantum verification suites and five executable Block 5 examples

## 0.4.0a1

Signals, semigroups, probability, and memory.

* Added one and multidimensional polyphase coordinates, finite convolution transfer blocks, and mixed radix addressing
* Added periodic and finite filter banks, lifting schemes, Haar packet trees, and exact additive best basis selection
* Added truncated power singular packets, retained packet reconstruction identities, overlap error formulas, and stable separated jump recovery
* Added nonlinear residue products, finite difference breakpoint bands, and cyclic Smith coordinates
* Added finite matrix and packet semigroups, Duhamel and resolvent refinement defects, and finite projective towers
* Added exact finite Markov lumping, obstruction diagnostics, stationary distributions, and transition models
* Added conditional expectation packet towers, Bayesian sufficiency, relative entropy decomposition, replicator flows, and coarse hidden Markov filtering
* Added positive spectral fractional operators, naturality and defect bounds, and discrete Dirichlet Green operators
* Added finite continuous and discrete Feynman Kac transfer
* Added Schur Feshbach reduction, rational memory kernels, auxiliary realizations, finite horizon stability bounds, and phase type survival
* Added installed signal and dynamics verification suites and five executable domain examples
* Added randomized Block 4 verification across signal and dynamics identities
* Added an independently testable NumPy matrix exponential fallback
* Added exact compact packet carry propagation and infinite polyphase kernel validation
* Added dimension aware vector jump confidence bounds and strict transient survival validation
* Separated exact Bayesian sufficient partitions from approximate likelihood profile clustering

## 0.3.0a1

Structured algorithms and exact finite compilation.

* Added contiguous packet layouts and a structured block sparse packet intermediate representation
* Added dense, diagonal, block diagonal, circulant, Toeplitz, Kronecker, and Walsh Hadamard operators
* Added exact block diagonal plus low rank products and Woodbury solves
* Added optimal rank constrained recompression through a rectangular small core singular value decomposition
* Added support multiplication, transitive closure, acyclic inverse support, Schur fill, and topological schedules
* Added Krylov bases, conjugate gradient, Lanczos tridiagonalization, and block Jacobi condition certificates
* Added continuous, clipped, and exact discrete precision allocation
* Added exact Lagrangian and integer budget tree compilers, Pareto frontiers, decision margins, and local tree updates
* Added task graph scheduling, shape batching models, and amortization thresholds
* Added arithmetic, storage, streaming traffic, and roofline cost models
* Added restricted fixed layout support and block diagonal plus low rank discovery
* Added NumPy execution and an optional CuPy packet executor
* Added matched benchmark contracts and timing scopes
* Added stricter validation for finite values, integer budgets, unique compiler names, packet sizes, and benchmark metadata
* Expanded installed verification with the complete Block 3 path

## 0.2.0a1

Refinement and obstruction foundation.

* Added exact packet refinement and finite refinement towers
* Added residue, nested residue, mixed radix, and Chinese remainder coordinates
* Added weighted transfer graphs and filtered inverse path expansions
* Added exact acyclic transfer inversion and weighted transfer series
* Added rational lattice packet data, p adic projector weights, gluing modules, and precision delays
* Added exact Smith decompositions and congruence solution counts modulo prime powers
* Added linear power, polynomial, resolvent, exponential, and multilevel refinement defects
* Added nonlinear packet coordinate conjugacy and nonlinear refinement defects
* Added exact finite dimensional cokernel obstruction computations
* Added exact linear and polynomial prime power lifting and finite lift trees
* Added finite matrix transport, gauges, fundamental holonomy, and intertwining certificates
* Added strict validation of prime parameters, Jacobian correction laws, and actual packet refinement hypotheses
* Expanded the command line verification interface for Block 2

## 0.1.0a1

Initial executable foundation.

* Added package metadata and command line entry point
* Added exactness aware result and certificate types
* Added NumPy and optional SymPy backends
* Added typed operator grammars and matrix realizations
* Added finite seed hull computation and seed rank utilities
* Added exact packet coordinates and packet operator blocks
* Added transfer graphs, support preorders, and morphism certificates
* Added the unified calculus system and feature registry
* Added extension discovery through Python entry points
* Added explicit placeholders for every later mathematical domain
