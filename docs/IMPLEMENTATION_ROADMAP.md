# Production implementation roadmap

## Block 1: finite foundation

Status: implemented in version `0.1.0a1` and retained in `0.13.0a1`.

Included:

* package metadata and wheel build
* NumPy backend
* optional exact SymPy backend
* result and certificate types
* provenance registry
* feature registry
* extension entry point discovery
* typed grammars and realizations
* generated matrix algebras
* finite seed systems
* exact packet systems
* packet operators
* transfer graphs
* support preorders
* operator, packet, and seed morphisms
* unified finite system
* command line inspection and verification

## Block 2: refinement and obstruction foundation

Status: implemented in version `0.2.0a1`.

Included:

* exact packet refinements
* finite refinement towers
* common refinements of commuting packets
* residue and nested residue coordinates
* Chinese remainder addresses
* weighted transfer graphs
* filtered inverse expansions
* exact acyclic transfer inversion
* lattice projector weights
* declared split lattice gluing modules
* Smith congruence solvers
* exact precision delay
* linear refinement defects
* nonlinear packet coordinate maps
* nonlinear refinement defects
* finite dimensional cokernel obstructions
* exact prime power correction spaces
* linear and polynomial lift trees
* finite matrix holonomy

These foundational nonlinear interfaces are retained for compatibility. Their higher-order extensions—packet jets, Faà di Bruno composition, rooted expansions, nonlinear elimination, filtered obstruction spectral sequences, and shared smooth transport—are implemented in Block 9.

## Block 3: structured computation

Status: implemented in version `0.3.0a1`.

Included:

* contiguous packet layouts
* packet intermediate representation with structured blocks
* dense, diagonal, block diagonal, circulant, Toeplitz, Kronecker, and Walsh Hadamard operators
* low rank operators, exact BDLR products, Woodbury solves, and optimal recompression
* support multiplication, powers, polynomial support, closure, Schur fill, and acyclic scheduling
* Krylov bases, conjugate gradient, Lanczos, and packet block Jacobi certificates
* continuous, clipped, and exact discrete precision allocation
* exact Lagrangian and quantized budget tree compilers
* Pareto frontiers, decision margins, shared objectives, and local tree updates
* task graph scheduling, shape batching, and amortization thresholds
* arithmetic, storage, streaming, and roofline cost models
* restricted fixed layout discovery
* NumPy packet execution
* optional CuPy packet execution when a compatible external CuPy package is installed
* matched benchmark contracts and timing scopes

The following remain outside Block 3:

* unrestricted automatic packet discovery
* global optimality for overlapping or nonlocal representation families
* a vendor neutral GPU array standard beyond the current executor boundary
* device measurements on hardware not present in the execution environment
* distributed and multi device scheduling

## Block 4: signals and dynamics

Status: implemented in version `0.4.0a1`.

Included:

* one dimensional and lattice polyphase arrays
* finite convolution packet blocks
* filter banks, lifting, wavelet packet trees, and best basis recursion
* singular packet reconstruction and certified separated jump recovery
* nonlinear signal packet products and Smith coordinates
* finite matrix and packet semigroups
* Duhamel, resolvent, and multilevel refinement defects
* Markov lumping and conditional expectation packets
* Bayesian sufficiency, entropy decomposition, and coarse filters
* fractional operators and Green reconstruction
* finite Feynman Kac transfer
* Feshbach reduction, rational memory, and survival models

The finite implementations do not claim continuum semigroup construction beyond the declared matrix and finite tower hypotheses. Signal recovery certificates keep unknown regularity and noise assumptions explicit.

## Block 5: analytic and quantum realizations

Status: implemented in version `0.5.0a1`.

Included:

* cyclic coefficient projections and exact residue reconstruction
* transfer and signed cyclic sections with finite companion matrix realizations
* reciprocal, logarithmic, congruential, Appell, and theta coefficient families
* numerical positive zero strings with residual certificates
* finite Hadamard products, coefficient diagnostics, zero counts, and truncated zeta sums
* finite quantum seed orbits and observable algebra rank formulas
* finite group and cyclic symmetry packets with tensor charge refinement
* circuit packet paths, interference identities, and block unitarity
* fixed leakage bounds, moving packet geometry, direct rotations, and finite Kato transport
* finite Kraus channels, Choi matrices, covariance, and Lindblad coherence sectors
* Knill Laflamme diagnostics and explicit finite recovery channels
* quantum Feshbach reduction and hidden packet memory
* finite Euclidean reflection positivity and periodic lattice spectral diagnostics
* finite constraint closure and bounded Hamiltonian tower diagnostics

Numerical zero routines certify the computed residuals and ordering only. They do not certify exhaustion of all large zeros. Quantum continuum endpoints remain outside the implemented finite matrix layer.

## Block 6: arithmetic and sequences

Status: implemented in version `0.6.0a1`.

Included:

* modular rings, prime fields, and truncated local domains
* exact valuations and Carmichael exponents
* Hurwitz series, exponential modes, subgroup selectors, and exact inverses
* uniform residual standard windows and rational periodic generating functions
* unified odd and dyadic Euler coordinate APIs with separate integral charts
* sharp dyadic kernel weights and local truncation certificates
* Mahler analysis, affine transfer, polyphase coordinates, and prime power sparsity
* local unit mode difference gains and simultaneous congruence certificates
* exact linear recurrences, residue packets, modular terms, and finite period detection
* lazy exact sequences and a classical integer sequence catalog
* binomial, inverse binomial, difference, product, and partial sum transforms
* finite automata for automatic sequences
* polynomial coefficient recurrences for holonomic sequences
* execution compilation that distinguishes declared models from finite pattern detection

The sequence compiler does not promote a recurrence guessed from finitely many terms to an exact infinite model. User supplied models remain conditional unless the caller supplies trusted provenance or the package catalog declares them certified.

## Block 7: deformation and Langlands finite models

Status: implemented in version `0.7.0a1`.

Included:

* residual lift forests and stable finite cores
* valuation visibility trees and exact local counting charts
* finite cochain complexes and packet cohomology
* Fitting generators, finite fibers, nested occupancy, and Frobenius separation
* exact finite fields, tame Frobenius packets, monodromy chains, and Gauss sums
* finite Heisenberg systems, symplectic forms, dyadic quadratic refinements, and projective envelopes
* explicit finite groups, induced monomial representations, Mackey packets, and orbit decompositions
* Clifford towers and conditional automorphic induction plans
* ramification adapted finite Clifford towers
* finite Hecke packets, support comparisons, and patching support reports
* five gate conditional completion and compatible propagation reports

The package does not construct arbitrary deformation rings, arbitrary wild local packets, an automorphic representation, or the full Langlands correspondence. It computes exact finite data and exposes unmet external assumptions.

## Block 8: portable output and LaTeX compilation

Status: implemented in version `0.8.0a1`.

Included:

* portable normalization of exact and numerical Python objects
* JSON, optional YAML, CSV, TSV, text, Markdown, HTML, LaTeX, PDF, NPY, and NPZ exports
* renderer independent mathematical documents
* prose, equations, tables, code, lists, figures, and explicit raw LaTeX blocks
* local LaTeX compiler discovery
* LuaLaTeX, XeLaTeX, PDFLaTeX, and latexmk support
* shell escape disabled by default
* compiler timeouts, logs, repeated passes, and auxiliary cleanup
* multi format ZIP bundles with normalized metadata and SHA256 manifests
* export command line tools and PowerShell smoke tests

## Block 9: integration, advanced nonlinear calculus, and release stabilization

Status: implemented in version `0.9.0a1`.

Included:

* derivative providers and validation
* packet jets and Faà di Bruno composition
* rooted nonlinear operator trees and majorant certificates
* finite graded operations and Maurer-Cartan packet identities
* Hilbert obstruction projection and right-inverse correction
* nonlinear Schur and Lyapunov Schmidt elimination
* finite filtered lifting and obstruction spectral sequences
* smooth transport shared between nonlinear and quantum namespaces
* public documentation source and pre-beta API stability review
* installed verification, clean-install validation, and independent deep audit

Optional GPU execution was not validated in the Block 9 release container because no CUDA device or CuPy installation was available. The optional CuPy executor remains isolated from the NumPy nonlinear runtime.


## Block 10: scalable nonlinear execution and pre-beta hardening

Status: implemented in version `0.10.0a1`.

Included:

* canonical sparse storage for symmetric derivative tensors
* sparse polynomial derivative providers with exact and numerical contractions
* framework-neutral directional JVP providers
* lazy JAX and PyTorch adapters through finite higher order
* projection-cached packet-jet execution plans
* conservative support pruning that requires complete certificates
* provider-independent packet-jet snapshots with finite evaluation guards
* explicit CuPy runtime status and no silent CPU fallback
* versioned, digest-checked persistence for selected finite nonlinear records
* safe built-in marker decoding without arbitrary class instantiation
* deterministic public API manifests and compatibility reports
* structured deprecation metadata and warnings
* Linux and Windows CI definitions for Python 3.11 through 3.13
* installed verification and an independent randomized deep audit

The Block 10 release container did not contain CuPy or a CUDA device. GPU execution remains an optional runtime path and has no hardware-backed release claim. API compatibility checks structural exports and reflected signatures, not mathematical semantics. Snapshot capture remains finite and can grow exponentially with derivative order.

## Block 11: resumable persistence, semantic regression, and beta compatibility

Status: implemented in version `0.11.0a1`.

Included:

* independently digested directory-backed packet-jet streams
* budgeted direct capture with pause, resume, and verified committed prefixes
* bounded chunking of existing packet-jet snapshots
* one-chunk-at-a-time iteration and guarded materialization
* explicit complete and partial-range evaluation certificates
* adjacent portable-codec migrations and packet-snapshot codec version 2
* a closed five-case behavioral golden fixture
* exact finite structural counts separated from local runtime observations
* an additive strict-signature `komorebi_math.beta` namespace
* separate research and beta API manifests
* executable local-environment and optional CUDA validation programs
* installed verification, CLI integration, examples, and randomized deep auditing

Streaming changes interruption tolerance and peak materialization, not the exponential number of standard-basis coefficients. Digest chains provide integrity, not signatures. Direct resume of an arbitrary Python derivative provider cannot establish unchanged callable semantics without an external provider identity mechanism. The local release environment did not execute Windows, hosted CI, CuPy, or CUDA paths.

## Block 12: production finite signal processing

Status: implemented in version `0.12.0a1`.

Included:

* deterministic finite window families, window metrics, periodic overlap profiles, COLA and NOLA diagnostics, and canonical dual windows
* finite uniform framing, STFT analysis, normalized overlap-add inverse synthesis, and spectrogram scaling
* one-sided and two-sided periodograms, Welch averaging, cross spectra, coherence, H1 transfer estimates, and Parseval certificates
* odd-length linear-phase windowed-sinc FIR design with direct and stateful streaming execution
* explicit rational polyphase resampling with direct-reference equivalence certificates
* finite LMS, normalized LMS, and recursive least-squares adaptation
* multilevel Haar decomposition, hard, soft, and garrote thresholding, universal and SURE rules, and observed-risk certificates
* chunk-invariant streaming STFT analysis with exact finite framing agreement
* finite direct-form II transposed FIR/IIR filtering, continuation state, stable biquad construction, and pole-radius certificates
* zero-padded FFT convolution and bounded-buffer stateful overlap-add execution
* dedicated source, installed-package, CLI, API, randomized deep verification, and optional SciPy reference comparison

The block provides finite offline and chunked computations. It does not assert real-time deadlines, stochastic adaptive-filter convergence, denoising optimality outside declared models, or device-backed GPU acceleration.

## Block 13: normalized generating functions and arithmetic compilers

Status: implemented in version `0.13.0a1`.

Included:

* exact coefficient domains and normalized finite series
* declarative custom generating-function sources
* Mahler mask construction, inversion, analytic realization, and Frobenius defects
* finite-mask and ordinary-product carry compilers
* rational Cartier linear representations and minimized automata
* Kummer support layers and prime-power kernel residues
* Bell digit reductions and residue automata
* Gaussian q-Kummer valuations and cyclotomic collapse
* trusted finite persistence and finite signal adapters
* CLI, compatibility, installed verification, and independent deep auditing

## Block 14: general coordinate atlas

Status: implemented in version `0.14.0a2`.

Included:

* arbitrary exact and numerical finite bases
* coordinate atlases with certified forward and inverse paths
* recurrence, companion-matrix, and state-space coordinates
* classical polynomial and combinatorial family connections
* affine, formal, finite-field, tensor, spectral, and packet coordinates
* portable trusted coordinate records
* dedicated source, CLI, installed-wheel, and compatibility verification

Block 14 was the final broad mathematical feature block before the original 1.0
freeze sequence. Blocks 18 and 19 later added research-only mathematics without
reopening the frozen stable surface.

## Block 15: canonical polyphase stabilization

Status: implemented in version `0.15.0a2`.

Included:

* canonical one-dimensional signal refinement and coarsening
* exact arithmetic polyphase refinement
* mixed-radix parent and child addresses
* rectangular lattice refinement
* multilevel refinement towers
* portable polyphase records
* curated stable namespace and strict manifest groundwork
* crash-isolated optional automatic-differentiation probes

## Block 16: API freeze candidate and release qualification

Status: implemented in version `0.16.0a2`.

Included:

* a general finite analysis--synthesis runtime with checked left inverses
* reconstruction components and finite transfer-block expansions
* portable analysis--synthesis records
* identity aliases `komorebi_math.stable`, `komorebi_math.v1`, and `komorebi_math.api.v1`
* strict stable API manifests with deterministic snapshots
* a ten-case closed semantic compatibility fixture
* a package-owned portable-codec inventory and adjacent migration checks
* aggregate structural, semantic, and persistence compatibility reports
* machine-readable platform and optional-backend support policy
* cumulative Block 15 and Block 16 installed verification
* stable, release, and GPU CLI gates
* crash-isolated PyTorch, CuPy, and Komorebi CUDA qualification
* source, installed-wheel, source-distribution, CI, documentation, and handoff scripts for the freeze-candidate line

## Block 17: public-beta contract freeze

Status: implemented in version `1.0.0b1`.

Included:

* the frozen 131-symbol stable namespace and its versioned aliases
* strict structural signatures, ten closed semantic cases, and 51 portable codecs
* predecessor upgrade, platform-evidence, documentation, and readiness gates
* a retained immutable `1.0.0b1` contract baseline

## Block 18: redundant-frame coefficient algebra

Status: implemented as a research-only addition in version `1.0.0b2`.

Included:

* coefficient-space lifts for redundant analysis--synthesis presentations
* quotient identities, cross-presentation bridges, and compositional coefficient blocks
* exact rational frame systems and an explicit distinction between physical and dual adjoints
* an additive research API with no change to the stable or beta namespaces

## Block 19: finite measure and integration calculus

Status: implemented as a research-only addition in version `1.0.0b3`.

Included:

* exact sparse rational polynomials and positive atomic, interval, product, and pushforward measures
* signed quadrature functionals kept distinct from positive measures
* exact moments, FTC, integration by parts, finite Fubini, affine pushforward, and coordinate-covector transport
* provenance-matched exact-rational Simpson bounds; arbitrary floating callables remain estimated or conditional
* 12 focused installed checks, a 24-check aggregate, 47 adversarial cases, and 16 independent mathematical checks
* the transported worked example with exact value `106/5`

## Block 20: typed finite expression calculus

Status: implemented as a research-only addition in version `1.0.0b4`.

Included:

* immutable semantic spaces identified by complete exact basis witnesses
* a typed finite linear-expression IR with exact rational values and maps
* replayable polynomial primitives for coordinates, differentiation,
  basepoint antiderivatives, evaluation, integration, affine substitution,
  finite differences, boundaries, and constant injection
* unique exact matrix/vector normal forms and decidable equality in the language
* conservative rewrite certificates, including the correct `J_a D` defect
* strict portable expression JSON with alpha-invariant lexical digests
* exact matrix-chain dynamic programming under a declared dense cost model
* independent direct and compiled evaluation paths with entrywise comparison

## Remaining path to 1.0

The `1.0.0b1` through `1.0.0b4` checkpoints are complete. The first froze the
stable contract; Blocks 18 through 20 are additive research betas and do not
promote their new names into that contract.

### `1.0.0rc1`

* build from a clean signed or otherwise authenticated release tag
* pass source, wheel, source-distribution, stable, persistence, and documentation gates
* execute the declared Python 3.11 through 3.13 CPU matrix on Windows and Linux
* pass a current recorded NVIDIA CUDA qualification when CUDA support is claimed
* produce final dependency, environment, and artifact manifests

### `1.0.0`

* change only approved version and release metadata from the successful release candidate
* create a new release candidate after any code change

## Release policy

A block is complete only when:

* public APIs have type annotations and docstrings
* exact, numerical, conditional, heuristic, and planned paths remain distinguishable
* every stable formula has behavioral regression coverage
* wheel installation passes outside the source checkout
* command line verification passes after installation
* optional native dependencies are probed in isolated child interpreters
* no production module imports paper-verification code
* the source distribution can rebuild an equivalent wheel
* portable codec changes include adjacent migrations
* stable API and semantic manifests are regenerated intentionally
* archive contents and SHA-256 records verify after fresh extraction
