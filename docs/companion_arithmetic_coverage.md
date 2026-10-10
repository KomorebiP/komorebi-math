# Current companion machinery: coverage and boundaries

This coverage-only audit reads the current October 8 traditional editions of
Quadratic-Form Engine and K-COS, Cubic Divisor Transport and Reconstruction
(Hilbert 12), Stratified Universality for Resolved Representations, and Signal
Reconstruction. The source hashes, exact sections, qualified APIs, relevant
tests, and construction boundaries are recorded in
`verification/paper_coverage/companion_arithmetic.json`. Every PDF was extracted
in full; the audit inventories its constructions and reads the corresponding
definitions and contracts. It is not an independent proof audit of every
ordinary theorem or a fresh replay of each paper's source bundle. The crosswalk
contains 70 scoped mechanism entries. All source hashes, cited test files and
qualified API references were checked against the current copies and checkout.

The old deep-audit crosswalk is a discovery aid. Its Stratified Blocks 17–26
semilattice, matroid, Čech, and extension-gauge scripts belong to separate
related research. Those sections are absent from the current 262-page
Stratified v2.0.0 manuscript and are not assigned invented current-paper
references. `core.semilattice` retains its own provenance.

## Quadratic forms: a substantial integrated subset

The optional `quadratic_forms` namespace already executes the positive central
directed backend. It encloses the finite tilted COS sum, adds support/alias and
infinite-mode bounds, and accepts a requested multiplicative width only from
outward literal endpoints. Automatic refinement fails closed at the supplied
caps. The matrix bridge certifies candidate eigensystems using directed
orthogonality and reconstruction residuals, then propagates coefficient
intervals monotonically into the tail.

The exact-spectrum domain is independent standard Gaussian coordinates with
strictly positive exact decimal coefficients and rank greater than two. The
automatic controller requires a threshold above the exact supplied mean. The
explicit evaluator additionally requires valid positive tilt, support, and
auxiliary Chernoff parameters. Strings or `Decimal` values express decimal
intent; float inputs use their string decimals. The stored-factor bridge
targets the Gram matrix of the entries after conversion to binary64, treated
as exact dyadic values. It uses a dense full basis and is intended for modest
dimensions. Precision/mode/support caps bound individual attempts, not total
elapsed time.

This enclosure depends on the qualified mpmath 1.3.0 interval arithmetic and
transcendentals, Python integers, and directed endpoint serialization. It is
separate from the external Lean companion's finite exact-real algebra and
supplied-bound consequences. Python does not inherit kernel verification
from that project.

| Paper construction | Current coverage |
|---|---|
| Positive central directed tails, requested width, independent Erlang control, stored-factor spectrum and whole-interval matrix tail | Implemented under the stated numerical trust and domain contracts |
| Matrix-free actions, finite phase paths, power perturbation bounds, small-core low-rank recompression | Reusable finite numerical interfaces; partial coverage of the paper's integrated streaming architecture |
| Reader-backed residualized forward/adjoint/dual actions, sample/variant tiling, phase-to-I/O scheduling and asynchronous buffers | External source machinery; no general package implementation |
| Phasewise transfer envelopes and query-weighted edge-risk scheduling | Scalar power bound exists; full envelope and scheduling frameworks remain outside the API |
| PSD core/residual moments, optimized residual MGF, rank bracket and streamed exact completion | External source machinery |
| Query-dependent centered spectral moment packets and their propagated tail-error passport | External analytic/binary64 prototype; the directed backend uses the exact supplied spectrum |
| Signed and positive noncentral Gaussian tail engines | External source engines; the integrated positive central API rejects those domains |

The source's stronger frequency-dependent omitted-mode result in §26 is also
broader than the integrated §4 rank-greater-than-two remainder. Archived
production spectra, application simulations, hardware timings and the full
extreme-tail benchmarks are preserved source evidence; they are not relabeled
as fresh results of this coverage task.

## H12: executable research remains external

The current H12 manuscript fixes a specific cubic embedding, fifth-smoothed
elliptic gamma function, principal-seven correction, and an independently
defined algebraic candidate. Its computable constructions include coprime
scalar normalization, fifth-torsion rational formulas, signed unit carries,
finite multiplication norms, integral lattice lifts and ceiling coefficients,
four cubic-subfield norms, and exact rational quotient reductions. The generic
package does not currently expose the specialized elliptic, number-field,
ceiling-divisor, or phase-transport constructors. General matrix, Smith,
finite-group, or polynomial primitives are supporting tools; their presence
does not implement these particular constructions.

The infinite machinery has further obligations: complete complex divisor
interpretation, cubic counting, the ungrouped absolute genus-three product,
fourth logarithmic derivative, reflection/transition phase coefficients and
branch-free special cube. The explicit product-tail bound requires an actual
all-radius counting constant and cutoff. A finite integer search does not
establish those hypotheses, and the current package does not evaluate the
regularized infinite product at one third.

The precise remaining arithmetic assertion is
`Im(C_pol * U_gamma**3) == 0`. The source leaves it open. A numerical interval
containing zero does not prove reality. Actual complex conjugation and the
separate arithmetic involution must remain distinct. The conditional torus
rigidity route additionally needs a periodic relative multiplier, its based
arithmetic identification, and its norm equation; scalar-plane divisor repair
does not supply them. This coverage makes no claim of a gamma/class-field
identity or a general Hilbert 12 solution.

Source-reported current/inherited Lean checks and exact/floating replays are
external evidence here. The current integer Lean subset does not formalize
complex orders, analytic counting, canonical products or gamma arithmetic.
The inherited individual declaration checks are distinct from the reported
incomplete combined import process. This audit runs no H12 source bundle or
Lean compiler.

## Stratified and signal interfaces

The current Stratified paper describes profiled strata, bridge passports,
normalized atlases, finite and completed lane transport, task-relative
quotients, carrier change, and metric/enrichment constraints. Existing runtime
APIs execute finite grammar relations, packet analysis and transfer,
refinement, masks/Cartier carries, supplied lattices and Smith data, Mahler
prefixes, marked fixed-point expansions, finite observers and carrier fibers.
The crosswalk records each narrower domain. These APIs do not construct a
general profiled-stratum/descent orchestrator, strict shortlex regular
normalizer, Ore/Fuchsian atlas, bridge-loss/relation-module wrapper, or global
tagged invariant compiler. Infinite, completed, nonsplit and heterogeneous
continuum resolutions still require their stated exactness, convergence and
coherence doctrines.

Signal Reconstruction's consecutive Fourier pipeline is executable through
`signals.fourier_recovery`: derivative moments, Hankel rank, annihilator roots,
Vandermonde amplitudes, and a periodic mean-zero right-continuous jump packet.
The implementation is numerical. Inferred order gives a fitted candidate;
declared-order approval remains conditional on the model and conditioning.
The finite fit does not certify that arbitrarily close hidden edges were
resolved. The paper's polyphase self-calibration, matched multi-offset branch
lifting, residualized sparse dictionary certificates, validated noisy root
disks, shared-location multichannel/confluent recovery and adaptive geometric
image pipeline remain partial or external. A finite FFT or known-edge
synthesis primitive does not supply those discovery algorithms. The corrected
periodic sawtooth L1 bound is distinct from the package's nonperiodic step
location error formula.

## Fresh verification

Run the integrated quadratic-form and related finite numerical controls with:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_quadratic_forms.py tests/test_quadratic_forms_source.py \
  tests/test_algorithms_structured.py tests/test_algorithms_low_rank.py \
  tests/test_algorithms_krylov.py
```

This checkout run passed 84 tests. Controls include five equal-coefficient
Erlang comparisons, ordinary SciPy comparisons, directed literal-width and
serialization failures, positive/domain rejection, resource exhaustion,
stored diagonal and nondyadic factor spectra, a complete matrix-tail control,
generator inputs, context restoration and concurrent operation serialization.
The related operator tests verify finite actions, Krylov residuals and
small-core recompression. They do not replay all paper experiments or prove
the analytical theorems universally.

Stratified and Signal coverage includes source-contract reading and validation
of API imports and cited test-file existence. This subtask did not rerun their
whole domain test suites; the file mapping is evidence of available controls,
not a fresh execution claim.
