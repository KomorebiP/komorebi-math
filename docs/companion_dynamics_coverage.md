# Mathematical companion coverage

The source-hashed crosswalk in
`verification/paper_coverage/companion_dynamics.json`
maps four traditional papers to actual finite APIs and tests. All four supplied
source texts were read. This is an implementation-coverage extraction, not a
full proof audit or a fresh Lean build.

| Source, edition | Existing reusable mathematical sectors | Remaining companion machinery |
|---|---|---|
| **Certified Bayesian Inference**, 8 October 2026, Core 1.0.0 | Rational factor-coded Gaussian model and exact gain solve; Joseph candidate and residual score; complete two-sided packet energy; distinct direct final-value dyadic score/parameter verification | Exact selected residual cache and fuel; source operation-by-operation dyadic graph, coefficient budgets, fallback and formal result-array equality |
| **Universal Generative Deformation Calculus**, 8 October 2026, archived 1.1.0 plus separate corrected reference | Finite typed linear paths; ordered trees/Catalan counts; exact sequence sums and prefix recurrence fitting; finite packet composition; prime polynomial lift fibers and residue forests | General admissible route explorer retaining actual route endpoints; continuation-domain/successor quotient checker; parameter-family laws; paired admissible covers and affine envelopes; fair countable scheduling |
| **Quantum–Arithmetic Bridge**, 8 October 2026, QA01–QA20 | Generic exact rational spectral packets, packet reconstruction, residue/polyphase lanes, finite circuit interference and dephasing | Dedicated Ramanujan/order-shell labeling and Fourier alias formulas; literal modulus-twelve bridge fixture; exact radical tuple coordinates; conditional Lüders branches and source phase-family probabilities |
| **Fourier Packets and Hilbert Stability**, 8 October 2026, PH01–PH12, Block 44 frontier | Four-spinor Clifford matrices, centered mode momentum, finite Dirac symbol and closed propagator; general finite semigroup/Duhamel numerics | Dedicated spectral projectors, torus packet/isometry fixture, quadratic consistency envelopes, normed coordinate bridge, sharp all-real-time Hermitian Duhamel bound and Hamiltonian-expectation contract |

The Bayesian paper concerns one **Gaussian** observation update. The new
[finite Gaussian verifier](finite_gaussian_verifier.md) supplies exact
factor/gain/Joseph/score and complete packet-energy algebra. Its independent
dyadic check first computes exact final rational values, then encloses them;
it is distinct from the source's operation-by-operation rounded program.
The existing finite probability-vector `BayesianUpdate`, `BayesianPartition`
and `HiddenMarkovFilter` use different objects. In the
paper, law tolerance bounds the distance from the returned Joseph law to an
explicit posterior version; parameter radii enclose that returned law's
parameters. Neither conclusion certifies a new Gaussian formed by discarding
the radii. Its exact cache also cannot survive independent rounding of gain and
residual centers without a new enclosure invariant.

For generative deformation, endpoint grouping and future expansion have
different obligations. A labeled partial action descends only when equivalent
representatives have the same enabled labels and equivalent successors. Ordered
tree and finite lift APIs cover useful restricted sectors; they do not supply a
general state-dependent explorer. A finite prefix, family sample or batch
diameter cannot establish an all-word family law, an infinite recurrence or an
omitted summable tail. Completion needs a complete target, and validity at a
limit needs a separate closedness argument.

Quantum character-order shells can be constructed exactly from a rational
cyclic shift and cyclotomic factors using `rational_spectral_projectors`.
That generic construction is recorded as related coverage; it does not imply
that the companion's Ramanujan kernel, order labels or literal bridge have
dedicated APIs/tests. Sample-residue lanes and character-order shells select
different coordinates. Numerical interference and dephasing remain distinct
from the source's exact radical arithmetic certificate.

Dirac APIs operate on supplied finite matrices and modes. The Physics source's
fixed-support limits, continuum Hilbert interpretation and unbounded domains
remain separate. Existing `dirac_modes.md` names the **6 October** source set;
this crosswalk pins the distinct **8 October** traditional edition. Later
**Block 49 Lean** material is a separate source layer and is not counted as a
Python API or a newly replayed kernel proof.

The mapped packet, operator, grammar, tree, sequence, lift-forest and discrete
Bayesian suites were replayed together: **47 tests passed**. The Gaussian
slice adds **96 tests**, with independent full inverse/Schur references,
completed-square and PSD controls, direct scalar packet sums, coarse-score
rejection and mutation/arithmetic/resource checks. The crosswalk lists
additional source-relevant tests without treating their finite checks as
universal verification of the papers.
