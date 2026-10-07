# Public API

Version `1.0.0b4` contains twenty numbered finite execution and stabilization blocks. The broad top-level and domain namespaces remain research surfaces. The versioned 1.0 candidate is the smaller 131-symbol `komorebi_math.stable` namespace and its aliases `komorebi_math.v1` and `komorebi_math.api.v1`; Blocks 18 through 20 do not add names to that frozen surface.

## Frozen 1.0 candidate

```python
import komorebi_math.stable as stable
import komorebi_math.v1 as v1
import komorebi_math.api.v1 as api_v1

assert stable.__all__ == v1.__all__ == api_v1.__all__
```

The committed stable manifest checks exported names, implementation kinds, qualified identities, and callable signatures. A separate closed fixture checks mathematical behavior, while the portable codec contract checks record tags, represented Python types, monotone codec versions, and every required adjacent migration.

```powershell
komorebi api check --stable
komorebi verify stable
komorebi release qualify
```

The stable surface supports a complete finite workflow: construct operators and seeds, analyze coordinates, decompose into packets, refine polyphase addresses, certify results, render mathematical records, and persist portable objects. Research namespaces remain available but do not acquire the same 1.x compatibility promise merely by existing in `komorebi_math.__all__`.

See `STABLE_API_1_0_CANDIDATE.md` for the exact policy.

## Research typed finite expression API

Block 20 is exposed through `komorebi_math.expression` and the broad research
namespace. It is not added to `komorebi_math.stable` or `komorebi_math.beta`.

```python
from komorebi_math.expression import (
    PolynomialCoordinateSpace,
    apply,
    derivative,
    evaluate_expression,
    exact_value,
)

monomials = PolynomialCoordinateSpace(
    "x", 2, ((1,), (0, 1), (0, 0, 1)), "display label"
)
polynomial = exact_value(monomials, (1, 2, 3))
result = evaluate_expression(apply(derivative(monomials), polynomial))
assert result.value == (2, 6)
```

The semantic identity of a polynomial coordinate type includes its complete
exact basis witness; labels do not participate. Closed terms normalize to exact
rational matrices or vectors, equality is decidable in this finite language,
and each reported rewrite is checked by exact normal-form replay. Expression
JSON uses a dedicated strict schema and is not silently inserted into the
frozen 51-codec persistence contract. Matrix-chain optimality is limited to
binary parenthesizations under the stated classical dense scalar-operation
model. See `TYPED_FINITE_EXPRESSION_CALCULUS.md` for scope and nonclaims.

## Research finite measure and integration API

Block 19 is exposed through `komorebi_math.integration` and the broad research
namespace, not through `komorebi_math.stable` or `komorebi_math.beta`.

```python
from komorebi_math.integration import (
    CompositeSimpson,
    ExactPolynomial,
    IntervalMeasure,
    integrate,
    polynomial_derivative_bound,
)

polynomial = ExactPolynomial.univariate((0, 0, 0, 0, 1))
measure = IntervalMeasure(0, 1)
bound = polynomial_derivative_bound(polynomial, measure)
result = integrate(polynomial, measure, provider=CompositeSimpson(16, bound))
result.certificate.require()
```

This Simpson result is rigorous because the integrand is the same
provenance-matched `ExactPolynomial`, the measure is the same exact rational
interval, and nodes, weights, evaluation, summation, derivative bound, and
error bound remain exact rational computations. Passing an arbitrary floating
callable instead gives an estimated Richardson statement without a bound or a
conditional statement with an explicitly assumed derivative bound. It does not
give a rigorous result merely because the callable implements a similar formula.

Positive measures (`AtomicMeasure`, `IntervalMeasure`, `ProductMeasure`, and
`PushforwardMeasure`) remain distinct from possibly signed quadrature
functionals. Exact FTC, integration by parts, finite Fubini, affine pushforward,
and integration-covector identities are available under their finite rational
hypotheses. See `FINITE_MEASURE_INTEGRATION.md` for the full scope boundary.

## General analysis--synthesis systems

```python
import numpy as np
from komorebi_math.stable import AnalysisSynthesisSystem

analysis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
system = AnalysisSynthesisSystem.canonical_dual(analysis)
system.certificate().require()

components = system.components()
assert np.allclose(sum(components), np.eye(2))
```

For a square invertible atlas, the maps `S E_I A` are algebraic projectors. For a redundant frame, they are additive reconstruction components and need not be idempotent. The compatibility methods `projector()` and `projectors()` remain available, but new code should use `component()` and `components()` unless idempotence is independently established.


## 1.0 freeze-candidate surface

```python
from komorebi_math.v1 import (
    AnalysisSynthesisSystem,
    OperatorGrammar,
    PolyphaseRefinement,
    check_stable_contract,
)

check_stable_contract().certificate.require()
```

Only the committed stable manifest receives the planned 1.x compatibility guarantee. Domain and research namespaces remain available without being silently frozen.

## Operator and packet foundation

```python
from komorebi_math import (
    OperatorAlgebra,
    OperatorGrammar,
    OperatorRealization,
    SeedSystem,
    PacketSystem,
    PacketOperator,
    TransferGraph,
    UnifiedSystem,
)
```

These classes implement typed operator systems, finite generated hulls, packet coordinates, transfer blocks, support graphs, and compatible finite runtime models.

## Refinement and obstruction

```python
from komorebi_math import (
    PacketRefinement,
    RefinementTower,
    WeightedTransferGraph,
    LatticePacketData,
    SmithCongruenceProblem,
    LinearRefinementDefect,
    NonlinearPacketMap,
    LinearObstructionProblem,
    LinearLiftProblem,
    PolynomialLiftProblem,
    TransportGraph,
)
```

The nonlinear namespace now includes finite derivative providers, packet jets, higher composition, rooted trees, graded operations, correction and elimination, filtered spectral sequences, and shared smooth transport. Infinite analytic and projective-limit conclusions remain conditional on explicit hypotheses.

## Advanced nonlinear calculus

```python
from komorebi_math import (
    CallableDerivativeProvider,
    PolynomialDerivativeProvider,
    PacketJetFamily,
    faa_di_bruno,
    PolynomialFixedPointProblem,
    NonlinearSchurElimination,
    LyapunovSchmidtReduction,
    FilteredVectorSpace,
    FiniteFilteredCochainComplex,
    KatoTransport,
)
```

Detailed result records and combinatorial utilities live under `komorebi_math.nonlinear`. See `NONLINEAR_CALCULUS.md`.

## Structured computation

```python
from komorebi_math import (
    PacketLayout,
    PacketIR,
    DenseOperator,
    BlockDiagonalOperator,
    LowRankOperator,
    BlockDiagonalLowRankOperator,
    TreeNode,
    DTypeChoice,
    Task,
    recompress_factors,
    conjugate_gradient,
    compile_tree_lagrangian,
    compile_tree_budget,
    assign_discrete_precisions,
    schedule_unlimited,
)
```

The complete structured API is under `komorebi_math.algorithms`.

## Signals

```python
import numpy as np
from komorebi_math.signals import (
    HaarPacketTree,
    PolyphaseSystem,
    SingularPacket1D,
    polyphase_analyze,
    recover_separated_jumps,
)

signal = np.arange(19, dtype=float)
coordinates = polyphase_analyze(signal, 4)
coordinates.certificate(signal).require()
assert np.array_equal(coordinates.reconstruct(), signal)

samples = np.concatenate((np.zeros(10), 2.0 * np.ones(10), -np.ones(10)))
recovery = recover_separated_jumps(
    samples,
    spacing=1.0,
    lipschitz_bound=0.0,
    noise_bound=0.0,
    minimum_jump=1.5,
)
recovery.certificate.require()
```

Full signal APIs include deterministic windows, finite framing, STFT and inverse STFT, spectrograms, periodogram and Welch estimators, cross spectra and coherence, FIR design and streaming, finite recursive filtering and stable biquads, FFT overlap-add convolution, rational resampling, LMS/NLMS/RLS adaptation, multilevel Haar denoising, polyphase coordinates, convolution blocks, filter banks, lifting, Haar packet trees, best basis selection, singular packets, scalar and vector jump recovery, finite difference break bands, nonlinear packet products, Fourier approximations, and Smith coordinates.

The finite convolution convention is signal first. `same` always returns the signal length, and `valid` returns zero samples when the kernel is longer than the signal.


## Normalized generating functions

```python
from komorebi_math.generating import (
    FiniteNormalizedSeries,
    FunctionRegistry,
    PrimePowerKernelCompiler,
    RationalDomain,
    compile_cartier_linear_representation,
    compile_finite_mask_carry_plan,
    factorial_power_basis,
)

basis = factorial_power_basis(2)
series = FiniteNormalizedSeries(basis, RationalDomain(), (1, 2, 3, 4))
series.certificate().require()

mask = compile_finite_mask_carry_plan((1, 1, 1), 2)
cartier = compile_cartier_linear_representation((1, 1), (1, -1), 5)
kernel = PrimePowerKernelCompiler(3, 5, 2)
```

The full namespace includes portable basis descriptors, exact domains, formal and domain-aware finite series, recurrence and algebraic sources, bounded closed-form recognition, residue sections, Mahler masks, finite carry matrices, rational Cartier automata, Bell digit reductions, Kummer layers, exact prime-power binomial residues, Gaussian q-Kummer support, Euler and cyclotomic collapse, portable frozen functions, and explicit FIR adapters. See `NORMALIZED_GENERATING_FUNCTIONS.md`.

## Dynamics

```python
import numpy as np
from komorebi_math.dynamics import (
    BayesianUpdate,
    FeshbachReduction,
    MarkovLumping,
    MatrixSemigroup,
    PhaseTypeSurvival,
    fractional_power,
)

generator = np.array([[-1.0, 0.0], [0.0, -2.0]])
semigroup = MatrixSemigroup(generator)
semigroup.semigroup_certificate(0.3, 0.7).require()

positive = np.diag([1.0, 4.0, 9.0])
root = fractional_power(positive, 0.5)
assert np.allclose(root, np.diag([1.0, 2.0, 3.0]))
```

Complete dynamics APIs include semigroups, refinement defects, finite towers, continuous and discrete Markov models, conditional expectations, martingales, Bayesian updates, exact sufficient partitions, approximate profile clustering, hidden Markov filtering, fractional operators, Green operators, Feynman Kac transfer, Feshbach reduction, memory kernels, and phase type survival.

The base wheel contains a NumPy matrix exponential fallback. It can be tested independently with

```python
from komorebi_math.dynamics import matrix_exponential
matrix_exponential(generator, method="taylor")
```

## Certificates

Every validation method returns a `Certificate`.

```python
certificate.valid
certificate.exactness
certificate.residuals
certificate.tolerance
certificate.assumptions_checked
certificate.assumptions_unchecked
certificate.diagnostics
certificate.require()
```

Execution classes are `exact`, `numerical`, `conditional`, `heuristic`, and `planned`.

## Configuration

```python
from komorebi_math import configure, get_settings, settings_context

configure(atol=1e-12, rtol=1e-10)
with settings_context(atol=1e-14):
    ...
current = get_settings()
```

## Provenance and feature status

```python
from komorebi_math import describe, feature, features

record = describe("komorebi_math.signals.recover_separated_jumps")
status = feature("signals")
all_statuses = features()
```

## Command line interface

```powershell
komorebi info
komorebi doctor
komorebi features
komorebi verify core
komorebi verify refinement
komorebi verify nonlinear
komorebi verify algorithms
komorebi verify signals
komorebi verify dynamics
komorebi verify block4
komorebi verify block9
komorebi verify hardening
komorebi verify block10
komorebi verify dsp
komorebi verify block12
komorebi verify generating
komorebi verify block13
komorebi verify coordinates
komorebi verify block14
komorebi api check --strict-signatures
komorebi verify block20
```

The `all` route is retained as a Block 17 compatibility alias. Use `block20`
for the current installed cumulative graph and the Block 21 publication
launcher for the source release gate.

## Extension discovery

Separately installed packages can register with the entry point group

```text
komorebi_math.extensions
```

See `EXTENSION_API.md` for the extension contract.

## Domain namespaces

All listed domain namespaces now contain production finite APIs. Some individual global or completion objects remain partial or conditional when they require external mathematical theorems. Feature status is available through `feature(...)`.

# Block 5 analytic and quantum API

## `komorebi_math.analytic`

The analytic namespace implements finite coefficient calculus and numerical diagnostics for the cyclic entire function families.

### Cyclic coefficient coordinates

```python
cyclic_project_coefficients(coefficients, order, residue)
cyclic_packets(coefficients, order)
cyclic_reconstruct(packets)
cyclic_project_value(function, z, order, residue)
cyclic_product_packets(left_packets, right_packets)
cyclic_projection_certificate(coefficients, order)
```

### Transfer and signed sections

```python
transfer_section(order, residue, z, scale=1)
signed_section(order, residue, z)
transfer_sections(order, z, scale=1)
signed_sections(order, z)
denominator(order, z)
denominator_derivative(order, z, derivative_order=1)
transfer_shift_matrix(order, scale=1)
signed_shift_matrix(order)
cyclic_addition(...)
section_certificate(...)
CyclicSectionFamily
```

### Reciprocal, Appell, and theta families

```python
logarithmic_coefficients(order, degree)
reciprocal_coefficients(order, parameter, degree)
reciprocal_polynomial(order, degree)
reciprocal_value(order, z, parameter=1)
transfer_tangent(order, z)
transfer_tangent_coefficients(order, degree)
congruential_inverse(order, residue, degree)
congruential_reciprocal_coefficients(order, residue, parameter, degree)
appell_polynomial(order, residue, parameter, degree, x)
appell_coefficients(order, residue, parameter, degree)
theta_polynomial(order, degree, parameter)
```

Symbolic polynomial constructors require the optional exact dependency.

### Zero diagnostics

```python
positive_zero_string(order, count, start=0)
asymptotic_positive_zero(order, index)
zero_count_asymptotic(order, radius, reduced=False)
hadamard_partial_product(order, z, positive_zeros)
reciprocal_coefficient_from_zeros(order, degree, positive_zeros)
reduced_zero_zeta(order, exponent, terms=64)
CyclicDenominator
```

A `ZeroStringResult` contains the computed representatives, asymptotic seeds, numerical residuals, and a certificate. The certificate does not assert exhaustion of the global zero set.

## `komorebi_math.quantum`

The quantum namespace implements finite dimensional matrix models only.

### Seeds and symmetry

```python
QuantumSeedSystem
schmidt_rank(state, left_dimension, right_dimension)
local_orbit_dimension(state, left_dimension, right_dimension)
observable_seed_rank(block_dimensions, multiplicities)
computational_basis_generation(qubits)
finite_group_projector(representation, character, irrep_dimension)
cyclic_symmetry_projectors(symmetry, order)
charge_certificate(operator, symmetry, order, charge)
charge_selection_certificate(packets, operator, charge)
total_charge_packets(left_packets, right_packets)
```

### Circuits and moving packets

```python
CircuitPacketExpansion
fixed_packet_leakage(hamiltonian, projector, time)
MovingPacketSystem
KatoTransport
direct_packet_rotation(source_packets, target_packets)
moving_geometric_coupling(hamiltonian, system, time, packet)
moving_packet_leakage_bound(...)
```

### Channels and error correction

```python
QuantumChannel
amplitude_damping(gamma)
operator_packets(operator, packets)
reconstruct_operator_packets(blocks)
dephase(operator, packets)
LindbladGenerator
QuantumCode
```

### Effective and spectral diagnostics

```python
QuantumFeshbachReduction
EuclideanTransfer
periodic_laplacian(shape, spacing=None)
periodic_lattice_eigenvalues(shape, spacing=None)
heat_trace(operator, time)
spectral_dimension(operator, time)
low_mode_dispersion_bound(shape, mode, spacing=None)
constraint_closure(constraints)
transported_constraint_defect(...)
BoundedHamiltonianTower
```

The quantum API returns finite matrix certificates. It does not report continuum quantum field theory, anomaly freedom, Lorentzian reconstruction, or gravitational infrared conclusions.

## Block 6 arithmetic API

The public arithmetic namespace includes `ModularRing`, `PrimeField`, `TruncatedPadic`, `HurwitzSeries`, `StandardWindow`, `EulerLocalCoordinates`, `MahlerSeries`, `UnitModeMahler`, `LinearRecurrence`, and `RecurrencePacket`.

Exact local computations use Python integers and fractions. Modular recurrence execution avoids constructing the full integer term when only a residue is required.

## Block 6 sequence API

The public sequence namespace includes `Sequence`, `CompiledSequence`, `AutomaticSequence`, and `HolonomicRecurrence`, together with the catalog and exact transforms. Compilation prefers a certified declared model, then a conditional declared model, then a finite detected recurrence, and finally the original evaluator.

## `komorebi_math.deformation`

```python
FiniteLiftForest
LiftBranch
LiftForestReport
VisibilityTree
RationalCountingSeries
FiniteCochainComplex
CohomologyData
PresentedModule
FittingGenerators
FiberSupportResult
FrobeniusSpine
CompanionForest
CompatibleCarrier
finite_frobenius_separator
nested_subspace_occupancy
```

Exact presentation and Fitting calculations require SymPy. Forests, visibility, counting, finite field cohomology, finite occupancy, and Frobenius separation run with the base NumPy installation.

## `komorebi_math.langlands.local`

```python
FiniteField
FiniteFieldElement
FrobeniusOrbit
TameWeilPacket
SpecialMonodromyBlock
GaussSumResult
HeisenbergModel
AlternatingForm
QuadraticRefinement
ProjectiveIntertwinerEnvelope
SymplecticBridge
LocalPacketBridge
```

These classes compute finite local packet data. They do not return an unrestricted local correspondence.

## `komorebi_math.langlands.finite_image`

```python
FiniteGroup
FiniteSubgroup
LinearCharacter
InducedRepresentation
MonomialAction
MackeyDecomposition
CliffordTower
AutomorphicInductionPlan
VirtualCharacterResolution
```

Finite group and orbit calculations are exact. Automorphic induction plans and unresolved Clifford extension steps are conditional.

## Ramification, global support, and completion

```python
ProjectiveMultiplicityPacket
RamificationStage
RamificationCliffordTower
FiniteHeckePackets
SupportComparison
PatchingSupportReport
AutomorphicCompletionReport
CompatiblePropagationReport
```

The completion API reports which external gates are supplied. It never treats a successful finite calculation as a proof of automorphy.


## `komorebi_math.export`

```python
DocumentMetadata
MathDocument
Heading
Paragraph
Equation
Table
CodeBlock
ListBlock
Figure
RawLatex
ExportFormat
ExportOptions
ExportResult
ExportBundleResult
LatexRenderOptions
HtmlRenderOptions
LatexEngine
LatexCompileOptions
LatexCompilation
to_serializable
table_rows
document_from_data
sequence_document
render_text
render_markdown
render_html
render_latex
export_data
export_document
export_many
export_bundle
available_export_formats
available_latex_engines
compile_latex
```

The base installation supports every format except YAML and PDF when their external backends are absent. YAML requires PyYAML. PDF requires a local supported LaTeX engine.

## Block 10 scalable nonlinear and compatibility API

### Sparse tensors and providers

* `SparseTensorEntry`
* `SparseSymmetricTensor`
* `SparsePolynomialDerivativeProvider`

### Automatic differentiation

* `DirectionalJVP`
* `AutomaticDifferentiationProvider`
* `JaxDerivativeProvider`
* `TorchDerivativeProvider`
* `autodiff_provider`
* `available_ad_frameworks`

### Packet-jet execution and snapshots

* `PacketJetInputGroup`
* `PacketJetExecutionPlan`
* `PacketJetEvaluation`
* `PacketJetBatchResult`
* `compile_packet_jet_plan`
* `execute_packet_jet_plan`
* `execute_packet_jets`
* `PacketJetCoefficient`
* `PacketJetSnapshot`
* `snapshot_packet_jet`

### Optional CuPy boundary

* `CuPyDeviceDerivativeProvider`
* `CuPyPolynomialDerivativeProvider`
* `CuPyPacketJetExecutor`
* `CuPySnapshotExecutor`
* `CuPyPacketJetSnapshotExecutor`
* `CuPyRuntimeStatus`
* `CuPyValidationResult`
* `cupy_runtime_status`
* `validate_cupy_snapshot`
* `nonlinear_cupy_available`
* `packet_jet_benchmark_contract`

### Portable persistence

* `PortableCodec`
* `register_portable_codec`
* `portable_codecs`
* `to_portable_data`
* `from_portable_data`
* `portable_digest`
* `dumps_portable`
* `loads_portable`
* `save_portable`
* `load_portable`
* `from_serializable`

### API compatibility

* `PublicApiSymbol`
* `PublicApiManifest`
* `ApiCompatibilityReport`
* `create_public_api_manifest`
* `load_public_api_manifest`
* `save_public_api_manifest`
* `compare_public_api`
* `check_public_api`
* `deprecated`
* `warn_deprecated`
* `KomorebiDeprecationWarning`

The version 0.12 research manifest covers explicit exports of `komorebi_math`, `komorebi_math.nonlinear`, `komorebi_math.export`, and `komorebi_math.signals`. It certifies structural compatibility only. The separate 83-symbol `komorebi_math.beta` manifest retains strict reflected signatures.

## Block 11 streams, migrations, semantics, and beta API

### Streamed packet jets

* `PacketJetStreamChunkRecord`
* `PacketJetStreamManifest`
* `PacketJetStreamWriteResult`
* `PacketJetStreamSelection`
* `StreamedPacketJetEvaluation`
* `PacketJetSnapshotStreamReader`
* `capture_packet_jet_stream`
* `save_packet_jet_snapshot_stream`
* `open_packet_jet_snapshot_stream`
* `load_packet_jet_snapshot_stream`

### Portable migrations

* `PortableMigration`
* `register_portable_migration`
* `portable_migrations`

### Golden semantics and profiling

* `BehavioralGoldenCaseResult`
* `BehavioralGoldenReport`
* `load_behavioral_golden_fixture`
* `run_behavioral_golden_fixture`
* `PacketJetComplexityEstimate`
* `RuntimeProfileSample`
* `RuntimeStageProfile`
* `PacketJetProfileReport`
* `estimate_packet_jet_complexity`
* `profile_operation`
* `profile_packet_jet_runtime`

### Curated beta surface

`komorebi_math.beta` re-exports 83 reviewed names from the finite certificate, packet, nonlinear execution, streaming, persistence, profiling, and compatibility layers. Existing beta signatures are checked strictly while additions remain permitted. The complete package remains alpha.

* `BETA_PUBLIC_NAMESPACES`
* `BETA_API_STABILITY_LEVEL`
* `DEFAULT_BETA_API_MANIFEST`
* `create_beta_api_manifest`
* `load_beta_api_manifest`
* `check_beta_api`

See [API compatibility contract for 0.11](API_COMPATIBILITY_0.11.md), [Resumable packet-jet streams](STREAMING_PACKET_JETS.md), and [Behavioral fixtures and finite profiling](BEHAVIORAL_GOLDEN_AND_PROFILING.md).

## General coordinate atlas

The complete coordinate API is available from `komorebi_math.coordinates`. It is organized around arbitrary coordinate systems rather than one named sequence conversion.

The namespace includes:

* exact finite bases, subspaces, atlases, path composition, and commuting diagrams
* normalized generating-function basis changes
* Stirling, Lah, binomial, forward-difference, divisor-zeta, and subset-zeta transforms
* classical, interpolation, and custom polynomial families
* arbitrary constant-coefficient recurrence models and state-space coordinates
* affine, barycentric, and formal changes of variable
* vector, covector, operator, bilinear-form, and higher tensor transport
* exact prime-field maps
* Fourier, cosine, Hadamard, Haar, and modal numerical transforms
* residue, digit-reversal, Chinese-remainder, tensor-axis, and custom packet permutations
* trusted portable exact and numerical coordinate records

```python
from komorebi_math.coordinates import (
    CoordinateAtlas,
    basis_from_matrix,
    connection_transform,
    get_polynomial_family,
    get_recurrence_model,
    standard_basis,
)

standard = standard_basis(3, name="standard")
shear = basis_from_matrix("shear", ((1, 1, 0), (0, 1, 1), (0, 0, 1)))
standard.connection_to(shear).certificate().require()

legendre = get_polynomial_family("legendre")
chebyshev = get_polynomial_family("chebyshev-t")
connection_transform(legendre, chebyshev, 8).certificate().require()

pell = get_recurrence_model("pell")
tribonacci = get_recurrence_model("tribonacci")
pell.coordinate_transform_to(tribonacci, 20).certificate().require()
```

```powershell
komorebi verify coordinates
komorebi verify block14
komorebi coordinates catalog
komorebi coordinates polynomial --source legendre --target chebyshev-t --degree 8
komorebi coordinates model --source pell --target tribonacci --count 20
```

Euler-to-Fibonacci helpers remain available as a named regression example. They do not define the coordinate namespace.
