> **Current development audit:** R4 Block 7 (`4b7`) repairs portable-certificate persistence so loaded validity is historical evidence requiring producer-specific revalidation. This is not final R4. See `REPAIR_BLOCK7.md` and `KNOWN_ISSUES_BLOCK7.md`.

# Komorebi Math

> **Release status:** `1.0.0b4` is the Block 20 research beta. The typed finite expression calculus is additive at the research tier; the 131-symbol stable namespace remains frozen.


`komorebi-math` is the executable Python package for the Komorebi Math research program. It implements the mathematical constructions as reusable objects, functions, certificates, and command line checks. The installed wheel does not contain article PDFs, LaTeX manuscripts, or paper figures.

Version `1.0.0b4` contains the prior finite execution blocks plus the Block 18 redundant-frame coefficient closure, Block 19 finite measure/integration calculus, Block 20 typed finite expression calculus, and the Block 21 external reproducibility gate:

1. typed operator grammars, finite seed systems, packet coordinates, transfer blocks, and support graphs
2. refinement towers, residue coordinates, weighted transfer, exact lattice separation, Smith congruences, finite obstructions, prime power lifting, and finite holonomy
3. packet intermediate representations, structured operators, low rank algebra, support propagation, Krylov methods, precision planning, exact tree compilation, scheduling, and optional CuPy execution
4. finite signal realizations, singular packets, semigroup refinement, Markov and Bayesian coarse models, fractional operators, Feshbach reduction, memory kernels, and survival models
5. cyclic analytic sections, reciprocal and Appell families, zero diagnostics, quantum symmetry packets, circuit paths, channels, Kato transport, error correction, effective dynamics, and finite spectral diagnostics
6. local modular domains, Hurwitz series, standard windows, odd and dyadic Euler coordinates, Mahler transfer, recurrence packets, automatic sequences, holonomic recurrences, and exact sequence compilation
7. residual lift forests, visibility and counting charts, finite cohomology packets, Fitting fiber tests, Frobenius companion diagnostics, tame and Heisenberg packets, finite image orbit decompositions, ramification towers, Hecke packets, and conditional completion gates
8. portable JSON, YAML, CSV, TSV, text, Markdown, HTML, LaTeX, PDF, NPY, and NPZ exports with document rendering, local LaTeX compilation, manifested bundles with normalized ZIP metadata, and PowerShell tests
9. derivative providers, multilinear packet jets, Faà di Bruno composition, rooted nonlinear operator trees, graded packet operations, correction and elimination, filtered lifting, finite spectral sequences, and smooth packet transport shared with the quantum namespace
10. sparse symmetric derivative storage, lazy JAX and PyTorch directional automatic differentiation adapters, projection-cached packet-jet execution, provider-independent packet-jet snapshots, digest-checked nonlinear persistence, explicit optional CuPy runtime boundaries, and a machine-readable pre-beta API compatibility contract
11. resumable packet-jet streams, adjacent schema migrations, closed semantic golden fixtures, exact structural counts, environment-scoped profiles, and a curated strict-signature beta namespace
12. deterministic windows, finite STFT and inverse synthesis, spectral estimation, FIR and IIR streaming, rational resampling, adaptive filtering, Haar denoising, chunk-invariant streaming time-frequency analysis, stable biquads, and FFT overlap-add convolution
13. normalized monomial bases, exact coefficient domains, declarative custom generating functions, local Mahler masks, carry and Cartier compilers, Bell congruence automata, prime-power kernels, Gaussian q-Kummer valuations, cyclotomic collapse, trusted finite snapshots, and signal adapters
14. a general coordinate atlas with arbitrary finite bases, recurrence and state-space realizations, classical and combinatorial families, affine and formal charts, tensor and operator transport, finite-field maps, spectral transforms, packet permutations, and trusted coordinate records
15. canonical polyphase refinement and coarsening, mixed-radix address maps, multilevel towers, rectangular lattice refinement, portable refinement records, and the curated 1.0-candidate API namespace
16. general finite analysis--synthesis systems, a 131-symbol versioned stable surface, closed semantic fixtures, portable-codec migration contracts, stable CLI verification, machine-readable platform policy, crash-isolated GPU qualification, and release-candidate packaging gates
17. public-beta upgrade closure, digest-checked predecessor records, exact stable documentation coverage, platform-evidence inheritance rules, beta and release-candidate readiness gates, and executable predecessor-wheel upgrade validation
18. redundant-frame coefficient algebra, gauge projections, quotient metrics, coefficient lifts, cross-presentation bridges, exact rational analysis--synthesis maps, and distinct physical and dual adjoints
19. exact polynomial integration, positive finite atomic and bounded-interval measures, products and pushforwards, signed quadrature functionals, explicit error guarantees, exact integral identities, and coordinate integration covectors
20. basis-witnessed polynomial coordinate types, exact rational expression normal forms, decidable equality, certified rewrites, strict portable expression JSON, and an exact dense matrix-chain planner
21. a deterministic public capability tour, one-command publication qualification, self-digesting evidence records, strict documentation checks, and tag-triggered release artifact construction

The nonlinear namespace is implemented for explicit finite-dimensional data. Exact, numerical, and conditional conclusions remain separated. In particular, the package does not infer Banach completeness, global convergence, analytic regularity, or infinite inverse-limit existence from finite samples. Those hypotheses remain visible in certificates.

Block 4 uses a signal first finite convolution convention. The `same` mode always returns the signal length. The `valid` mode returns no samples when the kernel is longer than the signal. Exact Bayesian sufficient partitions use identical likelihood profiles. Approximate profile clustering is a separate numerical utility and does not certify sufficiency.

## Installation

Install the wheel directly:

```powershell
python -m pip install --force-reinstall .\release\dist\komorebi_math-1.0.0b4-4b7-py3-none-any.whl
```

Install exact integer and rational functionality with SymPy:

```powershell
python -m pip install "komorebi-math[exact]"
```

Install optional SciPy and high precision analytic dependencies:

```powershell
python -m pip install "komorebi-math[scientific]"
```

Install one automatic differentiation adapter only when needed:

```powershell
python -m pip install "komorebi-math[jax]"
python -m pip install "komorebi-math[torch]"
```

CuPy is detected dynamically and is not installed by the wheel because the correct package depends on the local CUDA platform.


### Optional native-backend preflight

An installed JAX or PyTorch package is not automatically considered usable. Komorebi Math checks each runtime in a child interpreter and confirms the directional JVP primitive required by the adapter. This prevents a broken native import from terminating the main source-validation process.

```powershell
.\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

The command reports the Python executable, NumPy and pytest versions, and isolated JAX and PyTorch status before pytest begins. Backend integration tests are also executed in separate child interpreters.

## Version 1 candidate contract

The broad research namespace remains available, but the compatibility promise is intentionally concentrated in three aliases of one curated surface:

```python
import komorebi_math.stable as stable
import komorebi_math.v1 as v1
import komorebi_math.api.v1 as api_v1

assert stable.__all__ == v1.__all__ == api_v1.__all__
```

The frozen stable layer and Block 17 verify eight independent contracts:

1. exported names, kinds, and reflected signatures
2. committed mathematical behavior across the central construction pipeline
3. portable codec identity, monotone versions, and adjacent migrations
4. declared Python, NumPy, optional-backend, and hardware qualification policy

```powershell
komorebi verify stable
komorebi verify release
komorebi verify block17
komorebi api check --stable
komorebi release contract
komorebi release readiness --output .\beta-readiness.json --force
```

Those package-wide release commands now bind the live `1.0.0b4` beta contract.
Historical `1.0.0b1` through `1.0.0b3` manifests remain immutable predecessor
records. Block 21 adds an external publication gate over the current API,
mathematical verification, source tests, documentation, and release evidence.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

On an NVIDIA system with compatible PyTorch and CuPy installations:

```powershell
.\Run_Block17_GPU_Validation.ps1
```

The GPU gate runs in a child interpreter, so a broken native backend cannot terminate the parent release process. See `docs/BLOCK_16_RELEASE.md`, `docs/STABLE_API_1_0_CANDIDATE.md`, and `docs/GPU_VALIDATION.md`.

## Typed finite expression calculus

Block 20 adds immutable typed expressions for exact finite rational values and
linear maps. Polynomial coordinate spaces are identified by their variable,
degree, and complete rational basis witness—not by a display name. Every closed,
well-typed term in this language reduces to one exact matrix or coordinate
vector, so equality is decided by finite entrywise comparison. Conservative
rewrites are accepted only after source/target checks and exact denotation
replay. The execution compiler independently checks its result against direct
normalization, and its matrix-chain dynamic program is optimal over binary
parenthesizations under the declared classical dense scalar-multiplication cost.

```powershell
python -m komorebi_math verify expression
python -m komorebi_math verify block20
$env:PYTHONPATH="$PWD\src"; python .\examples\block20_typed_expression_worked_example.py
```

The worked example evaluates an Euler/Fibonacci coordinate pipeline. Its map
normal form is exactly `I4`, its integral is exactly `5/4`, and the retained
affine/weighted integration regression is `106/5`. Block 20 does not claim a
general computer-algebra system, arbitrary nonlinear or infinite-dimensional
theorem proving, floating-point stability, or a universally optimal execution
planner. See `docs/TYPED_FINITE_EXPRESSION_CALCULUS.md`,
`docs/BLOCK_20_WORKED_EXAMPLE.md`, and `docs/BLOCK_20_RELEASE.md`.

## Finite measure and integration calculus

Block 19 introduces a research-only integration layer for explicit finite-dimensional data. Positive measures and signed quadrature functionals are separate types: `AtomicMeasure`, `IntervalMeasure`, `ProductMeasure`, and `PushforwardMeasure` describe positive finite measures, while quadrature rules may have signed weights and do not become measures automatically. Exact polynomial moments, the fundamental theorem of calculus, integration by parts, finite Fubini identities, affine pushforwards, and integration-covector transport are checked with exact rational arithmetic where their hypotheses permit it. Numerical quadrature reports whether its error statement is rigorous, conditional, estimated, or unknown.

A rigorous composite-Simpson result is available only when the integrand is the provenance-matched `ExactPolynomial`, the domain is the same exact rational `IntervalMeasure`, and the fourth-derivative bound, nodes, weights, evaluations, summation, and error bound remain exact rational computations. An arbitrary floating callable is not promoted by a matching-looking sample table: without a bound its Richardson statement is estimated, and with an explicitly assumed derivative bound it remains conditional because applicability and floating-point roundoff are unchecked.

```powershell
python -m komorebi_math verify integration
python -m komorebi_math verify block19
$env:PYTHONPATH="$PWD\src"; python .\examples\block19_measure_integration_worked_example.py
```

The worked example transports Lebesgue measure on `[0,1]` through `y = 2x + 1`, integrates `y^4 - 2y + 1` exactly as `106/5`, checks the result in three polynomial bases, and compares it with provenance-matched exact-rational Simpson quadrature. The committed Block 19 adversarial runner covers 47 cases, and the independent mathematical verifier covers 16 checks. See `docs/BLOCK_19_WORKED_EXAMPLE.md`, `docs/FINITE_MEASURE_INTEGRATION.md`, and `docs/BLOCK_19_RELEASE.md`.

This block does not claim general Lebesgue integration, automatic measurability, countably additive constructions on infinite spaces, improper or stochastic integration, generic nonlinear change of variables, manifold integration, or automatic convergence theorems.




## General coordinate atlas

Block 14 implements coordinate changes across arbitrary exact bases, normalized series, recurrence systems, state-space realizations, polynomial families, combinatorial transforms, affine and formal charts, finite fields, tensors, spectral systems, and packet layouts. Euler-to-Fibonacci coordinates remain available as one named example inside the general recurrence and polynomial machinery.

```python
from komorebi_math.coordinates import (
    CoordinateAtlas,
    basis_from_matrix,
    get_polynomial_family,
    standard_basis,
)

standard = standard_basis(3, name="standard")
shear = basis_from_matrix(
    "shear",
    ((1, 1, 0), (0, 1, 1), (0, 0, 1)),
)
change = standard.connection_to(shear)
change.certificate().require()

atlas = CoordinateAtlas("example atlas")
atlas.register_basis(standard)
atlas.register_basis(shear)
atlas.register_transform(change, include_inverse=True)

legendre = get_polynomial_family("legendre")
chebyshev = get_polynomial_family("chebyshev-t")
legendre.connection_to(chebyshev, 8).certificate().require()
```

```powershell
komorebi verify coordinates
komorebi verify block14
komorebi coordinates catalog
komorebi coordinates basis --source-matrix "1,0;0,1" --target-matrix "1,1;0,1" --vector "3,5"
komorebi coordinates polynomial --source legendre --target chebyshev-t --degree 8
komorebi coordinates model --source pell --target tribonacci --count 20
komorebi coordinates spectral --kind haar --size 8
komorebi coordinates packet --kind residue --length 12 --modulus 3
```

See `docs/GENERAL_COORDINATE_ATLAS.md` for the complete coordinate model. Block 14 established the broad coordinate atlas; Blocks 15--17 stabilized polyphase refinement, the versioned API, persistence, semantic behavior, platform support, and release qualification. Blocks 18 and 19 are additive research-tier extensions and do not reopen the frozen stable contract.

## Normalized generating functions and arithmetic compilers

Block 13 implements custom generating functions as executable finite mathematics rather than static formula tables. A declaration separates its coefficient domain, normalized basis `z**n/gamma_n`, and coefficient source. The same namespace compiles local masks into carry matrices, rational masks into Cartier representations, and factorial-power kernels into exact prime-power valuations and residues.

```python
from komorebi_math.generating import (
    FiniteNormalizedSeries,
    PrimeFieldDomain,
    compile_finite_mask_carry_plan,
    exponential_basis,
)

series = FiniteNormalizedSeries(
    exponential_basis(),
    PrimeFieldDomain(5),
    tuple(range(12)),
)
product = series.multiply(series)
product.certificate().require()

mask = compile_finite_mask_carry_plan((1, 1, 1), 2)
mask.certificate(500).require()
assert len(mask.sequence(500)) == 500
```

The base wheel remains NumPy-only. Exact rational, modular, carry, Cartier, Bell, and prime-power operations require only the standard library and NumPy. Symbolic recognition imports SymPy only when requested. Analytic mask evaluation reports an explicit finite tail bound. Portable persistence captures validated finite records and never reconstructs executable callbacks.

```powershell
komorebi verify generating
komorebi verify block13
komorebi generating mask --mask "1,1,1" --radix 2 --count 20
komorebi generating cartier --numerator "1,1" --denominator "1,-1" --prime 3 --count 20
komorebi generating kernel --index 137 --subindex 52 --prime 3 --exponent 5 --power 2
```

See `docs/NORMALIZED_GENERATING_FUNCTIONS.md` for the complete API model.

## Resumable streams, migrations, and beta compatibility

Block 11 adds directory-backed packet-jet streams for finite derivative data that is too large or too costly to capture in one uninterrupted operation. Direct capture can stop at a declared evaluation budget and resume from a verified committed prefix. Existing snapshots can be written in bounded coefficient batches. Readers verify chunks independently and can iterate, select, or evaluate a declared coefficient range without loading the complete stream.

```python
from pathlib import Path

from komorebi_math.nonlinear import (
    capture_packet_jet_stream,
    open_packet_jet_snapshot_stream,
)

path = Path("quadratic.kmstream")
partial = capture_packet_jet_stream(
    jet,
    path,
    chunk_size=256,
    evaluation_budget=500,
)

if not partial.completed:
    capture_packet_jet_stream(jet, path, chunk_size=256, resume=True)

reader = open_packet_jet_snapshot_stream(path, verify_chunks=True)
value = reader.evaluate(directions)
value.certificate.require()
```

Portable persistence now supports registered adjacent migrations. The packet-jet snapshot codec is version 2, while version 1 records remain readable through an explicit v1-to-v2 transformation. Unknown future versions and missing migration steps fail closed.

The broader research namespace remains under the additive pre-beta contract. `komorebi_math.beta` exposes a smaller reviewed surface with strict signature checks. A closed five-case golden fixture supplements structural reflection with selected mathematical regression checks.

```powershell
komorebi verify block11
komorebi golden
komorebi api check --beta
komorebi stream verify .\quadratic.kmstream
```

Stream and portable-file digests provide integrity, not authorship or authenticity. Direct resume of an arbitrary live Python derivative provider also cannot prove that the implementation behind the provider label is unchanged across processes.

## Scalable nonlinear execution and persistence

Block 10 keeps the Block 9 directional derivative protocol and adds sparse storage, optional automatic differentiation, reusable execution plans, and portable frozen packet jets. Sparse tensors store one coefficient per symmetric input orbit. Execution plans cache packet projections and group addresses that share the same projected inputs. Pruning occurs only when a complete support certificate proves a packet block is zero.

```python
import numpy as np
from komorebi_math import PacketSystem
from komorebi_math.export import dumps_portable, loads_portable
from komorebi_math.nonlinear import (
    PacketJetFamily,
    PolynomialDerivativeProvider,
    SparsePolynomialDerivativeProvider,
    compile_packet_jet_plan,
    execute_packet_jet_plan,
    packet_jet_support,
    snapshot_packet_jet,
)

tensor = np.zeros((2, 2, 2))
tensor[0, 0, 0] = 2.0
tensor[1, 1, 1] = -3.0
dense = PolynomialDerivativeProvider({2: tensor}, source_dimension=2, target_dimension=2)
sparse = SparsePolynomialDerivativeProvider.from_dense_provider(dense)
packets = PacketSystem(
    (np.diag([1.0, 0.0]), np.diag([0.0, 1.0])),
    names=("x", "y"),
)
family = PacketJetFamily(sparse, np.zeros(2), packets, packets, order=2)
plan = compile_packet_jet_plan(family, packet_jet_support(family))
result = execute_packet_jet_plan(
    plan,
    family,
    (np.array([2.0, 3.0]), np.array([-1.0, 4.0])),
)
result.certificate.require()

snapshot = snapshot_packet_jet(family.jet("x", ("x", "x")))
restored = loads_portable(dumps_portable(snapshot))
assert restored == snapshot
```

JAX, PyTorch, and CuPy remain lazy optional dependencies. A missing backend does not prevent NumPy-only import or installed verification. The package reports CuPy installation, runtime access, and device count separately and makes no GPU correctness or performance claim without an available device.

The portable persistence layer uses an explicit codec registry, schema versions, canonical JSON, and a SHA-256 payload digest. It never imports a class named by untrusted input. The API manifest checks exported name presence and broad symbol kind. It does not certify semantic equivalence.

## Advanced nonlinear calculus

The finite nonlinear layer uses one directional derivative protocol across exact tensors, declared analytic derivatives, optional SymPy expressions, and finite differences.

```python
import numpy as np
from komorebi_math import Exactness
from komorebi_math.nonlinear import (
    PolynomialDerivativeProvider,
    PolynomialFixedPointProblem,
    faa_di_bruno,
    majorant_convergence,
)

# g(x) = x + x^2 and f(y) = y^2. T_r stores the r-th derivative tensor.
g = PolynomialDerivativeProvider(
    {
        1: np.array([[1]], dtype=object),
        2: np.array([[[2]]], dtype=object),
    },
    exactness=Exactness.EXACT,
)
f = PolynomialDerivativeProvider(
    {2: np.array([[[2]]], dtype=object)},
    exactness=Exactness.EXACT,
)

second = faa_di_bruno(
    f,
    g,
    np.array([2], dtype=object),
    (np.array([3], dtype=object), np.array([4], dtype=object)),
)
second.certificate.require()
assert second.value[0] == 888

problem = PolynomialFixedPointProblem(
    np.array([0.1]),
    {2: lambda values: np.array([values[0][0] * values[1][0]])},
)
finite_tree_sum = problem.tree_expansion(4)
finite_tree_sum.certificate.require()

convergence = majorant_convergence(0.1, {2: 1.0}, radius=0.2)
convergence.certificate.require()
```

Faà di Bruno evaluation is finite and exact when its providers are exact. Rooted-tree enumeration is also exact. A convergence certificate is conditional on the declared multilinear norm bounds and ambient completeness assumptions recorded in the certificate.

## Signal polyphase coordinates

```python
import numpy as np
from komorebi_math.signals import polyphase_analyze

signal = np.arange(17, dtype=float)
coordinates = polyphase_analyze(signal, 4)
coordinates.certificate(signal).require()
assert np.array_equal(coordinates.reconstruct(), signal)
```

## Filter banks and wavelet packets

```python
import numpy as np
from komorebi_math.signals import HaarPacketTree, haar_analyze, haar_synthesize

signal = np.arange(16, dtype=float)
low, high = haar_analyze(signal)
assert np.allclose(haar_synthesize(low, high), signal)

tree = HaarPacketTree.analyze(signal, depth=3)
tree.certificate().require()
assert np.allclose(tree.prune(("0", "10", "110", "111")).reconstruct(), signal)
```

## Stable singular packet recovery

```python
import numpy as np
from komorebi_math.signals import recover_separated_jumps

samples = np.concatenate((np.zeros(12), 3.0 * np.ones(12), np.ones(12)))
result = recover_separated_jumps(
    samples,
    spacing=1.0,
    lipschitz_bound=0.0,
    noise_bound=0.0,
    minimum_jump=1.5,
)
result.certificate.require()
assert result.cells == (11, 23)
```

The certificate is conditional because the program can check the numerical threshold gap but cannot infer the unknown signal regularity assumptions from the samples alone.

## Markov lumping and Bayesian coarsening

```python
import numpy as np
from komorebi_math.dynamics import BayesianPartition, MarkovLumping

generator = np.array(
    [
        [-0.4, 0.2, 0.2, 0.0],
        [0.3, -0.3, 0.0, 0.2],
        [0.1, 0.0, -0.5, 0.4],
        [0.0, 0.1, 0.3, -0.6],
    ]
)
partition = [0, 0, 1, 1]
lumping = MarkovLumping(generator, partition)
lumping.certificate().require()

prior = np.array([0.1, 0.2, 0.3, 0.4])
likelihood = np.array([2.0, 2.0, 0.5, 0.5])
coarse = BayesianPartition(partition).coarse_posterior(prior, likelihood)
coarse.certificate.require()
```

## Feshbach reduction and memory

```python
import numpy as np
from komorebi_math.dynamics import FeshbachReduction

operator = np.array(
    [
        [-0.4, 0.2, 0.3, 0.0],
        [0.0, -0.6, 0.0, -0.2],
        [0.4, 0.0, -1.0, 0.1],
        [0.0, 0.5, 0.0, -1.4],
    ]
)
reduction = FeshbachReduction(operator, retained=[0, 1])
reduction.resolvent_certificate(2.0 + 0.3j).require()
kernel = reduction.memory_kernel()
assert np.allclose(kernel.laplace(2.0 + 0.3j), reduction.self_energy(2.0 + 0.3j))
```

## Structured computational layer

```python
import numpy as np
from komorebi_math import PacketIR, PacketLayout

layout = PacketLayout((2, 1), ("bulk", "edge"))
matrix = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
operator = PacketIR.from_dense(matrix, layout)
x = np.array([1.0, 2.0, 3.0])
assert np.allclose(operator.apply(x), matrix @ x)
operator.certificate().require()
```


## Cyclic analytic functions

```python
from fractions import Fraction
from komorebi_math.analytic import (
    CyclicDenominator,
    reciprocal_coefficients,
    positive_zero_string,
)

coefficients = reciprocal_coefficients(2, Fraction(1), 5)
assert coefficients[:4] == (Fraction(1), Fraction(1), Fraction(5), Fraction(61))

family = CyclicDenominator(4)
zeros = family.positive_zeros(4)
zeros.certificate.require()
```

Computed zero strings are numerical objects. Their certificates validate residuals and ordering for the returned representatives. They do not certify a global zero classification.

## Quantum packet circuits

```python
import numpy as np
from komorebi_math.quantum import CircuitPacketExpansion, cyclic_symmetry_projectors

symmetry = np.diag([1, -1]).astype(complex)
packets = cyclic_symmetry_projectors(symmetry, 2)
hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

circuit = CircuitPacketExpansion(
    (hadamard, hadamard),
    (packets, packets, packets),
)

circuit.reconstruction_certificate().certificate.require()
circuit.block_unitarity_certificate().require()
```

## Quantum channels and error correction

```python
from komorebi_math.quantum import amplitude_damping

channel = amplitude_damping(0.2)
channel.certificate().require()
choi = channel.choi_matrix()
```

Finite quantum APIs validate the matrix hypotheses that can be checked from their inputs. They do not claim a continuum field theory or a quantum gravity construction.

## Local arithmetic packets

```python
from komorebi_math.arithmetic import EulerLocalCoordinates, StandardWindow

window = StandardWindow(prime=7, compression=2)
window.certificate(address=3, periods=4).certificate.require()
assert window.period == 3

coordinates = EulerLocalCoordinates(prime=2)
result = coordinates.truncate(index=20, depth=2)
result.certificate.require()
print(result.value, result.modulus)
```

The odd and dyadic Euler systems share one reconstruction API while retaining their different integral kernels and valuation laws.

## Mahler transfer and recurrence packets

```python
from komorebi_math.arithmetic import MahlerSeries, fibonacci_recurrence

series = MahlerSeries.from_values([n**4 + 2*n + 1 for n in range(9)])
refined = series.affine_subsequence(multiplier=3, shift=2)
assert refined.evaluate(5) == series.evaluate(17)

fibonacci = fibonacci_recurrence()
packet = fibonacci.residue_packet(modulus=7, address=3)
packet.certificate(count=12).certificate.require()
print(fibonacci.term_mod(1_000_000, 1_000_003))
```

## Exact sequence compiler

```python
from komorebi_math.sequences import (
    apery_sequence,
    compile_sequence,
    fibonacci_sequence,
    thue_morse_sequence,
)

for sequence in (fibonacci_sequence(), thue_morse_sequence(), apery_sequence()):
    model = compile_sequence(sequence)
    model.certificate.require()
    print(sequence.name, model.strategy, model.first(12))
```

The compiler distinguishes declared constant recurrences, finite automata, polynomial coefficient recurrences, detected finite patterns, and direct evaluation. A detected finite pattern is not reported as a theorem about the infinite sequence.

## Portable exports and LaTeX documents

```python
from fractions import Fraction
from komorebi_math.export import DocumentMetadata, ExportOptions, MathDocument, export_data

document = MathDocument(DocumentMetadata(title="Executable mathematics", author="Komorebi P."))
document.paragraph("Ordinary text is escaped by every renderer.")
document.equation(r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}", numbered=True)
document.table(("n", "value"), ((0, 1), (1, Fraction(1, 2))))

export_data(document, "report.tex", options=ExportOptions(overwrite=True)).certificate.require()
export_data(document, "report.pdf", options=ExportOptions(overwrite=True)).certificate.require()
```

PDF output requires a supported local LaTeX engine. Shell escape is disabled by default. The generated `.tex` source is preserved beside the PDF unless `keep_tex=False` is selected.

Command line examples:

```powershell
komorebi export formats
komorebi latex engines
komorebi export sequence fibonacci --terms 30 --compile --output fibonacci.tex --force
komorebi export sequence fibonacci --terms 30 --compile --output fibonacci.pdf --force
komorebi latex compile fibonacci.tex --output fibonacci_recompiled.pdf
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
komorebi verify analytic
komorebi verify quantum
komorebi verify block5
komorebi verify arithmetic
komorebi verify sequences
komorebi verify block6
komorebi sequence fibonacci --compile --terms 20
komorebi arithmetic standard-window --prime 7 --compression 2 --address 3
komorebi arithmetic euler --prime 2 --index 20 --depth 2
komorebi verify block9
komorebi verify hardening
komorebi verify block10
komorebi verify streaming-beta
komorebi verify block11
komorebi golden
komorebi api check --strict-signatures
komorebi api check --beta
komorebi stream inspect .\snapshot.kmstream --verify-chunks
komorebi stream verify .\snapshot.kmstream
komorebi verify block20
```

`komorebi verify all` remains available as a Block 17 compatibility alias; it
is not the current cumulative or publication gate.

## Public reproducibility gate

The deterministic public tour is the fastest end-to-end review path:

```powershell
$env:PYTHONPATH="$PWD\src"
python .\examples\block21_public_capability_tour.py --json
```

The complete publication gate executes the full source suite, strict API
contracts, stable and Block 20 verification, release and upgrade contracts,
the capability tour, local documentation links, and a strict MkDocs build:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

See [Block 21 public reproducibility](docs/BLOCK_21_PUBLIC_REPRODUCIBILITY.md)
and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Status discipline

Every substantial computation identifies its execution class.

| Class | Meaning |
|---|---|
| `exact` | exact algebra or arithmetic under checked hypotheses |
| `numerical` | floating point computation with a residual and tolerance |
| `conditional` | valid when the listed unchecked assumptions hold |
| `heuristic` | exploratory discovery or estimation without a complete guarantee |
| `planned` | reserved namespace without a production implementation |

A thresholded block is not reported as an exact zero. A jump recovery certificate does not claim to infer unknown regularity. A fixed layout singular value truncation is numerical and certified. The tree compiler is exact only for its finite additive model. GPU timing is not claimed unless a compatible CuPy installation and device are actually used.

## Documentation

* [Unified executable language](docs/UNIFIED_LANGUAGE.md)
* [Public API](docs/PUBLIC_API.md)
* [Block 21 public reproducibility](docs/BLOCK_21_PUBLIC_REPRODUCIBILITY.md)
* [API inventory](docs/API_INVENTORY.md)
* [Implementation roadmap](docs/IMPLEMENTATION_ROADMAP.md)
* [Block 1 release record](docs/BLOCK_1_RELEASE.md)
* [Block 2 release record](docs/BLOCK_2_RELEASE.md)
* [Block 3 release record](docs/BLOCK_3_RELEASE.md)
* [Block 4 release record](docs/BLOCK_4_RELEASE.md)
* [Block 5 release record](docs/BLOCK_5_RELEASE.md)
* [Block 6 release record](docs/BLOCK_6_RELEASE.md)
* [Block 7 release record](docs/BLOCK_7_RELEASE.md)
* [Block 8 release record](docs/BLOCK_8_RELEASE.md)
* [Block 9 release record](docs/BLOCK_9_RELEASE.md)
* [Block 10 release record](docs/BLOCK_10_RELEASE.md)
* [Block 11 release record](docs/BLOCK_11_RELEASE.md)
* [Eleventh major block report](docs/ELEVENTH_MAJOR_BLOCK_REPORT.md)
* [Resumable packet-jet streams](docs/STREAMING_PACKET_JETS.md)
* [Behavioral fixtures and profiling](docs/BEHAVIORAL_GOLDEN_AND_PROFILING.md)
* [Scalable nonlinear execution](docs/SCALABLE_NONLINEAR_EXECUTION.md)
* [Portable persistence schemas](docs/PERSISTENCE_SCHEMAS.md)
* [Public API compatibility contract for 0.11](docs/API_COMPATIBILITY_0.11.md)
* [Public API compatibility contract for 0.10](docs/API_COMPATIBILITY_0.10.md)
* [Deprecation policy](docs/DEPRECATION_POLICY.md)
* [Finite nonlinear packet calculus](docs/NONLINEAR_CALCULUS.md)
* [Pre-beta API stability review](docs/API_STABILITY_REVIEW.md)
* [Extension author API](docs/EXTENSION_API.md)

## Deformation forests and support diagnostics

```python
import numpy as np
from komorebi_math.deformation import residue_forest, nested_subspace_occupancy

forest = residue_forest(
    {1: (0, 1), 2: (0, 1, 2, 3), 3: (0, 1, 4, 5)},
    prime=2,
)
forest.report().certificate.require()

occupancy = nested_subspace_occupancy(
    (
        np.eye(3),
        np.array([[1, 0], [0, 1], [0, 0]], dtype=float),
        np.array([[1], [0], [0]], dtype=float),
    )
)
occupancy.certificate.require()
```

Exact Fitting ideal calculations require the optional SymPy backend. Support is computed from finite presentations and fibers. Deformation components are not treated as central idempotent packets.

## Finite local and finite image Langlands models

```python
from komorebi_math.langlands.local import FrobeniusOrbit, HeisenbergModel
from komorebi_math.langlands.finite_image import CliffordStep, CliffordTower

orbit = FrobeniusOrbit(2, 3, 1)
orbit.certificate().require()

heisenberg = HeisenbergModel(3, 1)
heisenberg.selector_certificate().require()

tower = CliffordTower(
    (
        CliffordStep(24, 12, 3, 3, "E", attached_descent=False),
        CliffordStep(12, 4, 3, 1, "I"),
        CliffordStep(4, 2, 1, 1, "E"),
    )
)
assert tower.report().depth == 1
```

These APIs execute finite group, finite field, tame packet, Heisenberg, orbit, Mackey, and Clifford calculations. Global automorphic conclusions are returned only through conditional gate reports whose unmet external assumptions remain visible.
