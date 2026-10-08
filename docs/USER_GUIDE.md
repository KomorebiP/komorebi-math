# Komorebi Math user guide

Release documentation edition: **8 October 2026**, for **`1.0.0b4`**.

Komorebi Math provides reusable finite mathematical objects: operators, packet
decompositions, coordinate changes, exact sequences, signal models, integration
rules, and quantum realizations. Results carry the assumptions and evidence of
the computation that produced them. This guide starts with a working environment
and two small programs, then shows how to choose a domain and interpret its
results.

The package is a research beta. Its frozen stable namespace contains 131
exports. The broader research API and the explicit modules added in October
have their own contracts. The repository and distribution retain the existing
private evaluation license. Use an authorized checkout or a supplied local
distribution; the installation instructions do not assume publication on PyPI.

Read this guide for first use, [the new machinery guide](NEW_MACHINERY_GUIDE.md)
for the October additions, and [the public reference](PUBLIC_API.md) for the
older domain APIs. The cumulative reference companion preserves the detailed
chapters and historical API cards. Dated inventories, source hashes, and
platform records in that companion refer to the revisions they name.

## Installation

Python 3.11 is the declared minimum. The CPU qualification matrix covers Python
3.11, 3.12, and 3.13 on Linux and Windows. The base dependency is NumPy 1.26 or
newer. Start in the repository root and use the same interpreter for pip and
the program.

On Linux or macOS with Python 3.12 available:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python -m pip check
python -m komorebi_math info
python -m komorebi_math verify stable
```

The commands are portable POSIX shell commands. macOS has no new qualification
claim in the recorded Linux and Windows matrix.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install --upgrade pip
& $py -m pip install .
& $py -m pip check
& $py -m komorebi_math info
& $py -m komorebi_math verify stable
```

Use another qualified Python minor when appropriate. Activation is unnecessary
when invoking the resolved Windows interpreter. The version reported by
`komorebi info` must be `1.0.0b4`.

### Optional dependencies

Install only the dependencies required by the workflow. These are the extras
declared by the current package metadata.

| Extra | Dependencies | Typical use |
|---|---|---|
| `exact` | SymPy >= 1.14 | Symbolic linear algebra and Smith decompositions |
| `scientific` | SciPy >= 1.11, mpmath >= 1.3 | Scientific and high precision numerical routines |
| `plot` | Matplotlib >= 3.8 | Plotting examples |
| `yaml` | PyYAML >= 6.0 | YAML import and export |
| `jax` | JAX >= 0.4 | JAX directional differentiation |
| `torch` | PyTorch >= 2.1 | PyTorch directional differentiation |
| `autodiff` | JAX and PyTorch | Both differentiation adapters |
| `test` | pytest >= 8 | Source tests |
| `dev` | pytest, build, mypy, Ruff, MkDocs | Development tools |
| `all` | Exact, scientific, plot, and YAML dependencies | Broad mathematical dependencies |

```bash
python -m pip install ".[exact,scientific,yaml]"
```

Standard-library integer, rational, modular, and many exact sequence operations
work without SymPy. The `all` extra does not include autodiff, CUDA, or development
tools. There are no `docs`, `build`, or `publication` extras in the current
metadata. Install the named tools explicitly when following a release workflow.

Directed quadratic-form enclosures require the qualified arithmetic backend
**mpmath 1.3.0**, even though the general scientific extra has a lower-bound
dependency declaration:

```bash
python -m pip install ".[scientific]" "mpmath==1.3.0"
```

CuPy is selected separately for the local CUDA version. Import presence does
not establish that a GPU or an autodiff adapter works. The doctor command reports
the environment, and hardware qualification has a separate scope:

```bash
python -m komorebi_math doctor
```

### Local wheel installation

The current wheel has build tag `4b21`. A checkout does not promise that built
artifacts already exist. A maintainer can create them with:

```bash
python -m pip install "setuptools==82.0.1" wheel
python verification/build_block21_distributions.py --root . --output-directory release/dist --report release/LOCAL_DISTRIBUTION_BUILD.json
```

Install the resulting wheel in a separate environment:

```bash
python -m pip install release/dist/komorebi_math-1.0.0b4-4b21-py3-none-any.whl
python -m pip check
python -m komorebi_math verify stable
```

The source distribution is `komorebi_math-1.0.0b4.tar.gz`. Installing it invokes
a source build. A fully offline installation also needs a local wheelhouse for
every dependency and build tool. Successful imports from an editable checkout
do not qualify an independently installed wheel.

## First program: packet analysis and reconstruction

A packet system stores a finite family of projectors. The following two
projectors separate the two coordinates of a vector. Their idempotence,
pairwise annihilation, and sum-to-identity conditions are checked by the
producer. NumPy execution gives a numerical certificate, even when the
observed residuals are zero.

```python
import numpy as np
from komorebi_math.stable import Exactness, PacketSystem

packets = PacketSystem(
    [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])],
    names=("left", "right"),
)
certificate = packets.certificate().require()
vector = np.array([2.0, 3.0])
coordinates = packets.analyze(vector)
reconstructed = coordinates.reconstruct()

np.testing.assert_array_equal(reconstructed, vector)
assert certificate.exactness is Exactness.NUMERICAL
print(reconstructed.real.tolist())
print(dict(certificate.residuals))
```

Expected output:

```text
[2.0, 3.0]
{'idempotence': 0.0, 'annihilation': 0.0, 'completeness': 0.0}
```

For a deliberately invalid family, construction normally raises. To inspect
the failed record in a diagnostic example, pass `validate=False`, inspect its
certificate, and call `.require()` at the acceptance boundary. Disabling
validation does not make the failed family satisfy the packet axioms.

## Exact coordinate changes

Use integers and `Fraction` values when the intended scalar domain is rational.
The stable exact transform below implements a shear and checks an inverse
roundtrip without a symbolic backend.

```python
from fractions import Fraction
from komorebi_math.stable import ExactCoordinateTransform, Exactness

Q = Fraction
transform = ExactCoordinateTransform(
    "A", "B", ((1, Q(1, 2)), (0, 1)),
    "GUIDE-EXACT-SHEAR", "exact rational shear",
)
vector = (Q(2), Q(3))
mapped = transform.apply(vector)
assert mapped == (Q(7, 2), Q(3))
assert transform.inverse().apply(mapped) == vector
assert transform.certificate().require().exactness is Exactness.EXACT
print(mapped)
```

Expected output:

```text
(Fraction(7, 2), Fraction(3, 1))
```

A valid transform record can still describe a singular map. Inspect the
invertibility diagnostics before calling an inverse. The
[coordinate atlas](GENERAL_COORDINATE_ATLAS.md) covers arbitrary finite bases,
polynomial families, recurrence coordinates, tensors, and formal charts.

## Certificates and saved evidence

`Certificate` separates validity from evidence class. `EXACT` describes exact
finite arithmetic under the producer's hypotheses. `NUMERICAL` describes a
floating computation and its error rule. `CONDITIONAL` retains unchecked
assumptions, and `HEURISTIC` or `PLANNED` describes weaker or unfinished claims.
API stability is a separate property.

The constructor requires a Boolean validity flag. A proposed true certificate
becomes invalid when a residual is nonfinite or nonreal, or its supplied
tolerance is nonfinite or negative. An invalid report may retain an infinite
residual as a failure diagnostic. The producer still owns the mathematical
comparison rule: the base class does not recompute every identity or impose one
universal norm on all residuals.

Residuals are copied into a read-only mapping. Use `dict(certificate.residuals)`
for an ordinary dictionary display. Diagnostics are copied only at the outer
dictionary level and are not deeply immutable. `.require()` accepts a live
true certificate; it does not discharge unchecked assumptions.

Portable loading returns historical evidence. A stored true flag appears as
`recorded_valid=True`, while `current_valid=False` and
`revalidation_required=True`. Rerun the producing operation to obtain a current
certificate. Equality with the old record and a valid SHA-256 envelope establish
record consistency, not fresh mathematical validation or authorship.

```python
from komorebi_math.stable import Certificate, Exactness, ValidationError
from komorebi_math.export import dumps_portable, loads_portable

original = Certificate(True, Exactness.EXACT, {"identity": 0})
restored = loads_portable(dumps_portable(original))
assert restored == original
assert restored.recorded_valid
assert not restored.current_valid
assert restored.revalidation_required
try:
    restored.require()
except ValidationError as error:
    assert "revalidation" in str(error)
else:
    raise AssertionError("a saved certificate cannot approve a new computation")
```

`ComputationResult.require()` returns its value after enforcing the certificate.
Its `.valid` property uses current validity. Its `.exact` property reports only
the evidence class and can be true for an invalid or historical result.
`ConditionalResult` enforces the conditional label and exposes its certificate
directly. Presentation JSON and typed portable persistence are different
formats. See [persistence schemas](PERSISTENCE_SCHEMAS.md).

## Choosing an API and a mathematical domain

Use `komorebi_math.stable`, `komorebi_math.v1`, or `komorebi_math.api.v1` for the
frozen 131-export compatibility surface. They are aliases of the same curated
objects. The 83-export beta manifest and the 1,312-export historical aggregate
research manifest have separate roles. Newly introduced explicit research
modules do not automatically join those frozen inventories.

| Goal | Reading route |
|---|---|
| Operators, seeds, packets, transfer blocks, refinement | [Public API](PUBLIC_API.md), [unified language](UNIFIED_LANGUAGE.md) |
| General analysis and synthesis, redundant frames | [Analysis-synthesis systems](ANALYSIS_SYNTHESIS_SYSTEMS.md), [frame algebra](REDUNDANT_FRAME_ALGEBRA.md) |
| Exact sequences and local OEIS comparisons | [Sequence and OEIS tools](sequences_oeis.md) |
| Generating functions, finite modular and digit algebra | [Normalized generating functions](NORMALIZED_GENERATING_FUNCTIONS.md) |
| Filters, streaming, STFT, and finite signals | [Production signal processing](PRODUCTION_SIGNAL_PROCESSING.md) |
| Fourier jump recovery | [Fourier recovery](fourier_recovery.md) |
| Exact integration and typed expression evaluation | [Integration](FINITE_MEASURE_INTEGRATION.md), [typed expressions](TYPED_FINITE_EXPRESSION_CALCULUS.md) |
| Derivatives, jets, nonlinear trees, and lifting | [Nonlinear calculus](NONLINEAR_CALCULUS.md) |
| Quadratic-form tails and factor spectra | [Quadratic forms](quadratic_forms.md) |
| Bernoulli-Hurwitz, Gamma, Meixner, and scalar Racah values | [Special functions](special_functions.md) |
| Exact finite join congruences and quotients | [Finite join semilattices](finite_join_semilattice.md) |
| Quantum memory, closure, and spectral dimension | [Quantum and memory contracts](QUANTUM_MEMORY_NUMERICAL_CONTRACTS.md) |
| Finite four-spinor Dirac modes | [Dirac modes](dirac_modes.md) |

The October guide provides worked examples for all six new module families.
Fourier recovery, finite semilattices, and Dirac modes use explicit submodule
imports. This keeps their supported scope visible and preserves the historical
aggregate API contract.

### Conventions that affect a computation

Exact finite arithmetic and floating linear algebra have different acceptance
rules. A result for a stored binary64 factor concerns that exact stored factor;
it does not include an unprovided measurement or rounding model. A finite
prefix matching a local OEIS record does not prove an infinite sequence
identity. Scalar Racah evaluation does not establish positive orthogonality.

For periodic Fourier recovery, the normalized coefficient is

\[
\widehat f(n)=\frac1T\int_0^T f(x)e^{-2\pi i n x/T}\,dx.
\]

Derivative moments are `2*pi*1j*n*fhat[n]` for any positive period. Recovery
needs the specified consecutive modes above the declared background bandlimit.
Its numerical consistency checks do not establish that the unknown background
really is bandlimited or that measured data are noiseless.

For the derivative Schrödinger equation `psi'=-i*H*psi`, the retained memory
kernel is

\[
K(t)=-H_{PQ}e^{-itH_{QQ}}H_{QP}.
\]

The Hamiltonian self-energy retains its resolvent convention. In the open
upper half-plane, `kernel.laplace(-1j*z)=-1j*self_energy(z)`. An infinite Laplace
integral, a rational continuation, and a finite-horizon transform are separate
operations:

```python
import numpy as np
from komorebi_math.dynamics.memory import MemoryKernel
from komorebi_math.stable import ValidationError

kernel = MemoryKernel([[1.0]], [[1.0]], [[1.0]])  # K(t)=exp(t)
np.testing.assert_allclose(kernel.rational_transform(0), [[-1.0]])
np.testing.assert_allclose(kernel.finite_laplace(1, 2), [[2.0]])
try:
    kernel.laplace(0)
except ValidationError:
    pass
else:
    raise AssertionError("the infinite Laplace integral diverges")
```

The Laplace half-plane test uses the full hidden realization and is conservative
when hidden modes cancel. These matrix calculations remain numerical.

## Command-line workflows and verification

Invoke the module through the resolved interpreter. Start with a relevant
installed suite, then broaden verification when changing the implementation.

```bash
python -m komorebi_math --help
python -m komorebi_math features
python -m komorebi_math api check --stable
python -m komorebi_math api check --strict-signatures
python -m komorebi_math verify stable
python -m komorebi_math verify block20
python -m komorebi_math release qualify
```

The CLI registry describes declared capabilities. It is not a live hardware
report, and an installed finite suite is not the complete source test suite.
The source-only examples and release tools require a checkout or unpacked
source distribution. Two connected tours are available:

```bash
python examples/block21_public_capability_tour.py --json
python examples/release_manual_machinery.py
```

For source development and release checks, install the explicit tools:

```bash
python -m pip install -e ".[test,exact,scientific,yaml]" "setuptools==82.0.1" wheel mkdocs
python verification/build_block21_distributions.py --root . --output-directory release/dist --report release/LOCAL_DISTRIBUTION_BUILD.json
python -m pytest -q --tb=short tests
python -m mkdocs build --strict
python verification/block21_publication_gate.py --full-tests --require-mkdocs --output release/LOCAL_PUBLICATION_GATE.json --markdown release/LOCAL_PUBLICATION_GATE.md
python -m pip check
```

On PowerShell, after installing these tools, the existing wrapper is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run_Block21_Publication_Gate.ps1 -Python $py
```

It uses the interpreter supplied by the caller and binds imports to `src`. It
does not create an isolated environment. `-Quick` omits the complete source
suite, and `-SkipMkDocs` omits strict documentation. These switches describe a
smaller diagnostic check and are not the full release command above.

Independent clean artifact checks also need a populated dependency wheelhouse:

```bash
python verification/release_artifact_gate.py --root . --runtime-tests --dependency-mode isolated --wheelhouse /absolute/path/to/wheelhouse --output release/LOCAL_ARTIFACT_GATE.json
```

The wheelhouse path is an explicit local input. See
[release process](RELEASE_PROCESS.md) and [readiness](GITHUB_READINESS.md) for
artifact provenance and qualification boundaries. Rebuild distributions after
source or documentation changes before checking source parity.

## Troubleshooting

| Observation | Next action |
|---|---|
| Wrong version or unexpected imports | Print `sys.executable` and `komorebi_math.__file__`; install through that interpreter |
| Missing optional dependency | Install the extra required by the named operation |
| Directed quadratic forms reject the backend version | Install the qualified `mpmath==1.3.0` backend |
| A certificate fails `.require()` | Inspect its message, residual rule, checked hypotheses, and retained assumptions |
| A loaded certificate requires revalidation | Rerun its producer; retain the loaded record as historical evidence |
| Exact input is rejected | Use the API's declared integer or rational domain; avoid fractional truncation and Boolean indices |
| Numerical recovery is unstable | Inspect singular values, thresholds, and residuals; the declared model and data conditioning still matter |
| The full gate cannot build documentation | Install MkDocs explicitly and rerun without `-SkipMkDocs` |
| Artifact parity fails after editing | Rebuild wheel and source distribution from the edited source |

Include the package version, interpreter, dependency versions, minimal input,
and complete certificate or exception when reporting an issue. Keep raw
measurements and numerical tolerances with the result so another reader can
replay the same finite computation.
