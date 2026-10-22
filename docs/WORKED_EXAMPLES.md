# Worked examples

This guide develops six mathematical workflows. Each section explains the
imports, supported inputs and result, then checks a small calculation against
an independent control. The examples are intended to be read, executed and
adapted to explicit finite inputs.

Use the explicit module paths shown below to make each mathematical interface
clear.

| Scope | Named exports covered here | Arithmetic and dependencies |
| --- | ---: | --- |
| `komorebi_math.quadratic_forms` | 12: seven operations and five result types | Directed intervals; qualified `mpmath==1.3.0` |
| `komorebi_math.sequences` | 15: twelve functions and three record types | Exact Python integers; offline local data |
| `komorebi_math.special_functions` | 18 functions | Exact rational scalars or numerical mpmath scalars |
| `komorebi_math.signals.fourier_recovery` | 5: three functions and two result types | NumPy numerical reconstruction |
| `komorebi_math.core.semilattice` | 2 types | Exact finite table checks |
| `komorebi_math.quantum.dirac` | 4 functions | NumPy finite matrices |

The numerical examples and the directed backend need the scientific extra.
From a source checkout, install and run them with:

```bash
python -m pip install ".[scientific]" "mpmath==1.3.0"
python examples/worked_examples.py
```

This installs the local source; it assumes no public PyPI release. The complete
worked-example script `examples/worked_examples.py`
contains the same twelve example bodies printed below. Run without Python's
`-O` flag so their assertions remain active. A successful run reports
`12 worked cases passed`; any failed assertion stops the run with a traceback.

## Positive quadratic-form tails

Use `komorebi_math.quadratic_forms` for the upper tail of

\[
Q=\sum_{j=1}^{r}\lambda_j Z_j^2,
\qquad \lambda_j>0,
\]

where the \(Z_j\) are independent standard Gaussian variables. The implemented
COS remainder bound requires \(r>2\). Automatic tail planning additionally
requires a threshold strictly above the exact sum of the supplied decimal
coefficients. Signed coefficients and noncentral Gaussian laws are outside
this API.

| Export | Return and purpose |
| --- | --- |
| `certified_positive_logsf` | `AccuracyDrivenTailResult`: refine an above-mean query to a requested multiplicative width |
| `design_positive_tail_parameters` | Dictionary of admissible tilt and support design values |
| `interval_positive_logsf` | `IntervalLogTailResult`: enclose a specified finite transform and its analytic error bounds |
| `even_df_scaled_chi_square_logsf_interval` | Dictionary containing an independent finite Erlang-tail enclosure |
| `directed_relative_accuracy` | `DirectedAccuracy`: check a width request from outward literal endpoints |
| `certify_float64_factor_spectrum` | `VerifiedSpectrum`: enclose a stored real binary64 Gram spectrum |
| `certified_float64_factor_logsf` | `MatrixTailCertificate`: combine spectrum and tail enclosures |

The five result types in the return column are also named exports. Importing
the namespace leaves mpmath unloaded. Requesting a directed operation or one of
these backend result types loads the qualified arithmetic implementation. A
missing backend or a version other than `mpmath==1.3.0` raises
`BackendUnavailableError` with an installation instruction.

Supply strings or `Decimal` objects to preserve exact decimal intent. A float
contributes its `str(value)` decimal; this scalar-input convention differs from
the stored-factor convention below. Spectrum iterables, including generators,
are materialized once before refinement.

For a result with `log10_interval="[l,u]"`, the authoritative bounds are the
outward decimal literals \(l,u\). Under the arithmetic trust assumptions,
`certified=True` proves

\[
10^{u-l}-1
=\frac{p_{\mathrm{upper}}}{p_{\mathrm{lower}}}-1
\leq \texttt{relative\_accuracy}.
\]

Midpoints and floating display summaries do not carry this enclosure and may
round inward. Pass the unchanged literal interval to
`directed_relative_accuracy`; it rejects interval objects whose ordinary string
conversion might collapse their width. When supplying your own interval
literal, you are responsible for its outward enclosure of the intended value.

### A tail with an independent Erlang control

Here the exact distribution is \(Q=\tfrac14\chi^2_{32}\). The example computes
the finite Erlang survival formula separately and checks that its high-precision
value lies between the returned literal endpoints. This numerical control is
independent of the COS transform, while still depending on mpmath arithmetic.

```python
from decimal import Decimal
from fractions import Fraction
import mpmath as mp
from komorebi_math.quadratic_forms import certified_positive_logsf

tail = certified_positive_logsf(
    "30", ("0.25" for _ in range(32)), relative_accuracy="1e-9",
    initial_K=64, max_K=512, initial_L="10", max_L=14,
    initial_interval_dps=40, max_interval_dps=80, working_dps=120,
)
assert tail.certified, tail.failure_reason
assert Decimal(tail.achieved_relative_ratio_minus_one) <= Decimal("1e-9")
lower, upper = map(Decimal, tail.log10_interval[1:-1].split(","))

# Q = 0.25 * chi-square(32): its survival is exp(-60)*sum(60**k/k!).
term = total = Fraction(1)
for k in range(1, 16):
    term *= Fraction(60, k)
    total += term
ctx = mp.mp.clone()
ctx.dps = 110
reference = (-60 + ctx.log(ctx.mpf(total.numerator) / total.denominator))
reference = Decimal(ctx.nstr(reference / ctx.log(10), 100))
assert lower <= reference <= upper
```

### A stored-factor spectrum and tail

The matrix bridge converts a real input factor \(A\) to binary64, then treats
those stored entries as exact dyadic numbers. It encloses the eigenvalues of
\(AA^{\mathsf T}\), using an eigensolver only to obtain candidates and directed
residual bounds to validate them. Uncertainty in the matrix before conversion
is outside this contract.

This example uses \(A=I_8\), so every eigenvalue is exactly one and the tail is
that of \(\chi^2_8\). Both controls are known independently of the candidate
solver. `VerifiedSpectrum.rank` counts all coefficient intervals, including
intervals for zero eigenvalues; it is not an algebraic matrix-rank report.

```python
from decimal import Decimal
import mpmath as mp
import numpy as np
from komorebi_math.quadratic_forms import certified_float64_factor_logsf

result = certified_float64_factor_logsf(
    "20", np.eye(8), relative_accuracy="1e-3",
    initial_candidate_digits=25, max_candidate_digits=25, interval_dps=40,
    tail_accuracy_fraction="0.5",
    tail_kwargs=dict(initial_K=128, max_K=512, initial_L="10", max_L=12,
                     initial_interval_dps=40, max_interval_dps=60,
                     working_dps=100),
)
assert result.certified, result.failure_reason
spectrum = result.spectral_enclosure
assert spectrum.rank == 8
for lower, upper in zip(spectrum.lower, spectrum.upper, strict=True):
    assert Decimal(lower) <= 1 <= Decimal(upper)
# A = I gives chi-square(8): exp(-10)*(1 + 10 + 50 + 1000/6).
ctx = mp.mp.clone()
ctx.dps = 100
reference = (-10 + ctx.log(ctx.mpf(683) / 3)) / ctx.log(10)
lower, upper = map(Decimal, result.log10_interval[1:-1].split(","))
assert lower <= Decimal(ctx.nstr(reference, 90)) <= upper
assert Decimal(result.achieved_relative_ratio_minus_one) <= Decimal("1e-3")
```

The matrix-tail controller requires a threshold above the exact stored trace
and at least three strictly positive lower spectral bounds. The spectrum
bridge uses a dense full basis and is intended for modest matrices. Unresolved
rank or planner-domain cases can raise an error. Valid accuracy budgets that
are exhausted return `certified=False` and a `failure_reason`; always inspect
those fields before using an enclosure as an accepted accuracy result.

Malformed domains and cap settings are rejected, usually with
`ValidationError`; backend admissibility failures can also raise `ValueError`.
When exact input domains are valid but planning or interval precision cannot
resolve admissible margins within the supplied caps, the automatic controller
returns `certified=False` with a failure reason. Matrix comparison-spectrum
obstructions trigger bounded candidate refinement. Reported candidate precision
is the actual last attempted precision. Direct domain and unresolved-rank errors
remain explicit.

Caps restrict each attempted transform and precision, not elapsed time or total
work. Public directed calls serialize their entire backend computation and
restore point and interval precision on success or failure. Concurrent external
changes to mpmath global contexts remain outside that guarantee.

The enclosure claim is conditional on mpmath 1.3.0 interval arithmetic, exact
binary endpoint storage, Python integer arithmetic, and directed Decimal
serialization. It is a numerical enclosure claim, distinct from a Lean proof
of the Python implementation. See [Directed quadratic-form tails](quadratic_forms.md)
for the full hypotheses, parameterized evaluator, and source provenance.

## Exact sequence families and offline OEIS comparisons

The additions to `komorebi_math.sequences` generate exact integer families and
compare finite windows against data that you supply explicitly. They introduce
no dependency or network request.

| Export group | Return and supported input |
| --- | --- |
| `universal_denominator_coefficients` | Dense integer EGF coefficients through an inclusive nonnegative degree |
| `reciprocal_sequence`, `reciprocal_coefficients` | Lazy `Sequence`, or exactly `count` compressed integer coefficients |
| `kbonacci_sequence`, `kbonacci_closed_form` | Lazy `Sequence`, or an integer term; order at least two |
| `fubini_sequence`, `derangement_sequence` | Lazy integer `Sequence` objects starting at index zero |
| `OEISRecord`, `LocalOEISCorpus`, `OEISPrefixMatch` | Local record, corpus, and finite-comparison types |
| `compare_oeis_prefix`, `find_local_oeis_matches` | One comparison, or a tuple of agreeing finite candidates |
| `parse_oeis_internal`, `parse_oeis_stripped`, `load_local_oeis` | `LocalOEISCorpus` parsed from text or an explicit file |

### Keep dense and compressed normalization separate

For spacing \(N\geq1\) and integer power \(r\geq0\), the normalization is

\[
D_N(t)=\sum_{q\geq0}\frac{(-1)^q t^{Nq}}{(Nq)!},
\qquad
D_N(t)^{-r}=\sum_{n\geq0}\frac{A_N^{[r]}(n)t^{Nn}}{(Nn)!}.
\]

`universal_denominator_coefficients(N, degree)` returns coefficients multiplying
\(t^j/j!\), with zero entries outside the selected residue lane. Its result has
`degree + 1` entries. `reciprocal_coefficients(N, count, power=r)` returns
exactly `count` values of \(A_N^{[r]}(n)\), starting at \(n=0\). These are
compressed exponential generating coefficients, not ordinary Taylor
coefficients. Power zero returns the identity sequence.

The k-bonacci convention starts with `k-1` zeros and then one. Fubini numbers
count ordered set partitions; derangements start with \(D_0=1,D_1=0\). The
example checks familiar initial terms and compares the finite k-bonacci formula
with recurrence execution at twelve indices.

```python
from komorebi_math.sequences import (
    derangement_sequence, fubini_sequence, kbonacci_closed_form,
    kbonacci_sequence, reciprocal_coefficients,
    universal_denominator_coefficients,
)

# These are compressed secant-number coefficients, not Taylor coefficients.
assert reciprocal_coefficients(2, 6) == (1, 1, 5, 61, 1385, 50521)
assert universal_denominator_coefficients(2, 6) == (1, 0, -1, 0, 1, 0, -1)
assert reciprocal_coefficients(3, 4, power=0) == (1, 0, 0, 0)
assert fubini_sequence().first(5) == (1, 1, 3, 13, 75)
assert derangement_sequence().first(6) == (1, 0, 1, 2, 9, 44)

# Check the finite binomial formula against direct recurrence execution.
recurrence = kbonacci_sequence(3).first(12)
assert recurrence[:7] == (0, 0, 1, 1, 2, 4, 7)
assert tuple(kbonacci_closed_form(3, n) for n in range(12)) == recurrence
```

### Verify the whole finite indexed window

Local OEIS records retain an explicit starting offset when one is available.
Supplying `query_offset` aligns sequence indices. Without it, comparison is
positional and does not claim verified index alignment. An unknown reference
offset cannot validate a query with a declared offset.

This example writes a small local JSON snapshot, loads it, and checks both a
complete indexed match and an exhausted reference. The temporary file is
removed after loading; the corpus retains its records and provenance.

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from komorebi_math.sequences import compare_oeis_prefix, load_local_oeis

# A user-supplied local snapshot; this example makes no network request.
payload = {
    "sequences": [{"id": "A000045", "terms": [1, 1, 2, 3, 5, 8],
                   "offset": 1, "name": "Fibonacci numbers"}],
    "provenance": {"purpose": "manual finite-window example"},
}
with TemporaryDirectory() as temporary:
    location = Path(temporary) / "snapshot.json"
    location.write_text(json.dumps(payload), encoding="utf-8")
    corpus = load_local_oeis(location, format="json")
reference = corpus.records[0]
matched = compare_oeis_prefix((2, 3, 5), reference,
                              query_offset=3, minimum_terms=3)
assert matched.passed and matched.index_alignment_verified
exhausted = compare_oeis_prefix((5, 8, 13), reference,
                                query_offset=5, minimum_terms=2)
assert exhausted.agrees_on_overlap and not exhausted.passed
assert exhausted.status == "reference-exhausted-prefix"
assert corpus.provenance["purpose"] == "manual finite-window example"
```

`agrees_on_overlap` checks all available compared terms against the requested
minimum. `complete_query_match` and its `passed` alias additionally require
the entire supplied query to have been compared. Candidate search can return
an exhausted agreeing prefix, so inspect `passed` before treating a candidate
as a check of the whole query. A known mismatch excludes the candidate even if
an earlier prefix agreed.

The loaders accept internal-format, stripped-format, and archive-schema JSON
text, including `.gz` files. Internal-format parsing assembles `%S/%T/%U` fields
or their signed `%V/%W/%X` counterparts; `%O` supplies the offset. Stripped text
has no offset field, so its offsets remain unknown. Internal-format parsing
ignores lines that do not match its field syntax, including malformed
identifiers; inspect the returned records before comparing them. Record
construction, stripped rows, and JSON snapshots validate identifiers. Invalid
terms in recognized fields, sizes, and indexing inputs raise `ValidationError`;
file access errors retain the usual filesystem exceptions. Booleans and
fractional integer coercions are rejected.

No reference corpus is bundled or downloaded. A finite match establishes only
the supplied finite comparison. A failed local search cannot establish absence
from the full OEIS, originality, or an infinite sequence identity. See
[Exact families and local OEIS comparisons](sequences_oeis.md) for the loader
schema, historical provenance, and recurrence conventions.

## Bernoulli packets, Gamma shapes, and terminating scalars

`komorebi_math.special_functions` contains two arithmetic layers. Exact
rational scalar functions return `Fraction` values without an optional backend.
Numerical functions return mpmath scalars at the requested precision. Shape
functions use isolated guard digits to resolve cancellation at large parameters,
then round their results. More than 10,000 working decimal digits, or excessive
positive window-exponent work, raises `ResourceLimitError`. These outputs retain
ordinary numerical precision rather than a directed enclosure.

Importing the namespace does not load mpmath, SciPy, or SymPy.

| Export group | Arithmetic and supported object |
| --- | --- |
| `bernoulli_packet_integer` | Exact rational Bernoulli polynomial residue packet |
| `bernoulli_hurwitz`, `bernoulli_hurwitz_packet` | Numerical complex-order interpolation and its residue packet, with real translation \(a>0\) |
| `q_pochhammer`, `q_racah_lattice`, `q_racah_scalar`, `racah_scalar` | Exact finite rational products, lattice sites, and terminating scalar sums |
| `log_gamma_window`, `log_gamma_fourier` | Numerical positive-shape unit window and its unitary Fourier transform |
| `gamma_sech_ambiguity`, `shape_affinity` | Numerical centered cross-shape overlap and zero-shift affinity |
| `meixner_density`, `meixner_characteristic`, `beta_logit_density` | Numerical scalar densities and characteristic function |
| `meixner_pollaczek` | Numerical normalized shape polynomial |
| `beta_binomial_refinement`, `shape_refinement_coefficient` | Tuple of two-child amplitudes, or one many-child amplitude |
| `shape_transfer_singular_value` | Numerical \(\sqrt{(s)_n/(s+t)_n}\), with \(s,t>0\) |

### Exact residue reconstruction and a q-Racah negative control

The exact Bernoulli packet is

\[
P_{n,M,r}(a)=M^{n-1}B_n\!\left(\frac{a+r}{M}\right),
\qquad B_1(x)=x-\tfrac12.
\]

It accepts rational translations, including zero and negative values. Degree
must be nonnegative, the modulus must be positive, and the residue must satisfy
\(0\leq r<M\). Integers, `Fraction` objects, and rational strings preserve exact
input; floating inputs are rejected.

All exact scalar functions accept integers, `Fraction` objects, and rational
strings and reject floating inputs. The q functions require rational
\(0<q<1\). Product lengths, scalar degrees `n`, and lattice indices `y` are
nonnegative strict integers. Scalar q-Racah sums terminate at `min(n, y)`.
A zero denominator in a retained term is rejected before any
cancellation. Termination by itself does not establish positive orthogonality.
The following corrected control has two equal lattice sites and identical rows.

```python
from fractions import Fraction as F
from komorebi_math.special_functions import (
    bernoulli_packet_integer, q_racah_lattice, q_racah_scalar,
)

a = F(3, 7)
packet_sum = sum(bernoulli_packet_integer(4, a, 3, r) for r in range(3))
# Independent polynomial control: B_4(x) = x^4 - 2*x^3 + x^2 - 1/30.
assert packet_sum == a**4 - 2 * a**3 + a**2 - F(1, 30)
assert bernoulli_packet_integer(1, 0, 1, 0) == F(-1, 2)

# A terminating q-Racah table can have repeated sites and no orthogonal rows.
q, alpha, beta, gamma, delta = F(1, 2), F(1, 2), F(1, 2), F(4), F(1)
assert gamma * q == q**-1
assert tuple(q_racah_lattice(y, gamma, delta, q) for y in range(2)) == (3, 3)
table = tuple(tuple(q_racah_scalar(n, y, alpha, beta, gamma, delta, q)
                    for y in range(2)) for n in range(2))
assert table == ((1, 1), (1, 1))
```

### Check the numerical Gamma convention

The positive-shape window and Fourier convention are

\[
g_s(y)=\Gamma(s)^{-1/2}\exp\!\left(\frac{sy-e^y}{2}\right),
\qquad
\widehat g_s(\omega)=\frac1{\sqrt{2\pi}}
\int_{\mathbb R}g_s(y)e^{-i\omega y}\,dy.
\]

`gamma_sech_ambiguity(alpha, gamma, h, eta)` instead uses the centered overlap
with factor \(e^{+i\eta y}\). The sign is part of that function's convention.
`meixner_density(s, omega)` is the Fourier energy density, and
`meixner_characteristic(s, h)` is \(\operatorname{sech}(h/2)^s\).

This example compares the Fourier value with direct quadrature of the defining
window after substituting \(u=e^y\). It also checks a unit overlap, the shape-one
density, and the squared refinement-amplitude normalization.

```python
import mpmath as mp
from komorebi_math.special_functions import (
    beta_binomial_refinement, gamma_sech_ambiguity,
    log_gamma_fourier, meixner_density,
)

ctx = mp.mp.clone()
ctx.dps = 45
shape, omega = ctx.mpf("1.75"), ctx.mpf("0.6")
# Direct Fourier integral after u = exp(y), independent of the Gamma formula.
def integrand(u):
    return ctx.exp((shape * ctx.log(u) - u) / 2) * u**(-ctx.j * omega) / u
integral = ctx.quad(integrand, [0, ctx.mpf(".25"), 1, 4, ctx.inf])
integral /= ctx.sqrt(2 * ctx.pi * ctx.gamma(shape))
assert abs(log_gamma_fourier(shape, omega, dps=45) - integral) < ctx.mpf("1e-32")
assert abs(gamma_sech_ambiguity(1, 1, 0, 0, dps=45) - 1) < ctx.mpf("1e-40")
density = meixner_density(1, omega, dps=45)
assert abs(density - 1 / ctx.cosh(ctx.pi * omega)) < ctx.mpf("1e-40")
amplitudes = beta_binomial_refinement(4, "1.25", "2.75", dps=45)
assert abs(sum(value**2 for value in amplitudes) - 1) < ctx.mpf("1e-40")
```

Numerical shapes must be positive finite real values; real coordinates must
be finite. Integer degrees reject booleans and fractional values. The
many-child shape and degree vectors must have the same positive length. Invalid
domains raise `ValidationError`, dimension mismatches raise `DimensionError`,
and a missing numerical backend raises `BackendUnavailableError`.

Each numerical call uses an isolated mpmath precision context. `dps` defaults
to 50 and must be an integer at least 15. It requests working precision; it
does not certify an error bound. The Bernoulli–Hurwitz functions accept finite
complex orders, use the real translation domain \(a>0\), retain exact removable
values at zero order, and
resolve tiny nonzero orders with adaptive isolated precision capped at 10000
digits. Their domain is narrower than the integer-order rational polynomial
function's translation domain.

These APIs evaluate scalar formulas and finite refinements. They do not
implement the full matrix recoupling engine, an infinite arithmetic transfer
estimate, or a continuous-series spectral transform. See
[Special-function machinery](special_functions.md) for the polynomial
normalization, correction register, and integration scope.

## Recover a finite periodic jump packet

Import from `komorebi_math.signals.fourier_recovery`. This explicit module is
the entry point for the five exports below.

| Export | Return and purpose |
| --- | --- |
| `PeriodicJumpPacket` | Canonical mean-zero periodic jump object |
| `FourierJumpRecovery` | Recovered packet, finite data arrays, and numerical certificate |
| `edge_moments_from_fourier` | NumPy vector of derivative moments |
| `hankel_moment_matrix` | NumPy matrix with entries \(H_{ij}=m_{i+j}\) |
| `recover_fourier_jumps` | `FourierJumpRecovery` from a consecutive high-mode block |

The model is a real periodic signal \(f=s+j\). The smooth background \(s\)
has Fourier degree at most the declared `background_degree`; \(j\) has finitely
many distinct active locations, real amplitudes summing to zero, and mean zero.
For period \(T>0\), the normalization is

\[
\widehat f(n)=\frac1T\int_0^T f(x)e^{-2\pi i n x/T}\,dx,
\qquad
m_n=2\pi i n\widehat f(n)
=\sum_\ell a_\ell e^{-2\pi i n\tau_\ell/T}
\]

above the background degree. In this normalization, the multiplier is
\(2\pi i n\) for every period.

### Recover three known edges on a nonstandard period

The recovery input must include `2 * maximum_jumps` consecutive positive modes
starting at `background_degree + 1`. This example generates ten coefficients
from three known jumps on period 3, then checks the recovered parameters against
the original input. It also requires the returned numerical certificate.

```python
import numpy as np
from komorebi_math.signals.fourier_recovery import recover_fourier_jumps

period = 3.0
locations = np.array([0.2, 1.1, 2.4])
amplitudes = np.array([2.0, -0.75, -1.25])
# Normalized coefficients use 2*pi*i*n for every period.
measurements = {
    n: np.sum(amplitudes * np.exp(-2j * np.pi * n * locations / period))
       / (2j * np.pi * n)
    for n in range(4, 14)
}
result = recover_fourier_jumps(measurements, background_degree=3,
                               maximum_jumps=5, period=period)
result.certificate.require()
assert result.order == 3
assert np.allclose(result.packet.locations, locations, rtol=0, atol=1e-10)
assert np.allclose(result.packet.amplitudes, amplitudes, rtol=0, atol=1e-10)
assert np.allclose(result.predicted_moments, result.measured_moments,
                   rtol=0, atol=1e-10)
```

### Represent a signal with no active jumps

An all-zero high-mode block returns an empty jump packet. Its evaluation and
all Fourier coefficients are zero. This is a valid model branch, rather than
a request to invent an edge from a vanishing signal.

```python
import numpy as np
from komorebi_math.signals.fourier_recovery import recover_fourier_jumps

result = recover_fourier_jumps({n: 0j for n in range(1, 5)},
                               background_degree=0, maximum_jumps=2)
result.certificate.require()
assert result.order == 0
assert np.array_equal(result.packet.evaluate([0, 0.5, 3]), np.zeros(3))
assert result.packet.fourier(0) == 0
```

The result contains `packet`, `annihilator_ascending`, `hankel_singular_values`,
`moment_indices`, `measured_moments`, `predicted_moments`, and `certificate`.
The packet wraps locations modulo its period, sorts them, copies its arrays,
and exposes read-only snapshots. `evaluate` uses the right-continuous
representative with its mean removed. `fourier(0)` is therefore zero.

The rank gate compares singular values with
`rank_tolerance * largest_singular_value`. A `known_order` hint must not exceed
the observed numerical rank. The default tolerances are `rank_tolerance=1e-10`
and `tolerance=1e-8`. Missing modes, invalid integer inputs, and out-of-range
real scalar parameters raise `ValidationError`. Unsupported scalar types may
raise Python conversion exceptions. Inconsistent real conjugacy, singular
recovery geometry, and invalid active-packet assumptions can raise
`AssumptionError`. Requiring a failed certificate raises `ValidationError`.

The certificate checks the represented model's finite moment fit,
annihilating recurrence, node-circle residual, real amplitudes, and zero sum.
It records Hankel and Vandermonde condition numbers. It does not verify the
unknown background's bandlimit, the truth of the finite jump model, the chosen
rank threshold's identification of the target order, or suitability of noise
and conditioning. Decimated samples cannot replace the required consecutive
block. A passing certificate establishes finite numerical consistency within
these assumptions. See [Fourier recovery of periodic jump packets](fourier_recovery.md)
for the algorithm and source boundaries.

## Form exact finite join quotients

`komorebi_math.core.semilattice` exports `FiniteJoinSemilattice` and
`SemilatticeQuotient`. A source object stores a nonempty square join table on
labels `0, ..., n-1`. The constructor checks idempotence, commutativity, and
associativity using exact integer comparisons, then keeps an immutable tuple
snapshot. It accepts Python and NumPy integer labels and rejects booleans,
floating labels, invalid ranges, and ragged tables.

### Generate the least congruence containing a relation

For the Boolean semilattice of subsets of a two-element set, labels are bit
masks and join is bitwise OR. Identifying zero with one also forces two to be
identified with three. The certificate replays the finite merge derivation and
checks the quotient homomorphism. The example checks every pair in the source
table as a separate homomorphism control.

```python
from komorebi_math.core.semilattice import FiniteJoinSemilattice

# Bit masks label the four subsets of a two-element set; union is bitwise OR.
source = FiniteJoinSemilattice(
    tuple(tuple(left | right for right in range(4)) for left in range(4))
)
result = source.generated_congruence(((0, 1),))
result.certificate().require()
assert result.classes == ((0, 1), (2, 3))
assert result.projection == (0, 0, 1, 1)
assert result.quotient.join_table == ((0, 1), (1, 1))
for left in range(4):
    for right in range(4):
        assert result.projection[left | right] == result.quotient.join(
            result.projection[left], result.projection[right])
```

### Validate a partition supplied by the caller

`quotient(partition)` accepts one arbitrary integer class label per source
element, validates compatibility with join, and constructs the quotient.
Canonical `partition` values are class minima; `projection` uses consecutive
quotient labels. `classes` lists source members, and `quotient` is another
validated `FiniteJoinSemilattice`.

```python
from komorebi_math.core.semilattice import FiniteJoinSemilattice

# The join of a finite chain is maximum; a convex partition gives a quotient.
chain = FiniteJoinSemilattice(
    tuple(tuple(max(left, right) for right in range(4)) for left in range(4))
)
result = chain.quotient((10, 10, -7, -7))
result.certificate().require()
assert result.partition == (0, 0, 2, 2)
assert result.projection == (0, 0, 1, 1)
assert result.classes == ((0, 1), (2, 3))
assert result.quotient.join_table == ((0, 1), (1, 1))
assert result.derivations is None  # This is not a leastness witness.
```

`generated_congruence(relations)` returns the least join congruence containing
the supplied pairs. Its derivation establishes leastness for this finite input:
every recorded merge follows from an input pair or a join context of an earlier
merge, and the output is itself a congruence containing the requested pairs.
A manually supplied quotient has no such leastness witness.

Dimension failures raise `DimensionError`; invalid labels or join axioms raise
`ValidationError`. A failed quotient certificate raises `ValidationError` when
required. Storage is quadratic in source size, and source axiom validation is
cubic. The exact decision concerns the supplied finite table. It does not imply
a universal matroid, reconstruction, or doctrine theorem. See
[Finite join congruence and quotient](finite_join_semilattice.md) for derivation
fields, complexity, and independent enumeration coverage.

## Evolve a fixed Dirac mode

Import from `komorebi_math.quantum.dirac` for finite four-spinor matrices in
units \(c=\hbar=1\). Fourier labels belong to the unit torus
\((\mathbb R/\mathbb Z)^3\), with phase \(e^{2\pi i n\cdot x}\) and continuum
momentum \(2\pi n\). Spinor coordinates use the order `e0, e1, e2, e3`.

| Export | Return and supported input |
| --- | --- |
| `dirac_matrices` | Independent read-only `alpha_x, alpha_y, alpha_z, beta` NumPy snapshots |
| `central_mode_momentum` | Real vector for a strict integer mode triple and a finite real mesh |
| `dirac_symbol` | Complex \(4\times4\) matrix for a finite real momentum triple and mass |
| `dirac_propagator` | Complex \(4\times4\) evolution matrix for a finite real momentum, mass, and time |

### Compare the closed formula with an eigensystem exponential

For physical momentum \(q\), the Hamiltonian and energy are

\[
H(q)=\sum_{j=1}^{3}q_j\alpha_j+m\beta,
\qquad E=\sqrt{m^2+\lVert q\rVert^2}.
\]

The Clifford identity gives \(H^2=E^2I\), so for \(E>0\)

\[
U(t)=\cos(tE)I-i\sin(tE)\frac{H}{E}.
\]

The implementation normalizes the real coefficients before constructing
\(H/E\), which avoids unstable division of complex entries at tiny positive
energy. At zero energy it returns the identity. This example compares the
closed formula with an independent Hermitian eigensystem exponential and checks
unitarity and the scalar-square identity.

```python
import numpy as np
from komorebi_math.quantum.dirac import dirac_propagator, dirac_symbol

momentum, mass, time = [0.3, -0.8, 1.2], -0.4, 0.7
hamiltonian = dirac_symbol(momentum, mass=mass)
evolution = dirac_propagator(momentum, time=time, mass=mass)
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
reference = (eigenvectors * np.exp(-1j * time * eigenvalues)) @ eigenvectors.conj().T
assert np.allclose(evolution, reference, rtol=0, atol=2e-14)
assert np.allclose(evolution.conj().T @ evolution, np.eye(4), rtol=0, atol=2e-14)
energy_squared = mass**2 + sum(value**2 for value in momentum)
assert np.allclose(hamiltonian @ hamiltonian, energy_squared * np.eye(4),
                   rtol=0, atol=2e-14)
```

### Check the Clifford and mesh conventions

`central_mode_momentum(mode, mesh)` computes
\(p\operatorname{sinc}(hp/\pi)\) coordinatewise, where \(p=2\pi n\),
\(h=\texttt{mesh}\), and NumPy uses
\(\operatorname{sinc}(x)=\sin(\pi x)/(\pi x)\). For nonzero mesh this equals
\(\sin(hp)/h\); at zero mesh the value is \(p\). The example checks all pairwise
Clifford anticommutators, continuum momentum, a central-difference value, and
zero-energy evolution at a large finite time.

```python
import numpy as np
from komorebi_math.quantum.dirac import (
    central_mode_momentum, dirac_matrices, dirac_propagator,
)

matrices = dirac_matrices()
for i, left in enumerate(matrices):
    assert not left.flags.writeable
    for j, right in enumerate(matrices):
        expected = 2 * np.eye(4) if i == j else np.zeros((4, 4))
        assert np.array_equal(left @ right + right @ left, expected)
assert np.array_equal(central_mode_momentum([1, 0, -2]), 2 * np.pi * np.array([1, 0, -2]))
central = central_mode_momentum([1, 0, 0], mesh=1 / 12)
assert np.allclose(central, [12 * np.sin(np.pi / 6), 0, 0], rtol=0, atol=1e-14)
assert np.array_equal(dirac_propagator([0, 0, 0], time=1e100, mass=0), np.eye(4))
```

Real negative mass, mesh, and time are allowed. Nonfinite or complex scalar
parameters and fractional or boolean mode labels raise `ValidationError`;
wrong vector shapes raise `DimensionError`. Unrepresentable energy or phase is
rejected. The central symbol can alias distinct Fourier labels: no Nyquist
restriction or inverse recovery of those labels is provided.

These helpers return ordinary finite numerical matrices. They do not construct
an unbounded Dirac operator, an infinite Fourier packet, or a sampled-grid
Hilbert space, and they do not certify a uniform continuum limit. See
[Fixed-mode Dirac evolution](dirac_modes.md) for source coordinate actions and
the larger finite comparison range.

## What the worked examples check

The script executes twelve cases without skips. Its controls are the finite
Erlang survival formula, known identity-factor spectrum, named integer prefixes,
a direct recurrence comparison, local indexed window statuses, a Bernoulli
polynomial, a degenerate q-Racah geometry, direct Fourier quadrature,
shape normalization, known Fourier jump data, an empty-packet branch, finite
quotient homomorphisms, Clifford coordinate identities, and an eigensystem
exponential. Each example body above is copied from that script.

These checks make the documentation reproducible on those inputs. They do not
replace the stated interval trust base, turn ordinary precision into an error
certificate, or prove an unrestricted mathematical theorem. The
[release guide](RELEASE_PROCESS.md) describes package and platform qualification
separately from these reader examples.
