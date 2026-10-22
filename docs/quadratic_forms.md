# Directed quadratic-form tails

The optional `komorebi_math.quadratic_forms` research namespace integrates the
positive directed backend from the Komorebi Quadratic-Form Engine / K-COS. It
evaluates tails for

\[
Q=\sum_{j=1}^{r}\lambda_j Z_j^2,\qquad Z_j\sim N(0,1)
\]

with independent standard Gaussian variables and strictly positive
coefficients. The current COS remainder bound requires `r > 2`. Automatic
planning requires the threshold to exceed the exact sum of the supplied
coefficients. This is a research API with explicit numerical and mathematical
assumptions. Original project software is licensed under MIT.

From a source checkout, install the scientific extra and qualified backend:

```bash
python -m pip install ".[scientific]" "mpmath==1.3.0"
```

Directed operations currently require
`mpmath==1.3.0`, the qualified arithmetic implementation used by the source
audit. Importing the package or this namespace does not import mpmath. A
missing or unqualified arithmetic backend raises an actionable
`BackendUnavailableError` when an operation is requested.

```python
from komorebi_math.quadratic_forms import certified_positive_logsf

tail = certified_positive_logsf(
    "30",
    ("0.25" for _ in range(32)),
    relative_accuracy="1e-9",
    initial_K=64,
    max_K=512,
    initial_L="10",
    max_L=14,
    initial_interval_dps=40,
    max_interval_dps=80,
    working_dps=120,
)
assert tail.certified
print(tail.log10_interval)
```

Inputs are interpreted through their decimal literals. Use strings or
`Decimal` values when exact decimal intent matters. A float input contributes
its `str(value)` decimal, not its exact binary fraction. Iterables are
materialized once, so generators work through every refinement.

`certified=True` means the returned literal endpoints prove
`p_upper / p_lower - 1 <= relative_accuracy`, conditional on the arithmetic
trust base and stated mathematical hypotheses. Exhausted valid resource caps
return `certified=False` with a failure reason. Invalid domains raise an error.
Caps constrain each attempted transform and precision, rather than elapsed
time or total work. Low-rank or poorly scaled spectra can require many modes.

The authoritative enclosure is the outward decimal string
`log10_interval`. Float fields such as `log10_lower`, `log10_upper`, and
`width_decades`, and point midpoints, are display summaries. They can round
inward or collapse a narrow interval. Certificate decisions use the literal
endpoints with directed arithmetic. `directed_relative_accuracy` supplies the
same width decision for a separately supplied outward bracketed literal
string. Interval objects are rejected because ordinary string conversion
can silently collapse their width at the ambient precision. The caller is
responsible for supplying valid outward endpoints of the intended interval.

| Operation | Supported purpose |
|---|---|
| `certified_positive_logsf` | Automatically refine an above-mean positive central tail and prove a requested multiplicative width |
| `design_positive_tail_parameters` | Choose admissible positive tilt/support design points for an above-mean query |
| `interval_positive_logsf` | Evaluate a specified finite tilted COS transform with support/alias and omitted-mode error bounds |
| `even_df_scaled_chi_square_logsf_interval` | Independent finite Erlang-tail control for a nonnegative threshold and positive even degrees of freedom |
| `directed_relative_accuracy` | Decide a multiplicative width from outward literal log10 endpoints |
| `certify_float64_factor_spectrum` | Enclose the eigenvalues of a stored binary64 Gram factor using directed residual checks |
| `certified_float64_factor_logsf` | Combine spectrum and tail enclosures with a whole-interval width decision |

The explicitly parameterized evaluator also checks positive tilt, a support
endpoint above the threshold, and a second tilt strictly between the first
tilt and every MGF singularity. Its `passed_positive_sum` field reports whether
the analytic and arithmetic error interval proves a positive lower bound for the
tail sum.
It does not assert that a requested relative tolerance was achieved.

## Stored-factor interpretation

```python
import numpy as np
from komorebi_math.quadratic_forms import certify_float64_factor_spectrum

spectrum = certify_float64_factor_spectrum(
    np.diag([1.0, 1.5, 2.0, 2.5]),
    candidate_digits=40,
    interval_dps=50,
)
print(spectrum.lower, spectrum.upper)
```

The bridge converts a real input matrix to binary64 and treats those stored
entries as exact dyadic numbers. It certifies the sorted eigenvalues of
`A @ A.T`. The eigensolver supplies a candidate only. Directed orthogonality
and reconstruction residuals establish the coefficient intervals. The
unknown matrix before conversion and any uncertainty in input data are outside
the contract. Complex and nonfinite matrices are rejected.

This implementation uses a dense full basis and is intended for correctness
on modest matrices. `VerifiedSpectrum.rank` is the number of coefficient
intervals, including zero eigenvalues, rather than the algebraic matrix rank.
The automatic matrix-tail bridge requires a threshold above the exact stored
trace and at least three strictly positive lower coefficient bounds.
Unresolved spectral rank and invalid planner domains can raise an informative
error. When valid input domains cannot be resolved at the supplied working or
interval precision, the automatic controller returns `certified=False` with
its failure reason. Conservative comparison-spectrum obstructions trigger
bounded candidate refinement. Reported candidate precision is the actual last
attempt, including on exhaustion. The public options preserve the supplied caps.

Public operations hold a reentrant lock for their entire backend computation
and restore the caller's point and interval precision in a `finally` block.
Concurrent calls through this namespace are serialized. Arbitrary concurrent
manipulation of mpmath global contexts by external code is outside this
guarantee. Private implementation modules are not the supported entry points.

## Numerical trust and recovered source

The enclosure claim assumes correctness of mpmath 1.3.0 interval arithmetic
and transcendental evaluations, its exact binary endpoint tuples, Python
integer arithmetic, and directed Decimal serialization. Finite-transform
rounding is enclosed, and analytic support/alias and COS remainder bounds are
added explicitly. This numerical claim is distinct from a Lean proof of the
Python engine. The source formal project covers finite algebra and arithmetic
consequences of supplied bounds. It does not establish Gaussian laws, COS
inversion, or correctness of the numerical libraries.

Source repository: `KomorebiP/Operation-KOMOREBI`, immutable commit
`5c2a0084db5c5441c88a15d31dc3711fbb967619`, selected final K-COS v1.9 engine.
The recovered source paths are:

- `papers/quadratic-form-kcos/engine/directed_accuracy.py`
- `papers/quadratic-form-kcos/engine/interval_kcos.py`
- `papers/quadratic-form-kcos/engine/accuracy_driven.py`
- `papers/quadratic-form-kcos/engine/verified_spectrum.py`

These become private relative-import modules. The integration adds the public
validation/context boundary, fixes repeated consumption of spectra, prevents
support-scale refinements from exceeding their cap, and makes overflow in a
display-only ratio return infinity. Historical floating signed/noncentral
engines, scalable streaming engines, and unconstrained root-finding helpers
are outside this integrated surface. No additional license grant is inferred
from the recovered source. Komorebi P. attribution and existing project terms
are preserved.

Validation includes the ten recovered directed-contract regressions, exact
finite-gamma controls for five equal-coefficient spectra, ordinary-tail SciPy
comparisons, two known/higher-precision stored-factor controls, a complete
matrix-tail control, malformed-input checks, and success/failure/concurrency
precision checks. These bounded tests establish behavior on their named cases
and do not substitute for the stated arithmetic trust base.
