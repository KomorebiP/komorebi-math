# Exact finite Tikhonov and discrepancy certificates

`komorebi_math.signals.tikhonov` implements the rational finite Euclidean
sector of *Sufficient Resolution: Exact reduction, hidden memory, and certified
reconstruction* (8 October 2026), Sections 11, 13, 14 and 16. Its matrices use
their ordinary Euclidean inner products. Nonorthogonal coordinates require
the metric to be transported before this interface is applicable.

```python
from fractions import Fraction as F
from komorebi_math.signals.tikhonov import tikhonov_system

model = tikhonov_system([[1, 1], [0, 0], [0, 0]], 1)
answer = model.solve([3, 4, 5])
answer.certificate.require()
```

The source is `(1, 1)`, the squared fitting residual is `42`, and the
regularized objective is `44`. The unregularized minimum-norm source
`(3/2, 3/2)` has regularized objective `91/2`. Positive regularization optimizes
its own objective. `objective_increment` verifies the exact quadratic
increment for any supplied rational competitor.

| API | Checked finite mathematical contract |
| --- | --- |
| `tikhonov_system(A, parameter)` | Constructs `N = A.T*A + parameter*I`, its inverse and the regularized map; checks inverse laws and complete kernel/range projectors. |
| `model.solve(data)` | Actual normal equation, energy identity, source orthogonality and squared source/fitting contractions. |
| `model.whole_map_bounds()` | Actual whole-map inverse, regularized, fitted and residual norm bounds through exact rational positive semidefiniteness. |
| `model.bias_noise(source, noise)` | Bias/noise vector identities and the orthogonal squared kernel split; bias and noise need not be mutually orthogonal. |
| `model.conditioned_bias_noise(source, noise, alpha_squared)` | Checks the whole identifiable-source coercivity premise before the conditioned error bound. |
| `model.source_condition_bias(source, witness)` | Checks `source = A.T*A*witness` before applying the normal-range source-condition bias rate. |
| `model.resolvent_difference(other)` | Two-operator/two-parameter inverse-difference identity. |
| `model.operator_sensitivity(other, ...)` | Verifies supplied whole-operator enclosures before the fixed-parameter regularized-map difference bound. |
| `normal_residual_correction` | Recomputes the actual normal residual and both source/image correction scales. |
| `bracket_discrepancy` | Strict attainable-floor and data endpoints, rational parameter enclosure with exact squared endpoint residual signs. |
| `select_discrepancy` | Total zero small-data branch and a rational positive source inside the discrepancy band. |
| `protected_discrepancy_window` | Protected endpoints, midpoint and the exact scalar nonemptiness condition. |
| `certify_squared_discrepancy` | Every SR-52 signed/nonnegative premise against freshly recomputed actual matrix squared norms. |
| `candidate_noise_acceptance` | Actual-source generalized acceptance, including nonsquare source norms through exact signed radical comparisons. |
| `candidate_norm_domination` | Feasible-comparison candidate-noise premise and the resulting source-norm domination. |
| `candidate_relative_feasibility` | Strict scalar relative-feasibility margin, retaining separate true data/model premises. |

## A discrepancy root can be irrational

```python
from komorebi_math.signals.tikhonov import bracket_discrepancy

root = bracket_discrepancy([[1], [0]], [1, 1], F(3, 2), width=F(1, 4096))
root.certificate.require()
```

The attainable squared residual floor is `1`. The target squared residual
`3/2` has positive parameter `1 + sqrt(2)`. The result contains rational
`lower` and `upper` parameters, their recomputed squared residuals, the
tested floor, the evaluation count and whether the requested parameter width
was reached. `exact_parameter` stays `None` here. It is populated only when
an evaluated rational parameter meets the exact target squared residual.

`select_discrepancy` seeks a rational source inside the band
`eta <= residual <= tau*eta`, using its interior target. This differs from
the paper's noncomputably chosen exact upper-edge root. In the positive branch
it checks the finite attainable floor is at most `eta`; in the zero small-data
branch it checks the defining `norm(data) <= tau*eta` inequality. Neither
branch authenticates `eta` as a true-data allowance for an unknown source.

## Source correction and image correction require different budgets

For `A = 1`, parameter `1/4` and data `1`, the exact source is `4/5`. A
computed source `81/100` has actual normal residual `1/80`. Its squared
source correction allowance is `1/400`, while its squared image correction
allowance is `1/6400`. The exact computed correction is smaller than both
allowances; the bounds remain the stated universal finite-map bounds.

```python
from komorebi_math.signals.tikhonov import certify_squared_discrepancy

record = certify_squared_discrepancy(
    tikhonov_system([[1]], F(1, 4)), [1], [F(81, 100)],
    eta=F(1, 10), tau=3, rho=F(1, 80), xi=F(1, 80),
    kappa=F(1, 20), measured=F(19, 100), omega=0,
)
record.certificate.require()
```

The producer recomputes `N*candidate - A.T*data` and `A*candidate - data`.
It checks the squared normal enclosure, both correction budgets, the clamped
fitting enclosure and both protected scalar margins. Supplying `rho = 0`
for this candidate fails. A displayed fitting norm inside the band cannot
hide a nonzero normal residual.

For the generalized candidate-noise rule, the allowance is
`eta + delta*norm(candidate)`. The producer uses exact rational sign tests
before squaring radical expressions. It checks that the positive candidate
is the actual regularized source. In the source's negative example
`A = 1`, available model `C = 0`, data `1`, `eta = 1/10`, `delta = 1`,
`tau = 2`, generalized acceptance fails. True data/model allowances alone
do not establish relative feasibility.

## Scope, resources and independent controls

Inputs accept integers, `Fraction` and SymPy rational values; floats, strings,
symbols and irrational entries are rejected. Matrices are owned and immutable.
`TikhonovLimits` caps dimensions, input rational bit size and a conservative
exact-elimination coefficient forecast before expensive work. Oversized shapes
are refused before array conversion. Root searches have an explicit evaluation
budget; exhaustion raises `ResourceLimitError` rather than returning a success.
Default limits are dimension `32`, input entry bits `4096` and working bits
`262144`; the conservative forecast can refuse a family below the individual
caps. Callers can choose stricter limits.

Tests use an independent standard-library `Fraction` two-coordinate inverse
and primal objective for all 81 matrices with entries in `{-1,0,1}`; exact
diagonal filters; principal-minor whole-map controls; nonsquare radical ordering
controls; an irrational root; strict floor endpoints; forged residual/measurement
records; and the zero-model infeasibility example. These are bounded
implementation checks. They do not prove infinite Hilbert-space convergence,
uniform rates across vanishing parameters, full globalization geometry or the
truth of physical calibration/noise measurements.

Run the worked example and its independent checks with:

```bash
PYTHONPATH=src python examples/tikhonov_certificates.py
PYTHONPATH=src python -m pytest -q tests/test_tikhonov_certificates.py
```
