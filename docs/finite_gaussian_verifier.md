# Exact finite Gaussian verification

`komorebi_math.dynamics.gaussian_verifier` implements a rational finite slice
of **Certified Bayesian Inference**, traditional edition 8 October 2026,
based on Certified Inference Core 1.0.0. The source PDF hash is recorded in
`verification/paper_coverage/companion_dynamics.json`.
Sections 2–7 supply the factor-coded model, Joseph candidate, residual score
and complete packet-energy identity.

The input model declares one independent Gaussian observation. It represents
the prior covariance as `L L.T`, the noise covariance as `N N.T + sigma I`,
and requires a positive rational `sigma`. Singular priors, rectangular
observation operators and factors with zero columns are supported. For a
supplied rational gain `B`, the candidate is the Joseph Gaussian with

\[
m_B=m+B(y-Hm),\qquad
C_B=(I-BH)P(I-BH)^T+BRB^T.
\]

The exact innovation equation is `S = H P H.T + R`. The finite score uses
`F = P H.T - B S` and equals

\[
a(B)=\|F\|_F^2\frac{\sigma+\|y-Hm\|^2}{\sigma^2}.
\]

Under the declared Gaussian interpretation, the source's Theorem 5.2 bounds
the squared Wasserstein distance by this score. A passing exact tolerance
check evaluates its rational cross-multiplied inequality. It does not compute
the optimal transport distance, and a failed sufficient test does not prove
that the actual distance is too large. `exact_gaussian_posterior` solves the
innovation equation with rational elimination; it never inverts the prior.

```python
from fractions import Fraction as F
from komorebi_math.dynamics.gaussian_verifier import (
    FactorGaussianModel, evaluate_gaussian_gain, exact_gaussian_posterior,
    check_gaussian_tolerance, verify_gaussian_gain,
)

model = FactorGaussianModel(
    prior_factor=[[1], [0]], observation_operator=[[1, 1]],
    noise_factor=[[]], noise_floor=1, prior_mean=[0, 0], observation=[1],
)
gain = [[F(1, 3)], [0]]
candidate = evaluate_gaussian_gain(model, gain)
assert candidate.mean == (F(1, 3), F(0))
assert candidate.covariance[0][0] == F(5, 9)
assert candidate.residual_energy == F(1, 9)
assert candidate.score == F(2, 9)
assert exact_gaussian_posterior(model).covariance[0][0] == F(1, 2)
check_gaussian_tolerance(model, gain, F(1, 2)).require()

checked = verify_gaussian_gain(
    model, gain, tolerance=F(1, 2), precision=8, output_radius=F(1, 100),
)
checked.certificate.require()
assert checked.score_interval.upper == F(57, 256)
```

The direct dyadic evaluator recomputes the model/gain quantities exactly, then
encloses each final rational with floor/ceiling endpoints on a `2**(-p)` grid.
At precision zero the score enclosure is `[0,1]`, which fails verification
against squared tolerance `1/4`. At precision eight its upper endpoint is
`57/256`, which passes. Score uncertainty never counts as success. An optional
positive output-radius request is checked separately for every mean and
covariance interval. Those intervals describe the returned Joseph law, whose
covariance here is `5/9`; they do not enclose the exact posterior covariance
`1/2`. Their centers are not a separately certified Gaussian covariance.

`gaussian_packet_energy` checks every packet's exact self-adjointness,
idempotence, mutual orthogonality and completeness, then compares the sum of
all two-sided residual energies with the whole residual energy. Partial,
oblique or overlapping families fail. Diagnostics and altered reports are
not trusted by the independent gain verifier.

All stored numbers are immutable `Fraction` values. Floats, strings, booleans,
symbols and irrational inputs are rejected. `GaussianBudget` limits each
matrix dimension, exact numerator/denominator bits, packet count and dyadic
precision. Derived arithmetic obeys the same bit cap. Shaped inputs are
checked before copying. The dyadic scaling guard is conservative and can
reject a value whose reduced endpoint would fit after cancellation.

This evaluator is distinct from the source's operation-by-operation dyadic
graph, coefficient budgets and automatic fallback theorem. The exact
selective cache, maximum-energy scan, fuel, rounded full-gain branch,
universal Lean soundness, posterior disintegration and host refinement remain
separate source machinery. No new kernel check, sampler, sequential filtering
contract, uncertain-input model or unrestricted statistical validity follows
from these finite checks.

The focused suite uses independent full inverse/Schur-complement references,
completed-square covariance comparisons, coupling-cost checks, principal
minors, direct scalar packet sums, positive and negative dyadic values,
singular and zero priors, and arithmetic/mutation/resource rejection controls.

```bash
PYTHONPATH=src python -m pytest -q tests/test_gaussian_verifier.py
PYTHONPATH=src python examples/finite_gaussian_verifier.py
```
