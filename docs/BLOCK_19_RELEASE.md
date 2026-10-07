# Block 19: Finite Measure and Integration Calculus

Block 19 advances the package to `1.0.0b3` and adds finite measure and
integration calculus only to the research tier. It does not promote new names
to the stable or beta namespaces and is not a release candidate.

## Mathematical scope

The central operation is

```python
result = integrate(integrand, measure, provider=optional_provider)
```

The integrand may be an exact polynomial or a callable. The measure has an
explicit finite dimension, mass, and moment interface. The implementation adds:

- exact sparse rational multivariate polynomial arithmetic and moments
- positive finite atomic measures and bounded constant-density interval measures
- finite product measures and affine pushforwards
- signed quadrature functionals kept distinct from positive measures
- exact moment tests for a quadrature rule's degree of exactness
- tensor composite Simpson evaluation with explicit error strength
- exact polynomial FTC, integration by parts, finite Fubini, pushforward, and
  one-dimensional affine change-of-variables identities
- integration covectors and exact contragredient transport between polynomial
  coordinate systems
- adapters for polynomial families, normalized formal series, derivative
  providers, and finite analysis--synthesis systems

Exact conclusions require exact data and satisfied hypotheses. An exact finite
quadrature sum is not by itself an exact integral certificate. Numerical
results label their error statement as rigorous, conditional, estimated, or
unknown. A rigorous Simpson result requires the provenance-matched
`ExactPolynomial`, the same exact rational `IntervalMeasure`, a fourth-derivative
bound derived from those objects, and exact rational nodes, weights, evaluation,
summation, and bound arithmetic. Arbitrary floating callables are estimated when
only a Richardson comparison is available and conditional when a derivative
bound is explicitly assumed; they are not labeled rigorous.

## Worked example

The release example pushes `dx` on `[0,1]` through the affine chart
`y = 2x + 1`, so the transported measure is `(1/2) dy` on `[1,3]`. For

\[
f(y)=y^4-2y+1,
\]

the exact pullback calculation gives

\[
\int f\,d(\phi_\#\mu)=\int_0^1 f(2x+1)\,dx=\frac{106}{5}.
\]

The example verifies the same pairing in monomial, Euler, and Fibonacci
polynomial coordinates, then compares it with provenance-matched exact-rational
composite Simpson quadrature. The Simpson value is `130253/6144`, and the exact
error and rigorous fourth-derivative bound both equal `1/30720`. It also uses a
degree-six polynomial that vanishes at all three Simpson nodes to demonstrate
why sampled agreement does not establish exactness.

Run it from the source handoff:

```powershell
$env:PYTHONPATH="$PWD\src"; python .\examples\block19_measure_integration_worked_example.py
```

Use `--json` for machine-readable output. The complete derivation is in
`docs/BLOCK_19_WORKED_EXAMPLE.md`; `docs/FINITE_MEASURE_INTEGRATION.md` describes
the API and mathematical boundary.

## Verification

After installation, run:

```powershell
python -m komorebi_math verify integration
python -m komorebi_math verify block19
python -m komorebi_math verify stable
```

The focused integration verifier exercises 12 installed checks. The Block 19
aggregate combines those checks with the Block 18 and stable suites for 24
checks. The committed adversarial runner passes 47 cases covering exact moment
laws, signed-rule boundaries, noninjective pushforwards, product-measure
semantics, coordinate covariance and provenance, derivative-bound misuse,
numerical overflow and underflow, strict provider shapes, tensor resource guards,
and the Simpson-node counterexample. The independent mathematical verifier
passes 16 checks using separate direct calculations.

## Frozen compatibility gates

The following public contracts remain unchanged from `1.0.0b2`:

- 131 stable exports, with the same names, kinds, signatures, and alias identities
- 83 beta exports
- 51 registered portable codecs and their migration contract
- 10 stable semantic fixtures

All Block 19 integration names are research-only. Research compatibility is
one-way additive against the saved `1.0.0b2` baseline.

## Deliberate limits

This finite beta does not claim a general Lebesgue theory, automatic
measurability, sigma-algebra construction, countably additive measures on
infinite spaces, improper or stochastic integration, generic nonlinear change
of variables, manifold integration, Radon--Nikodym automation, or automatic
dominated-convergence and interchange theorems. Product-measure identities are
proved for the implemented finite exact cases; they are not a blanket theorem
for arbitrary callables or dependent joint measures.

`1.0.0b3` remains a research beta. Promotion of any integration symbol requires
an explicit public-tier review and must not be inferred from its presence in the
top-level research namespace.
