# Finite measure integration

The integration API supplies a common operation for explicit finite measures:

```python
result = integrate(integrand, measure, provider=optional_provider)
```

The integrand may be an `ExactPolynomial` or a callable.  Measures have an
explicit finite dimension and expose total mass and monomial moments.

## Positive measures

- `AtomicMeasure` represents a finite positive weighted sum of point masses.
- `IntervalMeasure` represents a constant nonnegative density on a bounded
  interval, including zero-density and degenerate zero measures.
- `ProductMeasure` supplies exact factorized moments and finite Fubini checks.
- `PushforwardMeasure` implements integration by precomposition.  Exact affine
  maps pull exact polynomials back with rational arithmetic.

Negative weights are rejected by positive measures.

## Quadrature functionals

`QuadratureRule` and `ExactQuadratureRule` are finite linear functionals.  Their
weights may be signed, so they are not automatically measures.  An exact rule
computes its finite sum exactly, but this does not prove equality with a target
integral.

`degree_of_exactness` compares rule moments with measure moments.  Only an exact
moment match through the integrand degree allows
`certify_polynomial_integral` to return an exact integral certificate.

`CompositeSimpson` evaluates exact polynomials and arbitrary callables over
bounded interval factors, but those paths make different claims:

- `rigorous` is available only for a provenance-matched univariate
  `ExactPolynomial` and the same exact rational `IntervalMeasure`.  The bound
  created by `polynomial_derivative_bound` must match both objects, and Simpson
  nodes, weights, polynomial evaluation, summation, and the final error bound
  are all computed with exact rational arithmetic.
- `conditional` applies to an arbitrary callable when a derivative bound is
  explicitly assumed.  Applicability to that callable and interval, as well as
  floating-point node, weight, evaluation, summation, and roundoff enclosure,
  remain visible unchecked assumptions.
- `estimated` applies to a fine/coarse Richardson estimate without a rigorous
  provenance chain, including the ordinary arbitrary-floating-callable path.
- `unknown` is used when no integral error theorem exists.

An internally rigorous polynomial bound cannot be attached to an arbitrary
callable, even if that callable appears to implement the same formula.  The
caller must pass the matching `ExactPolynomial` itself to retain rigorous
provenance.

## Exact identities

The exact polynomial layer certifies:

- the fundamental theorem of calculus
- integration by parts with the boundary term and constant density
- finite Fubini identities for rational interval and atomic products
- affine pushforward and one-dimensional Jacobian density laws
- oriented polynomial bounds without treating orientation as a positive measure
- coordinate covariance of the integration covector

For a polynomial family with derivative matrix \(D\), integration covector
\(\ell\), density \(\rho\), and endpoint evaluation covectors \(e_a,e_b\),

\[
\ell D=\rho(e_b-e_a).
\]

If \(C\) maps source coefficients to target coefficients, then

\[
\ell_{\mathrm{target}}C=\ell_{\mathrm{source}}.
\]

This is the coordinate law that closes integration with the existing polynomial
coordinate calculus.

`polynomial_integration_covector` records and replays the exact graded basis
witness before issuing an exact certificate.  A covector constructed directly
from arbitrary weights remains a structurally valid finite linear functional,
but its relationship to the named basis is conditional and explicitly
unchecked.

## Existing-calculus adapters

- normalized `FormalSeries` values convert to exact ordinary polynomials
- arbitrary exact polynomial-family coordinates integrate directly
- `DerivativeProvider` values integrate numerically through a measure
- `AtomicIntegratedDerivativeProvider` differentiates a fixed finite weighted sum
- analysis and synthesis maps are checked for commutation with atomic integration

The derivative adapter assumes fixed weights.  A parameter-dependent measure has
an additional derivative-of-measure term and is not silently accepted.
Provider dimensions must be positive integers, every value and derivative output
must have shape `(target_dimension,)`, and exact object arithmetic is preserved.
The frame commutation certificate incorporates the underlying
analysis--synthesis-system certificate, including its validity, assumptions, and
diagnostics. Zero-valued samples cannot make an invalid system pass.

## Verification

Run the installed integration checks with:

```console
python -m komorebi_math verify integration
```

The source tests include independent direct calculations, exact moment checks,
boundary counterexamples, and controls for unsupported quadrature claims.

## Deliberate limits

This finite integration API does not claim general Lebesgue integration, automatic
measurability, countably additive infinite-space construction, improper or
stochastic integration, generic nonlinear change of variables, manifold
integration, Radon–Nikodym automation, or dominated-convergence proofs.
