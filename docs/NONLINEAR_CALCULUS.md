# Finite nonlinear packet calculus

## Derivative providers

A derivative provider supplies dimensions, exactness, map evaluation, directional derivatives, and order support.

```python
from komorebi_math.nonlinear import DerivativeProvider
```

The concrete providers are:

| Provider | Use |
|---|---|
| `CallableDerivativeProvider` | declared analytic derivatives |
| `PolynomialDerivativeProvider` | finite symmetric coefficient tensors |
| `FiniteDifferenceDerivativeProvider` | numerical directional finite differences |
| `SympyDerivativeProvider` | optional exact symbolic derivatives |
| `ComposedDerivativeProvider` | composition through Faà di Bruno |

For `PolynomialDerivativeProvider`, tensors satisfy

```text
F(x) = sum_r T_r[x,...,x] / r!
```

Thus `T_r` is the declared `r`-th derivative tensor at the origin, not the ordinary polynomial coefficient.

## Packet jets

Given source packets `P_a`, target packets `Q_b`, and an order-`r` derivative,

```text
J[b; a1,...,ar](h1,...,hr)
  = Q_b D^r F(x)[P_a1 h1,...,P_ar hr].
```

`PacketJetFamily` constructs these blocks lazily.

```python
family = PacketJetFamily(provider, point, source_packets, target_packets, order=3)
jet = family.jet("output", ("left", "middle", "right"))
value = jet.evaluate((h1, h2, h3))
family.reconstruction_certificate((h1, h2, h3)).require()
```

`packet_jet_support` evaluates addresses under an explicit limit. A complete scan is exact or numerical according to the provider. A truncated scan is conditional because untested addresses may be nonzero.

## Faà di Bruno composition

For `F = outer o inner`, the order-`n` directional derivative is a sum over set partitions of `{0,...,n-1}`. Each block feeds one derivative of the inner map into a derivative of the outer map.

```python
result = faa_di_bruno(outer, inner, point, directions)
result.certificate.require()
```

The result records every retained term and combinatorial counts. `packet_faa_di_bruno` adds packet labels at the input, intermediate, and output stages.

## Rooted trees

A leaf represents the seed. An internal vertex of arity `r` represents the declared `r`-linear operation. `PolynomialFixedPointProblem` evaluates all finite trees up to a depth bound and compares the sum with the corresponding Picard iterate.

```python
problem = PolynomialFixedPointProblem(seed, {2: bilinear, 3: trilinear})
result = problem.tree_expansion(3, max_trees=100_000)
result.certificate.require()
```

Enumeration is guarded because the number of trees grows rapidly. `majorant_convergence` separately checks a scalar invariant-ball and contraction criterion. It does not estimate operation norms from the operation callable.

## Graded operations

`FiniteAbelianGroup` supplies exact labels and addition. `GradedPacketOperation` checks that an `r`-linear operation maps grades `(a1,...,ar)` into their group sum.

The higher Maurer-Cartan utility evaluates a finite truncation

```text
sum_r l_r(x,...,x) / r!
```

and resolves the result by grade. It is a finite identity, not an assertion that unspecified higher operations exist.

## Correction and elimination

### Hilbert obstruction

`HilbertObstructionProjector` separates a residual into the range of a linearized operator and its orthogonal obstruction component. When the obstruction vanishes, it returns the least-norm correction.

### Right-inverse correction

`RightInverseCorrectionProblem` implements a contraction argument with explicit right inverse, radius, and Lipschitz bound. The certificate checks the supplied finite inequalities.

### Lyapunov-Schmidt reduction

`LyapunovSchmidtReduction` computes numerical kernel, complement, range, and obstruction spaces from the base derivative. It solves the range equation and exposes the reduced obstruction map.

### Nonlinear Schur elimination

`NonlinearSchurElimination` eliminates hidden variables locally. It returns the implicit first derivative and the reduced first and second derivatives. Exact providers produce exact SymPy matrices. Numerical providers use finite matrix solves and residual certificates.

## Filtered lifting

`FilteredVectorSpace` represents a finite decreasing filtration over `F_p`. It provides level bases, associated-graded bases, coordinates, and lifts.

`FilteredCorrectionProblem` solves the one-depth equation on the associated graded and returns the correction coset. `verify_finite_filtered_branch` checks a supplied sequence of approximations through a finite terminal depth.

`conditional_projective_limit_completion` records, but does not prove, the external infinite hypotheses.

## Filtered spectral sequences

`FiniteFilteredCochainComplex` requires an explicit finite complex and a filtration by subcomplexes. It computes finite pages from `Z_r` and `B_r` subspaces, induced differential ranks, and limiting dimensions.

```python
result = complex_.spectral_sequence()
result.certificate.require()
for page in result.pages:
    print(page.page, page.total_dimension, page.total_differential_rank)
```

The finite terminal filtration permits an exact dimensional convergence check.

## Smooth transport

`MovingPacketSystem` validates a complete orthogonal packet family and declared derivatives. The canonical generator is

```text
K(t) = sum_a dot(P_a(t)) P_a(t).
```

`KatoTransport` integrates this generator by midpoint exponential steps. It also forms products of direct polar rotations and compares nested meshes.

The same classes are re-exported from `komorebi_math.quantum`.
