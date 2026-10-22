# Normalized generating functions

The generating-function namespace implements finite normalized generating functions, local masks, carry compilers, Cartier operators, and prime power arithmetic.

## Normalized bases

A normalized monomial basis is determined by nonzero factors \(\gamma_n\):

\[
e_n(z)=\frac{z^n}{\gamma_n}, \qquad \gamma_0=1.
\]

For

\[
F(z)=\sum_n a_n e_n(z), \qquad G(z)=\sum_n b_n e_n(z),
\]

the product coefficient is

\[
(FG)_n=\sum_{k=0}^n
\frac{\gamma_n}{\gamma_k\gamma_{n-k}}a_kb_{n-k}.
\]

The package provides ordinary, exponential, factorial power, \(q\)-factorial, Fibonacci factorial, Pochhammer, finite table, and callback-defined bases. Portable bases use closed `BasisDescriptor` records. Callback bases remain usable for live finite computation and are snapshotted as finite tables when a portable series is requested.

```python
from komorebi_math.generating import (
    FiniteNormalizedSeries,
    RationalDomain,
    factorial_power_basis,
)

basis = factorial_power_basis(3)
left = FiniteNormalizedSeries(basis, RationalDomain(), (1, 2, -3, 5))
right = FiniteNormalizedSeries(basis, RationalDomain(), (2, -1, 4))
product = left * right
product.certificate().require()
```

## Coefficient domains

`FiniteNormalizedSeries` uses an explicit exact coefficient domain:

* integers
* rational numbers
* residue rings \(\mathbb Z/M\mathbb Z\)
* prime fields \(\mathbb F_p\)
* truncated \(p\)-adic rings \(\mathbb Z/p^r\mathbb Z\)

The implementation checks invertibility before reciprocal, logarithmic, exponential, composition, or basis-transfer operations that require division. Same-basis divided-power products in characteristic \(p\) do not invert \(n!\). Their binomial structure constants remain valid even when \(n!\) itself vanishes modulo \(p\).

## Declarative coefficient sources

`CustomFunctionSpec` separates a coefficient source from its normalization. Constructors are provided for:

* explicit finite tables
* constant-coefficient recurrences
* rational generating functions
* hypergeometric term ratios
* implicit algebraic branches
* Euler products
* finite composition graphs
* factorial-power Bell families
* weighted factorial-power Bell families

```python
from komorebi_math.generating import (
    FunctionRegistry,
    algebraic_function,
    rational_function,
)

registry = FunctionRegistry()
registry.register(
    rational_function("fibonacci", (0, 1), (1, -1, -1))
)
registry.register(
    algebraic_function(
        "catalan",
        {(0, 0): 1, (0, 1): -1, (1, 2): 1},
        1,
    )
)

assert registry.get("fibonacci").series(10).coefficients[10] == 55
```

Closed-form recognition routines are bounded exact searches. A returned relation is verified on the supplied finite data. No result means only that the declared search region found no relation.

## Local masks

For a radix \(p\ge2\), the local-mask transform is

\[
\mathcal M_p[R](z)=\prod_{j\ge0}R\!\left(z^{p^j}\right).
\]

Every formal unit series has the unique local mask

\[
R(z)=\frac{F(z)}{F(z^p)}.
\]

The finite APIs provide exact truncated product, exact mask inversion, functional-equation certificates, analytic finite products with conservative tail bounds, Frobenius defect masks, first lifts modulo \(p^2\), bounded polynomial coboundary solving, and Euler or cyclotomic orbit collapse.

## Finite mask carry compiler

For a finite mask \(R(z)=\sum_t r_tz^t\), the coefficient of \(\mathcal M_p[R]\) has an exact finite carry representation. The transition matrices are

\[
T_d(q,q')=r_{d+pq'-q}.
\]

```python
from komorebi_math.generating import compile_finite_mask_carry_plan

plan = compile_finite_mask_carry_plan((1, 1, 1), 2)
plan.certificate(500).require()
coefficients = plan.sequence(500)
```

The overfull binary mask \(1+z+z^2\) produces Stern's diatomic sequence shifted by one index. Ordinary Cauchy products of digit-local sequences are compiled by `compile_digit_cauchy_product`, which retains the required addition carry state.

## Rational Cartier compiler

If

\[
B(z)F(z)=A(z)F(z^p)
\]

over \(\mathbb F_p\), the package builds a bounded-dimensional Cartier linear representation and, when requested, its reachable minimized output automaton.

```python
from komorebi_math.generating import compile_cartier_linear_representation

representation = compile_cartier_linear_representation(
    numerator=(1, 1),
    denominator=(1, -1),
    prime=5,
)
value = representation.evaluate(10**12)
```

Evaluation reads the least-significant-first base-\(p\) digits of the index and therefore requires logarithmically many matrix transitions after compilation.

## Bell congruence compilers

The factorial-power Bell family is implemented with exact integer polynomial recurrences. Under the theorem hypotheses, the global polynomial modulo \(p\) reduces to the ordinary weighted Bell polynomial indexed by the base-\(p\) digit sum.

The package provides:

* pointed-subindex certificates
* weighted and unweighted digit-sum reductions
* the exact binary population-count parity automaton
* Bell period bounds
* finite Bell residue automata

## Prime power arithmetic

`PrimePowerUnitCompiler` computes the \(p\)-free factorial unit and exact binomial residues modulo \(p^r\). `PrimePowerKernelCompiler` adds factorial powers and optional unit twists.

Kummer layer matrices count subindices by borrow number. This gives exact support for \(\binom nk^s\) modulo \(p^r\), including zero-borrow and one-borrow formulas and carry-layer convolution records.

Gaussian binomial valuations use the all-prime mixed-radix decomposition. The two-adic branch uses radix two for odd \(q\). Support modulo prime powers follows directly from the valuation.

## Persistence

Portable generating-function records are data only. They use the package persistence envelope, explicit codec versions, canonical JSON, and SHA-256 payload integrity. No arbitrary import path or class name is executed from persisted input.

Supported records include normalized bases, budgets, formal series, domain-aware finite series, frozen custom functions, finite mask plans, Cartier automata and linear representations, Kummer distributions, and carry-layer convolution results.

## Signal bridge

A finite ordinary series can be interpreted as an FIR transfer polynomial under an explicit \(z\mapsto z^{-1}\) convention. `series_to_fir_adapter` records the convention, taps, response, and round-trip certificate. The bridge does not identify every formal series with a stable causal filter and does not hide truncation or numerical conversion.

## Resource contracts

Operations that accept `GeneratingBudget` check operation-specific limits on
degree, coefficient size, matrix dimension, reachable states, local combinations,
and modulus size. Exceeding an enforced limit raises `ResourceLimitError`.
Checks may require preliminary input conversion or arithmetic, so the budget is
not an overall memory or elapsed-time bound.

The `maximum_serialized_bytes` and `maximum_recursion_depth` fields are retained
in the budget record but are not currently enforced by generating-function
operations. Portable persistence has its own explicit `max_depth` parameter.
Bound serialized input and output sizes separately when needed.
