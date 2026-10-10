# Companion calculus machinery coverage

This is a computational crosswalk for the 8 October 2026 traditional editions of
AVOC, Finite Integration, Cartier Compilers and Sufficient Resolution. Complete
PDFs were extracted locally; chapter inventories, selected mechanism statements
and their surrounding arguments were compared with runtime source and tests.
This is not an independent audit of every proof or a fresh Lean kernel replay.
The machine-readable, section-specific record is
`verification/paper_coverage/companion_calculus.json`.

| Source | PDF SHA-256 |
| --- | --- |
| AVOC, 37 pages | `0426cccf0d2170e79fb495e41bd6d08879a2f2dd0086377021b1408d310c6959` |
| Finite Integration, 25 pages | `52fc8988aecb78146f3f7e3a7b5b43c2f5601b50c4d3088ab5b3dc62a21272e8` |
| Cartier Compilers, 81 pages | `a0924d62d63faa152f926d726e271523cb2b037141873ac1c52e87be2e96d88c` |
| Sufficient Resolution, 47 pages | `7767270c6a7757534c89aedca5578804edf316b92d307736e3a23ffeb836d336` |

“Implemented” means that the described finite sector has callable machinery and
tests, with the boundaries printed in the crosswalk. “Partial” means that useful
pieces exist but the complete mechanism or its source domain does not. “Missing”
identifies an absent executable construction; it is not a commitment or a claim
that the source has no implementation in its separate archives.

| Source | Present computational sector | Material gap |
| --- | --- | --- |
| Finite Integration | Rational polynomial moments, affine pushforward including singular maps, basis covectors, homogeneous-product Fubini, signed quadrature degree prefixes and matched rational Simpson bounds | Mixed atomic/interval iterated Fubini; a dedicated quadrature defect-row/kernel producer |
| Cartier | Finite normalized series, custom coefficient rules, digit kernels/Bell reductions, finite carry plans, rational masks modulo a prime or fixed prime power, prime-power unit/kernel residues and cyclotomic collapse | Chapter 24 canonical reachable-observable ring modules and finite-word comparison |
| Sufficient Resolution | Exact finite redundant frame algebra, rational predictive row-space closure, finite projected memory, numerical Feshbach/semigroup/SVD machinery | An integrated arbitrary-split closure certificate, globalization geometry, Tikhonov/discrepancy and calibration allowance contracts |
| AVOC | Numerical Bayesian/lumping pieces and exact rational affine coefficients/linear expression normalization | Rational utility channels, guarded adaptive Bellman evaluation, exact grid calibration, Boolean closure banks and recorder/recovery reducers |

Several neighboring APIs solve different problems. Matrix-chain planning
optimizes multiplication parenthesizations, not AVOC adaptive observation trees.
`PrimePowerUnitCompiler` computes factorial/binomial residues, not rational-mask
Cartier states. Deterministic automaton minimization does not compute the canonical
linear residual module; over `Z/4Z` that module may be nonfree. A numerical rank
threshold is not an exact finite-field or rational closure proof.

Finite Integration is particularly explicit about arithmetic and target error.
For `x**4` on `[0,1]`, the four-subinterval Simpson value is exactly `77/384`, the
target is exactly `1/5`, and both the error and proved bound are `1/1920`.
Exact arithmetic does not make the approximation equal to its target. The
provider requires subdivisions divisible by four and provenance matching the
same exact polynomial and interval. Floating callable and tensor paths carry
weaker guarantees.

Sufficient Resolution also separates conditions: prepared hidden-state closure
does not imply closure for arbitrary hidden initialization; compatibility does
not remove a nonzero realization obstruction; a small fitting residual does not
control source-kernel content; calibration on a proper source subspace does not
bound the whole operator. General Hilbert strong-limit statements and nonclosed
range boundaries remain theoretical, even when finite samples look favorable.

The newly added `generating/cartier_prime_power.py` implements Cartier Chapter
23's explicit redundant rectangle. Its coefficient modulus is `p**r`, layer
bound is `p*r`, and degree bound is
`ceil((deg(A)+(p*r-1)*deg(B))/(p-1))`. It applies the exact section transition to
any represented state, rather than enumerating the huge set of reachable states.
The compiler reduces masks to canonical integer representatives. It computes
`(B**p-B(z**p))/p` modulo `p**r` by first retaining coefficients modulo `p**(r+1)`;
premature reduction would lose a digit before the division. It prunes only
branches whose scalar or complete polynomial multiplier is zero at the chosen
precision.

For a worked example, let `F` count partitions into powers of two:
`F(z) = product_j (1-z**(2**j))**(-1)`. Its mask is `1/(1-z)`.

```python
from komorebi_math.generating.cartier_prime_power import (
    compile_prime_power_cartier_representation,
)

representation = compile_prime_power_cartier_representation(
    numerator=(1,), denominator=(1, -1), prime=2, exponent=3,
)
assert representation.sequence(16) == (
    1, 1, 2, 2, 4, 4, 6, 6, 2, 2, 6, 6, 4, 4, 2, 2,
)
assert representation.evaluate(2) == 2
representation.certificate(160).require()
```

These are residues modulo eight. The existing modulo-two compiler returns zero
at index two, so lifting its output alone would give the wrong residue. The
rectangle has 42 coefficient slots; that is an ambient representation size, not
a minimal automaton claim. The implementation also checks arbitrary-state
section identities against independently expanded `sum(P_l/B**l)*F`, validates
large-index rational collapses by a binomial closed form, and compares pruned
and unpruned branches. No nonfree ring-module minimizer has been added.

Resource controls validate raw state coefficient sizes before residue reduction
and check actual base-prime word lengths before any transition. A large custom
word allowance does not trigger exponentiation of that allowance. Polynomial
powering omits the unused final square, keeping its work inside the compiler's
checked degree bound. Regression sentinels cover all three cases.

The next useful finite constructions are, in priority order:

1. Sufficient Resolution SR-37–40 and SR-49–52: a rational Tikhonov constructor
   plus squared normal-residual and protected discrepancy checks. True noise
   allowances must remain separate inputs.
2. AVOC §§9–11: a rational guarded finite-horizon planner and grid-band builder,
   tested by complete small policy enumeration and exact binomial failure sums.
3. Sufficient Resolution SR-07–08: arbitrary rational split compression and its
   complete hidden-dimension moment certificate, including delayed-return and
   prepared-versus-arbitrary-initialization controls.
4. Cartier Chapter 24: nonfree finite-ring reachable/observable quotients. This
   requires ring-module machinery and should not be implemented using a field
   rank routine.

Fresh validation ran 13 relevant source-test modules: **447 passed in 37.98s**;
the prime-power controls include unit denominator constants other than one and
the three resource regressions. A combined compatibility run with existing
Cartier/p-adic, coefficient-domain and generating tests passed **158 tests in
6.69s**. An independent reviewer replay passed 54 rational masks, 54 full-integer
Frobenius-defect divisions, 162 arbitrary states, 540 exact section-series
identities, 108 closed-form queries up to `p**999-1`, and all three resource
sentinels. These finite controls do not prove the source's general theorem.
The native integration
verifier passed **12/12**, and a direct rational antiderivative independently
confirmed `106/5` in monomial, Euler and Fibonacci coordinates and four sharp
Simpson refinements. The crosswalk retains the exact commands and boundaries.
The publication's archived test counts, large adversarial scripts and Lean
evidence were not silently added to these fresh results. No claim here follows
merely from a matching module name.
