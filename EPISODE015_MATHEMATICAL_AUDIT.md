# Episode 015 Mathematical Audit — Arithmetic Precision and Mahler Certificate Semantics

## Scope

This episode audited the mature arithmetic execution layer independently of the preceding quantum work. The target was not a new arithmetic theorem. The target was semantic soundness: whether finite p-adic precision, exact p-adic valuation, a finite Mahler truncation, and an exact unit mode were being kept distinct by the executable certificate layer.

Two predecessor contracts failed.

## 1. Truncated p-adic visible valuation

Let `p` be prime and let the precision be `N >= 1`. The truncated domain stores only a residue class modulo `p^N`.

Define the visible, precision-capped valuation by

`v_N(a mod p^N) = min(v_p(a), N)`, with `min(+infinity, N) = N`.

### Theorem 1 — well-definedness

`v_N` is independent of the chosen integer lift.

If `a == b mod p^N` and `v_p(a) < N`, then `b = a + p^N k`; the two summands have unequal p-adic valuations, so the nonarchimedean valuation law gives `v_p(b)=v_p(a)`. If `v_p(a) >= N`, then both `a` and `b` are divisible by `p^N`, so both capped valuations equal `N`.

Therefore a finite residue class can determine a valuation below the precision cap exactly, but the zero residue gives only the statement `v_p >= N`.

### Predecessor defect

Both `arithmetic.TruncatedPadic.valuation` and `generating.TruncatedPadicDomain.valuation` returned `+infinity` whenever the normalized residue was zero. Consequently, for precision 4 at p=3, the predecessor returned infinity not only for integer 0, but also for 81 and 162.

That behavior can be a legitimate convention for the zero element of an exact valued object, but it contradicted Komorebi's declared *visible valuation capped by the declared precision* contract. It also erased the epistemic distinction between exact zero and a nonzero lift divisible by the full stored modulus.

### Repair

The two truncated domains now return the precision cap for a zero residue. The exact rational/integer function `padic_valuation` remains unchanged and still returns `+infinity` for exact zero.

Thus the API now distinguishes:

- exact valuation of an exact integer/rational;
- visible valuation of a finite residue modulo `p^N`.

## 2. Finite unit-mode Mahler certificates

For nonzero integer `b`, positive jump `h`, residue `j >= 0`, write

`delta = b^h - 1`.

Then, for every nonnegative integer `x`,

`b^(h*x+j) = b^j (1+delta)^x = sum_{r=0}^x b^j delta^r binom(x,r)`.

The stored degree-d Mahler truncation is

`T_d(x) = sum_{r=0}^d b^j delta^r binom(x,r)`.

### Theorem 2 — first failure of an undertruncation

For every `x <= d`, `T_d(x)` is exact because `binom(x,r)=0` for `r>x`.

At `x=d+1`,

`b^(h(d+1)+j) - T_d(d+1) = b^j delta^(d+1)`.

Because `b` is nonzero, the undertruncation is exact beyond degree `d` iff `delta=0` (equivalently `b^h=1`). Therefore, if a requested maximum `M>d` and `delta != 0`, the first failure is exactly `d+1`.

### Predecessor defect

`UnitModeMahler.certificate(maximum_value, degree)` checked only

`0 <= x <= min(maximum_value, degree)`

but returned an `Exactness.EXACT`, `valid=True` certificate for the whole requested call, placing the untested tail in `assumptions_unchecked`. `Certificate.require()` does not reject such unchecked assumptions.

For `b=2`, `h=1`, `j=0`, `degree=0`, `maximum_value=5`, the predecessor certified the constant truncation

`1,1,1,1,1,1`

while the exact mode is

`1,2,4,8,16,32`.

### Repair

The certificate now checks the complete requested finite range exactly. It reports the first failing integer when one exists. Undertruncated nondegenerate modes are invalid. Degenerate zero-increment modes such as `b=1` or `b=-1` with even jump remain correctly certifiable at degree zero.

## 3. What was not changed

The Euler local-coordinate reconstruction, standard-window recurrence, Hurwitz arithmetic, prime-power Mahler kernel formula, and exact affine Mahler transfer were not modified.

A separate numerical weakness was observed but deliberately left outside this repair: `MahlerSeries.reconstruction_certificate()` uses exact tuple equality on floating coefficients even though it labels those cases numerical, causing false negatives as degree and conditioning grow. Choosing a tolerance requires an explicit conditioning/error model and is therefore deferred rather than patched heuristically.

## 4. Computation

The deterministic Episode 015 probe used seed `20260923`.

- 6,000 finite-precision valuation trials across p in `{2,3,5,7,11}` and precisions 1 through 7 produced zero failures of capped valuation, cross-domain consistency, or refinement compatibility.
- 6,000 unit-mode Mahler trials produced zero disagreements between certificate validity and direct exact finite evaluation.
- 2,367 sampled nondegenerate undertruncations were all rejected.
- 400 sampled low-degree zero-increment cases remained valid.
- The inherited Block 6 deep audit passed all 61,609 exact checks.

These computations audit the implementation; the universal statements above are proved independently.

## Epistemic classification

- **PROVED:** precision-capped valuation is well-defined on `Z/p^N Z`.
- **PROVED:** the first undertruncation failure is `d+1` for a nonzero Mahler increment.
- **REFUTED:** the predecessor's `UnitModeMahler.certificate` certified the full requested range.
- **CONTRACT REPAIR:** zero residue in the truncated domains now means visible valuation at least the precision cap, not exact infinite valuation.
- **COMPUTATIONALLY VERIFIED:** finite implementation checks described above.
- **NOT CLAIMED:** novelty of p-adic valuation or Mahler-expansion mathematics.
