# Discovery 1.1 exact machinery

`komorebi_math.special_functions.paper_machinery` implements the reusable exact
coefficient and resolution machinery in *Special Function Discovery 1.1*.
It uses the Python standard library. Rational inputs are integers, `Fraction`
values, or rational strings. Floating inputs and booleans are rejected.

The module implements finite formal coefficients, exact-data inverses, and
rational approximation bounds. A floating curve, small fit residual, or
agreement between two truncated sums does not approve a recovery.

## A coupled worked example

The normalized pair satisfies

\[
 U'(z)=1+U(qz)V(rz),\qquad V'(z)=1+U(az)V(bz),\qquad U(0)=V(0)=0.
\]

```python
from fractions import Fraction as F
from math import factorial
from komorebi_math.special_functions.paper_machinery import (
    coupled_odd_derivatives,
    coupled_certified_evaluation,
    recover_coupled_rational_scales,
)

scales = (F(1, 5), F(2, 5), F(3, 5), F(7, 10))
U, V = coupled_odd_derivatives(*scales, last=3)
jet = tuple(value for n in range(1, 4) for value in
            (U[n] / factorial(2*n+1), V[n] / factorial(2*n+1)))
assert recover_coupled_rational_scales(*jet) == scales

enclosure = coupled_certified_evaluation(
    *scales, x=1, last=30, outer_radius=2,
)
assert enclosure.tail_bound < F(16, 10**14)
print(enclosure.u_interval, enclosure.v_interval)
```

The two arrays contain **derivatives** of orders `1, 3, ..., 2*last+1`.
Divide entry `n` by `(2*n+1)!` to obtain an ordinary Taylor coefficient.
For the worked pair the first four ordinary coefficients are:

| Index | U | V |
| --- | --- | --- |
| 0 | 1 | 1 |
| 1 | 2/75 | 7/50 |
| 2 | 88/234375 | 8211/1250000 |
| 3 | 18814/8544921875 | 85664601/625000000000 |

`coupled_cleared_derivatives` returns the common denominator `D` of the scales
and two integer arrays `D**(n*(n+1))*A_n`. That clearing factor need not be
minimal. Existing residue, polyphase, finite-tap, and Hurwitz-product APIs
apply to the full derivative arrays after inserting zero even entries.
Ordinary convolution of derivative values and binomial convolution of
exponential generating functions are distinct operations.

## Formal and analytic domains

`feedback_odd_derivatives` computes
`F'=1+sum(h*F(qz)*F(rz))`; channels are `(h,q,r)` tuples. Empty channels give
`F=z`. `asymmetric_odd_derivatives` is its single-channel specialization.
These functions and the coupled recurrence accept arbitrary rational scales
as **formal series data**. They do not assign an entire-growth theorem to a
negative or noncontracting example.

`coupled_disk_majorant` and `coupled_certified_evaluation` require every scale
strictly between zero and one. A finite contraction ladder gives an exact
bound `B(S)` on both components. The tail after odd index `N` is bounded by
`B(S)*(R/S)**(2*N+3)` throughout the complex disk `|z|<=R<S`. Positive
coefficients sharpen the real enclosure at `x>=0` to
`[partial_sum, partial_sum + tail_bound]`. The output endpoints and bound are
`Fraction` values; converting them to floats loses directed enclosure semantics.

`recover_coupled_squared_scales` consumes the six **ordinary** coefficients
`(u1,v1,u2,v2,u3,v3)` at degrees three, five, and seven. Its implemented domain
requires rational quadratic candidates, and it returns rational squared
scales. `recover_coupled_rational_scales` additionally requires rational square
roots. It rejects inconsistent data and the reduction locus
`D=(ab)**2*u2-(qr)**2*v2=0`. In the positive contraction domain this locus is
exactly `q=b` or `qb=ar`, checked by `coupled_reduction_discriminant`.

The inverse makes no noise tolerance substitution. Near the reduction locus,
candidate seventh coefficients can approach each other. Exact recovery does
not establish uniform finite-precision stability. General algebraic-number
recovery, the numerical Perron root solver, ordinary type-limit existence,
angular growth, and complex zero geometry are outside this module.

`coupled_integer_growth_anchor` verifies the exact characteristic identity at
a supplied positive integer exponent and returns the positive Perron ratio
`w_U/w_V`. The worked order-one pair returns `1/2`. This checks a rational
anchor directly; it does not estimate growth from a finite Taylor plot or solve
the equation at a general real exponent. The source theorem identifies a
verified anchor with the common order of the normalized analytic pair.

## Resolution examples

```python
from fractions import Fraction as F
from komorebi_math.special_functions.paper_machinery import (
    prime_exclusion_disk_bound,
    sieve_interpolation_certificate,
    sieve_negative_detector,
    weak_tap_disk_bound,
)

# A fast feedback weight changes global order while its unit-disk effect is small.
assert weak_tap_disk_bound(F(1, 10**12)) < F(241890267, 10**21)

# Bound the excluded-prime difference and its derivatives on |s|<=5.
assert prime_exclusion_disk_bound(10007) < F(1330, 10**19)
second_derivative_bound = prime_exclusion_disk_bound(10007, derivative_order=2)

# All polynomials of degree <=4 have positive signed norm at this defect size.
low_degree = sieve_interpolation_certificate((1, 3, 5, 7, 11))
assert low_degree.positivity_margin(F(1, 10**12)) > 0

# A normalized degree-seven polynomial has a rigorously negative norm.
detector = sieve_negative_detector(F(1, 10**12))
assert detector.degree == 7 and detector.detects_negativity
```

The prime difference belongs to
`A_Q(s)=sum n**(-4)*(1-n**(-2))**(2*s)` over squarefree `n>=2` coprime to
`Q`. `prime_exclusion_disk_bound(p,R,k)` supplies the rational bound
`4/(3*(p**4+1)) * (p**2/(p**2-1))**(2*R) * (2/(p**2-1))**k`
for integer disk radius and derivative order. The bound uses the whole
arithmetic series, rather than a silently truncated sum.

For the signed sieve norm, prime-free squarefree atoms have mass `n**(-4)`;
atoms divisible by the selected prime have mass `-delta*n**(-4)`.
`sieve_interpolation_certificate` constructs exact Lagrange polynomials and
their uniform bounds. A strictly positive margin proves positivity for
every nonzero polynomial through the stated degree. A zero or negative margin
is inconclusive. `sieve_negative_detector` bounds the entire positive tail and
keeps the negative prime atom. Its killed labels must form the **complete
ordered initial segment** of positive squarefree labels. Omitting an earlier
positive node is rejected because the tail proof would then fail.

The degree-four and degree-seven certificates leave degrees five and six
undecided. These are sufficient bounds, without an optimal detecting-degree
claim. `detects_negativity=False` is also inconclusive.

## Resource guards and verification

`PaperMachineryBudget` caps the last odd index, exact rational bit length,
contraction-ladder depth, channel count, selected-node count, and integer
label. A cap raises `ResourceLimitError`; it never returns a shortened list
or weakens a certificate. The checks are conservative, and they are not a
uniform wall-clock or memory guarantee for arbitrary user inputs.

The implementation suite independently compares dense Picard integration,
the sine/cosine tangent quotient, a known hyperbolic-sine case, exact inverse
and reduction cases on a 256-point rational grid, Lagrange coefficient
identities, a finite signed norm, and analytic tail bounds. It also checks
wrong component swapping, noisy coefficients, invalid domains, incomplete
killed-label lists, and resource exhaustion. Passing this finite suite does
not replace the paper's universal analytic proofs.

The coverage records in `verification/paper_coverage/special_functions.json`
pin the source archive, selected source members, contracts, APIs, and checks.
Earlier Special Functions I–VIII APIs remain available in the existing
`shape`, `bernoulli`, and `terminating` modules. Full matrix recoupling,
continuous-series transforms, and unrestricted signed arithmetic estimates
are separate research machinery, and are not represented by these scalar APIs.
