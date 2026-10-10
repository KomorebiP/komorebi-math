# Papers IX–XII: executable machinery and exact boundaries

This audit uses both Paper IX manuscripts, followed by the latest traditional
editions of Papers X, XI, and XII. Their exact source identities and the
construction crosswalk are recorded in
`verification/paper_coverage/late.json`. The PDFs were extracted in full for
inventory. The crosswalk separates finite executable constructions, numerical
models, formal mathematical statements, and external arithmetic inputs. It
does not independently reprove every statement or inherit a Lean certificate
for the new Python code.

## Added finite polynomial lifting

`arithmetic.polynomial_lifting` now executes the one-digit polynomial
obstruction from Paper IX §4.1, the shared correction test from §4.2, the
correction torsor and cardinality from §4.3, finite compatible towers from
§4.4, and prescribed parameter digits from §4.6. In the Euler companion these
are §§10.1–10.4 and 10.6.

Its coefficients, polynomial evaluation, Jacobians, row reduction, and
residue arithmetic use Python integers. Inputs are frozen. A prime-field
correction fiber exposes its particular digit, a full kernel basis, rank,
cokernel dimension, and exact count. An inconsistent system has an empty
fiber and a failed existence certificate. The count can be inspected without
enumeration. Enumeration has `max_count`; finite towers have `max_steps`.

For example, the equation `Y² − T = 0` at `(1,1)` modulo three has three next
lifts. Prescribing the parameter digit two leaves one:

```python
from komorebi_math.arithmetic.polynomial_lifting import (
    IntegerPolynomialSystem, SparseIntegerPolynomial,
)

equation = SparseIntegerPolynomial(2, {(2, 0): 1, (0, 1): -1})
system = IntegerPolynomialSystem((equation,))
fiber = system.lift_fiber((1, 1), prime=3, depth=1)
assert fiber.count == 3
prescribed = system.lift_fiber(
    (1, 1), prime=3, depth=1, fixed_digits={1: 2},
)
prescribed.certificate.require()
assert prescribed.lifts() == ((4, 7),)
assert system.evaluate((4, 7)) == (9,)  # zero modulo nine
```

`first.stack(second)` constructs a simultaneous correction problem, so
separate liftability does not imply a common lift. The paper's control
`F(X)=X`, `G(X)=X−2` over zero modulo two gives separate next roots zero and
two modulo four, while the stacked fiber is empty.

The domain is integer polynomial coefficients and positive depth over a
prime residue field. Composite residue rings, arbitrary p-adic coefficients,
infinite endpoints, local homeomorphisms, weighted inverse-Jacobian bounds,
and analytic root graphs are separate boundaries. The general composite
residue-ring statement of the paper is broader than this new API. A singular
fiber can lift once and fail later: `X²−5` has lifts from modulus two to four
and no lift from four to eight. The finite-tower API raises at that later
obstruction.

## Existing Euler packets and added rational stress controls

The full-address family was already executable through
`analytic.appell.congruential_inverse(N,j,degree)`. It returns the normalized
packet `C[N,j](n)` for rows zero through `degree`, with ordinary degree `N*n`.
The original coefficient is `U[N,j](n) = j! * C[N,j](n)`. In particular, its
constant coefficient is `j!`. This differs from the classical signed Euler
family represented by `EulerLocalCoordinates`.

The audit reuses that implementation. Independent tests compute the cleared
triangular recurrence from Paper IX §3.1, check denominator clearing, compare
the two classical width-two addresses with signed Euler and Bernoulli
coefficients, and test original-coefficient Nishibuchi displacements.

`arithmetic.packet_differences` adds exact rational mixed stencils, combining
repeated offsets before evaluation. It accepts integer and `Fraction` values
and has explicit `max_stencil_size` and `max_order` limits. This complements
`arithmetic.local_modes.finite_difference_value`, whose function values must
be integers. `certify_packet_difference` checks one supplied finite valuation
bound; its diagnostic explicitly records `finite_statement_only=True`.

```python
from komorebi_math.analytic.appell import congruential_inverse
from komorebi_math.arithmetic.packet_differences import certify_packet_difference

packet = congruential_inverse(4, 1, 8)
evidence = certify_packet_difference(
    packet, 0, (2, 2), prime=2, valuation_bound=5,
)
evidence.certificate.require()
assert not certify_packet_difference(
    packet, 0, (2, 2), prime=2, valuation_bound=6,
).certificate.valid
```

The latest Euler companion's §§4–7 are tested with exact rational arithmetic:
prime-power signed mixed transfer, four-address factorial gains, the mixed
cancellation class, homogeneous exact laws in addresses zero and one,
address-two power orders, and the upper-half counterexamples at `(16,9)`,
`(32,18)`, and `(32,19)`. Those counterexamples prevent a test framework from
silently promoting a lower bound to a universal isometry law. Finite samples
do not prove infinite interpolation or exact Taylor radii.

## Finite occupancy repair

`deformation.support.nested_subspace_occupancy` incorrectly used the first
`rank` columns of unpivoted QR. A leading zero or dependent column could
select a vector outside the supplied column space and report a false empty
intersection. It now obtains the column space from left singular vectors.
An independent exhaustive control checks every pair of coordinate subspaces
in dimensions one through four after a common nonorthogonal change of basis,
including leading zero and repeated columns.

The former `stabilized_at` also reported the first adjacent equality even if
a later supplied intersection shrank. It now reports the first unchanged
step in the final constant run of the supplied finite prefix. It is `None`
when the final step shrinks or only one level is supplied. This is a numerical
finite observation, not evidence of stability at unseen future levels.

## Exact tame tensor characters and local polynomials

`langlands.local.tame_tensor.tame_tensor_character_packets` now executes the
actual character data in Paper XI §§5.1–5.3. It constructs the simultaneous
translation cycles, norm-lift exponents, exact rational closing scalars, and
the product of `1 − a_i*u^L` over precisely the inertia-trivial cycles. It
also records inertia-fixed and Weil-fixed dimensions and the tame Artin
conductor. An interaction cycle can be reducible, so its dimension is
retained rather than asserted to be an irreducible atom.

```python
from komorebi_math.langlands.local.tame import FrobeniusOrbit
from komorebi_math.langlands.local.tame_tensor import tame_tensor_character_packets

tensor = tame_tensor_character_packets(
    FrobeniusOrbit(2, 2, 1), FrobeniusOrbit(2, 2, 2),
)
tensor.certificate.require()
assert tensor.inertia_fixed_dimension == 2
assert tensor.weil_fixed_dimension == 1
assert tensor.local_polynomial_coefficients == (1, 0, -1)  # 1 − u²
```

Both input orbits must be regular with the same prime-power residue
cardinality. Scalars must be nonzero integers or `Fraction` values. Explicit
dimension, lifted-modulus bit, and residue-cardinality caps bound the
construction. The bit cap is checked using the conservative preconstruction
bound `lcm(n,m) * q.bit_length()`, which can reject a modulus whose actual bit
length would fit. The scalar convention is the forward orbit cycle
`F(e_i)=e_(i+1)` with the closing edge `a`; comparisons with actual arithmetic
data must retain its Frobenius normalization. The code verifies finite
character/cycle data, while arithmetic attachment remains an explicit
unchecked input. This does not supply arbitrary wild pair factors or a local
correspondence.

## Coverage boundaries beyond the added primitives

| Paper | Executable layers | Broader machinery that remains separate |
|---|---|---|
| IX | Residue/Hurwitz/Mahler/recurrence transfers; all-address reciprocal packets; classical local Euler reconstruction; exact prime-field polynomial correction fibers and finite towers; rational mixed differences | Composite-ring polynomial fibers; arbitrary p-adic endpoints; weighted nonunit-Jacobian root/deformation theorem; analytic charts and exact Taylor radii; unrestricted singular persistence |
| X | Finite lift forests and visibility; smooth/weighted/singular counting models; finite cochains and packet cohomology; directed support graphs; finite Frobenius trace separation; supplied compatible carriers; repaired numerical intersections | Infinite inverse limits and horizontal companion existence; actual number-field lattice estimates and ideal-class/norm-character attachments; universal arithmetic occupancy |
| XI | Finite fields and numerical Gauss phases; tame orbits and cycle matrices; exact norm-lift tensor character cycles and rational local polynomials; finite Heisenberg/symplectic/Arf data; declared finite bridges and ramification towers; matrix transport graphs | Complete convention-aware Weil–Deligne constructors; arbitrary local-field/type construction; Artin–Schreier restricted arithmetic appendices; the general realization of Conjecture 66 and complete pair-factor comparison |
| XII | Presented-module Fitting minors and point fibers; supplied finite Hecke projectors; finite declared support comparisons; coefficient-two sign example; finite companion forests; conditional five-gate reports | General completed tensor/quotient rings and support descent algorithms; canonical multivariate nonsemisimple Artin packet extraction; changing-carrier vector-space stable cores; actual automorphy lifting, classicality, attached Galois representations and missing companions |

The latest Paper XI uses geometric Frobenius and the scalar chain
`F = diag(1,q,...,q^(m−1))`, with `F*N*F^−1 = q^−1*N` (§2.4).
`SpecialMonodromyBlock` currently uses its argument `q` as the conjugacy scale
in `F*N*F^−1 = q*N`; it is a numerical scale model, so that argument must be
the inverse residue cardinality when matching the geometric convention.
Likewise, the existing tame matrix relation is stated with the inverse
Frobenius convention. The orbit sets are unchanged, but the convention and
closing scalar must be retained in any actual local-factor comparison.

## Reproducible controls

Run against the checkout explicitly:

```bash
PYTHONPATH=src python -m pytest -q tests/test_paper_late_machinery.py
```

The controls include prime-field polynomial lifts at primes 2, 3, and 5 and
depths one and two, exact direct enumeration, prescribed parameter digits,
shared incompatibility, later singular obstruction, finite smooth towers
through depth twelve, integer coefficients larger than 128 bits, snapshot
mutation, rank/count caps, composite/depth/domain rejection, all addresses at
widths one through eight through row sixteen, dyadic rows zero through four
and displacements around powers of two, selected Nishibuchi primes through
eleven, and the current Euler factorial/mixed-step boundaries. The numerical
occupancy tests use well-separated exact subspaces. They do not assert a
validated error enclosure for nearly coincident subspaces.

The exact tame controls compare every degree pair one through four at residue
cardinalities two, three, and four with independently constructed tensor
permutation matrices, exact restricted determinants and fixed-space ranks.
They also multiply the actual edge scalars around each cycle and reject
nonregular inputs, incompatible cardinalities, inexact/zero scalars, and
exceeded resource caps.
