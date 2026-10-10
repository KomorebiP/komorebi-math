# Papers I–IV: executable machinery coverage

This audit compares the current runtime with the 8 October 2026 traditional
editions. The corresponding source identities, numbered references, APIs,
tests, and limitations are recorded in `verification/paper_coverage/core.json`.
It is a machinery coverage audit, not a new proof audit or a claim that finite
Python checks reproduce the Lean developments or every general theorem.

The finite foundation is substantial. Paper I's typed paths, finite generated
algebras, seed hulls, free evaluation presentations, packet coordinates,
transfer blocks, support preorders, and declared invariant quotient maps have
callable implementations. Paper II adds refinements, residue/CRT addresses,
weighted paths and rational walk series, supplied lattice gluing data, Smith
congruences, and propagated operator defects. Paper III includes nonlinear
coordinates, jets, higher composition, tree sums, graded operations,
correction/lifting, finite spectral sequences, Schur elimination, holonomy,
and numerical smooth projector transport. Paper IV supplies the cyclic,
reciprocal, Appell, and theta coefficient machinery and numerical zero
diagnostics.

## Added finite constructions

`komorebi_math.core.paper_machinery` now supplies two previously missing generic
constructions. Both use the existing packet/complex implementations.

### Exact rational spectral projectors

Paper II Theorems 7.1–7.2 construct spectral projectors from a coprime
characteristic factorization by a polynomial Bézout identity. The new
`rational_spectral_projectors` implements that construction over the rational
field. It checks the supplied factorization rather than trusting a list of
labels. Repeated roots inside one factor are allowed, while distinct factors
must be coprime.

`rational_primary_packets(A)` provides the single-operator adapter: it factors
the rational characteristic polynomial and uses each irreducible factor to
its full multiplicity. The resulting primary projectors are discovered from
the supplied matrix. This does not discover a general commutative Hecke
algebra decomposition or global transfer data.

```python
import sympy as sp
from komorebi_math.core.paper_machinery import rational_spectral_projectors

x = sp.Symbol("x")
A = sp.Matrix([[0, 1], [0, 2]])
packets = rational_spectral_projectors(A, [x, x - 2])
packets.certificate().require()
assert packets.projectors[0] == sp.Matrix([[1, -sp.Rational(1, 2)], [0, 0]])
assert packets.projectors[1] == sp.Matrix([[0, sp.Rational(1, 2)], [0, 1]])
```

The example produces oblique exact projectors. Their sum is the identity,
their pairwise products vanish, and each is idempotent. Nonmatching,
noncoprime, nonmonic, constant, and floating-point factors are rejected.
This function constructs neither an arbitrary DVR realization nor a lattice
intersection basis. Use `LatticePacketData` with explicitly supplied and
validated lattice data for the existing separation/gluing calculations.

### Finite mapping cones

Paper III Definition 21.10 fixes the cochain convention
`Cone(f)^n = E^n ⊕ C^(n+1)` with differential
`[[d_E, f_(n+1)], [0, -d_C]]`. The new `finite_mapping_cone` constructs it over a
common prime field, checks each cochain-map square, and retains exact Python
integer arithmetic. Sparse and negative cochain degrees are supported.

```python
from komorebi_math.core.paper_machinery import finite_mapping_cone
from komorebi_math.deformation.complexes import FiniteCochainComplex

C = FiniteCochainComplex({0: 1}, {}, prime=7)
E = FiniteCochainComplex({0: 2}, {}, prime=7)
cone = finite_mapping_cone(C, E, {0: [[1], [1]]})
cone.certificate().require()
assert cone.cohomology(-1).cohomology_dimension == 0  # kernel
assert cone.cohomology(0).cohomology_dimension == 1   # cokernel
```

The result is a `FiniteCochainComplex`, so its existing cohomology dimension and
compatible packet-cohomology operations apply. This constructor does not
compute general cohomology representatives, all connecting homomorphisms, or
an infinite-dimensional mapping cone. Missing component maps mean zero maps,
and the cochain-map squares must still commute.

## Corrected exactness failure

The existing filtered prime-field runtime used fixed-width NumPy integer
arrays. Products overflowed before modular reduction. For the prime
`4294967291`, the matrix

```python
[[1, 4294967290], [4294967290, 1]]
```

has exact rank one. The old `FilteredVectorSpace` nevertheless accepted it as
a full two-dimensional `F^0` and emitted a valid exact certificate. This was a
runtime soundness failure.

`nonlinear.filtered` now stores Python integers in object arrays for matrix
products, row reduction, coordinates, corrections, branch differences, and
spectral-sequence computations. The same invalid `F^0` is rejected. Valid
large-prime chain identities and correction equations are preserved exactly.

The new tests compare 243 public filtered-cohomology calculations with an
independent scalar determinant/rank oracle. They also exercise a genuinely
nonzero obstruction, valid and invalid differential compositions, noncanonical
filtration coordinates, and input coefficients larger than 200 bits. Cone
controls check the differential sign, invalid commuting squares, kernel and
cokernel dimensions, identity-cone acyclicity, and input ownership. Spectral
controls use Jordan blocks and an irreducible quadratic and compare the
projectors with independent change-of-basis formulas.

## Remaining boundaries

| Paper | Executable boundary |
|---|---|
| I | Supplied semisimple multiplicities can be evaluated, but there is no generic Jacobson radical or division-ring decomposition solver. Single-operator seed rank is implemented over rational matrices. Countable unconditional sums, general completion, and universal functor preservation remain mathematical hypotheses rather than executable proofs. |
| II | Weighted truncations require the declared filtration/weight assumptions. Gluing calculations require supplied split-lattice data. General DVR operations, automatic resultant bounds, and arbitrary holomorphic functional calculus are not supplied by finite matrix examples. |
| III | Finite field computations do not establish an infinite compatible branch. Nonlinear contraction and analytic bounds retain their external hypotheses. Smooth projector transport and Hilbert corrections are numerical. Cone cohomology currently exposes dimensions, not a general class-representative/connecting-map API. |
| IV | Zero routines certify finite numerical residuals, not complete root classification or the finite exceptional set. Partial products and zeta sums do not implement the full canonical product, Fredholm operator, meromorphic continuation, or derivative-weighted entire zeta theory. Formal coefficients and analytic branch choices remain distinct. |

These boundaries are included in the machine-readable crosswalk. Reusing a
packet class or theorem name is not evidence that an entire paper theorem is
implemented.
