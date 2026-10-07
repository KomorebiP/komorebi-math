# Block 2 release record

Version `0.2.0a1` implements the common execution layer needed by Papers II and III.

The completed path is now:

```text
seed
  -> address
  -> operate
  -> project
  -> transfer
  -> refine
  -> constrain
  -> finite reconstruct
```

Completion through inverse limits remains a later domain specific block.

## Refinement towers

`PacketRefinement` records one exact relation between a coarse packet system and a fine packet system. Every fine packet has one coarse parent, and each coarse projector is the sum of its children.

```python
refinement = PacketRefinement(coarse, fine, parent_of_fine)
fine_coordinates = refinement.refine(coarse_coordinates)
coarse_again = refinement.coarsen(fine_coordinates)
```

`RefinementTower` composes adjacent refinements and provides exact coordinate transport between any two declared levels.

`common_refinement` constructs the nonzero products of two commuting packet systems and returns the two refinement maps.

## Residue and Chinese remainder coordinates

The residue layer implements ordinary finite polyphase analysis:

```python
coordinates = residue_analyze(values, modulus=4)
values_again = coordinates.reconstruct()
```

It also implements:

* nested residue refinement
* mixed radix address composition
* finite diagonal residue packet systems
* pairwise coprime Chinese remainder addresses
* dyadic and odd address factorization

## Weighted transfer

`WeightedTransferGraph` combines an invertible packet diagonal operator with positive weight transfer edges.

```python
graph = WeightedTransferGraph(packets, diagonal, edges, prime=2)
truncation = graph.truncated_inverse(precision=8)
```

With exact rational data and a declared prime, the package verifies the p adic valuation of the omitted part. Without an exact valuation backend, the path sum remains available with a conditional certificate.

The implementation includes:

* weighted path enumeration
* minimum path weight
* path operator products
* filtered inverse truncation
* exact finite inversion for acyclic transfer graphs
* weighted adjacency matrices
* rational transfer series
* an explicit path enumeration safety limit

## Integral separation

`LatticePacketData` measures rational packet projectors against one full lattice basis.

```python
data = LatticePacketData(
    lattice_basis,
    projectors,
    prime=2,
    split_basis=split_lattice_basis,
)
```

It computes:

* each projector denominator weight
* the exact worst separation weight
* input precision delays
* the Smith invariants of a supplied split lattice inclusion
* the p primary gluing length and exponent
* a certificate comparing the gluing exponent with direct projector denominators
* monotonicity only after verifying that the fine projectors actually refine the coarse projectors

The split lattice basis is supplied by the caller. The package verifies integral containment and the resulting Smith data. It does not infer arbitrary lattice intersections automatically in this release.

## Smith congruences

`SmithCongruenceProblem` solves

```text
A x = b mod p^L
```

for an integer matrix and a genuine prime `p`.

The result includes:

* Smith invariant factors
* p primary exponents
* transformed right hand side
* exact obstruction rows
* one particular solution
* kernel generators
* the exact solution count
* optional enumeration when the count is below a declared limit

Signed Smith diagonal entries are handled correctly when constructing a particular solution.

## Linear refinement defects

For

```text
Delta = C T_f - T_c C
```

`LinearRefinementDefect` computes and verifies:

* power defects
* polynomial defects
* resolvent defects
* exponential defects through a block exponential
* exact multilevel telescoping across a finite chain

SciPy is required for numerical matrix exponentials. SymPy provides an exact matrix path when the exponential is symbolically evaluable.

## Nonlinear coordinate maps and defects

`NonlinearPacketMap` conjugates an ambient nonlinear map through exact packet analysis and synthesis.

`NonlinearRefinementDefect` evaluates

```text
C_Y F_f(x) - F_c(C_X x)
```

and supplies:

* the exact composition identity at a point
* iterated orbit error bounds from a user supplied Lipschitz constant
* a fixed point error bound that verifies the supplied fine point by default

Packet jets, rooted trees, nonlinear elimination, and smooth transport are not yet implemented.

## Cokernel obstruction

`LinearObstructionProblem` computes the obstruction to

```text
D u = -r
```

as a finite dimensional cokernel class.

The exact SymPy path returns:

* algebraic left nullspace coordinates
* cokernel dimension
* a correction when the class vanishes
* a basis of the correction kernel

The NumPy path uses least squares and an orthonormal numerical left nullspace. Its certificate records residuals and tolerance.

`ObstructionSquare` verifies a commuting correction square before transporting a defect.

## Prime power lifting

The lifting layer contains:

* exact row reduction over a prime field
* affine correction spaces
* one step linear lifts
* one step polynomial lifts
* finite lift trees
* terminal branch reconstruction

For polynomial systems, the Jacobian correction law is checked over every residual correction vector whenever the finite check is below the declared limit. If the user explicitly trusts a Jacobian that is too large to verify, the result is marked conditional.

## Finite holonomy

`TransportGraph` stores invertible matrix transports on a connected graph. It computes:

* spanning trees
* root to vertex transports
* tree gauges
* gauge transformed systems
* fundamental cycle holonomies
* gluing certificates
* propagated vertex intertwiners
* edge and cycle residuals

Certificate labels remain unique even when several user supplied edges have the same name.

## Verification

The release is accepted only after:

* the complete Block 1 suite remains green
* all Block 2 unit tests pass
* the installed verification commands pass
* exact Smith systems agree with brute force enumeration on deterministic finite samples
* lift children agree with brute force one step searches
* random acyclic weighted inverses agree with direct exact inversion
* holonomy remains invariant under finite gauge transformations
* a clean wheel installs outside the source tree
* the source distribution rebuilds to an installable wheel
* the wheel contains no paper PDFs, manuscript sources, or research archive payload
