# Block 20: Typed Finite Expression Calculus

Block 20 is the research scope shipped in `1.0.0b4`. It turns a bounded part
of Komorebi Math into a typed exact expression language. The new layer is
additive and remains outside the stable and beta namespaces unless a later
promotion review says otherwise.

This document states the mathematical and release boundary. It does not treat
an unexecuted package, wheel, documentation, or platform gate as passed.

## Scope

The supported semantic objects are:

* the rational scalar space
* finite univariate polynomial coordinate spaces through a declared degree
* exact graded basis witnesses with `Fraction` coefficients
* typed exact rational vectors and matrices
* exact linear maps between declared spaces
* finite expression DAGs built from values, maps, symbols, typed bindings,
  identities, zero maps, composition, addition, scaling, and application

Polynomial primitives cover coordinate changes, tight derivatives, basepoint
antiderivatives, evaluation, constant-density interval integration, affine
substitution, finite differences, endpoint functionals, and constant injection.
Each primitive is replayed from its exact basis and rational parameters.

Closed expressions reduce to exact matrix, vector, or scalar normal forms.
Equality in that closed language is decidable by type equality and entrywise
rational equality. Free symbolic terms retain decidable canonical structural
equality, but Block 20 does not infer arbitrary equations about unknown symbols.

## Structural type identity

A polynomial space is identified by its variable, degree, and complete exact
basis witness. A name is not a type. This prevents two same-name, same-dimension
bases from sharing coordinate-change or inverse evidence accidentally.

Canonical type and expression data are versioned and deterministic. The decoder
accepts only the declared finite schema. It does not load arbitrary Python
objects or invoke arbitrary callbacks.

## Certified algebra

The exact matrix model checks:

\[
C_{T\to U}C_{S\to T}=C_{S\to U},
\qquad
C_{T\to S}C_{S\to T}=I,
\]

\[
DJ_a=I,
\qquad
J_aD=I-u e_a,
\]

the naturality of differentiation, antiderivation, evaluation, and integration,
and the density-weighted fundamental theorem

\[
\ell_{[a,b],\rho}D=\rho(e_b-e_a).
\]

A certified rewrite records the exact before and after digests, theorem
identifier, side conditions, and composed evidence. A valid-looking user
certificate is not a substitute for replay. Inverse cancellation requires an
actual exact inverse. An unchecked condition prevents an exact conclusion.

## Dense matrix-chain planner

For a fixed compatible matrix chain, the planner minimizes the standard dense
scalar multiplication count over binary parenthesizations that preserve
factor order. The independent oracle compares the dynamic-programming answer
with exhaustive enumeration on several finite chains.

This is a precise but narrow optimality claim. It does not include sparsity,
low rank, memory traffic, parallel scheduling, device transfer, algebraic
factorization, or hardware timing. It is not a universal optimizer.

## Worked closure

The primary example begins with the Fibonacci-basis polynomial

\[
G_3(x)=F_4(x)=x^3+2x.
\]

It applies the zero-basepoint antiderivative in Fibonacci coordinates, changes
from degree-four Fibonacci to Euler coordinates, differentiates in the Euler
basis, and changes back to degree-three Fibonacci coordinates. The compiled
four-dimensional operator is exactly

\[
C_{E_3\to F_3}D_E^{(4)}C_{F_4\to E_4}J_{0,F}^{(4)}=I_4.
\]

The recovered polynomial integrates on \([0,1]\) to

\[
\int_0^1(x^3+2x)\,dx=\frac54.
\]

The complete matrices and every intermediate vector are recorded in
`BLOCK_20_WORKED_EXAMPLE.md`.

The same example replays the affine pushforward regression

\[
\frac12\int_1^3(y^4-2y+1)\,dy
=\int_0^1\bigl(16x^4+32x^3+24x^2+4x\bigr)\,dx
=\frac{106}{5}.
\]

## Independent mathematical evidence

`verification/block20_math_verifier.py` is intentionally separate from the
production expression engine. It uses standard-library `Fraction`, local
monomial, Euler, and Fibonacci formulas, local exact matrix routines, and an
independent matrix-chain implementation.

It currently passes 15 of 15 checks covering:

* same-name structural basis fingerprints
* exact coordinate cocycles and inverses through degree four
* both tight \(D,J_a\) laws
* coordinate naturality of \(D,J_a,e_a,\ell\)
* the density-weighted fundamental theorem row
* matrix-chain dynamic programming against exhaustive parenthesizations
* the full \(I_4\) Fibonacci pipeline and its \(5/4\) scalar result
* the affine and weighted \(106/5\) workflow
* false \(J_aD=I\), omitted density, and singular inverse adversaries

Run that focused oracle from the source tree with

```powershell
python .\verification\block20_math_verifier.py
```

This 15-check result is the only verification result claimed by this document.
It does not imply that source, API, stable, wheel, sdist, fresh-install,
documentation, Windows, CUDA, or other platform qualification has run.

## Compatibility requirements

The saved `1.0.0b3` boundary contains 1,229 research symbols, 131 stable
symbols, 83 beta symbols, 51 package-owned portable codecs, and 10 stable
semantic cases. Final Block 20 qualification must demonstrate all of the
following before distribution:

* research changes are additive relative to the saved `1.0.0b3` manifest
* the stable namespace remains exactly 131 symbols with no incompatible change
* the beta namespace remains exactly 83 symbols with no incompatible change
* the portable codec and migration contract remains compatible
* all 10 stable semantic cases remain unchanged and pass
* source, wheel, and fresh-sdist rebuilt-wheel executions agree

These are release requirements, not results asserted in advance.

## Resource boundary

Expression budgets bound graph nodes, depth, traversal work, polynomial degree,
exact entries, coefficient bit length, serialized structural size, and symbol
length. A cycle, malformed child, ill-typed edge, or exceeded limit is rejected
before an exact execution claim is issued.

Shared DAG nodes are memoized during checking, normalization, simplification,
and planning. Exact outputs are budgeted as well as inputs. Portable encoders
replay their expanded JSON trees through the paired strict decoder before
returning them, and runtime validation replays exact built-in class and
constructor invariants rather than trusting subclasses or mutated objects.

## Deliberate nonclaims

Block 20 does not claim arbitrary symbolic algebra, arbitrary user rewrite
theories, nonlinear or infinite-dimensional theorem proving, function-space
equality, floating-point numerical stability, analytic convergence, or a
universal optimizer. It does not extend the finite measure layer into general
Lebesgue or stochastic integration.

`1.0.0b4` remains a research beta. The existence of a research export does not
promote it into the frozen public tiers.
