# Typed finite expression calculus

The typed expression calculus composes finite polynomial and coordinate
operators using exact rational values. It supplies explicit types, replayable
transformations, canonical normal forms, and decidable equality within its
finite semantic domain.

The implemented semantic domain is finite and rational:

\[
\mathbb Q,
\qquad
\mathcal P_n^B(x)=\operatorname{span}_{\mathbb Q}
\{b_0(x),\ldots,b_n(x)\},
\qquad
\operatorname{Hom}_{\mathbb Q}(X,Y).
\]

Every coefficient, matrix entry, endpoint, density, and affine parameter in
the exact fragment is an integer or a reduced `Fraction`.

## Exact polynomial spaces

A polynomial coordinate space is not identified by a family name. It is
identified by

\[
(x,n,B),
\qquad
B=(b_0,\ldots,b_n),
\]

where each basis polynomial has exact rational coefficients,

\[
b_k(x)=\sum_{j=0}^k B_{jk}x^j,
\qquad B_{kk}\ne0.
\]

Thus the coefficient matrix

\[
B_n=(B_{jk})_{0\le j,k\le n}
\]

is triangular with a nonzero diagonal and is exactly invertible over
\(\mathbb Q\). The display label is metadata. Two spaces with the same label
and dimension but different coefficient witnesses are different types.

This distinction prevents a common error. A statement such as "both vectors
use the Fibonacci basis" is not a type proof. The type checker compares the complete
variable, degree, and basis witness. Its structural fingerprint is a
deterministic digest of that canonical exact data. Exact type equality still
comes from the underlying witness, not from assuming that a cryptographic hash
can never collide.

The primary type objects are `RationalSpace`,
`PolynomialCoordinateSpace`, and `LinearMapType`. A linear map type records its
source and target spaces explicitly.

## Typed expressions

The expression layer contains immutable nodes for exact values, exact maps,
symbols, typed `let` bindings, identities, zero maps, composition, sums, rational
scaling, and map application. The constructors reject an expression before it
can enter the graph when any of the following fails:

* a vector length does not equal its space dimension
* a matrix shape does not equal target dimension by source dimension
* adjacent composition endpoints do not agree
* a sum contains unlike types
* a map is applied to a value outside its source
* a bound value does not have the symbol's declared type

Composition is stored outermost to innermost. If

\[
A:X\to Y,
\qquad B:Y\to Z,
\]

then `compose(B, A)` denotes \(B\circ A:X\to Z\). The `then(A, B)` helper
accepts the same maps in the order in which they act.

`Symbol` and `Let` provide a finite first-order binding mechanism. They are not
a lambda calculus and do not introduce higher-order functions. Evaluation of a
closed term checks scope and declared types. Free symbols remain explicit and
cannot be silently replaced by similarly named objects. Binder identity is
object identity, not dataclass field equality, so a free symbol with the same
printed name and declared type as a binder does not become bound accidentally.

## Replayable polynomial primitives

For a basis matrix \(B_n\), a coefficient vector \(c_B\) represents the
monomial coefficient vector

\[
c_M=B_nc_B.
\]

If \(S\) and \(T\) are exact bases through the same degree, the coordinate
change from source to target is

\[
C_{S\to T}^{(n)}=B_{T,n}^{-1}B_{S,n}.
\]

It follows immediately that

\[
C_{T\to U}^{(n)}C_{S\to T}^{(n)}=C_{S\to U}^{(n)},
\qquad
C_{T\to S}^{(n)}C_{S\to T}^{(n)}=I_{n+1}.
\]

The other primitive matrices are also reconstructed from their exact
structural data rather than accepted from a label or a certificate supplied by
a caller. The public primitive constructors cover:

* coordinate change
* the tight derivative
* a basepoint antiderivative
* point evaluation
* constant-density interval integration
* affine substitution
* finite difference
* the endpoint boundary functional
* constant injection

`replay` recomputes the matrix. When a caller supplies a claimed matrix, exact
certification compares every entry with that replay.

### Derivative and basepoint antiderivative

For \(n\ge1\), the tight derivative is

\[
D_B^{(n)}:\mathcal P_n^B\longrightarrow\mathcal P_{n-1}^B.
\]

It has shape \(n\times(n+1)\), not \((n+1)\times(n+1)\). Let

\[
J_{a,B}^{(n)}:\mathcal P_{n-1}^B\longrightarrow\mathcal P_n^B
\]

be the unique primitive whose value at \(a\) is zero. Then

\[
D_B^{(n)}J_{a,B}^{(n)}=I_n.
\]

The reverse product is not the identity. If \(u_B\) is the coordinate column
of the constant polynomial \(1\), and \(e_{a,B}\) is evaluation at \(a\), then

\[
J_{a,B}^{(n)}D_B^{(n)}=I_{n+1}-u_Be_{a,B}.
\]

This rank-one defect records the lost integration constant.

### Evaluation, integration, and the fundamental theorem

Evaluation and integration are exact covectors. For constant density
\(\rho\ge0\),

\[
e_{a,B}c_B=p(a),
\qquad
\ell_{B,[a,b],\rho}c_B
=\rho\int_a^b p(x)\,dx.
\]

They transport contragrediently under coordinate changes:

\[
e_{a,T}C_{S\to T}=e_{a,S},
\qquad
\ell_{T,[a,b],\rho}C_{S\to T}
=\ell_{S,[a,b],\rho}.
\]

The derivative and antiderivative naturality squares are

\[
D_T^{(n)}C_{S\to T}^{(n)}
=C_{S\to T}^{(n-1)}D_S^{(n)},
\]

and

\[
J_{a,T}^{(n)}C_{S\to T}^{(n-1)}
=C_{S\to T}^{(n)}J_{a,S}^{(n)}.
\]

The finite exact fundamental theorem is the row identity

\[
\ell_{B,[a,b],\rho}^{(n-1)}D_B^{(n)}
=\rho\bigl(e_{b,B}^{(n)}-e_{a,B}^{(n)}\bigr).
\]

These are matrix equalities over \(\mathbb Q\). No sampling tolerance is used.

## Exact normal forms and decidable equality

A closed, well-typed map expression normalizes to its type and one exact
rational matrix. A closed value expression normalizes to its space and one
exact rational vector, or one rational scalar in \(\mathbb Q\). `Fraction`
normalization gives each entry a unique numerator and positive denominator.
Matrix dimensions and the complete space witnesses are part of the normal
form.

Consequently, equality is decidable for this closed exact language:

1. type-check both expressions
2. replay every primitive
3. evaluate finite sums, scales, applications, bindings, and products
4. compare the resulting types and rational entries

There is no numerical threshold and no dependence on a printed family name.
For expressions with unresolved free symbols, canonical structural equality is
decidable, but the expression calculus does not decide every universally quantified
identity that a user might intend those symbols to satisfy.

## Canonical data and digests

Types and expression graphs have a closed, versioned JSON representation. The
portable entry points are `type_to_dict`, `type_from_dict`,
`expression_to_dict`, `expression_from_dict`, `expression_to_json`,
`expression_from_json`, and `expression_digest`.

The decoder accepts only the declared expression schema and exact rational
tokens. It does not import arbitrary classes, call arbitrary functions, or
deserialize arbitrary Python objects. Deterministic field order, normalized
Unicode text, normalized fractions, exact basis witnesses, and canonical child
order make repeated serialization stable. Noncommutative composition order is
never sorted away.

## Certified rewrites

A rewrite is more than a before-and-after pair. A `RewriteStep` records the rule
name, theorem identifier, before digest, after digest, checked side conditions,
and its certificate. Built-in exact rewrites are replayed in the finite rational
model. Type preservation and equality of exact normal forms are checkable.

Certificate composition never strengthens a child claim. Invalid evidence
remains invalid. Numerical, conditional, heuristic, and planned inputs cannot
become exact merely because a surrounding algebraic identity is exact. Any
unchecked side condition makes the composed conclusion at most conditional.

The important guarded rewrites include coordinate cocycles, derivative and
antiderivative laws, coordinate naturality, evaluation and integration
transport, and the finite fundamental theorem. Inverse cancellation requires a
replayed exact inverse. A square shape or a certificate whose message says
"exact" is not enough.

## Matrix-chain planning

Once an expression has become a compatible chain of dense matrices, the planner
may choose a multiplication parenthesization. For matrices with dimension chain

\[
p_0\times p_1,
p_1\times p_2,
\ldots,
p_{r-1}\times p_r,
\]

the dynamic program uses

\[
m[i,j]=\min_{i\le k<j}
\left(m[i,k]+m[k+1,j]+p_i p_{k+1}p_{j+1}\right).
\]

This proves the minimum dense scalar-operation count among binary
parenthesizations that preserve the matrix order. It does not optimize across
algebraic rewrites, sparsity patterns, low-rank structure, memory traffic,
parallel scheduling, device transfer, or hardware-specific kernels. Equal-cost
plans may exist even when deterministic tie-breaking selects one of them.

## Resource boundary

`ExpressionBudget` checks a finite expression DAG before execution. It bounds
node count, depth, traversal steps, polynomial degree, exact-entry count,
coefficient bit length, serialized structural size, and symbol length. Cycles
and malformed children are rejected. These guards are part of the semantics:
exceeding a declared limit does not produce a partial exact certificate.

Graph checking, normalization, simplification, and plan construction memoize
shared nodes in their applicable lexical context, so a DAG is not evaluated as
an exponentially expanded tree. Exact intermediate and final values are
rechecked against entry, coefficient-bit, and serialized-size limits. The
dense matrix-chain dynamic program also has a fixed work guard, and expression
compilation charges planning work to the caller's step budget.

Portable JSON necessarily expands shared nodes into occurrences. Before an
envelope is returned, the encoder replays that generated tree through the same
strict decoder and the same budget. Thus an encoder cannot emit a transport it
would reject itself because of expanded node, edge, exact-entry, label, or type
witness counts. Runtime validation also accepts only the exact built-in node
and type classes and replays constructor invariants, preventing subclasses or
mutated frozen objects from spoofing a certified denotation.

## Deliberate nonclaims

The expression calculus does not claim:

* arbitrary symbolic algebra or arbitrary user-defined rewrite theories
* nonlinear, infinite-dimensional, or function-space theorem proving
* analytic convergence or equality of general functions
* floating-point error analysis or numerical stability
* general Lebesgue, stochastic, manifold, or distributional calculus
* a universal optimizer

It is a finite exact linear language for a deliberately bounded polynomial
coordinate calculus. Within that boundary, types, replay, normal forms, and
equality are explicit and mechanically checkable.
