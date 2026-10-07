# Komorebi Math Block 20

Block 20 introduces a typed finite expression calculus at the research tier in
`1.0.0b4`.

The language works with finite univariate polynomial coordinate spaces whose
complete basis witnesses are stored with exact `Fraction` coefficients. It
supports typed rational values and linear maps, replayable polynomial
primitives, canonical data, exact matrix, vector, and scalar normal forms,
certified rewrites, and bounded execution.

A polynomial basis is identified by its variable, degree, and complete exact
witness. A display name is not a type. This prevents two same-name bases from
sharing coordinate-change or inverse evidence accidentally.

For closed well-typed expressions, equality is decidable by reducing both
expressions to their typed exact rational normal forms and comparing every
entry. The supported calculus includes coordinate changes, tight derivatives,
basepoint antiderivatives, evaluation, constant-density interval integration,
affine substitution, finite difference, endpoint functionals, and constant
injection.

The principal exact laws are

\[
C_{T\to U}C_{S\to T}=C_{S\to U},
\qquad
DJ_a=I,
\qquad
J_aD=I-u e_a,
\]

and

\[
\ell_{[a,b],\rho}D=\rho(e_b-e_a).
\]

The worked pipeline starts with

\[
G_3(x)=F_4(x)=x^3+2x
\]

in Fibonacci coordinates. It applies \(J_0\), changes to Euler coordinates,
differentiates, and changes back. The complete operator is exactly \(I_4\), and
the final integral on \([0,1]\) is \(5/4\). The example also replays the affine
and weighted integration identity yielding \(106/5\).

The standalone mathematical verifier is independent of the production
expression and polynomial-family implementations:

```powershell
python .\verification\block20_math_verifier.py
```

It passes 15 of 15 exact checks. Those checks include the full worked pipeline,
coordinate naturality, exhaustive matrix-chain comparisons, and adversaries for
a false \(J_aD=I\) rewrite, omitted pushforward density, same-name basis
confusion, and singular inversion.

The dense matrix-chain planner is optimal only for scalar multiplication count
among binary parenthesizations of one fixed ordered dense chain. It is not a
hardware, memory, sparsity, or universal optimizer.

Block 20 does not claim arbitrary symbolic algebra, nonlinear or
infinite-dimensional theorem proving, general function-space equality,
floating-point numerical stability, or universal optimization. It remains a
finite exact rational research layer.

The execution boundary is explicit: shared expression DAGs are memoized,
intermediate exact outputs remain inside the caller's resource budget, and a
portable JSON envelope is replayed through its own strict decoder before it is
returned. Exact built-in classes and their constructor invariants are replayed
at the trust boundary.

The independent 15-check oracle is only one release gate. Package, API, wheel,
sdist, documentation, and platform results are recorded separately by the
Block 20 qualification workflow; unexecuted Windows or CUDA paths are not
silently claimed.

See `docs/TYPED_FINITE_EXPRESSION_CALCULUS.md` for the mathematical boundary,
`docs/BLOCK_20_WORKED_EXAMPLE.md` for every exact matrix and intermediate
vector, and `docs/BLOCK_20_RELEASE.md` for the release contract.
