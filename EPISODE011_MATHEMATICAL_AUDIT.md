# Episode 011 Mathematical Audit — Carrier-Change Obstructions

## Research decision

Episode 010 showed that a continuation-state packet is not meaningful without a
carrier algebra: convex/max aggregation and additive signal reconstruction obey
different laws. Episode 011 therefore tests maps *between* carrier semantics.

The working question is:

> Given two exact semantics of the same raw computation, when can one semantic
> carrier replace the other compositionally, and what information must be
> retained when exact replacement fails?

## 1. Factorization criterion

Let `X` be a raw algebra of expressions/histories and let

```
s : X -> S,
t : X -> T
```

be homomorphisms for the same operation signature. Then there exists a unique
homomorphism

```
phi : image(s) -> T
```

with `t = phi o s` iff

```
s(x) = s(y)  =>  t(x) = t(y).
```

Equivalently, the congruence kernel of `s` is contained in the congruence kernel
of `t`.

### Proof

If `t = phi o s`, equality of `s(x)` and `s(y)` immediately implies equality of
`t(x)` and `t(y)`. Conversely, if target values are constant on source fibers,
define `phi(s(x)) = t(x)`. The fiber condition makes this well-defined. Since
`s` and `t` preserve every primitive operation, `phi` does as well. Uniqueness
follows because `s` is surjective onto its image.

If factorization fails, any pair `x,y` with `s(x)=s(y)` and `t(x)!=t(y)` is an
exact obstruction witness.

The joint semantic map

```
j(x) = (s(x), t(x))
```

has kernel `ker(s) intersect ker(t)`. Its image is the coarsest quotient of the
raw computation that preserves both semantics. This is standard quotient /
factorization mathematics, not a novelty claim.

## 2. Pushdown versus post-observation

A query may always be evaluated *after* an exact source computation. Replacing
intermediate source nodes by target nodes requires more: every primitive
operation square must commute.

For binary aggregation this means

```
phi(a plus_S b) = phi(a) plus_T phi(b).
```

If the square fails once, the target carrier cannot be pushed through arbitrary
expression trees with that operation.

This distinction matters for Boolean support. `x != 0` is a legitimate final
query of a signed sum. It is not a valid intermediate replacement of signed
addition by Boolean OR because cancellation can change nonzero inputs into zero.

## 3. Cancellation barrier

Let `R` be a nonzero ring and `S` a zerosumfree semiring, so

```
a plus b = 0  =>  a = b = 0.
```

There is no unital semiring homomorphism `R -> S`.

### Proof

A unital homomorphism would give

```
0 = phi(1 + (-1)) = 1_S plus phi(-1).
```

Zerosumfreeness forces `1_S=0`, contradicting nontriviality.

Thus a signed additive carrier cannot be exactly replaced by Boolean OR,
max-plus, or any other zerosumfree target while preserving the ring operations.
The Haar cancellation counterexample from Episode 010 is a concrete instance.

By contrast, the support map from the nonnegative reals to the Boolean semiring
is an exact semiring homomorphism: zero maps to false and every positive value
to true. The difference is precisely the absence of additive cancellation.

## 4. Idempotence barrier

There is a dual obstruction. Let the source be a nontrivial unital semiring with
idempotent addition. Then

```
1 plus 1 = 1.
```

A unital homomorphism into an additively cancellative target would imply

```
1 + 1 = 1,
```

hence cancellation gives `1=0`, again impossible.

Therefore a Boolean/max carrier cannot be pushed into ordinary additive
semantics while recovering multiplicity. Episode 010's duplicate-route example
is this obstruction in signal form.

The loss can require an unbounded residual: all positive multiplicities
`n=1,2,...` have identical Boolean support but distinct additive counts. No
fixed finite residual alphabet can reconstruct exact unbounded multiplicity from
Boolean support.

## 5. Signed cancellation completion

`core.carrier_change.SignedMassPacket` stores a pair `(p,n)` of nonnegative
positive and negative masses.

Addition and multiplication are

```
(p,n) + (q,r) = (p+q, n+r)
(p,n) * (q,r) = (pq+nr, pr+nq).
```

The observation

```
value(p,n) = p-n
```

preserves both operations exactly.

For a finite signed decomposition,

```
total_variation = p+n
cancellation_defect = total_variation - |p-n| = 2 min(p,n).
```

This packet is an exact scalar cancellation completion. It is not claimed to be
a universal minimal signed representation; if only the final signed scalar is
queried, the scalar itself is sufficient.

## 6. Sum-product to max-plus residual completion

For finite scores `x_i` and temperature `tau>0`, set

```
m = max_i x_i
r = sum_i exp((x_i-m)/tau).
```

Then

```
F_tau = tau log(sum_i exp(x_i/tau)) = m + tau log r.
```

`TropicalResidualPacket` stores the max-plus value `m` together with the exact
log residual mass `log r` and the number of represented terms.

For alternative-union / sum-product addition, packet merge maps `m` by `max`
and updates the residual exactly. For product expressions, pairwise scores add,
so

```
m(PQ) = m(P)+m(Q)
correction(PQ) = correction(P)+correction(Q).
```

Thus the packet simultaneously preserves the finite-temperature sum-product
semantic and its max-plus projection under the generated `+`/`*` grammar.

The correction obeys

```
0 <= F_tau - m <= tau log n.
```

Equal scores attain the upper bound. Consequently there is no uniform finite
error bound for finite-temperature max replacement when fan-in is unbounded.
The quantity

```
effective_multiplicity = exp((F_tau-m)/tau)
```

lies between 1 and `n` and records the soft amount of alternative-path mass that
max-plus discards.

## 7. Mutual non-factorization of sum and max

Neither finite-temperature sum-product nor max-plus determines the other on raw
finite decompositions.

Same max, different sum:

```
[0]       versus [0,0].
```

Same finite-temperature sum at `tau=1`, different max:

```
[0,0]     versus [log 2].
```

Therefore a scalar translation in either direction is impossible on this raw
class. The joint packet is necessary for preserving both scalar semantics.

## 8. Query boundary: scalar completion is not provenance completion

At `tau=1`, compare

```
[0, 0, log(1/2)]
[0, log(3/4), log(3/4)].
```

Both have:

- maximum `0`;
- term count `3`;
- partition sum `5/2`;
- the same tropical correction.

But the first has two maximizing branches and the second only one. Therefore the
scalar residual packet does not determine argmax multiplicity or identity.
Those are separate observer/provenance semantics and require additional state.

## 9. Computational audit

The deterministic probe uses seed `20260923` and checks:

- 3,000 nonnegative Boolean-support trials;
- 3,000 signed cancellation trials;
- 5,000 tropical residual trials;
- 1,000 repeated random expression trees.

Observed failures: zero in every category.

Maximum relative errors against direct stable scalar formulas were below
`5e-16` in the recorded run. The computation validates the implementation in
that finite range; the universal theorems above are proved separately.

## 10. Literature boundary

The episode intentionally does not make a broad novelty claim.

- The semiring-homomorphism and congruence/factorization language is standard.
  A checked semiring reference explicitly gives the nonnegative-to-Boolean
  support homomorphism and the `Z -> Boolean` cancellation obstruction.
- The deformation

  `h log(exp(u/h)+exp(v/h)) -> max(u,v)`

  is standard Maslov dequantization / tropical mathematics.
- Stable shifted log-sum-exp formulas and their numerical behavior are standard
  numerical analysis.
- Signed/symmetrized tropical structures are established in tropical algebra.

The potentially useful Komorebi contribution is therefore not these individual
facts. It is the explicit typed separation between source carrier, target
carrier, factorization obstruction, residual completion, continuation grammar,
and downstream query semantics.

## Status

- **PROVED:** factorization criterion, cancellation barrier, idempotence barrier,
  signed-pair algebra identities, tropical residual composition identities,
  sharp `tau log n` bound, and the provenance counterexample.
- **COMPUTATIONALLY VERIFIED:** implementation on the deterministic finite probe.
- **REFUTED:** universal exact signed-to-zerosumfree translation; universal
  Boolean/max-to-additive multiplicity translation; scalar sum/max semantics as
  sufficient for argmax provenance.
- **NOVELTY:** low for the component mathematics; unresolved for the typed
  cross-domain organization as a machine-operational interface.
