# Carrier-change obstructions

A carrier translation preserves downstream semantics only when it respects the
operations and observations used by the computation. The cancellation in
[polyphase routing](POLYPHASE_ROUTE_PACKETS.md) gives a concrete example.

## Factorization criterion

Suppose one raw computation space `X` carries two exact algebraic semantics

```
s : X -> S,    t : X -> T.
```

If both are homomorphisms for the expression grammar, then an exact carrier map
`phi : image(s) -> T` with `t = phi o s` exists exactly when

```
s(x) = s(y)  =>  t(x) = t(y).
```

Equivalently, the kernel congruence of `s` is contained in that of `t`.  If the
condition holds, define `phi(s(x)) = t(x)`; kernel inclusion makes this
well-defined and the homomorphism law follows from those of `s` and `t`.  If the
condition fails, any collision `s(x)=s(y)` with `t(x)!=t(y)` is a complete
carrier-change obstruction witness.

When neither semantic factors through the other, the joint map

```
x -> (s(x), t(x))
```

is the coarsest raw quotient sufficient to recover both semantics; its kernel is
`ker(s) intersect ker(t)`.  `finite_factorization_audit` implements the finite
fiber test without presenting sample success as a proof of the universal claim.

A useful compiler interpretation follows.  A query may always be applied *after*
an exact source computation.  It can be pushed down through intermediate nodes
and used as a replacement carrier only when the corresponding operation squares
commute.  Signed nonzero-support is a valid final query of a signed sum, but it
cannot be pushed through addition as Boolean OR because cancellation breaks that
square.

For a source carrier `(M, plus_M, times_M)` and target carrier
`(N, plus_N, times_N)`, a translation `phi : M -> N` is exact for every finite
expression built from those operations only when it commutes with the required
primitive operations (and with any branch actions / observations used by the
query).  This is the ordinary homomorphism requirement expressed as a packet
contract.

### Cancellation barrier

Let `R` be a nonzero ring and `S` a nontrivial zerosumfree semiring, meaning

```
a plus_S b = 0  =>  a = b = 0.
```

There is no unital semiring homomorphism `R -> S`.  Indeed,

```
0 = phi(1 + (-1)) = 1_S plus_S phi(-1),
```

which would force `1_S = 0` by zerosumfreeness.

Thus signed additive routing cannot be exactly replaced by Boolean OR, max-plus,
or another zerosumfree/idempotent carrier.  The [Haar cancellation example](POLYPHASE_ROUTE_PACKETS.md) is an instance of
this algebraic obstruction.

By contrast, on a zerosumfree domain such as the nonnegative reals the support
map `x -> (x != 0)` does preserve `+` as Boolean OR and `*` as Boolean AND.
The obstruction is cancellation.

### Idempotence barrier

There is a dual obstruction.  If the source carrier has idempotent addition,
then `1 plus 1 = 1`.  A unital homomorphism into a nontrivial additively
cancellative carrier would force

```
1 + 1 = 1,
```

and cancellation gives `1 = 0`, contradicting nontriviality.  Thus Boolean or
max-like semantics cannot be pushed into ordinary additive semantics while also
recovering route multiplicity. This is the algebraic form of the
[duplicate-route counterexample](POLYPHASE_ROUTE_PACKETS.md).

The loss is not bounded by a fixed finite residual alphabet on unbounded inputs.
For raw packets containing `n >= 1` indistinguishable unit histories, Boolean
support is always `true` while additive multiplicity is `n`.  Any residual that
recovers exact multiplicity from Boolean support must therefore distinguish
arbitrarily many states.

## Signed cancellation packet

`core.carrier_change.SignedMassPacket` stores positive and negative mass
separately.  Its observation

```
value = positive - negative
```

preserves both addition and multiplication exactly under the packet operations.
The quantity

```
total_variation - abs(value) = 2 min(positive, negative)
```

is the exact unsigned-mass loss caused by cancellation.

This is a small exact completion for the scalar signed-routing query; it is not
claimed to be a universal minimal signed representation.

## Sum-product to max-plus

For scores `x_i` and temperature `tau > 0`, define

```
m = max_i x_i
r = sum_i exp((x_i - m) / tau).
```

Then

```
tau log(sum_i exp(x_i / tau)) = m + tau log r.
```

`TropicalResidualPacket` stores `(m, log r, term_count, tau)`.  Its tropical
projection keeps only `m`; its correction is `tau log r`.

The packet has exact compositional operations:

- merging alternatives maps `m` by `max` and reconstructs the finite-temperature
  sum exactly;
- multiplying positive expressions maps `m` by `+`, and the corrections add.

Moreover

```
0 <= correction <= tau log(term_count).
```

The upper bound is sharp: if all `n` scores tie at the maximum, the correction
is exactly `tau log n`.  Hence finite-temperature max-plus replacement has no
uniform error bound when fan-in is unbounded.  The quantity

```
effective_multiplicity = exp(correction / tau)
```

lies in `[1, n]` and records how much alternative-path mass the tropical
projection has discarded.

So the max-plus carrier is the zero-temperature shadow, while the residual is an
exact finite-temperature defect packet.

This is the standard Maslov-dequantization and log-sum-exp construction,
expressed as a compositional carrier-change contract.

## Query boundary of the residual completion

The tropical residual packet is exact for the paired scalar semantics
`(finite-temperature sum-product value, tropical max value)`.  It is not a
provenance packet.  For example, at `tau=1`, the score multisets

```
[0, 0, log(1/2)]
[0, log(3/4), log(3/4)]
```

have the same maximum `0`, the same partition sum `5/2`, and the same total
term count `3`, but the first has two maximizing branches and the second has
one.  Thus exact scalar carrier completion does not determine argmax
multiplicity or identity.  Those are separate future-query semantics and need
additional provenance state.
