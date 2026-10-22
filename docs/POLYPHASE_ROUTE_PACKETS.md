# Polyphase additive route packets

Perfect reconstruction in polyphase and filter-bank routing is additive and
depends on cancellation. Convex or maximum-based branch aggregation generally
changes that reconstruction. For the normalized Haar bank and input `(1, 0)`, the two branch contributions to
the first reconstructed sample are `1/2` and `1/2`, while those to the second
are `1/2` and `-1/2`. Sum semantics reconstruct `(1, 0)`; max/convex semantics
would produce `(1/2, 1/2)`.

A route packet therefore pairs its continuation state with a declared carrier
algebra. See [carrier-change obstructions](CARRIER_CHANGE_OBSTRUCTIONS.md) for
the conditions under which an aggregation law can be replaced.

For causal FIR polyphase routing the carrier is additive. If `E_h` is the
polyphase route packet of `h`, with

```
E_h[target, source][q] = h[N q + target - source],
```

then a cascade satisfies

```
E_(g*h)[target, source]
  = sum_middle E_g[target, middle] * E_h[middle, source],
```

where `*` on scalar entries is convolution in the coarse-lag variable.
Intermediate phase/tap histories can therefore be summed exactly into a finite
matrix of lag kernels. This is the standard polyphase matrix multiplication
law expressed as a route-packet contract.

`komorebi_math.signals.route_packets.PolyphaseRoutePacket` implements this
finite additive routing law.
