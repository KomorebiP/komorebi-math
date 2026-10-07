# R4 Block 14 — Polyphase Additive Route Packets

Block 14 is a cross-domain falsification of the Block 13 provenance-packet idea.
It asks whether observer-conditioned convex packets can be carried unchanged into
polyphase/filter-bank routing.

They cannot. Perfect reconstruction is additive and depends on cancellation.
For the normalized Haar bank and input `(1, 0)`, the two branch contributions to
the first reconstructed sample are `1/2` and `1/2`, while those to the second
are `1/2` and `-1/2`. Sum semantics reconstruct `(1, 0)`; max/convex semantics
would produce `(1/2, 1/2)`.

The reusable layer is therefore not a fixed convex carrier. It is a continuation
state together with a declared carrier algebra.

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
matrix of lag kernels. This is the standard polyphase matrix law expressed in
the packet/provenance vocabulary; it is not claimed as new mathematics.

The research-tier `signals.route_packets.PolyphaseRoutePacket` implements this
law without widening the frozen aggregate API.
