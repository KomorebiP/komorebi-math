# Komorebi Math R4 Block 12 — support packets and labeled branch transport

Checkpoint: `1.0.0b4-repair-r4-block12`.

## Scope

This research block tests whether the facet information exposed in Block 11 is
an independent geometric packet or a compiled view of a smaller semantic
object.

## Added

- research-tier finite `PolyhedralSupportPacket` in
  `komorebi_math.nonlinear.convex_packets`;
- exact support pullback under linear images,
  `h_{A C}(u) = h_C(A^T u)`;
- exact branch/Bellman envelope
  `B_A h(x) = max_i h(A_i x)` represented by
  `conv_i A_i^T C`;
- active-atom and branch-provenance queries;
- compositional branch words under repeated Bellman transport;
- deterministic 2,000-case support/Bellman probe;
- adversarial examples separating value semantics from argmax provenance.

## Main correction

For an exact convex body, normal-fan and exposed-face geometry is derivable
from its support function.  It is therefore not an independent semantic packet.
For a finite polytope, extreme atoms are the irredundant value representation,
and removing nonextreme atoms is safe under every future linear/Bellman
transport.

Branch provenance is different.  Two branch systems can produce the same
support function while having different maximizing action sets.  If the task
asks only for the value, the convex hull is sufficient.  If it asks which
branch attained the maximum, labels must survive compaction.

The resulting distinction is:

1. support/convex-body packet — value semantics;
2. normal fan — derivable execution index;
3. labeled support packet — policy/provenance semantics.
