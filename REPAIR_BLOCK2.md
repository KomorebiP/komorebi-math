# R4 Block 2 — scoped exact arithmetic repair

Development checkpoint `1.0.0b4-repair-r4-block2`, wheel build `4b2`.
The package version and frozen 131-export stable surface remain unchanged.
The authoritative predecessor is the complete R4 Block 1 archive with SHA-256
`22ce2dac1fdc404b9460a7c1bacd372cc3f46525bd45984b5f31707b06341f9d`.

## Exact residual comparisons

The repaired transfer, nonlinear coordinates, holonomy, derivative symmetry,
packet jets, Faà di Bruno, graded operations and refinement comparison paths
use residual objects for the acceptance decision. Scalar, vector and matrix
values keep their exact entries until the decision. A tiny nonzero rational
cannot become equality just because its diagnostic floating norm underflows.
Unknown symbolic equality is rejected as unresolved, not proved nonzero.
The diagnostic infinity sentinel inherited from Block 1 means unavailable norm,
not an infinite mathematical residual. Simplification uses SymPy; symbolic
singular domains, assumptions and pre-rounded Float inputs are not certified.

Numerical callback comparisons remain numerical even on exact storage. A
successful finite comparison does not prove a user-supplied callback globally
correct, symmetric or multilinear. Faà di Bruno retains those premises after
matching a separately supplied provider. Jet reconstruction and support records
identify supplied mathematical assumptions rather than reporting multilinearity
as actually tested universally.

Packet-jet support now separates proved zero, witnessed nonzero and unresolved
exact probes. The existing `untested` field also holds addresses evaluated but
not classifiable. A later witnessed nonzero basis evaluation can still certify
nonzero support. Inspect `complete`, `exactness` and assumptions: valid=True on
a partial support record does not mean every address was classified.

## Prime fields

Finite cochain/packet cohomology and modular Gaussian elimination now store
Python integers in NumPy object arrays. Every product, sum and row operation is
exact before modular reduction, avoiding int64 overflow. At p=4294967311 and
q=2^32, q^2=225 mod p, not zero. This fixes false complex-identity acceptance,
false rejection of valid complexes/projectors, and an observed rank-one matrix
being assigned rank two. Returned field arrays deliberately use object dtype.
Fractional and nonfinite entries are rejected before field conversion.

The inherited primality check is trial division. No speedup, huge-prime input
qualification or formal time/space complexity is claimed. Already-returned
objects/certificates are not automatically invalidated after external mutation.

## Verification and reproduction

The outer checkpoint report and command receipts are the verification authority.
Run the full suite only with the matching rebuilt distributions present:

    python -B -m pytest tests -q -p no:cacheprovider

Focused tests:

    python -B -m pytest tests/test_repair_r4_block1.py tests/test_repair_r4_block2.py -q -p no:cacheprovider

Select the intended source or installed package path explicitly. The outer
checkpoint contains a bounded runner, source/installed replay, exact field
oracle, hashes, patch and immutable predecessor. Its receipts distinguish fresh
runs from retained history. No Lean, Windows, GPU or independent referee claim.

## Scope

Forty inherited static-comparison rows map to the rewritten and tested scope.
Together with five rows from Block 1, 175 of the 220 original candidates remain
unreviewed. These are location rows, not counts of bugs, theorems or unique
certificate producers. This block does not finish the exactness audit.
