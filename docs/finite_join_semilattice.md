# Finite join congruence and quotient

The API lives in `komorebi_math.core.semilattice`. It constructs finite join
semilattices, computes generated congruences, and supplies independently
replayable quotient certificates.

```python
from komorebi_math.core.semilattice import FiniteJoinSemilattice

# Subsets of a two-element set, labeled by bit masks.
source = FiniteJoinSemilattice(
    tuple(tuple(left | right for right in range(4)) for left in range(4))
)
result = source.generated_congruence(((0, 1),))
result.certificate().require()
assert result.classes == ((0, 1), (2, 3))
assert result.projection == (0, 0, 1, 1)
assert result.quotient.join_table == ((0, 1), (1, 1))
```

`FiniteJoinSemilattice` requires a nonempty square table whose entries are strict
integers in the source label range. It checks idempotence, commutativity and
associativity. Its storage is an immutable tuple snapshot. Python and NumPy
integer labels are accepted. Booleans, floating labels, truncation, ragged tables
and invalid entries are rejected.

`generated_congruence(relations)` returns the least equivalence relation closed
under joining with every source element and containing the requested pairs.
Each successful merge records a finite derivation: an input pair, or a join
context of an earlier merge. `SemilatticeQuotient.certificate()` independently
replays those derivations and checks the quotient homomorphism and requested
relations. Every derived merge belongs to any congruence containing the input
relations. Conversely, the verified output is itself such a congruence. Together
these finite checks establish leastness for this supplied finite table.

`quotient(partition)` validates a supplied equivalence partition, allowing
arbitrary integer class labels, and constructs its quotient. It does not claim
that a manually supplied partition is least for any set of relations.

The output contains canonical class minima in `partition`, consecutive quotient
labels in `projection`, original members in `classes`, and the validated quotient
table in `quotient`. The certificate has exact status because table validation,
union-find and witness replay use Python integer comparisons. Residual floats are
only the zero or one display of that exact decision.

For source size `n`, table storage is `O(n^2)` and axiom validation is `O(n^3)`.
With `m` supplied relations, congruence closure performs at most `n-1` successful
merges and queues `O(m+n^2)` pairs. Union-find operations have near constant
amortized cost. Constructing and validating the quotient retains a cubic worst
case table-validation cost. This is a finite-table API, suitable when that cost
and the explicit table storage are acceptable.

The regression suite compares all pairs of relations on two different size-four
semilattices against an independent enumeration of every equivalence partition.
It also checks relation order independence, join homomorphisms, a size-eight
Boolean context closure, forged witnesses, integer validation and caller
mutation. These computations do not certify a universal matroid colimit,
nonadditive packet reconstruction theorem or doctrine closure statement.
