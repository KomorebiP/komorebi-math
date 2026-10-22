# Exact families and local OEIS comparisons

The `komorebi_math.sequences` domain provides exact family generation and
comparison with explicit local sequence corpora. These operations work offline
and need no additional dependency.

## Related APIs

| Operation | Package implementation |
| --- | --- |
| Bernoulli numbers and signed Euler coefficients | `arithmetic.euler`, `sequences.catalog` |
| Fibonacci and Lucas recurrences and companion jumps | `arithmetic.recurrence` |
| Generic Hurwitz products and inversion | `arithmetic.hurwitz` |
| Ordinary and normalized formal series, reciprocals, powers | `generating.series`, `generating.finite_series` |
| Catalan, Bell, partitions, Apéry, and automatic sequences | `sequences.catalog`, `generating.sequences` |
| Residue channels, refinement, and polyphase reconstruction | `refinement.residue`, `refinement.polyphase` |

`LinearRecurrence` orders companion states from oldest to newest and uses
`x[n+d] = sum(c[i] * x[n+i])`. The convenience functions in this namespace use
the same convention.

## Exact families

```python
from komorebi_math.sequences import (
    reciprocal_coefficients, reciprocal_sequence,
    universal_denominator_coefficients,
    kbonacci_sequence, kbonacci_closed_form,
    fubini_sequence, derangement_sequence,
)

assert reciprocal_coefficients(2, 6) == (1, 1, 5, 61, 1385, 50521)
assert kbonacci_sequence(3).first(7) == (0, 0, 1, 1, 2, 4, 7)
assert kbonacci_closed_form(3, 6) == 7
assert fubini_sequence().first(5) == (1, 1, 3, 13, 75)
assert derangement_sequence().first(6) == (1, 0, 1, 2, 9, 44)
```

For spacing `N` and nonnegative integer power `r`, the normalization is

\[
D_N(t)=\sum_{q\geq0}\frac{(-1)^q t^{Nq}}{(Nq)!},\qquad
D_N(t)^{-r}=\sum_{n\geq0}\frac{A_N^{[r]}(n)t^{Nn}}{(Nn)!}.
\]

`reciprocal_coefficients(N, count, power=r)` returns exactly `count` compressed
coefficients starting at `n=0`. Power zero gives `(1,0,0,...)`.
`universal_denominator_coefficients(N, degree)` returns dense EGF coefficients
through the inclusive ordinary degree, including the zero residue lanes. It does
not return ordinary Taylor coefficients. The compressed inversion recurrence
uses Python integers throughout. The existing dense Hurwitz implementation
provides an independent normalization control.

The k-bonacci family has `k-1` initial zeros and then one. Its finite binomial
formula is tested against the recurrence and modular companion execution.
Fubini numbers use ordered set partitions. Derangements start at `D_0=1`.
All integer sizes, offsets, and terms reject booleans and fractional coercions.

## Explicit local corpora

```python
from komorebi_math.sequences import (
    OEISRecord, compare_oeis_prefix, find_local_oeis_matches, load_local_oeis,
)

reference = OEISRecord("A000045", (1, 1, 2, 3, 5, 8), offset=1)
match = compare_oeis_prefix((2, 3, 5), reference, query_offset=3, minimum_terms=3)
assert match.passed
assert match.index_alignment_verified

# An explicit file can be internal-format text, stripped text, or archive JSON.
# .gz files are supported. No download occurs.
# corpus = load_local_oeis("dated_snapshot.json", format="json")
# matches = find_local_oeis_matches(query_terms, corpus, query_offset=0)
```

`parse_oeis_internal` reads `%S/%T/%U` fields, preferring signed `%V/%W/%X`
fields when supplied. Continuations are assembled in field order. `%N` provides
the name and `%O` supplies the starting index. Missing offsets remain `None`.
Lines that do not match the internal field syntax are ignored, including lines
with malformed identifiers. Inspect the returned records before comparing them.
`parse_oeis_stripped` reads the terms and optional names. The stripped format
contains no index offset, so the loader preserves an unknown offset.

Without `query_offset`, matching compares the first stored positions and does
not claim verified index alignment. With a declared query offset, the matcher
aligns actual indices. A reference with an unknown offset cannot validate an
explicit indexed query. A query beginning before the available reference is
reported as incomplete rather than silently dropping its leading terms.

`agrees_on_overlap` means every available compared term agrees and the minimum
length is met. `complete_query_match`, and its `passed` alias, additionally mean
that the entire supplied query was compared. If a reference ends early, the
status is `reference-exhausted-prefix` and `passed` is false. Candidate search
can retain that incomplete evidence, but excludes any known mismatch even if
an earlier prefix met the minimum. Every result concerns a finite window.
Neither a match nor a failed local search establishes infinite identity,
originality, priority, or absence from the full OEIS.

## Local data and licensing

No OEIS corpus or atlas is bundled with the library. Users load a local
snapshot explicitly, and JSON provenance is retained on
`LocalOEISCorpus.provenance`. The original software is licensed under MIT.
OEIS data are attributed to the OEIS Foundation under CC BY-SA 4.0, separate
from the software license. A software grant does not change the terms of
separately supplied data.
