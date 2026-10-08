# Exact families and local OEIS comparisons

The `komorebi_math.sequences` domain now includes reusable parts of the
Komorebi OEIS 1.0.0 release recovered on 2026-10-08. These additions work offline
and require no new dependency. They integrate exact family generation and local
sequence comparison without importing the standalone atlas, CLI, or publications.

## Existing machinery retained

| Recovered operation | Existing package implementation |
| --- | --- |
| Bernoulli numbers and signed Euler coefficients | `arithmetic.euler`, `sequences.catalog` |
| Fibonacci and Lucas recurrences and companion jumps | `arithmetic.recurrence` |
| Generic Hurwitz products and inversion | `arithmetic.hurwitz` |
| Ordinary and normalized formal series, reciprocals, powers | `generating.series`, `generating.finite_series` |
| Catalan, Bell, partitions, Apéry, and automatic sequences | `sequences.catalog`, `generating.sequences` |
| Residue channels, refinement, and polyphase reconstruction | `refinement.residue`, `refinement.polyphase` |

The archive's companion convention orders states from newest to oldest. The
existing `LinearRecurrence` orders them from oldest to newest and uses
`x[n+d] = sum(c[i] * x[n+i])`. Recovered convenience functions use the existing
convention. The archive's matrix routines were not duplicated.

## New exact families

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

## Source and scope

The software source is `Komorebi_OEIS_v1.0.0`, released 2026-10-07. Its
`SOURCE_PROVENANCE.json` identifies the recovered
`Transfer_Calculus_OEIS_Polished_Release.zip` baseline with SHA-256
`43a6b736d53944459d9b50c7c2b43052a8d043f7b14c36a4b1d973803ae0429a`.
The recovered archive records proprietary research-draft software status. This
integration leaves the package license unchanged and introduces no software
license grant.

The archive's reference corpus records 44 anchors and a historical OEIS export
commit `29a947c1ea590c34717d08bc4fc8083093d433ae`, retrieved 2026-07-10. Those are
source metadata, not a claim that this commit is current. No OEIS corpus or
atlas is bundled in this integration. Users can load a local snapshot explicitly.
Loaded JSON provenance is retained on `LocalOEISCorpus.provenance`. The archive
attributes OEIS data to the OEIS Foundation under CC BY-SA 4.0, separate from
software licensing.
