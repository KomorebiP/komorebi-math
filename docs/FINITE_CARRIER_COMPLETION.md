# Finite trace separation

`komorebi_math.deformation.finite_frobenius_separator` works with the supplied
finite trace table. If its selected trace matrix has full column rank, the
constituent multiplicity vector is uniquely recoverable from those rows.
Rank-deficient data does not separate every declared constituent.

```python
from komorebi_math.deformation import finite_frobenius_separator

separator = finite_frobenius_separator({"place_a": (1, 0), "place_b": (0, 1)})
separator.certificate.require()
assert separator.recover((2, 3)) == (2, 3)
```

## Certificate contract

The Boolean `separating`, certificate validity, and exact column-rank condition
agree:

\[
\text{valid}\iff\text{separating}\iff\operatorname{rank}(A)=r,
\]

where `A` is the supplied finite trace matrix and `r` is the constituent count.
A rank-deficient result has an invalid certificate. Both
`certificate.require()` and `recover(...)` reject that result.

The certificate has `Exactness.EXACT` because rank and recovery use exact
finite algebra. Its checked assumptions concern the supplied finite table and
exact column rank.

## Arithmetic scope

Finite trace separation does not establish a global Galois or automorphic
identification. The unchecked assumption
`Chebotarev density beyond the supplied table` is recorded explicitly.
Any global conclusion requires the relevant external arithmetic hypotheses.
