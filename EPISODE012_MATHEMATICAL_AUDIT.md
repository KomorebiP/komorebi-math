# Episode 012 Mathematical Audit — Hankel-Rank Falsification

## Research decision

Episode 012 tests whether the finite continuation observer introduced in Block 13
should be generalized as Komorebi's default provenance-compression primitive.
The answer is **no** for field-valued semantics.

The correct classical distinction is:

1. **set-valued/discrete semantics:** quotient histories by equality of all
   future observations;
2. **field-linear semantics:** quotient the span of future residual functions,
   whose dimension is the Hankel rank.

A field-linear realization may therefore have finite dimension even when the
exact discrete future quotient has infinitely many classes.

## 1. Classical weighted-series theorem

For a scalar series `f : Sigma* -> K`, define

`H_f[u,v] = f(uv)`.

For each prefix `u`, its residual is the function

`f_u(v) = f(uv)`.

Over a field, the vector space spanned by the residuals is the natural linear
continuation state.  Classical weighted-automata theory identifies its dimension
with the rank of the infinite Hankel matrix and with the dimension of a minimal
linear weighted automaton.

This theorem is not new Komorebi mathematics.  Episode 012 uses it as an
external comparison standard.

## 2. Exact polyphase route example

Take the existing Block 14 route packet for the FIR filter

`h=(1,1)`

with polyphase order `N`.  Evaluate its coarse-lag polynomial at a positive
scalar `z`.  Directly from

`E_h[r,s][q] = h[Nq+r-s]`

the evaluated matrix is

`M_N = I + S_z`,

where

`S_z e_j = e_(j+1)` for `j<N-1`,

`S_z e_(N-1) = z e_0`,

hence

`S_z^N = z I`.

Consider the scalar route query

`f_N(n) = e_0^T M_N^n e_0`.

### Theorem 2.1 — exact Hankel rank

For every `N>=1` and `z != 0`, the `N x N` Hankel block

`H[i,j] = f_N(i+j)`, `0<=i,j<N`,

is nonsingular.  Therefore the Hankel rank of the scalar series is at least `N`;
since `M_N` is an `N`-dimensional realization, its rank is exactly `N`.

### Proof

Factor

`H = O C`,

where row `i` of `O` is `e_0^T M_N^i` and column `j` of `C` is
`M_N^j e_0`.

For `j<N`,

`M_N^j e_0 = sum_(k=0)^j binom(j,k) S_z^k e_0`

and `S_z^k e_0=e_k`.  Relative to the standard basis, the columns of `C`
therefore differ from `(e_0,...,e_(N-1))` by a unit triangular binomial matrix,
so `det(C)=1`.

Likewise

`e_0^T M_N^i = sum_(k=0)^i binom(i,k) e_0^T S_z^k`.

For `1<=k<N`,

`e_0^T S_z^k = z e_(N-k)^T`.

The row basis `(e_0^T, e_0^T S_z, ..., e_0^T S_z^(N-1))` therefore has
determinant `+/- z^(N-1)`.  The binomial change of basis is unit triangular,
so

`det(H)=det(O)det(C)=+/- z^(N-1) != 0`.

QED.

This result is a direct route-specific proof; the interpretation of the rank as
minimal weighted-automaton dimension is the classical external theorem.

## 3. Infinite discrete quotient versus finite linear quotient

The same example produces an arbitrarily large separation from a deterministic
future-equivalence observer.

Using `(I+S_z)^n` and `S_z^N=zI`,

`f_N(n) = sum_(j=0)^floor(n/N) binom(n,Nj) z^j`.

For positive `z`, `f_N(n)=1` for `n<N`, while for `n>=N-1`

`f_N(n+1)-f_N(n) = sum_(j>=1) binom(n,Nj-1) z^j > 0`.

Thus the sequence is strictly increasing from index `N-1` onward.

Let the residual after prefix length `n` be

`r_n(k)=f_N(n+k)`.

If both prefixes are at least `N-1`, their residuals differ already at `k=0`.
If `n<m<N`, choose `k=N-m`: then `r_m(k)=f_N(N)=1+z` while
`r_n(k)=1`.  The mixed cases are distinguished similarly.

Hence

`r_n != r_m` whenever `n != m`.

So the exact discrete future quotient has **infinitely many classes**, while the
minimal field-linear continuation state has dimension exactly `N`.

For `N=2,z=2`,

`M=[[1,2],[1,1]]`

and

`f(n)=1,1,3,7,17,41,...`.

Since `(M-I)^2=2I`,

`M^2-2M-I=0`,

so

`f(n+2)=2f(n+1)+f(n)`.

The existing Komorebi recurrence subsystem represents this with

`LinearRecurrence((1,2),(1,1))`.

Thus the low-dimensional realization was already present in the package before
Episode 012.

## 4. What the finite computation showed

The deterministic exact probe used seed `20260923`.

* For route orders `2` through `10`, the two-tap family at `z=2` had exact
  sampled Hankel rank equal to the phase dimension in every case.
* For every order, 65 sampled prefix residuals were already pairwise distinct on
  a 65-suffix window.
* The order-two route sequence matched the existing recurrence object for 80
  exact terms.
* In 120 random nondegenerate two-symbol route systems, every sampled Hankel
  block had rank equal to the route phase dimension.  This is finite evidence,
  not a generic theorem.
* A deliberately degenerate six-phase identity route had Hankel rank one,
  demonstrating ordinary reachable/observable reduction when syntactic phase
  slots carry no independent query information.

## 5. Belief revision

Episode 009's language of a universal "continuation observer" was too narrow.
The continuation representation depends on the algebra of observations.

A more accurate table is:

| Query semantics | Natural continuation compression |
|---|---|
| Boolean/set-valued | future-equivalence classes / Myhill--Nerode |
| field-linear | residual vector space / Hankel rank |
| linear recurrence | companion/Krylov state |
| polyphase linear routing | phase-space polynomial matrix state |

The last three are already mutually familiar through classical linear systems,
rational series, and weighted automata.

## 6. Research conclusion

Hankel minimization is a useful **audit** for Komorebi numerical/provenance
systems, but it does not justify another core abstraction.

The large separation between discrete state count and linear dimension is real,
but classical.  In the generic route family tested here the Block 14 phase state
was already Hankel-minimal, while the recurrence subsystem already encodes the
unary low-rank state.

Therefore Episode 012 recommends **compressing the packet theory rather than
expanding it**.

The next high-value inquiry should leave this trajectory and use the accumulated
typing rules as an adversarial audit tool on an independent mature Komorebi
subsystem.
