# Exact finite Lehmer shells

`komorebi_math.arithmetic.lehmer_shells` implements finite constructions from
the Lehmer reader edition 1.0 (9 October 2026), Chapters 6, 7 and 13. It uses
Python integers throughout, with no numerical tolerance or optional dependency.

For a positive integer index $n$, the complete shell consists of the five-tuples
$v$ with $v_i\equiv i\pmod 5$, $\sum_i v_i=0$ and $\sum_i v_i^2=10n$.
Every signed weight is $W(v)=\prod_{i<j}(v_i-v_j)/288$.
The corresponding five-core partition has size $n-1$.

```python
from komorebi_math.arithmetic.lehmer_shells import (
    enumerate_five_core_shell, tau_from_q_product,
    primitive_residuals_from_tau, tau_from_primitive_residuals,
)

shell = enumerate_five_core_shell(121)
assert shell.content_masses == {1: -25439104668, 11: 25937424601}
assert shell.weighted_sum == 498319933
tau = tau_from_q_product(128)  # Entries include tau(0)=0.
primitive = primitive_residuals_from_tau(tau)
assert primitive[121] == shell.primitive_residual
assert tau_from_primitive_residuals(primitive) == tau
```

The quadratic enumerator chooses three coordinates, solves the remaining sum
and squared norm by an integer discriminant, and checks parity and residues.
The `four_coordinate` method independently chooses four coordinates and fixes
the fifth from the zero-sum condition. Candidate, packet and integer-size caps
are checked before coordinate ranges are allocated. A resource refusal raises
`ResourceLimitError`; it never returns a truncated shell.
Direct `FiveCoreShell` construction requires a finite tuple or list, checks its
exact expected cardinality before copying, and bounds index arithmetic.

A packet's positive coordinate gcd $g$ is its content. The source residue
transport is $u_j=v_{gj\bmod5}/g$. `packet.primitive()` performs this transport;
`primitive.scale_content(g)` reverses it for positive $g$ coprime to five.
The weight factor is $\chi_5(g)g^{10}$, including its sign.
The finite primitive transform is
$R(n)=\sum_{d^2\mid n}\mu(d)\chi_5(d)d^{10}\tau(n/d^2)$.
Its inverse omits the Möbius factor. Both accept supplied coefficient lists and
retain exact zeros. Neither interprets arbitrary input as a modular form.

`auxiliary_shell_polynomial(shell)` builds the degree-at-most-ten polynomial
$P_n(T)=\sum_v\prod_{i<j}[(i-j)+T(c_i-c_j)]$, with $c_i=(v_i-i)/5$.
It returns increasing-degree coefficients, the exact 5-adic content and the
normalized integral polynomial $G_n(U)=5^{-\mu_n}P_n(5U)$.
It satisfies $P_n(5)=288\tau(n)$ and $P_n(0)=288|V_n|$.
The variable $T$ is auxiliary; it does not interpolate the shell index.

Independent tests compare complete shells through 32, signed tau coefficients
and primitive residuals through 128, and binomial-basis determinants. They also
reproduce the source's polynomial coefficients at 499 and the opposed content
masses at 121. At the separate cancellation regime 2411, every packet weight
is a 2411-adic unit while the signed total is divisible by 2411.

`prime_unit_certificate()` requires an index prime greater than 20, exactly the
source hypothesis. Its termwise conclusion does not imply that the sum is
nonzero. These APIs do not prove universal null-polynomial conservation,
modularity, global Galois properties or Lehmer's conjecture. At a prime the
primitive residual remains the original tau coefficient.
