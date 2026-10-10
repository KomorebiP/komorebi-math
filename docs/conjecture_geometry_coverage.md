# Conjecture readers: finite machinery and remaining constructions

The current Strong Artin, Hodge, Catalan constant and Inverse Galois reader
PDFs have 65 construction-to-runtime entries in
`verification/paper_coverage/conjecture_geometry.json`.
The inventory records 3 implemented finite mechanisms, 23 partial mechanisms,
29 external constructions or research obligations and 10 theory-only statements.
Each entry gives the current source section, exact API and test-file references,
and the part those references do not establish. An implemented entry covers its
stated finite mechanism, not an entire theorem or conjecture.

| Current source | Edition | Pages | SHA-256 |
|---|---|---:|---|
| `Komorebi_Strong_Artin_Conjecture_Piece_v1.0.pdf` | 1.0, 8 Oct 2026 | 58 | `a232cdc1aee848f50cf3f65763ff33b0cd2a008250c685ac04d8cda1e61ecc0f` |
| `Komorebi_Hodge_Conjecture_Piece_v1.0.pdf` | 1.0, 9 Oct 2026 | 126 | `607f0a25fc27ab36379147292c8ac803076498bdde3d766329ee814a0b4eab87` |
| `Komorebi_Catalan_Constant_Conjecture_Piece_v1.0.pdf` | 1.0, 9 Oct 2026 | 70 | `217cfccee04ac533a22f9db75146b47e3c3a162df5f0af9be0f4ebf1c029d057` |
| `Komorebi_Inverse_Galois_Conjecture_Piece_v1.0.pdf` | 1.0, 9 Oct 2026 | 45 | `cb1ec0db4cd4a733ff9de795a2de1de72b7983eba73c54ee91ce4fc8bac93706` |

| Paper | Executable finite layers | Remaining source mechanisms |
|---|---|---|
| Strong Artin | Explicit groups and numerical induced representations; declared prime-cyclic Clifford towers; rational finite trace separators; supplied prime-field cohomology and polynomial digit fibres; conditional completion reports | Exact type multiplicity spectra and cyclic alignment, cyclotomic coefficient selectors, actual residual/pseudocharacter realization, automorphic fibres, positive character-cone search, ring-kernel cocycle constructors, moving syzygies and integral two-sided matrix-power inverse certificates |
| Hodge | Exact rational matrix ranks, frame presentations and maps; supplied rational primary operators; finite numerical range/obstruction splitting | Fermat unit-orbit/conductor/certificate grammar constructors, actual Chow correspondences and cycle classes, weighted carriers, signed transfers, global capacity certificates, five-phase surfaces and Weil representative geometry |
| Catalan constant | The existing exact holonomic engine runs the reader's recurrence with supplied coefficients; exact prefix transforms and scalar valuations provide ingredients | A named constant-specific packet/clearing/content scan API, universal integral/analytic source contracts and the missing asymptotic content-growth margin |
| Inverse Galois | Explicit finite groups, subgroups and induced permutation actions; supplied numerical group averages and finite cochain systems | Actual rational number fields, splitting-field and automorphism certificates, invariant-field rationality, regular specialization, arithmetic proper embedding-problem solutions and field multiplication tables |

Strong Artin §27 distinguishes H2 lift obstruction from H1 conjugacy gluing and
requires real ring, action and compatibility hypotheses. Prime-field matrix
cohomology and finite digit lifting implement useful restricted layers; they
do not produce those objects automatically. The matrix-power contraction also
needs a certified **two-sided** derivative inverse and the specified
nonarchimedean norm estimates. Current numerical correction frameworks do not
provide that certificate. A condition-labelled completion report preserves
external assumptions rather than proving them.

For Hodge, balanced character tuples identify a Hodge type. They do not produce
a rational algebraic cycle. The exact frame layer needs actual cycle classes
and the correct target subspace as input. The positive-frame argument requires
containment; otherwise the applicable finite criterion is
`rank([C H]) == rank(C)`. The current reader's degree-33 correction, degree-70
carrier, degree-210 signed transfer, five-phase replacement and Weil-class
failed-route corrections remain explicit source-specific constructions. A
failed grammar, matrix method or representative cannot certify nonalgebraicity.
The reader also records a Lean build blocked before elaboration by missing
compiler/Mathlib; this audit claims no fresh Lean pass or general Hodge proof.

For Inverse Galois, a faithful group representation does not construct a field.
Moreover, faithfulness as a group representation is weaker than an injective
group-algebra representation. The subgroup-average equivalences in §4 require
the regular/group-algebra setting; a zero projector in another representation
can hide nonnormality. An actual realization certificate must retain field
multiplication and prove the rational polynomial's irreducibility, splitting
and distinct full-group automorphisms.

## Worked exact Catalan-constant recurrence

The combinatorial Catalan sequence and the Apéry recurrence for ζ(3) are
different objects. Catalan constant Chapter 3 uses

\[
a_n w_{n+1}-q(n)w_n-c_n w_{n-1}=0,\qquad n\geq1,
\]

with `u[0:2] = (1, 7/4)` and `v[0:2] = (0, 13/8)`. The library convention is
`sum(P[j](k) * w[k+j]) == 0`. Thus the adapter must use source index `n=k+1`
and the ascending coefficient lists for `(-c(k+1), -q(k+1), a(k+1))`.
Applying the source polynomial at `n=0` would incorrectly impose an extra
equation on `v`.

```python
from fractions import Fraction as F
from math import comb, gcd, lcm
from komorebi_math.sequences.holonomic import HolonomicRecurrence

polynomials = (
    (-260, -1848, -5188, -7344, -5536, -2112, -320),
    (-10699, -56088, -120196, -134592, -83024, -26752, -3520),
    (1872, 8976, 17428, 17520, 9632, 2752, 320),
)
u = HolonomicRecurrence(polynomials, (1, F(7, 4)), name="Catalan constant u")
v = HolonomicRecurrence(polynomials, (0, F(13, 8)), name="Catalan constant v")

def generalized_binomial(x, j):
    result = F(1)
    for t in range(j):
        result *= F(x - t, t + 1)
    return result

def p(n):
    return 20*n*n - 8*n + 1

# Independent Chapter 5 half-binomial formula, rather than rechecking the
# recurrence using its own forward equation.
for n in range(25):
    direct_u = sum((
        F(comb(n, j))
        * generalized_binomial(F(2*n - 1, 2), j)
        * generalized_binomial(F(2*n + 2*j - 1, 2), j)
        for j in range(n + 1)
    ), F(0))
    assert u.term(n) == direct_u
    assert (16**n * u.term(n)).denominator == 1
    clearing = 4 * 16**n * lcm(*range(1, 2*n + 1))**2
    assert (clearing * v.term(n)).denominator == 1
    if n < 24:
        wronskian = u.term(n+1)*v.term(n) - u.term(n)*v.term(n+1)
        expected = F((-1)**(n+1)*p(n+1), 2*(2*n+1)**2*(2*n+2)**2)
        assert wronskian == expected

# Exact packet/content checks on 63 supplied finite addresses.
for n in range(9):
    for m in range(7):
        U = sum((F(comb(m, i))*16**(n+i)*u.term(n+i)
                 for i in range(m+1)), F(0))
        V = sum((F(comb(m, i))*16**(n+i)*v.term(n+i)
                 for i in range(m+1)), F(0))
        assert U.denominator == 1
        N, D = V.numerator, V.denominator
        Q = 8 * lcm(*range(1, 2*(n+m)+1))**2
        assert Q % D == 0
        content = gcd(int(Q*U), int(Q*V))
        h = gcd(D*int(U), N)
        assert content == (Q//D)*h
        assert F(int(Q*V)//content, int(Q*U)//content) == V/U

print(tuple(int(16**n*u.term(n)) for n in range(6)))
# (1, 28, 2596, 311536, 41759524, 5953698928)
```

This example passed with exact rational arithmetic. It checks 25 independent
half-binomial rows, 25 integrality/denominator rows, 24 Wronskians and 63 packet
content identities. It does not prove a universal denominator bound, identify
an analytic limit, establish a limiting growth rate or prove irrationality.
Chapter 7's residue warning also matters: a zero residue modulo `2**K` gives
only a valuation **lower bound**. Theorem 9.2 still requires the strict
asymptotic content-growth margin that the reader leaves open.

## Validation scope

The following representative checkout suites passed **97 tests in 1.33 s**:
finite-image groups/Clifford data, conditional Langlands completion, supplied
cohomology, filtered and numerical corrections, exact primary projectors and
frame maps, generic exact sequences and numerical symmetry projectors. The
exact command is stored in the JSON fragment. All four source hashes, every
named API and every cited repository test-file path were checked separately.

These checks do not replay the external conjecture research bundles, rebuild
Lean or certify the source geometry, arithmetic realization or universal
claims. Existing tests are references for the finite APIs; they are not
represented as specialized proof tests for every source construction.
