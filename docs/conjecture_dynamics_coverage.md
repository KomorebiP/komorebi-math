# Collatz and Riemann machinery coverage

This is a coverage audit of the Collatz and Riemann reader editions 1.1. It adds no mathematical API, formal build, or conjecture proof. The machine-readable inventory is `verification/paper_coverage/conjecture_dynamics.json`; its 37 mechanisms record source sections, available APIs, tests, and boundaries. Mechanism counts are not completion percentages.

Both complete extracted source texts were inspected against the current source and tests. This checks what the library supplies, rather than independently verifying every proof or historical reference in the papers.

| Source PDF | SHA-256 |
| --- | --- |
| `Komorebi_Collatz_Conjecture_Piece_v1.1.pdf` | `5ac6e27d5f5179ed0b3b82191de392cc8680cfa7597bf3c4580fb6f069951048` |
| `Komorebi_Riemann_Conjecture_Piece_v1.1.pdf` | `d486d21a24a336a1596e6a4f9a028a4ed6f7e14f3390ea0b64452553ff0e0495` |

`implemented` means the bounded finite mechanism has a named implementation and relevant tests. `partial` records reusable pieces with an explicit missing mechanism or domain restriction. `missing` means the audited mechanism has no named implementation. `theory_only` covers analytic implications, infinite statements, and open conditions. `external` identifies dependencies outside this package. `separate_reference` identifies historical evidence that the current implementation does not reproduce.

## Collatz

The new `komorebi_math.arithmetic.collatz` module supplies exact arithmetic for a supplied finite valuation word. If its length is \(k\), depth is \(S\), and affine offset is \(C\), it uses

\[
2^S x_k=3^k n+C,
\qquad x_k<n\iff (2^S-3^k)n>C.
\]

The exact realization cylinder is modulo \(2^{S+1}\), including the terminal parity bit. An affine quotient that happens to be integral is insufficient: for example, seed 3 does not realize the word `(1, 3)`, whose affine quotient is even. Coefficient contraction alone also does not imply descent below the original source; the offset is checked explicitly.

| Source mechanism | Status | Package coverage and boundary |
| --- | --- | --- |
| Positive odd acceleration and valuation | implemented | `odd_collatz_step` uses exact positive integer arithmetic. |
| Three step clocks | partial | Word length/depth give \(k,S,S+k\); separate shortcut/classical orbit APIs are absent. |
| Affine word coordinates | implemented | `CollatzWord`, affine value, and exact orbit replay, within construction caps. |
| Exact realization cylinder | implemented | `realizes`, `apply`, and finite realization certificate. |
| Cylinder densities | theory_only | No infinite partition or probabilistic orbit law is certified. |
| Word composition | implemented | `concatenate` gives the composed offset and cylinder. |
| Offset-sensitive descent | implemented | Strict descent, threshold, and first descending member for a fixed word. |
| Bounded first descent | implemented | `first_odd_descent` reports success or explicit inconclusive exhaustion. |
| Source-27 coordinates and Appendix A | partial | Generic exact replay is available; the complete table and all clocks are not shipped. |
| Growing-run exit mechanism | partial | Generic words/search are available; no symbolic run-family builder or fixed-horizon theorem API. |
| Delayed terminating family | partial | Individual words can be replayed; the named family constructor and divisibility induction are absent. |
| Uniform run lifts | partial | A tested fixed progression gives an affine lift; arbitrary symbolic families are absent. |
| Coarse residue atlas | separate_reference | The historical 1804/2048 atlas is not reproduced; a final-valuation bound differs from an exact cylinder. |
| Noncrossing prefix coordinates | partial | Finite slope/depth/offset exist; no permanent-branch certificate. |
| Bounded-deficit packing | theory_only | The theorem is about bounded-deficit tails, not exclusion of every noncrossing branch. |
| Merger and cycle constraints | partial | Supplied word coordinates can be evaluated; no merger or cycle solver. |
| Universal first descent | theory_only | The remaining universal condition is open in the source and unproved here. |
| Historical formal closure | external | No fresh Lean build or imported proof artifact. |

The source's bounded-deficit result must keep its quantifiers: unbounded deficit on every tail does not assert convergence of the deficit to infinity, and does not rule out an entire noncrossing branch. Likewise, a successful certificate for one word or seed cannot establish universal first descent. Step-budget exhaustion is inconclusive; bit/depth cap violations are explicit resource errors.

The relevant repository tests are `tests/test_collatz_words.py`. Independent reviewer controls also compared 86,614 bounded word/seed cases and checked first descent for 1,024 positive odd seeds below 2048. Those are finite controls, not conjecture evidence for all integers.

## Riemann

The package has exact generic Möbius and finite transform machinery, plus finite Hermite polynomials. The source's actual zero heat packet, complete prime-power explicit-formula packet, coherent compression, and stable Báez-Duarte coefficient mechanism are not named implementations. Their continuum estimates remain boundaries of this inventory.

| Source mechanism | Status | Package coverage and boundary |
| --- | --- | --- |
| Completed zeta and its actual zeros | external | Classical analytic input; no certified zero multiset or all-zero tail enclosure. |
| Zero heat packet | missing | No named evaluator with multiplicities and controlled infinite tails. |
| Heat growth criteria | theory_only | All-scale equivalences are not established by finite numerical samples. |
| Prime-power heat packet | missing | No combined von Mangoldt, pole, and digamma implementation with the source constants. |
| Pole annihilator | missing | No named application of \(1/4-\partial_u^2\) to the analytic packet. |
| Local heat energy | theory_only | The continuum integral and its uniform bounds are not certified. |
| Finite Hermite polynomials | partial | Physicists/probabilists polynomial APIs exist; the source's physical Gaussian Hilbert space is not supplied by those alone. |
| Coherent Poisson compression | missing | No named source-normalized prime-power coherent packet/compression API. |
| Ground channel front | theory_only | Its asymptotics are not inferred from a finite Hermite truncation. |
| Integer Möbius function | implemented | `mobius_integer` has exact finite tests. |
| Báez-Duarte coefficients | missing | No named stable evaluator for \(c_K\), its signed infinite sum, or its tail. |
| Prime-cube grouping | partial | Divisor/subset zeta transforms and Möbius values exist; source-weighted cube regrouping is not directly implemented. |
| Squarefree coprime support | partial | Möbius gives finite indicators; uniform interval counting is absent. |
| Absolute cube barrier | theory_only | Its all-scale lower bound and dimension condition are not computational certificates. |
| High-pass reconstruction | external | Historical formal support is not an ordinary DSP filter. |
| Neutral diagonal modes | theory_only | No analytic cancellation theorem follows from finite diagonal computations. |
| Hardy and fourth-moment consequences | theory_only | The source analytic implications have no package proof artifacts. |
| Signed closure estimate | theory_only | The source's remaining cancellation condition is open, with all-scale constants required. |
| Historical sources | separate_reference | Reference results/formal ledgers are not rebuilt here. |

For prime-cube regrouping, the source weight is \(w_K(n)=n^{-2}(1-n^{-2})^K\). A finite sum split as \(n=bm\) must preserve its cutoff \(bm\leq N\); replacing it with independent unrestricted factor sums changes the computation. A lower bound for an absolute majorant is not a lower bound for the signed sum. The squarefree factor \(\mu^2\), coprimality restriction, and the source condition \(d=o(\log M)\) must be retained.

The classical polynomial tests are `tests/test_coordinates_general_classical.py` and `tests/test_paper_middle_machinery.py`; Möbius/transform tests are `tests/test_coordinates_general_combinatorial.py`. The physical \(L^2(du)\) coherent Gaussian in the Riemann source differs from the standard Gaussian probability measure in `dynamics.hermite`. Also, `analytic.positive_zero_string` and `analytic.reduced_zero_zeta` concern other special-function zeros; they do not provide the actual Riemann zero packet.

## Validation boundary

`komorebi_math.audit.coverage.validate_coverage_catalog` can verify this inventory against the repository and paper-source roots, including PDF byte hashes, API resolution, and test-path existence. Validation does not execute those tests, verify their mathematical oracles, rebuild formal dependencies, or prove any listed analytic implication. An accepted inventory is an auditable crosswalk, not a mathematical qualification certificate.
