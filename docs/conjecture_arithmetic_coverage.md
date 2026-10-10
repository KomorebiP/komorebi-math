# Conjecture arithmetic coverage

This coverage-only audit uses the current **abc**, **Beal** and **Ramanujan tau nonvanishing Lehmer** reader PDFs. The JSON inventory records source hashes, section references, finite runtime interfaces, test paths and mathematical boundaries. No new arithmetic API or conjecture-solving feature was added.

| Source | Current edition | Pages | Runtime coverage and remaining arithmetic obligation |
| --- | --- | ---: | --- |
| abc | Reader 1.0, 8 October 2026 | 71 | Generic valuation, polyphase, polynomial lift, CRT, lattice, cokernel and exact-matrix primitives exist. The abc prime differential, cross-prime regulator, mixed polylog filter and depleted determinant constructors are absent. The corrected nonzero integral witness and all-place height bridge remain open. |
| Beal | Reader 1.0, 8 October 2026 | 48 | Supplied polynomial systems, lattices and rational/symbolic cokernels are supported. The source-specific genus-two arithmetic and F49 fifteen-row evaluator are external. No authenticated arithmetic localization target is supplied for the eight patterns, and later integral/global obligations remain. |
| Lehmer (tau) | Reader 1.0, 9 October 2026 | 103 | The exact Euler-product framework computes finite Delta coefficients; supplied recurrences, polynomials and finite transfer matrices are supported. Complete five-core shells, primitive content, harmonic null families and the source CRT/NTT engine are absent or external. The actual-Delta nonvanishing obligation at primes congruent to 4 modulo 5 remains open. |

The canonical machine-readable rows are in `verification/paper_coverage/conjecture_arithmetic.json`. Generic infrastructure is marked **partial** when the paper needs an arithmetic assignment or complete enumeration that the package does not supply. **External** means a source-specific computation, formal layer or arithmetic theorem is outside the installed runtime; it does not assert a fresh execution. **Theory only** records a source mathematical or conditional argument, including explicit unresolved obligations.

## Normalizations that affect correctness

For abc, write `R=rad(abc)` and `Q=abc/R`. The corrected target has `Q/(ab)=c/R`, and the conditional witness bound includes `(ab)^s_r`. The current source explicitly does not establish its global witness and comparison hypotheses. Its rational truncation negative control fails the required output contact; finite local determinants cannot replace this obligation.

For Beal, homogeneous rank and image calculations cannot determine the affine input. The current finite evaluator works with encoded F49 data and a pattern-specific fifteen-row map, whereas the generic exact backend uses its rational/symbolic domain. The actual arithmetic target `t(p0)` and its obstruction `N t(p0)` remain unauthenticated; the evaluator's arithmetic-authentication flag is false.

For Lehmer, `euler_product_function` uses the convention `product (1-z^m)^(-c_m)`. Use the constant exponent **-24** to obtain `Delta/q`, restore the leading q, and interpret coefficient index `j` as `tau(j+1)`. Local trace-zero compatibility and a synthetic countermodel do not identify an actual Delta Frobenius or an actual zero of tau. This paper addresses tau nonvanishing rather than Mahler measure.

## Validation boundary

Validation passed with both repository and current paper-source roots: all three PDF hashes matched, all 26 unique APIs resolved (43 row references), and all 14 unique referenced test files exist. This checks metadata and links, not theorems or execution of every listed suite. The focused existing Euler-product partition test passed (one test). A separate exact replay of tau(1) through tau(12) matched the known integer values and an independently multiplied finite binomial product. No wider tau range was tested in this pass.

The audit did not execute the papers' bespoke programs, build their formal projects, certify universal shell enumeration or authenticate missing arithmetic inputs.

Inventory: 42 mechanism rows; external=14, implemented=1, missing=8, partial=13, theory_only=6.

- `Komorebi_ABC_Conjecture_Piece_v1.0.pdf`: SHA-256 `48c3320567c4351774fbed1cf02daddbdee6acb1a43f2c90001106171085a3aa`.

- `Komorebi_Beal_Conjecture_Piece_v1.0.pdf`: SHA-256 `301b67ed8b6a05c767a537cb28c4387c5595453c7500e6d9d6424c34a9b5bde3`.

- `Komorebi_Lehmer_Conjecture_Piece_v1.0.pdf`: SHA-256 `30f377a14324b596e09eab9402bc771ff29c84aef2f0181d97506386b986d14f`.
