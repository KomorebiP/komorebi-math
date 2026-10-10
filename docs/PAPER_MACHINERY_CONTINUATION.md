# Finite mathematical machinery continuation

This continuation adds five mathematical modules to the paper machinery
draft. Source sections and exact supported domains are recorded in the
installed crosswalk. The existing beta identity and 131-symbol root API remain
unchanged. Coverage belongs to the recorded editions and source hashes.

| Module | Mathematical construction |
| --- | --- |
| [AVOC decisions](avoc_decisions.md) | Finite fiber decoders, rational channels, guarded Bellman policies and frozen quote bounds |
| [Boolean synthesis](avoc_boolean.md) | Eager three-input trees, cost-preserving compaction and a checked minimum bank for all 256 signatures |
| [Tikhonov certificates](tikhonov_certificates.md) | Rational positive normal inverses, source/image bounds, discrepancy brackets and actual residual certificates |
| [Residual modules](residual_modules.md) | Reachable ring spans, actual nonfree observable quotients and complete noncommuting-word comparisons |
| [Five-core shells](lehmer_shells.md) | Complete signed arithmetic shells, gcd sectors, Möbius transforms and auxiliary polynomials |

## Worked example

Run `examples/paper_machinery_continuation.py` with the installed exact extra.
It produces these exact finite results:

| Construction | Result |
| --- | --- |
| De Morgan Boolean tree | Six eager nodes become three; the complete depth-one search costs exactly 6,890 modeled units |
| Perfect two-state channel, then one-observation garbling | Optimal guessing value falls from 1 to 1/2 |
| Cartier source Example 24.4 | Ambient cardinality 16, reachable cardinality 8, residual cardinality 2; the quotient is nonfree over Z/4Z |
| Rank-deficient discrepancy model | Squared attainable floor 16 and target 18; the unique parameter lies in [913/1024, 457/512] |
| Shell at index 121 | 133 signed packets sum to 498,319,933; content masses are -25,439,104,668 and +25,937,424,601 |

The discrepancy interval includes an irrational parameter; its endpoints are
rational and their residual inequalities are checked exactly. The shell
retains cancellation. Termwise prime-unit weights do not imply a nonzero or
prime-unit total.

## Independent checks

The installed `--papers` campaign now executes 26 checks. Five new controls
enumerate channel and guarded policies, compute Boolean minimum costs by node
count, compare diagonal inverse problems with scalar formulas, build additive
ring spans and future-word classes, and multiply the defining q-product
independently of the shell construction.

```bash
python -m komorebi_math.audit --papers --coverage --samples 16 \
    --seed 20261010 --timeout 30 --total-timeout 180 --output mathematics.json
python examples/paper_machinery_continuation.py
```

The separate `verification/check_paper_continuation_review.py` records a fresh
review with source hashes, versions, seed and bounded regimes. Tests deliberately
alter optima, banks, module factors and signed totals to verify that the
independent installed controls reject them.

Review found two constructor defects: nonreflexive AVOC labels could be declared
sufficient but fail decoding, and a direct shell constructor could consume an
unbounded iterator. Both now reject those inputs. Exact scalar review also found
fixed-width integer components retained inside Fraction inputs, including a
Hermite inner product incorrectly reduced to zero. Accepted exact scalars now
own Python integer components before arithmetic or bit checks; large-input
regressions cover the repair.

The previous GitHub run exposed Windows packaging assumptions, a stale manual
profile, a numeric-count/Boolean-certificate mismatch and a worker teardown
race. Corrections retain the original check deadlines and require all 26 named
checks in both installed artifacts. The observed failures and local corrections
are retained in `verification/PAPER_MACHINERY_CI_CORRECTIONS.json`.

## Mathematical boundary

The crosswalk now has 169 implemented finite mechanisms, 181 partial and 22
missing constructions. The other categories remain 65 theory only, 70 external
and 21 separate reference. The catalog contains 55 recorded source objects and
705 distinct API references. Local source-byte validation matches the recorded
hashes, and all 705 references resolve in the installed package.

The catalog still reports incomplete coverage. Exact real selectors, genuine
noise/model allowances, arbitrary abstract modules, infinite-family limits,
formal Gaussian operation graphs, further quantum/arithmetic constructors and
conjecture obligations are not established by these finite runs. The initial
21-check qualification is historical evidence for its named input snapshot;
the continuation has its own execution records.

## Executed qualification

The final source regression passed 3,805 tests, with seven explicit optional
JAX/PyTorch skips and six warnings. The bounded source and isolated-wheel
campaigns each passed all 26 checks at 16 samples per check, with the same
scientific replay digest. The source campaign pinned 432 execution inputs and
verified that their bytes stayed unchanged.

An offline isolated installation gate separately qualified the direct wheel
and the wheel rebuilt from the source distribution. Each passed 1,906 runtime
tests across 97 selected files, with no skips or failures, and all 26
mathematical checks at one sample per check. The rebuilt wheel was byte-identical
to the direct wheel. These are Linux CPython results; fresh Windows, GPU and
optional-backend qualification remain outside this evidence.

Strict documentation builds, 467 mathematical expressions across 36 pages,
64 selected manual examples and the five-construction worked example passed.
The run summaries and exact reports are retained in
`verification/PAPER_MACHINERY_CONTINUATION_QUALIFICATION.json`,
`verification/PAPER_MACHINERY_CONTINUATION_PYTEST.txt` and
`verification/paper_runs/continuation_installed/artifact_gate.json`.
The final source archive additionally includes these execution records; its
support-file parity is checked after that rebuild. No tagged release or merge
is declared by these results.
