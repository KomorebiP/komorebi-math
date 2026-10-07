# Thirteenth major block report

## Scope

Block 13 turns the normalized generating-function paper into an executable finite subsystem. The implementation preserves the paper's separation between normalization, coefficient source, local mask, carry state, Cartier state, and prime-power arithmetic. It does not encode the manuscript as static formulas. Each layer is represented by reusable data types, exact algorithms, validation certificates, resource contracts, tests, and installed verification paths.

## Architecture

### Normalization layer

`WeightedMonomialBasis` implements \(e_n=z^n/\gamma_n\). Closed descriptors make ordinary, exponential, factorial-power, `q`-factorial, Fibonacci-factorial, Pochhammer, and finite-table normalizations portable. Callback bases remain available for finite live computation.

`FiniteNormalizedSeries` adds an explicit coefficient domain. Its multiplication uses the exact structure constants \(\gamma_n/(\gamma_k\gamma_{n-k})\). Same-basis divided-power multiplication remains valid in positive characteristic without attempting to invert a vanishing factorial. Transfers to a different basis require both normalization ratios in the selected domain.

### Source layer

`CustomFunctionSpec` provides one finite series interface for explicit tables, linear recurrences, rational functions, hypergeometric ratios, implicit algebraic branches, Euler products, compositions, and factorial-power Bell families. `FunctionRegistry` prevents silent replacement. `FrozenGeneratingFunction` captures a provider-independent finite record.

### Mask and carry layer

The local-mask transform, inverse, functional equation, analytic approximation, Frobenius defect, and coboundary solvers are implemented directly. Finite masks compile into carry matrices. Ordinary products of digit-local sequences compile into addition-carry matrices rather than being mislabeled as one-state digitwise products.

### Cartier layer

Rational masks over finite fields compile into denominator-stabilized Cartier matrices. The package provides both a fixed-dimensional linear representation and a reachable output automaton with exact partition-refinement minimization. Finite-mask carry matrices are certified as the polynomial-mask specialization of the Cartier transition.

### Congruence layer

The Bell subsystem implements the weighted factorial-power recurrence, pointed-subindex identity, digit-sum reductions, parity automaton, period bound, and residue automata. General congruence tools include Mahler coefficients, valuation profiles, residue lanes, modular recurrences, Lucas audits, Gaussian Lucas audits, and finite relation ledgers.

### Prime-power layer

Kummer matrices count borrow layers exactly. Prime-power unit compilers reconstruct binomial residues, and normalized-kernel compilers add factorial powers and unit twists. The Gaussian `q`-Kummer decomposition covers every prime, including the separate two-adic odd-`q` branch. Euler and cyclotomic orbit tools test finite collapse without symbolic expansion when exponent data suffices.

### Persistence and signal bridge

The portable codecs are closed data-only schemas built on the versioned package persistence envelope. The signal adapter maps a finite formal transfer polynomial to FIR taps using an explicit convention and round-trip certificate.

## Validation program

The source suite combines the inherited package tests with the original exact generating-function tests, positive-characteristic domain tests, production persistence and CLI tests, and Block 13 integration tests. The installed verifier exercises the complete public compiler chain with lightweight deterministic examples.

The independent deep audit uses separate direct products, direct convolutions, integer binomial coefficients, borrow counting, exact Gaussian-binomial recurrences, and rational-mask recurrences. It records 383,336 exact comparisons with zero failures in the current source checkpoint.

The final release process additionally validates the installed wheel in full and NumPy-only environments, checks every public export, compiles every source file, audits docstrings and annotations, rebuilds from the source distribution, verifies byte reproducibility, tests archive CRC and manifests, and restores the tagged commit from the Git bundle.

### Source checkpoint

| Validation layer | Result |
|---|---:|
| Complete source suite | 841 of 841 passed |
| Block 13 generating and CLI tests | 193 of 193 passed |
| Aggregate installed verifier | 146 of 146 passed |
| Generating verifier | 14 of 14 passed |
| Independent arithmetic audit | 383,336 of 383,336 passed |
| Public research API | 901 symbols with zero strict differences |
| Curated beta API | 83 symbols with zero strict differences |
| Production Python files | 233 |
| Imported package modules | 232 of 232 |
| Generating namespace exports | 181 |
| Missing audited public docstrings | 0 |
| Missing audited function annotations | 0 |

The complete source suite was executed in bounded independent batches because the release environment terminates sufficiently long single commands. Only batches that reached a normal passing exit were counted. The batch ledger is committed as `verification/BLOCK13_SOURCE_TEST_BATCHES.json`.

## Boundaries

The package implements finite formal and modular computation. Analytic mask evaluation is numerical and carries a conservative tail estimate. Closed-form discovery is bounded. Arbitrary callback semantics cannot be authenticated across processes. Prime-power Cartier lifting for every rational mask is not represented as a single universal minimized automaton. Global analytic continuation and infinite-state claims remain outside the finite runtime unless separately supplied and certified.
