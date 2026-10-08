# Deeper mathematical audit and source integration

This audit continues the GitHub readiness repairs with independent finite
controls, adversarial inputs and a comparison against newer Komorebi sources.
The starting pull-request revision is
`a362d4722bbabcd9486ccc039b77dd413338e3cf`; the default-branch baseline is
`50447a253c2d00d4a7a4108440e090844b7148b2`. The package remains research beta
`1.0.0b4` under its existing license. The 131-symbol stable namespace and
historical API manifests remain frozen.

## Machinery brought into the Python library

| Source family | Reusable addition | Supported boundary |
| --- | --- | --- |
| Quadratic Form K-COS | [Directed tails and stored-factor spectrum bridge](quadratic_forms.md) | Positive central Gaussian quadratic forms; rank and planner restrictions, outward decimal bounds and qualified arithmetic trust base |
| OEIS 1.0.0 | [Exact coefficient families and local matching](sequences_oeis.md) | Factorial normalization and explicit offsets; finite matches against user-supplied local corpora |
| Corrected Special Functions I–VIII | [Gamma/Meixner shape transforms, Bernoulli–Hurwitz refinement and scalar Racah functions](special_functions.md) | Exact rational functions and ordinary numerical evaluations distinguished; scalar termination does not imply positive orthogonality |
| Signal Reconstruction, 2026-10-06 | [Fourier/Hankel jump recovery](fourier_recovery.md) | Finite real periodic jump model, consecutive high modes, declared background bandlimit and numerical conditioning assumptions |
| Stratified Universality Block 26 | [Finite join congruence and quotient](finite_join_semilattice.md) | Supplied finite join table, exact replayable closure derivation and quotient homomorphism |
| Physics finite-mode source | [Clifford symbols and closed Dirac propagation](dirac_modes.md) | Four-spinor fixed mode and central sine momentum; explicit units, zero-energy and continuum limits |

Existing Bernoulli/Euler, Hurwitz, recurrence, generating-function, coordinate,
polyphase and packet APIs are reused. Standalone plotters, manuscript builders,
historical CLI bundles and duplicated matrix conventions are not runtime
dependencies. No online OEIS fetch or bundled third-party corpus was added.

H12's cubic/conductor-specific evaluator and divisor checks remain research
replay source. Their fixed arithmetic construction and open comparison do not
provide a general elliptic/gamma API with a validated error contract. Stratified
subgroup, Čech, extension and matroid census scripts also retain their bounded
research role. The generic finite join closure is the extracted primitive.
The Physics release contributes its finite Clifford/Dirac mode computation;
publication and Lean runners, growth/cutoff claims and continuum completion
remain outside that runtime API.
Polyphase Float remains a separate C/GMP runtime with its own license and no
Python binding. See [companion C integration](COMPANION_C_INTEGRATION.md).

## Reproduced defects and repairs

| Counterexample or failure | Resulting behavior |
| --- | --- |
| Integer analysis/synthesis products wrapped to identity; infinite residual and scale passed `inf <= inf`; malformed groups and arbitrary blocks passed | Promote integer arithmetic, use finite scaled checks, verify partitions and every declared operator block |
| `ceil((2**53+1)/1)` lost one in floating arithmetic | Semisimple seed ranks use strict integers and exact ceiling division |
| Caller or witness mutation invalidated cached Smith and lattice evidence | Immutable snapshots and fresh checks of witness identity, kernels, rationality, invertibility and packet conditions |
| A true live certificate contained NaN/infinite residuals or tolerance; its caller changed its residual mapping | Nonfinite evidence cannot retain true validity; residuals are owned and read-only; persisted evidence still requires revalidation |
| A two-state Schrödinger reduction produced `cosh(t)` | Two `-i` couplings produce the derivative kernel and independently recover `cos(t)` |
| `exp(t)` at Laplace parameter zero received a convergence certificate; a finite resonant integral failed | Separate rational continuation from the convergence half-plane; augmented exponentials evaluate finite integrals at resonance |
| Spectral dimension underflowed and constraint closure changed under harmless generator scaling | Shift heat weights and decide closure in an orthonormal resolved span; expose numerical rank limits |
| Mutable/nonfinite Lindblad inputs and overflowing density normalization produced misleading results | Snapshot matrices, recheck finite data and reject nonfinite normalization |
| Underdetermined filter banks had positive lower frame bounds; integer Haar/jump/route arithmetic overflowed; complex stream state was discarded | Include the full domain's zero singular values, promote integer arithmetic and preserve complex state across chunks |
| Window, spectral, STFT and recovery edge cases accepted invalid parameters or nonfinite data | Validate finite domains and degenerate branches; retain explicit numerical certificate limits |
| A low-precision interval display collapsed before the K-COS accuracy decision | Require authoritative outward literal bounds at that API boundary; generators and precision contexts are preserved |
| A special-function shortcut replaced tiny nonzero complex orders by zero | Resolve the removable pole using increased isolated precision, with explicit resource limits |

These regressions are supplemented by independent exact convolution/Hurwitz
controls, finite chi-square tail references, direct trajectories and integrals,
Fourier synthesis controls, quadrature, polynomial addition and exhaustive
finite partition comparisons. Every numerical result retains its declared
tolerance, model and arithmetic assumptions. Finite computation does not prove
an unrestricted mathematical theorem or certify the external numerical stack.

## Reproduction and evidence

From an authorized source checkout, install the CPU test dependencies:

```bash
python -m pip install -e ".[test,exact,scientific,yaml]" "setuptools==82.0.1" wheel mkdocs
python verification/deep_audit_regressions.py --root . --revision YOUR_COMMIT
python verification/build_block21_distributions.py --root . --output-directory release/dist --report release/LOCAL_DISTRIBUTION_BUILD.json
python -m pytest -q tests
python -m mkdocs build --strict
python verification/block21_publication_gate.py --full-tests --require-mkdocs --output release/LOCAL_PUBLICATION_GATE.json
```

The audit driver records all pinned source/test input hashes before and after
execution, software versions, the executed finite range, seeded randomness,
timeout, outputs and residual risks in a version-two computation manifest.
Its `repository.dirty` flag records that an input snapshot, rather than an
unverified clean checkout assumption, supports the run. The existing installed
artifact profile includes all new mathematical regression files, so the clean
wheel and the wheel rebuilt from the source distribution both execute them.

The CPU CI continues to cover Linux and Windows on Python 3.11–3.13, the minimum
NumPy 1.26.4 / SymPy 1.14.0 / SciPy 1.11.4 line, strict documentation and clean
installed artifacts. Results apply only to their named revision and input
snapshot. No fresh GPU, JAX, PyTorch, LaTeX or Lean qualification is inferred.

The original archive identities and SHA-256 hashes are recorded in
`komorebi_math/data/research_source_provenance_20261008.json`. Detailed source
restrictions and correction conventions are in the linked domain documents.
