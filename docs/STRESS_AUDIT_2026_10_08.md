# Release-candidate stress audit

This pass starts from the manual release candidate
`d63263251c43d11379997e7c102a5e13a29e1687`. All 1,170 recovered Git blobs
were verified before testing. The package remains `1.0.0b4` under its existing
license. Frozen public signatures and aggregate manifests are retained.

## Confirmed failures and repairs

| Area | Reproduced failure | Resulting behavior |
| --- | --- | --- |
| Exact recurrence and matrix arithmetic | A six-step recurrence returned 205128748323 instead of zero modulo `10**12 + 39`; NumPy integer inputs overflowed inside object arrays | Native arbitrary-precision integer and rational intake; modular matrix products remain object arithmetic |
| Exact transforms and coordinates | NumPy integer binomial/Hurwitz operations overflowed; invalid moduli entered arithmetic; a three-element residue ordering enumerated `10**30` empty lanes | Exact input normalization, early modulus validation, strict integral axes including NumPy integers, and finite-prefix lane enumeration |
| Live certificates | Mutable scalar arrays were accepted as evidence and stayed approved after being changed to NaN | Evidence must be supported immutable finite scalars |
| Portable evidence | An integer `2**54 + 1` rounded on load; finite JSON became infinite or truncated when cast to a declared array dtype | Exact integer/rational preservation and checked dtype conversion |
| Portable writes and contracts | Concurrent exclusive saves both succeeded and overwrote each other; version fields accepted coercible nonintegers | Atomic exclusive publication and strict integer contract versions |
| Automatic differentiation | NaN symmetry norms became zero; `log(-1)` had invalid primal output but finite derivatives; derivative orders were coerced before validation | Finite input, primal and residual checks; order validation before conversion; consistent finite tolerance handling |
| Gamma shape functions | Equal shapes `1e100` produced a degree-one transfer value of one rather than `1/sqrt(2)` | Isolated guard precision, requested-precision results, and explicit bounded-work rejection |
| Finite quantum numerics | Caller mutation changed channel data; extreme central phases changed under a sinc argument roundtrip; spectral overflow produced misleading finite answers | Owned read-only Kraus snapshots, finite output checks, represented-phase evaluation, and explicit range/dimension rejection |
| Directed matrix-tail provenance | An exhausted result reported a precision cap that was never attempted | Report the actual last attempted precision |

The directed controllers also improve supported failure behavior. An automatic
query whose exact domain is valid can return `certified=False` when its planning
or interval margins remain unresolved within the supplied precision budget.
Conservative comparison spectra are refined only up to their candidate cap.
Direct invalid-domain and unresolved-rank exceptions remain explicit. The audit
found no unsafe directed tail certificates in the tested range.

## Independent finite controls

| Campaign | Recorded range and independent check |
| --- | --- |
| Finite quotients | 4,126 candidate operation tables, 88 semilattices, 4,941 relation sets checked against a separate partition-intersection oracle, and 4,853 corrupted witness rejections |
| Exact arithmetic | 864 independent modular matrix comparisons, 4,800 recurrence/packet comparisons, 4,097-bit data, a 1,001-digit index, and 1,200 indexed local OEIS windows |
| Directed tails | 30 independently directed Erlang controls over coefficient scales `1e-300` through `1e300`, 12 exact dyadic diagonal controls, 60 dense numerical spectrum comparisons, and 48 concurrent public calls |
| Numerical machinery | 3,486 scenarios with 5,700 checks covering Fourier synthesis, independent Dirac matrices/eigensystems, channel adjoints, closed damping solutions, diagonal spectra, Bernoulli polynomials, and direct overlap quadrature |
| Optional CPU autodiff | JAX 0.11.2 and PyTorch 2.14.1+cpu, 337 analytic/domain/dtype controls per backend through derivative order four, plus the four existing isolated smoke tests |

The 216 ill-conditioned or noisy Fourier probes produced 54 passing finite
consistency certificates and 162 failed certificates. Those 54 passes do not
establish recovery of an unrestricted unknown target. The numerical model,
bandlimit, rank threshold and conditioning assumptions still apply. Large Dirac
phases are checked at their represented floating-point values.

No defect was found in the finite semilattice algorithm or the well-separated
Fourier recovery controls. Ordinary special-function and Fourier evaluations
remain numerical results. Directed enclosures retain their documented arithmetic
trust base. Historical theorem dependencies were not re-proved.

## Replay and resource evidence

From the repository checkout, using the scientific, exact and test dependencies:

```bash
python verification/stress_release_candidate.py --revision YOUR_COMMIT
```

The default replay runs the five backend-independent campaigns and eleven
resource examples. It starts separate processes, enforces a per-campaign wall
timeout and a total wall-time budget, rejects skips in requested campaigns,
and records input hashes before and after execution. A changed input, missing
result, timeout or failed control prevents a successful report. The manifest
records the actual command, software versions, finite ranges and output hashes.

The resource examples validate join tables of size 4 through 64, stored-factor
spectra of size 4 through 16, and positive tails of ranks 4, 8 and 16. A properly
reported tail-budget exhaustion is recorded alongside independently checked
intervals. Timings concern those inputs and the measured machine. Tracemalloc
reports Python allocations, not a bound on native allocations or total RSS.
Native thread environment limits are cooperative. No operating-system memory
or CPU-affinity cap is claimed.

To replay the optional CPU cases after installing the qualified dependency line:

```bash
python verification/stress_release_candidate.py --campaign optional_backends --skip-resources --revision YOUR_COMMIT
```

The optional-backends CI job installs the named CPU versions and requires all
six selected integration cases to execute without skips. Linux CPU qualification
does not designate GPU, Windows optional-runtime, or minimum optional-version
qualification. The ordinary Linux/Windows source matrix continues to cover the
backend-independent repairs. The clean installed-artifact profile includes the
five new backend-independent test files.

Shape-function guard work is capped at 10,000 decimal digits, and excessive
window-exponent work raises `ResourceLimitError`. The tail-library caps still
bound individual attempts; the replay's total wall timeout is an external test
budget. Exclusive file publication uses Windows no-overwrite rename and POSIX
hard links. A POSIX target filesystem must support hard links for that operation.
This does not assert remote-filesystem or power-loss durability.

## Audit incidents

The initial resource control assumed ascending spectrum order and was corrected
to the documented implementation order. Its first rank-four tail exhausted the
configured caps correctly; the control now records that outcome and checks its
interval against the independent Erlang formula. A numerical evidence harness
initially tried to parse an mpmath complex display string; direct complex
conversion corrected it before the complete sweep was rerun.

One source-file fetch briefly disconnected and succeeded on retry with its Git
blob verified. The execution service also briefly disconnected and recovered.
An interim artifact check rejected a distribution after source changes, as
intended. Final qualification uses rebuilt artifacts from the frozen input
snapshot. These incidents are distinct from the reproduced library defects.
