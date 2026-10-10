# Installed mathematical stress framework

The `komorebi_math.audit` namespace runs reproducible finite mathematical
checks against the installed library. It requires the package's base NumPy
dependency, Python 3.11 or newer and no network. The built-in suites do not
depend on pytest, a source checkout, Lean, GPU libraries, SymPy, SciPy or mpmath.

This complements the repository's larger `verification/stress_release_candidate.py`
and `verification/deep_audit_regressions.py` campaigns. Those drivers audit
repository tests and release resources. This framework is shipped in the wheel
and accepts additional mathematical controls from an installed user program.

## Offline commands

```bash
python -m komorebi_math.audit --list
python -m komorebi_math.audit --samples 16 --seed 20261009 --output math-stress.json
python -m komorebi_math.audit --suite arithmetic --suite signals --samples 32
python -m komorebi_math.audit --coverage --output math-stress-with-coverage.json
python -m komorebi_math.audit --papers --coverage --output paper-machinery-stress.json
```

The default CLI isolates each check in a fresh interpreter using cross-platform
`multiprocessing` spawn. The default wall-time bounds are 30 seconds per check
and 180 seconds for execution of all requested checks. `--timeout` and
`--total-timeout` set those bounds. Interpreter startup counts against each
check's wall budget. Source hashing occurs outside the execution budget.

The process exits zero only if every requested check passed and its pinned
source inputs remained unchanged. A false mathematical decision, assertion
failure, exception, explicit skip, worker crash, timeout or unexecuted check
causes nonzero status. Invalid arguments and inability to write a requested
report also exit nonzero. Every executed CLI run prints its full JSON report
to stdout, and `--output` additionally writes that report to a file.

`--coverage` also checks the paper-to-machinery catalog's metadata and installed
API links, then attaches `paper_coverage` to the report. Invalid links or metadata
cause nonzero exit. A valid catalog can explicitly report partial coverage.
That records missing or deferred machinery and does not assert paper completion
or theorem validity. When coverage is requested, `stress_status` preserves the
execution result separately from the combined top-level status.

Python's optimized `-O` mode is rejected because it could erase assertions in
custom check functions. No CPU, memory, thread, filesystem or descendant-process
resource sandbox is claimed. Run only trusted mathematical check functions.

## Built-in finite assertions

`samples` must be an integer from 1 through 128. Changing it changes the
declared range and the input digest. All intervals below have inclusive integer
endpoints where their inputs are integer-valued. The report records actual
sample inputs and/or exact output digests as well as the seed and code hashes.

| Suite | Arithmetic | Independent comparison and fixed controls |
| --- | --- | --- |
| `core` | numerical | Random 2–6 dimensional packet block squares compared with direct scalar multiplication. Reject `2I` as a projector and as a realization of `T²=T`. |
| `arithmetic` | exact | 120-bit signed binomial data, order 1–5 recurrences through index 23, modular matrix powers with large moduli, rational Hurwitz products versus ordinary Taylor convolution. Also 350 k-bonacci values and 35 reciprocal values compared with direct rolling sums and rational inversion. |
| `special_functions` | exact | q-Racah values at degrees/sites 0–6 against a full terminating sum. The C11 rank-one table with a repeated lattice is retained as a counterexample to automatic positive orthogonality. Bernoulli degrees 0–4 against explicit polynomials. |
| `coordinates` | exact | Rational matrices of size 1–4 compared with the Leibniz determinant and scalar inverse identities. Singular inverse and consistent/inconsistent linear-system controls. |
| `finite_joins` | exact | Four-element chain and Boolean join tables. Every single relation and sampled double relations compared with all 15 set partitions, independently of the production union-find closure. |
| `signals` | numerical | Well-separated three-jump models from independently integrated Fourier data, plus close-edge, declared-rank and nonfinite-input controls. The close-edge fit is expected to pass while recovery approval is withheld and the pointwise error exceeds 0.69. |
| `quantum` | numerical | Four-spinor Dirac evolution compared with spectral decomposition. The two-state cosine memory trajectory fixes the coupling sign. Nonclosure at three constraint scales and large-time spectral dimension are checked. |

Numerical tolerances are stated in each check record. They describe finite
binary64/complex128 consistency, and are not rigorous enclosures of exact real
inputs. Some discrete witnesses in numerical suites happen to use exactly
representable entries. The suite's declared arithmetic remains numerical.

The true jump count in the resolved signal controls is part of the supplied
model. Its recovery approval is conditional on that model, not a claim that
the library authenticated the count from measurements.

These suites are a useful cross-domain baseline. They do not cover every API,
every paper claim, every parameter regime or the optional hardware backends.
Use the paper-machinery coverage inventory and the full repository campaigns
to identify further tests needed for a particular mathematical claim.

## Python use and custom exact oracle

`--papers` appends ten finite paper-machinery checks and the four added
special-function checks. These audit prime-field mapping cones, rational primary
projectors, exact compiler decisions, Gaussian/Hermite polynomial transfer,
polynomial lift fibers and rational stencils, tame tensor cycles, Collatz word
cylinders, exact predictive signal closure, factor-coded Gaussian gain and
outward dyadic verification, and fixed-precision prime-power Cartier sections.
Gaussian controls compare the Joseph law with an independent Schur complement,
and Cartier controls compare arbitrary states with rational-series inversion
including noncanonical mask lifts. The appended special-function
controls cover coupled rational inverses, exact tail anchors, sieve detectors
and compact resolution bounds. Each record states its finite ranges and any
external analytic assumptions. No completion of the source papers follows.

Rational projectors and predictive signal foundations require the `exact`
SymPy extra. If it is absent, those checks are explicitly skipped and the run
exits nonzero. The base seven suites keep their NumPy-only dependency contract.
Python callers can append `paper_checks(samples=...)` and
`special_function_checks(samples=...)` explicitly. The complete installed
worked example is `examples/paper_machinery_stress.py`.

```python
from komorebi_math.audit import CheckOutcome, StressCheck, builtin_checks, run_stress
from komorebi_math.certificates import Exactness
from komorebi_math.sequences.transforms import binomial_transform_values

def constant_binomial(context):
    cases = []
    for n in range(1, context.parameters["maximum_count"] + 1):
        c = context.rng.randrange(-2**128, 2**128)
        actual = binomial_transform_values((c,) * n)
        expected = tuple(c * 2**k for k in range(n))
        if actual != expected:
            return CheckOutcome(False, {"count": n, "constant": c})
        cases.append({"count": n, "constant": c})
    return CheckOutcome(True, {"cases": cases})

if __name__ == "__main__":
    custom = StressCheck(
        "my.constant-binomial-v1", "custom", constant_binomial, Exactness.EXACT,
        {"counts": [1, 16], "coefficient_bits": 128},
        parameters={"maximum_count": 16},
    )
    report = run_stress(
        (*builtin_checks(samples=8), custom), seed=20261009,
        timeout_seconds=30, total_timeout_seconds=180,
    )
    print(report.counts)
    print(report.replay_digest)
    raise SystemExit(0 if report.passed else 1)
```

See `examples/math_stress_framework.py` for a complete executable version.
Additional domains can export tuples of `StressCheck` objects and append them
to `builtin_checks()` without replacing existing controls. Identifiers must be
unique, because they define stable independent random streams.

For interactive functions defined in a notebook or local closure, use
`run_stress(checks, seed=...)` without timeouts. This runs in process and cannot
terminate a hanging callback. Hard timeout execution requires a picklable,
module-level function from an importable module and an entry-point `__main__`
guard on Windows and other spawn platforms. An unpickleable function is recorded
as an error rather than silently falling back to unbounded execution. A total
wall-time budget requires isolated execution.

## Outcome and evidence contracts

A check returns `CheckOutcome(passed, evidence, detail)` or a Python Boolean.
`None`, truthy objects and NumPy Boolean scalars are errors. The framework does
not guess whether a mathematical object is a successful result. A custom check
can raise `SkipCheck(reason)` when its required oracle is unavailable, but that
run cannot qualify as passed.

Evidence supports owned JSON data, Python integers, exact `Fraction` values and
finite floating or complex values. Fractions use `{"rational": [p, q]}` and
complex numbers use `{"complex": [real, imag]}`. Arbitrary objects are rejected
instead of being stringified into apparently meaningful evidence. Evidence in
an exact check cannot contain floating-point values. A numerical check must
declare a positive finite tolerance, and an exact check cannot declare one.

The framework cannot prove that a caller's function actually used the declared
arithmetic or implemented a correct oracle. Those are review obligations. All
declared assumptions, finite ranges, parameters, generated seeds and function
identities remain visible in the report.

`CheckOutcome`, `StressCheck` and `StressReport` snapshot their retained data.
Reading parameters, ranges, evidence or report dictionaries returns fresh
copies. Later changes to caller mappings cannot promote or alter an old result.

## Provenance and interpretation

Each report records Python/dependency versions, source locations, SHA256 hashes
before and after execution for installed package Python/JSON files and custom
check source files, per-check declarations and input digests, actual statuses,
durations, effective wall budgets, aggregate counts and explicit non-claims.
Custom data files and third-party oracle code are not discovered automatically.
Record their contents or hashes in the check parameters and extend provenance
for any claim that depends on them.

`replay_digest` hashes the scientific records, source content and versions while
excluding wall clocks and local filesystem paths. Replays with the same code,
versions, seeds and finite assertions should match it. Numerical reproducibility
can still differ across dependency builds and platforms, which are not fully
identified by package version strings alone.

Report status `passed` means every named finite assertion executed and satisfied
its check contract. It does not establish a universal theorem, revalidate Lean
proofs, authenticate source hypotheses or grant a live package `Certificate`.
Reports are historical data and remain so when loaded from JSON.
