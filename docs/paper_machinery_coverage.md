# Mathematical machinery from the papers

This crosswalk compares the current paper-source snapshot with the Python
library. It identifies mathematical constructions, their APIs, independent
controls, and the precise domain supported by each implementation. It is not
a proof audit or a declaration that the entire paper program is implemented.

The installed inventory is
`komorebi_math/data/paper_machinery_coverage.json`. Each source has a SHA-256
digest. Each mechanism names its source section, status, APIs, tests, and
remaining mathematical boundary. Coverage applies to the listed editions and hashes; later source revisions require a new comparison. Archive members have their own hashes and
are read as data, never executed by the inventory validator.

The audited snapshot contains 55 source objects (including archive members),
528 mechanism records and 705 distinct API references. Metadata, API links,
repository test paths and original source bytes were checked together.

| Snapshot status | Mechanisms |
| --- | ---: |
| Implemented finite scope | 169 |
| Partial | 181 |
| Missing finite construction | 22 |
| Theory only | 65 |
| External objects or research boundary | 70 |
| Separate reference/formal machinery | 21 |

## Inspect coverage

```python
from komorebi_math.audit.coverage import (
    load_coverage_catalog, validate_coverage_catalog,
)

catalog = load_coverage_catalog()
report = validate_coverage_catalog()
assert report.valid
for mechanism in catalog["mechanisms"]:
    if mechanism["status"] == "missing":
        print(mechanism["key"], mechanism["summary"])
```

`report.valid` checks inventory metadata and live API references. It does not
approve the mathematics. `all_catalogued_mechanisms_implemented` remains false
while any recorded mechanism is partial, missing, theoretical, external, or
maintained separately. Installed-wheel validation does not claim to include
the original papers or repository tests.

| Status | Meaning |
| --- | --- |
| `implemented` | The stated finite mechanism has an API and tests; its boundary still applies. |
| `partial` | A restricted or related API leaves some of this mechanism absent. |
| `missing` | A dedicated finite construction is absent from the runtime. |
| `theory_only` | The statement needs abstract or infinite hypotheses without a general finite evaluator. |
| `external` | External mathematical objects, data, or unresolved research assumptions remain. |
| `separate_reference` | Research or formal source exists separately from this Python package. |

In a checkout with the original authorized sources available, rebuild or
verify the crosswalk with actual source hashes and repository test paths:

```bash
PYTHONPATH=src python verification/build_paper_coverage.py --source-root /path/to/paper_sources
PYTHONPATH=src python verification/build_paper_coverage.py --check --source-root /path/to/paper_sources
```

The builder keeps the reviewed fragments and their hashes. It refuses invalid
source/API/test references. It does not download or publish the paper files.

## Source crosswalks

| Source group | Mathematical machinery and limits |
| --- | --- |
| Papers I–IV | [Typed operators, packets, obstructions and analytic families](paper_core_coverage.md) |
| Papers V–VIII | [Masks, transfer, dynamics, quantum structure and compilers](paper_middle_coverage.md) |
| Papers IX–XII | [Lifting, arithmetic addresses, deformation and tame packets](paper_late_coverage.md) |
| Special-function sources | [Exact feedback, inverse data, growth, prime disks and sieves](special_functions_paper_machinery.md) |
| Finite predictive foundations | [Compression, row closure, exact memory and delay criteria](finite_predictive_foundations.md) |
| AVOC, finite integration, Cartier, sufficient resolution | [Calculus and resolution companion coverage](companion_calculus_coverage.md) |
| Bayesian, generative deformation, quantum arithmetic, physics | [Dynamics companion coverage](companion_dynamics_coverage.md) |
| KCOS, H12, stratified universality, reconstruction | [Arithmetic and reconstruction companion coverage](companion_arithmetic_coverage.md) |
| Collatz and Riemann | [Conjecture dynamics coverage](conjecture_dynamics_coverage.md) |
| ABC, Beal and Lehmer | [Conjecture arithmetic coverage](conjecture_arithmetic_coverage.md) |
| Strong Artin, Hodge, Catalan constant, inverse Galois | [Conjecture geometry coverage](conjecture_geometry_coverage.md) |

The edition/hash distinction matters: an older deep-audit report or a related
Lean research block is not automatically a section in the current traditional
paper. The fragments retain those boundaries instead of assigning nonexistent
source sections.

## Added finite constructions

The new modules are explicit imports. They do not enlarge the frozen stable
root API or declare a new release.

| Module | Finite construction |
| --- | --- |
| `core.paper_machinery` | Prime-field mapping cones and exact rational spectral/primary projectors |
| `algorithms.compiler_audit` | Bounded exhaustive independent pruning and budget oracles |
| `dynamics.hermite` | Rational Gaussian moments, polynomial transfer and OU generator |
| `dynamics.gaussian_verifier` | Factor-coded rational Gaussian gain, Joseph law, residual score, packet energies and direct outward dyadic verification |
| `arithmetic.polynomial_lifting` | Simultaneous one-digit polynomial correction fibers and finite towers |
| `arithmetic.packet_differences` | Rational mixed stencils and finite valuation witnesses |
| `langlands.local.tame_tensor` | Exact tame tensor character cycles, fixed spaces and local polynomials |
| `arithmetic.collatz` | Exact valuation words, affine cylinders and bounded first descent |
| `generating.cartier_prime_power` | Rational-mask denominator-layer transitions modulo prime powers |
| `signals.foundations` | Exact finite predictive completion, memory and delay-factor criteria |
| `special_functions.paper_machinery` | Scalar/coupled coefficient recurrences, exact inverse and finite growth/prime/sieve controls |
| `decision.avoc` | Finite task fibers, rational channels, guarded Bellman values and frozen quote bounds |
| `algorithms.boolean_synthesis` | Three-input eager Boolean trees, budgeted synthesis and checked 256-signature minimum bank |
| `signals.tikhonov` | Rational regularization, actual residual correction and certified discrepancy intervals |
| `generating.residual_modules` | Finite ring spans, actual nonfree reachable-observable quotients and complete word comparison |
| `arithmetic.lehmer_shells` | Complete signed five-core shells, gcd sectors, Möbius transforms and auxiliary polynomials |

Existing machinery was reused when it already matched the source, including
congruential Appell/Euler inverses and probabilists' Hermite coordinates.

Two existing mathematical defects were repaired: finite-field elimination
now avoids fixed-width overflow, and occupancy uses the actual span when input
columns are dependent. Finite occupancy no longer declares permanent
stabilization from one equal pair before a later shrink.

## Stress the installed mathematics

Install the exact/scientific extras, then run:

```bash
python -m komorebi_math.audit --papers --coverage --samples 16 \
    --seed 20261009 --timeout 30 --total-timeout 180 --output math_stress.json
```

The base suites cover operators, arithmetic, special functions, coordinates,
finite joins, signals and quantum operations. `--papers` adds source-defined
finite constructions and the coupled special-function controls. Each check
declares its tested range, exactness, tolerance if relevant, and assumptions.
Independent oracles compare different constructions; negative controls check
that known false assertions and invalid inputs remain rejected.

Failures, exceptions, missing optional dependencies, hard timeouts, and checks
left unexecuted by the total time budget remain distinct. A run qualifies only
when every requested check passes. The runner records deterministic per-check
seeds, input and source hashes, environment, and retained evidence. Its report
is historical bounded evidence, not a live certificate or universal theorem.

See [stress framework](math_stress_framework.md),
[Collatz words](collatz_words.md), [finite Gaussian verifier](finite_gaussian_verifier.md),
and the source-group worked examples above.

## Remaining mathematical work

The catalog exposes absent finite constructions as well as theorem boundaries.
Examples include AVOC grid calibration, affine fusion-width witnesses and recorder/recovery reducers,
automatic adapters from large Cartier representations to bounded residual
modules, general route and
quotient explorers, dedicated quantum arithmetic coordinates, and specialized
arithmetic/conjecture evaluators. Partial runtime coverage must not be promoted
to completion because a related generic packet API exists.

General countable completions, arbitrary-ring decomposition, continuum
convergence, global Galois/automorphic realization, and unresolved conjecture
implications require further mathematics or external formal research artifacts.
Finite replay cannot establish those hypotheses or replace their proof.
