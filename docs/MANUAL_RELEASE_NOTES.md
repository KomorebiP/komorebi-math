# Manual revision notes

The 8 October 2026 reader edition replaces the older handoff instructions with
the current repository workflow. It provides a compact entry point while the
cumulative reference preserves the detailed chapters and API cards.

## Corrections and additions

Installation now uses declared extras, the `4b21` wheel name, the actual
PowerShell wrapper, and explicit development tools. The older manual named
`docs`, `build`, and `publication` extras that are absent from the current
metadata, and described an isolated Windows launcher absent from this checkout.
The directed quadratic-form path now explicitly pins its qualified mpmath
backend.

Certificate prose now covers read-only owned residuals, finite-evidence guards,
current versus recorded validity, and producer-specific revalidation after
portable loading. The historical 51-codec compatibility contract is
distinguished from the live certificate codec version and its additional
migration.

The numerical chapters now match the negative derivative Schrödinger memory
kernel, the `-i` self-energy transform relation, the Laplace convergence domain,
rational continuation, and finite integration at a resolvent pole. Other
corrections cover undercomplete frame bounds, complex streaming state,
constraint-span scaling, spectral underflow and asymptotics, and signal input
validation. Recipe return types, CLI counts, cross-references, spelling, and
malformed mathematical separators were also corrected.

The [new machinery guide](NEW_MACHINERY_GUIDE.md) covers all 56 named exports
in the six added families. Its twelve worked cases include independent exact,
finite-matrix, Fourier, and Erlang controls. Each section states its assumptions,
return types, error behavior, and numerical or arithmetic trust limits.

## Implementation repairs found during manual replay

Nine bounded defects disclosed by the older chapters were reproduced and
repaired. They have 46 new regression cases in
`tests/test_manual_release_regressions.py`.

| Operation | Corrected behavior |
|---|---|
| Digit reversal | Reject a noninteger radix or radix below two before iteration |
| Simultaneous unit difference | Preserve infinite valuations for a zero difference |
| Empty q-factorial | Validate the integer base before returning the empty product |
| Detected integral recurrences | Execute integral `Fraction` data modulo an integer |
| Formal coefficient pushforward | Preserve constants and support degree-zero projection |
| Monic polynomial extended gcd | Reject the pair of zero polynomials with `ValidationError` |
| Sparse algebraic generating functions | Reject coerced, fractional, Boolean, or negative degrees |
| Lucas digit helpers | Require a prime before applying the digit identity |
| Five finite-prefix recognition routines | Require a nonnegative integer holdout count |

The already repaired `RecurrenceModel.synthesize` path was checked and its
obsolete defect warning removed. Public signatures and historical aggregate
manifests remain unchanged.

## Example checks and review scope

The current reader guides have **16 independently executable Python examples**.
The replay tool starts a fresh interpreter and temporary working directory for
each example, retains its output and source digest, and rejects disabled Python
assertions:

```bash
python verification/check_release_manual.py --output release/MANUAL_EXAMPLE_CHECKS.json
```

The documentation CI job runs this replay alongside strict MkDocs. The clean
installed-artifact profile includes the 46 new regression cases, so they are
also checked against the wheel and the source-rebuilt wheel.

The cumulative review separately checked 164 core examples and fragments,
65 numerical examples including one explicitly dependent continuation, and
35 integration, persistence, export, and workflow examples. Four core examples
correctly raise their documented exceptions. Continuations retain their named
prerequisites. Twenty independent numerical contract demonstrations passed.
The first eight chapters also have 24 independently replayed import-containing
Python examples; an application-dependent catch pattern is labeled pseudocode.
Twelve installed CLI suites passed 145 finite checks, and `block20` passed
47 checks. These counts describe the recorded runs, rather than universal
mathematical coverage.

Review covered reader prose, current contracts, worked examples,
cross-references, signatures, and the affected repeated reference-card
templates. It did not re-prove historical theorem dependencies. Historical
production ledgers and their proposed Appendices E–O remain dated project
records. Their broader definition of a complete research reference is not
silently relabeled as achieved by the reader edition.

The package remains beta `1.0.0b4`. Current CPU source, artifact, dependency,
and documentation qualification belongs to its named source revision.
Older platform and hardware records keep their dates and scope. This manual
does not designate a final 1.0 release or record a public package upload.
