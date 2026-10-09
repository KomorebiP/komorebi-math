# Release closeout: 9 October 2026

This pass continues the [stress audit](STRESS_AUDIT_2026_10_08.md) from
revision `46267f52513c17eba0f599edbdbed896a8adcfd3`. The changes remain
part of [pull request 1](https://github.com/KomorebiP/komorebi-math/pull/1).
Package version `1.0.0b4`, repair identity, and stable API are unchanged.

## Repairs and regression controls

- Portable persistence normalizes NumPy integer components of exact fractions
  and preserves sequence-valued cells in object arrays without expanding them.
- Exact automatic-differentiation symmetry checks compare exact differences
  with zero. A discrepancy below floating-point range can no longer receive a
  valid exact certificate. Numerical residual norms use magnitude scaling to
  avoid avoidable underflow and overflow.
- Stress, publication, and installed-runtime qualification ignore inherited
  `PYTEST_ADDOPTS` test filters. Stress reports reject missing or inconsistent
  JUnit evidence and incomplete resource controls, and replace stale success
  records after child failures. Optimized Python cannot disable resource checks.
- The manual checker recognizes explicitly labelled Python fences and requires
  examples in each current reader guide.
- CPU PyTorch controls explicitly select CPU tensors. A simulated CUDA-build
  metadata control checks that CPU qualification does not reject that metadata;
  it does not qualify a CUDA wheel or GPU execution.

The new controls comprise 36 library contract cases, 29 repository-tool cases,
and one optional CPU-backend case. The generic stress selection contains 351
cases plus 11 bounded resource controls. The optional selection contains seven
cases. The installed runtime profile contains 91 test files. These counts
describe selected controls; successful execution must be established by the
reports for the corresponding revision.

## Manual and offline equations

Reader-facing instructions now distinguish historical evidence from current
qualification, use interpreter-consistent installation commands, and explain
the build prerequisite for source/artifact parity tests. Mathematical claims
retain their documented finite scope, precision limits, and hypotheses.

The documentation build uses pinned Markdown math preservation and a local,
unmodified MathJax 3.2.2 SVG renderer with its license and provenance. Assets
are included in the source distribution. No external script or stylesheet is
needed by the built manual. CI checks the original formula corruption case,
code exclusion, local asset references, and asset hashes.

During this pass the actual renderer converted 428 expressions across 30 pages
in an offline DOM runtime, with zero TeX errors and no external requests. Six
standalone rendered equations were visually inspected. Chromium installation
failed because its download was invalid; full browser layout QA is not claimed.

## Reproduce qualification

Follow [local validation](LOCAL_VALIDATION.md) to install the development
dependencies and build distributions before the full source suite. Then run:

```sh
python verification/stress_release_candidate.py --timeout 180 --total-timeout 300
python verification/check_release_manual.py --output release/MANUAL_EXAMPLES.json
python -m mkdocs build --strict
python verification/check_math_docs.py --site-dir site --output release/MATH_DOCS_CHECK.json
```

For isolated wheel and source-rebuilt-wheel qualification, use the local
wheelhouse procedure in the [release process](RELEASE_PROCESS.md). Optional
backends run separately as described in [backend isolation](OPTIONAL_BACKEND_ISOLATION.md).
The CI workflow also injects a deliberately filtering `PYTEST_ADDOPTS` value
to verify that required stress and installed-runtime cases still execute.

## Scope and remaining release gates

Local closeout checks bind revision `4a6473723af8679cb4882858ee21c6b995704dd5`.
At the subsequent review, no GitHub Actions run for that revision was available
and PR metadata still named the earlier head. The preceding successful runs
therefore remain evidence only for their named revision. These finite tests
are regression evidence, not universal mathematical or performance proofs.

CPU beta qualification does not satisfy the separate final-release policy.
The embedded platform registry still needs current Windows evidence. Its Linux
record matches the package version but predates this revision; the record's
digest authenticates its JSON, not the latest source. The RC/final process also
requires current hardware CUDA qualification, which is a separate check from
`release readiness --final`. Existing Windows CPU CI and simulated CUDA metadata
do not substitute for these records. No
macOS, ARM, expanded optional-version matrix, or fresh GPU qualification is
claimed. See [release qualification](RELEASE_QUALIFICATION.md) before promotion.
