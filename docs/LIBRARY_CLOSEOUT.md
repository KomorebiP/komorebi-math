# Library release closeout

This closeout prepares the library for its final release while preserving the
frozen 131-symbol stable API. The package remains `1.0.0b4` until its release
candidate and hardware requirements have passing evidence for the new source.

## Reconstruction certificates

Ordinary and lattice polyphase certificates now retain small reconstruction
errors without squaring them at their original scale. A nonzero error such as
`1e-200` is rejected at zero tolerance. Integer differences are measured before
fixed-width subtraction can wrap, or conversion to floating point can erase
them. Mixed integer, floating and complex comparisons retain their component
differences. Nonfinite or unrepresentable diagnostics cannot produce a valid
certificate.

The 28 new regression controls produced 22 failures on the preceding source.
All 79 focused polyphase and refinement tests pass after repair. These are
finite implementation checks, with the written scope of each certificate.

## Release stages and evidence

The release-contract parser recognizes consistent beta, release-candidate and
final version/stage pairs. This does not change the installed version or grant
release eligibility. The repository promotion review binds the exact source
and artifacts to the CPU matrix, hardware record, documentation and test
evidence. See the [candidate guide](STABLE_RELEASE_CANDIDATE.md) and
[release process](RELEASE_PROCESS.md) for the current commands.

The preceding candidate `8f6c096e044416e7ecdd9f599a754f7d6028a343`
has recorded passing Linux and Windows CPU qualification. Its evidence is
historical after the source changes in this closeout. New package files must
receive their own qualification. Missing NVIDIA hardware cannot be replaced by
a CPU result or a successful missing-backend control.

## Open-source distribution

Original software and its associated documentation are now licensed under MIT.
The grant is recorded consistently in `LICENSE.md`, project metadata and the
release identity. Third-party notices, including the bundled MathJax assets,
are preserved. Separately supplied OEIS data retain their own terms.

## Reproduction

Qualification uses `pytest==8.4.2`. Pytest 9.1.1 counts passing unittest
subtests without creating a separate XML testcase for each one, which fails
the release check's concrete-case validation. The pinned runner executes the
same subtest assertions within their parent cases and records consistent XML
counts. A failed subtest still fails its parent. Historical reports retain
their original runner versions and counting conventions. See
[the upstream reporting issue](https://github.com/pytest-dev/pytest/issues/13876).

Use one interpreter for installation and all validation commands:

```console
python -m pip install -e ".[test,exact,scientific,yaml]" "setuptools==82.0.1" wheel mkdocs "pymdown-extensions==10.16.1"
python verification/build_block21_distributions.py --root . --output-directory release/dist --report release/LOCAL_DISTRIBUTION_BUILD.json
python -m pytest -q --tb=short tests --junitxml=release/LOCAL_SOURCE_TESTS.xml
python -m komorebi_math api check --strict-signatures
python -m komorebi_math api check --stable
python -m komorebi_math verify stable
python verification/check_release_manual.py --profile verification/release_manual_profile.json --output release/LOCAL_MANUAL_EXAMPLES.json
python verification/block21_publication_gate.py --require-mkdocs --output release/LOCAL_PUBLICATION_GATE.json --markdown release/LOCAL_PUBLICATION_GATE.md
python verification/stress_release_candidate.py --root . --output-directory release/LOCAL_STRESS --timeout 180 --total-timeout 300
```

The full source suite records any optional backend, TeX or platform exclusions.
The installed-artifact runtime profile is a separate check and rejects skips.
Prepare the platform-compatible wheelhouse and run that check as described in
the release process. Keep the exact artifacts and complete reports together.

Public repository visibility, merging a pull request, creating a release tag,
and publishing distributions remain separate actions. This closeout performs
no release publication.
