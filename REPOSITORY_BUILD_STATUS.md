# Repository preparation status

## Current library closeout

Original software is now licensed under MIT. The release-stage and polyphase
certificate changes are described in [the closeout guide](docs/LIBRARY_CLOSEOUT.md).
The installed package remains a beta until new-candidate CPU and actual CUDA
evidence satisfy the final release policy. The dated records below describe
earlier revisions, including their earlier licensing and execution status.

## Current GitHub qualification

Reviewed on 2026-10-08 against `KomorebiP/komorebi-math` main commit `50447a253c2d00d4a7a4108440e090844b7148b2`, Git tree `faf5d17d4ab43090ab02bb4b2b53d2d5b87f2081`. The source identity remains `1.0.0b4-repair-r4-block21` under the existing private evaluation license.

The [CPU release qualification workflow](https://github.com/KomorebiP/komorebi-math/actions/runs/37698654422) succeeded on 2026-10-07. All eight jobs passed:

- full source tests and contract checks on Linux and Windows, each with Python 3.11, 3.12, and 3.13
- strict MkDocs documentation and the documentation-required publication gate
- reproducible distributions, source parity, isolated wheel and source-distribution regressions, and clean installed-wheel contract checks

The Linux Python 3.13 source job reported 1,725 passed, 7 skipped, and 4 warnings. Its skipped tests require absent JAX, PyTorch, or LaTeX installations. CUDA qualification remains separate. The source matrix includes the exact, scientific, and YAML extras, while clean base-wheel checks in this run use Linux Python 3.13.

This GitHub evidence supersedes the local-environment limitations recorded below for source tests, documentation, CPU platform coverage, and installed distributions. The original preparation results remain historical evidence. See [GitHub readiness](docs/GITHUB_READINESS.md) for the artifact identity, precise scope, current installation instructions, and outstanding license and publication decisions. No GitHub Release was listed at review, and the repository remains private.

## Historical local preparation

The remainder of this document records the preparation environment on 2026-10-07 before the successful GitHub CI run. References below to missing pytest, skipped MkDocs, pending Windows execution, or unresolved repository URLs describe that earlier local preparation and do not describe the currently qualified GitHub commit.

Prepared on 2026-10-07 from the selected Block 21 certificate-consistency checkpoint. This is a repository candidate. It has not been published, and its existing license has not been changed.

### Original source identity

- Source archive: `Komorebi_Math_R4_Block21_Certificate_Consistency_2026-09-24.zip`
- Archive SHA-256: `324632eb1fcd559469571173c6bbcfc840b6214786ef58f5a8205f8c2f97cf71`
- Package version, independently checked in project metadata, runtime metadata, and citation: `1.0.0b4`
- Release identity: `1.0.0b4-repair-r4-block21`
- Recorded release channel: `development_checkpoint`
- Author pseudonym in metadata: `Komorebi P.`

This identity comes from the archive contents and `RELEASE_ID.json`. Selection of this archive does not establish that it is the newest checkpoint across the entire Komorebi program.

The repository preserves substantive source, tests, documentation, historical qualification evidence, compatibility predecessor fixtures, workflows, and license files. Python caches, the current wheel and source distribution, and an unreferenced Block 9 distribution backup were excluded from the repository candidate. All excluded distribution bytes and the five sibling evidence files remain in the preparation handoff with SHA-256 provenance. Shell entry points have executable POSIX modes. No mathematical implementation was changed.

### Original local validation

Validation used Linux x86-64, CPython 3.12.14, and NumPy 2.3.5.

| Check | Actual outcome |
|---|---|
| Prescribed Block 21 publication gate with `--full-tests --skip-mkdocs` | **Failed:** 16 of 17 checks passed; the complete source suite could not start because `pytest` is absent. |
| Stable verification suite within that gate | 6 of 6 passed. |
| Block 20 verification suite within that gate | 47 of 47 passed. |
| Research, beta, stable API, release, upgrade, and beta-readiness contracts | Passed the gate's finite checks. |
| Public capability tour | Passed; digest `0e20e32f4ad9004d5248d258813b6acfec7b9e96cc30c001aa6865aba9039fdf` matches the archived report. |
| Source parity against archived current wheel and source distribution | All 376 package files matched both artifacts; wheel RECORD and shell-mode checks passed. |
| Source and artifact snapshot during qualification | Unchanged; digest `72c4c06b134e2ce1ec11d122473f34ee9098f9b66ff48a46662b217dfdfd0f3e`. |
| Available Block 21 exact-oracle producer probes | 600 seeded cases each for finite groups, Haar best basis, tree Lagrangian compilation, tree budget compilation, and precision assignment; zero failures in all five categories. |
| Complete Episode 018 consistency probe | Could not complete: its validation-bypass checks require the absent optional `sympy` dependency. |
| Dependency installation | Failed: pip could find no distributions for `pytest` in this execution environment. |

The archived current distributions were restored temporarily for the parity checks, then removed from the repository candidate. They retain their original hashes:

- Wheel: `a7b41efe91524ad1e6e641b0567b571a04247c0b17994c1d019e7d6ac2c4e52d`
- Source distribution: `969d16b0235dc2e35e16d13a1fca044512d27bed3e7a2d7149613f0ff0b5653b`

The available producer probes call the original probe functions without modifying package code. They do not replace the unexecuted validation-bypass section or the complete source suite. Full command receipts, output, dependency-installation failure, and partial-probe selection are retained in the preparation handoff.

Historical reports under `release` and `verification` retain their historical scope and dates. In particular, the original final Block 21 handoff explicitly says the complete source and full installed-runtime suites were not run. The gate's platform-evidence check reviews those records; it does not constitute fresh Windows or CUDA execution.

### Original publication blockers and remaining gates

- **License:** `LICENSE.md` grants private evaluation and reproducibility testing only, reserves all rights, and prohibits redistribution without separate written permission. `pyproject.toml` uses `LicenseRef-Proprietary`, and `RELEASE_ID.json` preserves those terms. An explicit replacement license is needed before representing this package as open source.
- **Repository identity:** `CITATION.cff`, `pyproject.toml`, and `mkdocs.yml` reference `github.com/KomorebiP/komorebi-math`. These URLs need to match the actual connected owner and repository. No destination was assumed or written remotely.
- **Fresh complete qualification:** Install the declared test and exact dependencies, build current distributions, and rerun the full publication gate, the complete Episode 018 probe, and installed-artifact qualification.
- **Documentation:** Strict MkDocs validation was explicitly skipped because MkDocs is absent.
- **Platforms:** Fresh Windows, other Python versions, optional backends, dependency-isolated installations, and CUDA qualification remain unexecuted here.
- **Identity review:** A location-only text scan found no credential markers or private-service URLs. Apparent email matches were matrix expressions. The name Victor occurs in a literature citation to Victor V. Albert; Aster labels the assistant inquiry protocol. Historical evidence contains local `/root` paths that should be reviewed before publication. This heuristic scan is not a comprehensive secret-detection guarantee.

To validate a clean checkout, first create the distributions as the CI workflow does:

```bash
python -m pip install -e '.[test,exact,scientific,yaml]' setuptools wheel mkdocs
python verification/build_block21_distributions.py --root . --output-directory release/dist --report release/CI_REPAIR_DISTRIBUTION_BUILD.json
PYTHON=python ./run_block21_publication_gate.sh
python verification/episode018_certificate_consistency_probe.py --root . --output release/FRESH_CERTIFICATE_CONSISTENCY_PROBE.json --seed 20260924 --trials 600
```

A passing finite gate establishes the checks it records. It does not establish mathematical claims beyond the hypotheses and finite semantics of the individual certificates.
