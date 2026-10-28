# Release process

This guide is for maintainers preparing a source release. Use one unchanged
checkout and one set of distribution files throughout qualification. Keep
reports outside the installed package so that recording a wheel's hash does
not alter the wheel being identified.

The public project remains **beta and work in progress**, as declared by the
development-status classifier in `pyproject.toml`. Version `1.0.0`, the
`stable` channel in `RELEASE_ID.json`, and the release contract's `final` stage
identify the prepared artifacts and version format. Production readiness still
requires the evidence listed under [Publication review](#publication-review).
Keep the public beta status until that review supports a change.

## Prepare the environment

From the checkout root, create an environment for a supported Python minor.
Install the development dependencies and the pinned build tools:

```bash
python -m pip install -e ".[dev,exact,scientific,yaml]" "setuptools==82.0.1" wheel
python -m pip check
python verification/build_distributions.py --root . --output-directory dist --report qualification/distributions.json
```

The builder creates `komorebi_math-1.0.0-py3-none-any.whl` and
`komorebi_math-1.0.0.tar.gz`. Generated files belong in `dist/` and
`qualification/`. Both directories are excluded from the published sources.
Rebuild after changing any bound source or documentation file.

Qualification uses `pytest==8.4.2`. Its JUnit output records concrete parent
cases consistently while executing their unittest subtest assertions. Failed
subtests fail their parent. Do not run qualification with assertions disabled.

## Source and documentation

```bash
python verification/check_release.py --root . --full-tests --require-mkdocs --output qualification/source.json --markdown qualification/source.md
python verification/check_release_manual.py --profile verification/release_manual_profile.json --output qualification/examples.json
```

The full source report retains the actual run's JUnit XML, digest, counts and
failure diagnostics. Strict documentation requires a successful MkDocs build
and valid mathematical rendering. A quick or skipped check is recorded with
its smaller scope and cannot replace this command.

The installed functional check is separate:

```bash
python -m komorebi_math api check --stable
python -m komorebi_math verify all
python -m komorebi_math release check
```

These commands inspect the installed implementation. They do not establish
source cleanliness, a platform matrix or external hardware qualification.

## Clean installed distributions

Prepare a wheelhouse for the operating system, architecture and Python minor
that will run the check. The download step can use the network. The subsequent
checks run offline:

```bash
python -m pip download --only-binary=:all: --dest qualification/wheelhouse "setuptools==82.0.1" wheel numpy "pytest==8.4.2" sympy scipy mpmath PyYAML
python verification/release_artifact_gate.py --root . --runtime-tests --dependency-mode isolated --wheelhouse qualification/wheelhouse --output qualification/installed.json
```

The build interpreter and wheelhouse must use the same setuptools version.
The gate creates clean environments without inherited host packages, checks
the original wheel, rebuilds a wheel from the source distribution, compares
exact bytes and executes the installed profile against both wheels. Missing
local dependencies cause failure rather than an online fallback. Installed
runtime qualification rejects skipped cases.

## One release, six CPU targets

The hosted workflow builds one common wheel and source distribution for Linux
and Windows on Python 3.11, 3.12 and 3.13. The same files are tested on all six
targets, together with the source matrix, minimum dependencies, documentation,
bounded stress and optional CPU backends.

A candidate manifest records the clean Git commit and tree, release identity,
source/tool binding and exact distribution hashes. Replace `COMMIT_SHA` with
the actual checked-out commit:

```bash
python verification/candidate_evidence.py create --root . --revision COMMIT_SHA --output qualification/candidate.json
python verification/candidate_evidence.py cpu --root . --candidate qualification/candidate.json --artifact-report qualification/installed.json --output qualification/cpu-local.json
```

The producer observes the actual host. A Linux record cannot fill a Windows
cell by changing its filename. Aggregate six separate paths for the same
candidate in place of `SIX_CPU_RECORDS`:

```bash
python verification/candidate_evidence.py aggregate --root . --candidate qualification/candidate.json --records SIX_CPU_RECORDS --output qualification/cpu.json
```

The reviewer rejects missing or duplicate cells, changed bytes, mismatched
identities, incomplete installed profiles and inherited dependencies. Preserve
the original records and hosted run details. Hashes identify content. They do
not identify who executed a program.

## Stress and hardware qualification

All requested stress campaigns must execute. For a promotion review, include
both base campaigns and the optional JAX/PyTorch CPU campaign:

```bash
python verification/stress_release_candidate.py --root . --include-optional-backends --revision COMMIT_SHA --output-directory qualification/stress
```

The output directory must be inside the checkout. Reports retain bounded
resource controls, selected source hashes, raw outputs and concrete test cases.
A missing optional backend fails this strict campaign.

On the NVIDIA host, select the existing interpreter with compatible
CUDA-enabled PyTorch and CuPy. The hardware producer installs the exact local
wheel in a temporary environment, clears source-path overrides and verifies
the installed bytes before executing both backend checks:

```powershell
.\Run_Hardware_Qualification.ps1 -Python "C:\path\to\python.exe" -Candidate qualification\candidate.json -Output qualification\hardware.json
```

Dependencies are inherited from that selected hardware environment and the
report says so. This is a different isolation claim from the clean CPU gate.
Retain successful and failed reports. The checked computations are described
in [GPU validation](GPU_VALIDATION.md).

## Publication review

Preparing `1.0.0` metadata does not establish final publication qualification.
A final release still requires current six-target CPU records, full source and
strict documentation, all stress campaigns, positive NVIDIA hardware results,
a clean release tag and the successful qualified RC predecessor. This source
preparation does not claim a new tagged RC or hardware run. Keep these
obligations separate from the installed package's self-check.

`verification/release_promotion.py` revalidates the original nested records
rather than trusting outer passed flags. Supply the six paths, exact tag and
successful RC review in place of the uppercase fields:

```bash
python verification/release_promotion.py --root . --candidate qualification/candidate.json --records SIX_CPU_RECORDS --hardware qualification/hardware.json --publication qualification/source.json --stress qualification/stress/manifest.json --tag RELEASE_TAG --predecessor QUALIFIED_RC_REVIEW --output qualification/promotion.json
```

The final tracked inventory and Git modes must match the RC after the permitted
version/stage metadata normalization. Other source changes require another
qualified RC. The historical embedded registry remains a record of its own
runs. A matching version number does not qualify a changed source tree.

`release readiness --final --require-gpu` is a separate diagnostic of that
registry and the current hardware host. The external promotion review supplies
the exact source/distribution binding.

## Build the source handoff

After the full same-source gate passes:

```bash
python verification/build_source_release.py --root . --output-directory dist --artifact-directory dist --gate qualification/source.json --report qualification/source_release.json
```

The deterministic source ZIP contains clean source files, the distributions and
a source manifest. It excludes local environments, cached bytecode, historical
reports and generated qualification output. A clean extraction is replayed to
check the handoff. Preserve its checksum with the reviewed records. Creating
these files does not itself merge a pull request, create a release tag or
publish to a package index.
