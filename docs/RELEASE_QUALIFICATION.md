# Release Qualification

Block 21 is the current qualification and publication handoff for
`komorebi-math==1.0.0b4`. It keeps five claims separate: source correctness,
installed behavior, frozen compatibility, executed platform evidence, and
optional hardware qualification.

## Source publication gate

From the checkout root, install the test dependencies and tools, then rebuild
the distributions before running the gate:

```powershell
python -m pip install -e ".[test,exact,scientific,yaml]" "setuptools==82.0.1" wheel mkdocs "pymdown-extensions==10.16.1"
python verification/build_block21_distributions.py --root . --output-directory release/dist --report release/LOCAL_DISTRIBUTION_BUILD.json
```

Use that same interpreter for the wrapper. The following Conda example assumes
the intended environment is active:

```powershell
.\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

This default route runs the complete source test suite and a strict MkDocs
build in addition to the base publication checks. The bounded smoke route is:

```powershell
.\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe" -Quick
```

Quick mode is useful for diagnosis, but it is not the strict handoff
prerequisite because it omits the complete source test suite.

## Machine-readable qualification

```powershell
python -m komorebi_math release environment
python -m komorebi_math release contract
python -m komorebi_math release upgrade
python -m komorebi_math release evidence
python -m komorebi_math release documentation
python -m komorebi_math release readiness
```

For a final release-candidate claim, also run:

```powershell
python -m komorebi_math release evidence --current
python -m komorebi_math release readiness --final
```

The final forms deliberately fail when any declared platform lacks a
current-version, digest-checked execution record. Here, "current" means the
record's package version matches the installed version. The digest checks the
record's source JSON, not a Git commit or the current package contents. The
embedded Linux record reports 1,138 tests from an earlier `1.0.0b4` checkpoint;
it does not qualify subsequent repairs. Beta readiness may use an authorized
predecessor record through the frozen upgrade contract; final readiness may
not. Revision-specific CPU CI and the embedded evidence registry are separate
records. Passing CI does not update the registry automatically. See
[GitHub readiness](GITHUB_READINESS.md) for the tested revisions and
[beta to release candidate](BETA_TO_RC.md) for the remaining registry boundary.

`--final` does not request GPU execution or check tag cleanliness. On the CUDA
qualification host, use `release readiness --final --require-gpu` to require
both the platform-evidence and hardware checks. The complete release policy
also requires the artifact and source-revision checks in the
[release process](RELEASE_PROCESS.md).

## Distribution and handoff reproducibility

`Build_Block21_Handoff.ps1` consumes a passing full publication-gate record,
rebuilds the wheel and source distribution twice, compares exact bytes, builds
the repository ZIP twice, audits every archive member, and replays the quick
gate from a clean extraction. These checks qualify artifacts; they do not
publish to PyPI, create a GitHub release, or create an authenticated tag.

## CUDA gate

```powershell
.\Run_GPU_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe" -Output release\CURRENT_GPU_QUALIFICATION.json
```

A missing GPU does not invalidate the base NumPy wheel. It invalidates only a
claim that explicitly requires CUDA qualification. This generic wrapper checks
the package installed for the chosen interpreter. Its default output name
contains `BLOCK16` for compatibility; the explicit path above avoids confusing
a new run with a historical report. Block 16 source and distribution launchers
remain historical `0.16.0a2` reproduction paths.
