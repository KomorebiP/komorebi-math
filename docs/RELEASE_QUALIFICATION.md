# Release Qualification

Block 21 is the current qualification and publication handoff for
`komorebi-math==1.0.0b4`. It keeps five claims separate: source correctness,
installed behavior, frozen compatibility, executed platform evidence, and
optional hardware qualification.

## Source publication gate

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
current-version, digest-checked execution record. Beta readiness may use an
authorized predecessor record through the frozen upgrade contract; final
readiness may not.

## Distribution and handoff reproducibility

`Build_Block21_Handoff.ps1` consumes a passing full publication-gate record,
rebuilds the wheel and source distribution twice, compares exact bytes, builds
the repository ZIP twice, audits every archive member, and replays the quick
gate from a clean extraction. These checks qualify artifacts; they do not
publish to PyPI, create a GitHub release, or create an authenticated tag.

## CUDA gate

```powershell
.\Run_GPU_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

A missing GPU does not invalidate the base NumPy wheel. It invalidates only a
claim that explicitly requires CUDA qualification. Block 16 launchers remain
in the repository solely as historical `0.16.0a2` reproduction paths.
