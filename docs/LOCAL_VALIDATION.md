# Local Windows validation

Block 21 is the current source-publication gate for `komorebi-math==1.0.0b4`.
It combines repository identity and archive-safety checks with strict API,
release, upgrade, evidence, readiness, capability-tour, documentation-link,
test, and MkDocs checks. The full launcher runs the source test suite and
requires strict MkDocs.

## Complete source qualification

```powershell
Set-Location C:\path\to\komorebi-math
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install -e ".[test,exact,scientific,yaml]" "setuptools==82.0.1" wheel mkdocs "pymdown-extensions==10.16.1"
& $py verification/build_block21_distributions.py --root . --output-directory release/dist --report release/LOCAL_DISTRIBUTION_BUILD.json
.\Run_Block21_Publication_Gate.ps1 -Python $py
```

Replace the checkout path with the actual location. An existing qualified
Conda environment can be used by setting `$py` to its Python executable instead
of creating a virtual environment. The distribution build must precede the
full suite because publication tests check source/artifact parity. Rebuild
after editing source or documentation.

The run writes `release\BLOCK21_PUBLICATION_GATE.json`, its Markdown report,
and `release\BLOCK21_PUBLIC_CAPABILITY_TOUR.json`. Use `-Quick` only for a
bounded smoke replay; it omits the full pytest run. Use `-SkipMkDocs` only when
the documentation build is intentionally outside the claim being tested.

## Installed and contract checks

The current installed cumulative verifier is `block20`. The historical
`verify all` route is a Block 17 compatibility alias and is not the current
cumulative or publication gate.

```powershell
& $py -m komorebi_math api check --stable
& $py -m komorebi_math release contract
& $py -m komorebi_math release upgrade
& $py -m komorebi_math release evidence
& $py -m komorebi_math release documentation
& $py -m komorebi_math release readiness
& $py -m komorebi_math verify stable
& $py -m komorebi_math verify block20
```

`release readiness --final` is stricter than beta readiness: it requires
current-version evidence on every declared platform family. A nonzero result
must be preserved until those records are committed and digest-bound.

## CUDA qualification

Run the hardware gate only in an environment containing a working CUDA
PyTorch build and exactly one compatible CuPy distribution:

```powershell
.\Run_GPU_Validation.ps1 -Python $py
```

PyTorch and CuPy computations execute in separate child interpreters. A native
import failure, timeout, or abnormal exit becomes a failed hardware record
instead of terminating the parent release process. Missing CUDA does not
invalidate the NumPy-only base wheel.

## Historical Block 16 reproduction

`Run_Block16_Source_Validation.ps1`, `Run_Block16_Batched_Validation.ps1`,
`Build_Block16_Distributions.ps1`, and `Install_And_Verify_Block16.ps1` remain
for reproducing the `0.16.0a2` freeze-candidate workflow. They are historical
entry points, not the canonical `1.0.0b4` publication route.
