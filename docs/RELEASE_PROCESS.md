# Release process

## Current beta checkpoint

`1.0.0b1` froze the 131-symbol stable namespace, stable semantic fixtures, and portable persistence contract. Blocks 18 through 20 were added only to the research tier in `1.0.0b2` through `1.0.0b4`; they do not silently expand the frozen stable or beta namespaces.

Every beta correction must regenerate the current API manifests and release evidence while comparing against retained immutable predecessor contracts. Stable signature changes require explicit review and normally a deprecation path.

## Current Block 21 publication gate

The canonical `1.0.0b4` source handoff is `Run_Block21_Publication_Gate.ps1`
on Windows or `run_block21_publication_gate.sh` on POSIX. A strict handoff runs
the complete source tests and strict MkDocs in addition to repository identity,
archive safety, API tiers, stable and Block 20 verifiers, release and upgrade
contracts, beta evidence and readiness, the public capability tour, and local
documentation links. `Build_Block21_Handoff.ps1` or
`build_block21_handoff.sh` then performs reproducible distribution and
repository-ZIP builds plus clean-extraction replay.

The committed beta evidence covers both declared platform families, but only
Linux has current `1.0.0b4` evidence. Windows is inherited from `0.16.0a2`
through the upgrade contract, so `release readiness` passes while
`release readiness --final` correctly fails until current Windows evidence is
committed and digest-bound.

## Historical Block 20 qualification gate

A Block 20 distribution is eligible for handoff only after all of the following
pass from a source tree bound to a deterministic digest:

* the focused installed expression suite and Block 20 aggregate
* 6/6 frozen stable checks and 174/174 cumulative Block 15 checks
* the independent 15-check mathematical verifier and committed adversarial runner
* worked results `5/4`, compiled `I4`, and the retained `106/5` regression
* additive research API comparison with no stable, beta, semantic, or codec drift
* strict expression-JSON, plan-tamper, type-spoof, resource, and archive checks
* complete wheel `RECORD`, safe paths, clean sdist rebuild, and fresh-venv
  installs that inherit the declared build host dependencies
* byte-identical wheel and canonicalized-sdist replay under the recorded
  `SOURCE_DATE_EPOCH`
* a safe single-root source handoff archive and SHA-256 manifest

The source tree in this workspace has no authenticated release-tag metadata, so
this beta gate binds a deterministic source digest and does not claim a clean or
signed tag. Windows, CUDA, and strict MkDocs results are claimed only when they
are actually run and recorded.

## Block 19 qualification gate

A Block 19 distribution is eligible for handoff only after all of the following pass from the frozen source:

* 12/12 installed integration checks
* 24/24 aggregate Block 19 checks
* 6/6 frozen stable checks
* 174/174 cumulative Block 15 checks
* 16/16 independent mathematical checks
* the complete committed adversarial suite
* the exact worked result `106/5`
* additive research API comparison with no stable or beta drift
* complete wheel `RECORD`, safe paths, and byte equality with the frozen package source
* clean sdist extraction, wheel rebuild, normalized payload comparison, and isolated installation of both wheels
* a clean single-root source handoff archive and SHA-256 manifest

The rigorous composite-Simpson path is restricted to a provenance-matched `ExactPolynomial` on the same exact rational interval. A generic floating callable cannot receive a rigorous result unless a future interval or ball backend encloses evaluation, conversion, summation, and truncation errors.

## Release candidate

A release candidate must additionally be built from a clean tag and pass the declared CPU platform matrix, hardware-backed CUDA qualification on the recorded NVIDIA environment, a strict documentation build, and archive-integrity verification. Historical Windows or CUDA evidence is not recast as a current Block 20 result.

## Final release

`1.0.0` should differ from the successful release candidate only in approved version and release metadata. Any code change after `rc1` produces `rc2` or later.

## Historical local Block 20 commands

POSIX:

```console
./run_block20_source_validation.sh
./build_block20_distributions.sh
./install_and_verify_block20.sh
python verification/build_block20_handoff.py --root . --output-directory ..
```

This POSIX sequence reproduces the complete Block 20 handoff path. The install venvs
use `--system-site-packages` and `--no-deps`, so the evidence is fresh-install
evidence with inherited dependencies, not dependency-isolation evidence.

PowerShell smoke paths:

```powershell
.\Run_Block20_Source_Validation.ps1
.\Build_Block20_Distributions.ps1
.\Install_And_Verify_Block20.ps1 -Recreate
```

The PowerShell scripts exercise source validation, distribution construction,
and the primary wheel on a Windows host. They do not currently rebuild and
install the sdist or produce the evidence chain required by the handoff
builder. Accordingly, this checkpoint makes no current Windows qualification
claim; the POSIX builder will reject incomplete or stale PowerShell evidence.

## Local Block 19 commands

POSIX:

```console
./run_block19_source_validation.sh
./build_block19_distributions.sh
./install_and_verify_block19.sh
```

PowerShell:

```powershell
.\Run_Block19_Source_Validation.ps1
.\Build_Block19_Distributions.ps1
.\Install_And_Verify_Block19.ps1 -Recreate
```
