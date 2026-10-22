# Qualification toward a stable release

The package remains research beta `1.0.0b4`. A passing candidate CPU report is
one prerequisite for promotion. It does not change the version, publish a
release, or establish CUDA qualification.

## One candidate, six installed environments

The CPU workflow builds one wheel and source distribution, records their
hashes in `CANDIDATE.json`, and supplies those same files to six clean-install
jobs: Linux and Windows on Python 3.11, 3.12, and 3.13. Each job installs the
wheel, rebuilds a wheel from the source distribution, and runs the installed
runtime profile on both. Dependencies come from an explicit local wheelhouse.
The qualification phase does not download packages or inherit host packages.

Wheel ZIP metadata is normalized, and generated package metadata uses LF
newlines. This makes the byte-identical rebuild check portable across the
declared environments. Package payloads retain their source bytes. A changed
payload still fails parity and reproducibility checks.

The separate six-way source matrix, minimum-dependency job, documentation
checks, stress replay, and optional CPU backend checks remain required. The
installed matrix uses current dependency versions available to each Python
minor. It is not an exhaustive dependency-version matrix.

## Evidence identity

`verification/candidate_evidence.py` creates an external candidate manifest
from a clean tracked Git revision and passing distribution integrity checks.
It records the commit, Git tree, release identity, source/tool binding, and
exact wheel and source-distribution hashes. CPU records additionally identify
the executing platform, Python version, nested installed-test results, and CI
run context.

The aggregate requires all six distinct platform/Python cells. Missing cells,
duplicates, changed candidate bytes, mismatched identities, skipped installed
tests, incomplete profiles, and inherited dependencies fail qualification.
Source and artifact bindings are checked before and after artifact execution.

Evidence stays outside the installed package. Embedding a wheel's own hash
inside that wheel would change the bytes being identified. The older embedded
platform registry remains historical, with its original scope and dates.

Hashes detect changed content. They do not authenticate who ran a program.
For hosted evidence, retain the GitHub run and artifact provenance. An offline
checkout without Git can be compared with the supplied manifest, but its
commit label is an external attestation rather than an independently queried
Git identity. The source/tool binding covers the inputs selected by
`release_binding`; it is not a claim that every repository file is in that
binding. The Git commit and tree provide the separate repository identity.

## Replaying a candidate

The completed workflow's `cpu-candidate-evidence-<revision>` artifact contains
the candidate distributions, manifest, and six CPU records. Download that
artifact from the named GitHub run and extract it into the checkout's
`release` directory. Its `dist` directory contains the original wheel and
source distribution. Access follows the repository's permissions.

Use the source checkout at the revision named in `CANDIDATE.json`, and copy
the candidate's original wheel and source distribution into `release/dist`.
Keep exactly those two distribution files there. Copy the manifest to
`release/CANDIDATE.json`. Rebuilding or editing the candidate requires new
evidence if any bound bytes change.

After preparing a compatible wheelhouse as described in the
[release process](RELEASE_PROCESS.md), run:

```powershell
python verification/release_artifact_gate.py --root . --runtime-tests --dependency-mode isolated --wheelhouse release/wheelhouse --output release/LOCAL_ARTIFACT_GATE.json
python verification/candidate_evidence.py cpu --root . --candidate release/CANDIDATE.json --artifact-report release/LOCAL_ARTIFACT_GATE.json --output release/LOCAL_CPU_CANDIDATE.json
```

The producer observes the actual host. A Linux record cannot fill a Windows
cell by renaming its output file. One local record does not satisfy the full
matrix. The aggregate command consumes all six records from the same candidate:

```console
python verification/candidate_evidence.py aggregate --root . --candidate release/CANDIDATE.json --records <six-record-paths> --output release/CPU_CANDIDATE.json
```

`<six-record-paths>` is a placeholder for six separate JSON paths. CI supplies
the concrete paths after downloading the records from its installed jobs.

## Windows CUDA handoff

Use an existing Python environment containing the intended CUDA-enabled
PyTorch and compatible CuPy installation. The NVIDIA driver and hardware must
be available. The following command uses the candidate files and source
checkout described above:

```powershell
.\Run_Hardware_Qualification.ps1 -Python "$env:CONDA_PREFIX\python.exe" -Candidate release\CANDIDATE.json -Output release\CURRENT_HARDWARE_QUALIFICATION.json
```

If Conda is not active, supply the full path to the intended `python.exe`.
The wrapper runs the Python hardware producer, which creates a temporary
environment and installs only the exact local Komorebi wheel. It does not
install or update packages in the chosen environment. Optional CUDA
dependencies are inherited from that environment, and this inheritance is
explicitly recorded. Hardware qualification therefore does not make the CPU
gate's stronger dependency-isolation claim.

The producer clears source-path overrides, verifies imports and installed
package bytes, and executes the existing PyTorch and CuPy CUDA checks. Both
backends must pass. Missing hardware, import failures, native-worker failures,
timeouts, stale candidate files, or failed numerical checks produce a failing
report and nonzero exit status. Retain the complete JSON report, including
failed runs. The small hardware checks establish their stated equivalence
controls, not unrestricted GPU stress or performance guarantees.

## Promotion review

The release contract now recognizes matching `beta`, `release-candidate` and
`final` stage/version pairs. The package remains `1.0.0b4`. This machinery does
not change a version or supply missing execution evidence.

`verification/release_promotion.py` is the authoritative repository promotion
review. It revalidates all six original CPU records, the exact installed-wheel
CUDA report, a full publication record with strict documentation, and all
bounded stress campaigns and resource controls. An outer `passed` flag alone
is insufficient. The full source record embeds the actual run's JUnit XML,
digest and concrete case counts. Existing optional LaTeX/backend and Windows
POSIX exclusions retain their stated scope. The installed runtime profiles
and the separately requested optional backend campaign must have zero skips.

After building the current candidate distributions, produce the source and
stress evidence from the same unchanged checkout:

```console
python verification/block21_publication_gate.py --root . --full-tests --require-mkdocs --output release/FULL_PUBLICATION.json --markdown release/FULL_PUBLICATION.md
python verification/stress_release_candidate.py --root . --include-optional-backends --output-directory release/PROMOTION_STRESS --revision <candidate-revision>
python verification/release_promotion.py --root . --candidate release/CANDIDATE.json --records <six-record-paths> --hardware release/CURRENT_HARDWARE_QUALIFICATION.json --publication release/FULL_PUBLICATION.json --stress release/PROMOTION_STRESS/manifest.json --output release/PROMOTION_REVIEW.json
```

The angle-bracket values are placeholders. Supply the exact candidate commit
and six separate CPU record paths. The stress output directory must be inside
the checkout. The ordinary stress command omits the optional backend campaign
by default, so it cannot substitute for the command above at promotion.

For an RC or final candidate, also supply `--tag v<package-version>`. The tag
must already point at the candidate's actual clean Git commit. For final
promotion, add `--predecessor <successful-RC-review.json>`. The reviewer compares
the tracked source and Git file modes with that RC, permitting the checked
version/stage metadata changes only. Other source changes require another RC.
The final release contract must name that successful RC as its predecessor.

The preserved embedded platform registry remains historical. RC/final source
publication checks validate that history's integrity and explicitly defer
current execution qualification to the external promotion review. The legacy
installed `release readiness` interface remains fail closed when current
embedded platform evidence or current-host CUDA execution is absent. Its
version-only registry checks are not a substitute for exact candidate review.

A deliberate stage transition must synchronize the project/runtime version,
release identity and channel, current API/documentation/codec version labels,
upgrade target and release-contract pins, citation and version-specific test
expectations. Preserve the historical evidence and predecessor artifacts.
Build and qualify new exact artifacts after these edits. The old beta's CPU
or hardware records do not qualify changed RC/final artifacts.

Positive CUDA execution, a newly qualified candidate and a clean release tag
remain execution obligations. This source change does not claim those runs.

The existing license and repository visibility are separate publication
decisions. This workflow does not alter either or publish to GitHub Releases
or PyPI.
