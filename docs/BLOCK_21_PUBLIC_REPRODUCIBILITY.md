# Block 21: Public Reproducibility and Publication Gate

Block 21 is a release-engineering and research-audit layer. It does not add symbols to the frozen stable namespace. Its purpose is to make the existing mathematical implementation independently reviewable through one finite execution path.

## Public capability tour

`examples/block21_public_capability_tour.py` executes six connected capabilities:

1. canonical polyphase refinement from order 4 to order 12 with exact reconstruction
2. redundant finite analysis and canonical-dual synthesis with portable roundtrip
3. Euler–Fibonacci recurrence coordinates with certified reconstruction
4. exact pushforward integration with the value `106/5` and an adversarial false-certificate guard
5. typed finite expression evaluation with identity pipeline and the exact value `5/4`
6. the 131-symbol stable API manifest and its deterministic digest

The output is canonical JSON with its own SHA-256 digest. Timing is intentionally excluded from the digest.

## Publication gate

`verification/block21_publication_gate.py` checks:

- build, runtime, and citation version identity
- repository-facing publication files and archive-safe paths
- strict research, beta, and stable API manifests
- stable and Block 20 verification suites
- release, upgrade, evidence, and beta-readiness contracts
- the public capability tour
- local Markdown links
- strict MkDocs output when requested or available
- the complete pytest suite in full mode

The complete PowerShell invocation is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

The Linux invocation is:

```bash
PYTHON=python ./run_block21_publication_gate.sh
```

## Evidence discipline

A passing gate proves only the finite checks recorded in the report. It does not assert current Windows execution when run on Linux, CUDA qualification without a device, or global mathematical conclusions beyond the hypotheses carried by individual certificates.

## Deterministic publication handoff

`verification/build_block21_handoff.py` packages the qualified repository only after a passing full publication-gate record exists. It records every payload path, normalized mode, size, and SHA-256, embeds the current wheel and source-distribution hashes, verifies a byte-identical rebuild, and replays the quick gate from a clean extraction.

PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Build_Block21_Handoff.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

Linux:

```bash
PYTHON=python ./build_block21_handoff.sh
```
