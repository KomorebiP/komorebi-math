# Reproducibility

Komorebi Math separates mathematical verification, package qualification, and platform evidence.

## Fast review

```powershell
$env:PYTHONPATH="$PWD\src"
python .\examples\block21_public_capability_tour.py --json
python .\verification\block21_publication_gate.py --skip-mkdocs
```

The capability tour is deterministic. It covers exact polyphase refinement, redundant analysis and synthesis, Euler–Fibonacci coordinates, exact finite integration, typed expressions, portable persistence, and the frozen stable API.

## Complete source gate

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

The complete gate runs the full pytest suite, strict research, beta, and stable API checks, stable and Block 20 verification, release and upgrade contracts, the capability tour, local documentation-link checks, and a strict MkDocs build when MkDocs is installed.

Linux:

```bash
PYTHON=python ./run_block21_publication_gate.sh
```

## Evidence outputs

The gate writes:

```text
release/BLOCK21_PUBLICATION_GATE.json
release/BLOCK21_PUBLICATION_GATE.md
release/BLOCK21_PUBLIC_CAPABILITY_TOUR.json
```

The JSON report contains a SHA-256 digest over its canonical content. Runtime details are recorded explicitly. A passing Linux run is not represented as a Windows or CUDA qualification.

## Determinism boundaries

Exact algebraic outputs and canonical JSON digests are expected to match across supported platforms. Floating-point backend versions, executable paths, operating-system strings, build timestamps, and optional native-backend availability are environment evidence rather than cross-platform deterministic values.

## Deterministic handoff build

After installing the `dev` extra, one command performs the full gate, rebuilds the distributions, creates the repository handoff, verifies a byte-identical rebuild, and replays the publication gate from a clean extraction:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Build_Block21_Handoff.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

Linux:

```bash
PYTHON=python ./build_block21_handoff.sh
```

The resulting manifest binds every payload file by relative path, normalized mode, size, and SHA-256. Absolute paths, parent traversal, symlinks, special files, non-NFC names, and case-fold collisions are rejected.
