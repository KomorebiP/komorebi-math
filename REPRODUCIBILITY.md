# Reproducibility

Komorebi Math separates mathematical verification, package qualification, and platform evidence.

## Prepare an authorized checkout

Run from the repository root with one chosen Python 3.11–3.13 interpreter:

```powershell
python -m pip install -e ".[test,exact,scientific,yaml]" "setuptools==82.0.1" wheel mkdocs "pymdown-extensions==10.16.1"
python verification/build_block21_distributions.py --root . --output-directory release/dist --report release/LOCAL_DISTRIBUTION_BUILD.json
```

The gate compares source with the built distributions. Rebuild after changing
source or documentation so the artifact-parity checks use the current bytes.
The `dev` extra alone does not include the exact, scientific, and YAML test
dependencies. For an explicit Windows interpreter path, follow
[local validation](docs/LOCAL_VALIDATION.md).

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

The PowerShell wrapper requires a strict MkDocs build by default and runs the
full pytest suite, strict research, beta, and stable API checks, stable and
Block 20 verification, release and upgrade contracts, the capability tour,
and local documentation-link checks. Use the prepared environment's
interpreter in the `-Python` argument.

For a strict POSIX gate that fails if MkDocs is missing:

```bash
python verification/block21_publication_gate.py --full-tests --require-mkdocs --output release/BLOCK21_PUBLICATION_GATE.json --markdown release/BLOCK21_PUBLICATION_GATE.md
python examples/block21_public_capability_tour.py --output release/BLOCK21_PUBLIC_CAPABILITY_TOUR.json
```

`run_block21_publication_gate.sh` is also available, but automatically skips
MkDocs when it is absent. That skipped build is a reduced diagnostic result.

## Mathematical documentation

The documentation build preserves LaTeX math with PyMdown Extensions 10.16.1.
Built pages use the bundled MathJax 3.2.2 SVG renderer and font data, so equation
rendering needs no CDN, network font, or npm installation. The upstream license
and source hashes accompany the bundle under `docs/assets/mathjax-3.2.2`.
Code blocks remain literal code.

```bash
python -m mkdocs build --strict
python verification/check_math_docs.py --site-dir site --output release/MATH_DOCS_CHECK.json
```

This check verifies math preservation, code exclusion, local asset references,
and the copied asset hashes. It does not replace browser rendering checks or
validate the mathematical assertions themselves.

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

After the preparation above, one command rebuilds the distributions, performs
the full gate, creates the repository handoff, verifies a byte-identical
rebuild, and replays the publication gate from a clean extraction:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Build_Block21_Handoff.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

Linux:

```bash
PYTHON=python ./build_block21_handoff.sh
```

Keep MkDocs installed for the strict POSIX handoff. The shell wrapper otherwise
requests a documentation skip, which cannot establish strict documentation
qualification. The default handoff's installed-artifact checks may inherit
build-host dependencies. A separate dependency-isolated replay uses
`release_artifact_gate.py --runtime-tests --dependency-mode isolated` with a
populated local wheelhouse, as shown in the [user guide](docs/USER_GUIDE.md).

The resulting manifest binds every payload file by relative path, normalized mode, size, and SHA-256. Absolute paths, parent traversal, symlinks, special files, non-NFC names, and case-fold collisions are rejected.
