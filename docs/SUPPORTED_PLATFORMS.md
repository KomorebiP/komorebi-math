# Supported Platforms and Dependency Policy

The declared support policy and embedded historical environment records are in
`src/komorebi_math/data/support_matrix_1_0.json`. Newer revision-specific CI and
stress results are recorded in [GitHub readiness](GITHUB_READINESS.md); a
successful external run does not automatically update that embedded file.

## Declared base contract

- Python 3.11, 3.12, or 3.13
- NumPy 1.26 or newer
- no mandatory JAX, PyTorch, CuPy, SciPy, SymPy, or LaTeX installation

The optional `exact` and `all` extras require SymPy 1.14 or newer. Smith
decomposition uses the transformation-matrix API introduced in that line.
NumPy 1.26 remains supported: the CPU CI workflow also exercises NumPy 1.26.4,
SymPy 1.14.0, SciPy 1.11.4, and mpmath 1.3.0 on Python 3.11. This checks one
minimum dependency combination rather than every permitted version combination.

Every declared Python minor must pass the base installed-wheel contract before the final 1.0 release. Declared metadata is not treated as executed evidence.

The earlier GitHub Linux/Windows Python 3.11–3.13 matrix installed editable
source, with clean artifact qualification on Linux Python 3.13. The expanded
workflow now requires a common candidate wheel and source distribution to pass
clean installed-runtime qualification on all six platform/Python cells.
The [candidate guide](STABLE_RELEASE_CANDIDATE.md) defines the evidence and
replay procedure. Only completed records for the named candidate establish
coverage; workflow configuration alone does not.

## Optional backends

JAX and PyTorch availability is determined by a child-interpreter import and a capability check for the directional JVP primitive. Package discovery alone is insufficient.

The stress-tested candidate separately qualified JAX/JAXlib 0.11.2 and
PyTorch 2.14.1+cpu on Linux Python 3.12. This CPU result does not establish
support for every version permitted by the optional dependency lower bounds,
Windows optional runtimes, or GPU execution.

CuPy is not a base dependency because its wheel must match the local CUDA line. CUDA qualification is a separate hardware gate.

## Recorded and unqualified environments

The support matrix records successful environments individually. Platforms that have not executed the release gates remain explicitly unqualified. The current record does not claim qualification for macOS, Windows ARM64, Linux ARM64, native Windows JAX GPU, or unrecorded CUDA versions.

The embedded `1.0.0b4` Linux record predates the October repairs and contains
1,138 passing source tests. The Windows record comes from `0.16.0a2`. In
`release evidence`, "current" checks package-version equality, and the stored
digest authenticates a record's JSON. Neither condition establishes that a
later commit has been tested. Release evidence must also identify the source
revision and artifacts actually executed.

## Constraints

The `constraints` directory distinguishes minimum package policy from one reproduced validation environment. A constraints file is not a claim that every listed package is required by the base wheel.
