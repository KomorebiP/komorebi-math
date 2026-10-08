# Supported Platforms and Dependency Policy

The machine-readable source of truth is `src/komorebi_math/data/support_matrix_1_0.json`.

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

## Optional backends

JAX and PyTorch availability is determined by a child-interpreter import and a capability check for the directional JVP primitive. Package discovery alone is insufficient.

CuPy is not a base dependency because its wheel must match the local CUDA line. CUDA qualification is a separate hardware gate.

## Recorded and unqualified environments

The support matrix records successful environments individually. Platforms that have not executed the release gates remain explicitly unqualified. The current record does not claim qualification for macOS, Windows ARM64, Linux ARM64, native Windows JAX GPU, or unrecorded CUDA versions.

## Constraints

The `constraints` directory distinguishes minimum package policy from one reproduced validation environment. A constraints file is not a claim that every listed package is required by the base wheel.
