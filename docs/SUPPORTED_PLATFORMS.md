# Platforms and dependencies

## Base package

The supported CPU targets are Linux and Windows on Python 3.11, 3.12 and 3.13.
The base package requires NumPy 1.26 or newer. JAX, PyTorch, CuPy, SciPy, SymPy
and LaTeX are optional.

The release matrix checks both source execution and clean installations of
one common wheel and source distribution on all six platform/Python targets.
Each installed job rebuilds a wheel from the source distribution and executes
the same runtime profile against both wheels. The
[release guide](RELEASE_PROCESS.md) defines those checks and their records.

## Optional dependencies

| Extra | Requirement | Use |
|---|---|---|
| `exact` | SymPy 1.14 or newer | Symbolic algorithms and Smith transformation matrices |
| `scientific` | SciPy 1.11 or newer, mpmath 1.3 or newer | Scientific and high precision routines |
| `plot` | Matplotlib 3.8 or newer | Plotting examples |
| `yaml` | PyYAML 6.0 or newer | YAML input and output |
| `jax` | JAX 0.4 or newer | Directional automatic differentiation |
| `torch` | PyTorch 2.1 or newer | Directional automatic differentiation |
| `autodiff` | JAX and PyTorch | Both derivative adapters |
| `all` | Exact, scientific, plotting and YAML extras | Combined mathematical dependencies |

Directed quadratic-form enclosures specifically require `mpmath==1.3.0`.
That arithmetic trust requirement is narrower than the general scientific
extra's version range. The `all` extra does not install autodiff, CUDA or
development tools.

The minimum CPU dependency job uses Python 3.11 with NumPy 1.26.4, SymPy 1.14.0,
SciPy 1.11.4 and mpmath 1.3.0. It tests that combination, rather than every
combination allowed by the metadata. Exact environments used for a release
are recorded with its qualification results.

## Native backends

JAX and PyTorch are loaded lazily. Availability requires a successful
child-interpreter import and the directional JVP capability used by the
adapter. Finding an installed package is insufficient. See
[backend isolation](OPTIONAL_BACKEND_ISOLATION.md).

CuPy is installed separately because its distribution must match the local
CUDA line. The [CUDA installation guide](GPU_INSTALLATION.md) describes the
environment boundary, and [GPU validation](GPU_VALIDATION.md) describes the
checked computations.

## Additional environments

macOS, Windows ARM64, Linux ARM64, native Windows JAX GPU and other unrecorded
platforms are outside the declared release matrix. They require their own
execution records before a support claim is made. A constraints file records
one reproduced environment; it does not make every listed dependency mandatory
for the base package.
