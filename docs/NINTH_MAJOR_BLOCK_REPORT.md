# Ninth major block report

## Objective

Block 9 closes the largest remaining executable gap in the package: the transition from first-order nonlinear coordinate and obstruction tools to a coherent finite higher-order calculus.

The implementation follows the dependency order:

1. derivative providers
2. multilinear packet jets
3. higher chain rules
4. rooted nonlinear trees
5. convergence certificates
6. graded operations
7. correction and elimination
8. filtered lifting and spectral sequences
9. shared smooth transport
10. release integration and validation

## Runtime architecture

The new code is divided into focused modules:

| Module | Responsibility |
|---|---|
| `nonlinear.derivatives` | derivative protocols and exact or numerical providers |
| `nonlinear.jets` | packet-resolved multilinear derivatives and Taylor terms |
| `nonlinear.composition` | set partitions and Faà di Bruno composition |
| `nonlinear.trees` | rooted trees, finite Picard expansions, and majorants |
| `nonlinear.graded` | finite group gradings and Maurer-Cartan packet identities |
| `nonlinear.correction` | obstruction projectors, right inverses, and Lyapunov-Schmidt reduction |
| `nonlinear.elimination` | nonlinear Schur reduction through second order |
| `nonlinear.filtered` | finite filtrations, lifts, and spectral sequences |
| `nonlinear.transport` | shared moving-packet transport |

Each module has its own public inventory and provenance records. The nonlinear package root re-exports the complete finite API. A curated subset is also available at the top-level package namespace.

## Exactness model

The implementation uses three principal execution classes.

### Exact

Exact outputs are returned for finite combinatorics, prime-field linear algebra, integer or rational polynomial tensors, and symbolic expressions when SymPy is installed.

### Numerical

Numerical outputs include finite differences, SVD obstruction projectors, damped hidden solves, Lyapunov-Schmidt decompositions, and smooth transport. Every result includes residuals and tolerances.

### Conditional

Conditional outputs are used where finite computation cannot establish an external analytic or topological hypothesis. Examples include incomplete support enumeration and infinite projective-limit completion.

No conditional result is silently promoted to exact.

## Defects found during implementation

Two implementation defects were found and corrected before release:

1. packet Faà di Bruno validation initially compared a packet-resolved result with derivatives evaluated on unprojected directions. Validation now uses the same projected directions as the packet formula.
2. exact factorial scaling in polynomial and Taylor paths initially risked floating conversion. Exact branches now use rational factorial scaling.

A first editable install attempt also failed because pip attempted network-backed build isolation. The package installed successfully with `--no-build-isolation`, which is the correct offline path for the supplied handoff.

## Test growth

The baseline contained 421 passing tests. Block 9 adds 35 focused tests:

* 8 derivative, jet, and composition tests
* 6 rooted-tree tests
* 5 graded-operation tests
* 5 correction and elimination tests
* 7 filtered-lifting and spectral-sequence tests
* 4 shared-transport tests

The complete suite passes 456 tests.

## Installed verification

The original nonlinear foundation contributes five checks. Block 9 adds nine:

* Faà di Bruno composition
* multilinear packet jets
* rooted nonlinear tree expansion
* graded packet equations
* obstruction and Lyapunov-Schmidt reduction
* nonlinear Schur elimination
* finite filtered lifting
* filtered obstruction spectral sequence
* shared smooth packet transport

The aggregate `block9` and `all` suites pass 106 checks in the full validation environment. A clean wheel installed with only the required NumPy dependency passes 104 checks; the two omitted checks require optional SymPy.

## Deep audit

`verification/block9_deep_audit.py` executes 21,890 checks in nine groups:

| Group | Checks |
|---|---:|
| combinatorial identities | 121 |
| polynomial Faà di Bruno | 3,600 |
| packet jet reconstruction | 800 |
| rooted tree expansions | 2,081 |
| graded packet equations | 1,687 |
| nonlinear elimination | 4,800 |
| filtered lifting and spectral sequences | 8,036 |
| smooth packet transport | 750 |
| source integrity | 15 |

All checks pass.

Ruff and mypy were not installed in the release container, so their command-line suites were not run. The release instead includes successful byte compilation, import-all validation, public docstring and annotation inspection, forbidden-import checks, and `git diff --check`. MkDocs and PowerShell were also unavailable; their source/configuration files are included but were not executed in this environment.

## Release boundary

The Block 9 finite API is complete enough for the `0.9.0a1` release. The broader package remains an alpha release because several arithmetic and global completion domains still expose conditional or partial research interfaces. This block does not convert those external hypotheses into executable theorems.
