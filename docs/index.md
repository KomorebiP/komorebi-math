# Komorebi Math

Komorebi Math provides finite mathematical constructions as Python objects,
algorithms and certificates. Its common language connects operator realizations,
coordinates, packet decompositions, refinement, arithmetic and numerical models.

## Start a calculation

The [user guide](USER_GUIDE.md) covers installation, the first programs and the
meaning of a certificate. The [worked examples](WORKED_EXAMPLES.md) develop
complete calculations with independent controls. The
[public API reference](PUBLIC_API.md) lists the available domain interfaces.

## Choose a mathematical guide

| Goal | Guide |
|---|---|
| Construct operators, seeds and packets | [Unified language](UNIFIED_LANGUAGE.md) |
| Change bases or transport finite data | [Coordinate atlas](GENERAL_COORDINATE_ATLAS.md) |
| Analyze and synthesize coordinates | [Analysis and synthesis](ANALYSIS_SYNTHESIS_SYSTEMS.md), [redundant frames](REDUNDANT_FRAME_ALGEBRA.md) |
| Process finite or streamed signals | [Signal processing](PRODUCTION_SIGNAL_PROCESSING.md), [Fourier recovery](fourier_recovery.md) |
| Compute derivatives and nonlinear packet jets | [Nonlinear calculus](NONLINEAR_CALCULUS.md), [scalable execution](SCALABLE_NONLINEAR_EXECUTION.md) |
| Integrate polynomials or compare exact expressions | [Finite integration](FINITE_MEASURE_INTEGRATION.md), [typed expressions](TYPED_FINITE_EXPRESSION_CALCULUS.md) |
| Work with recurrences and generating functions | [Sequences](sequences_oeis.md), [generating functions](NORMALIZED_GENERATING_FUNCTIONS.md) |
| Bound positive quadratic-form tails | [Quadratic forms](quadratic_forms.md) |
| Evaluate the supported special-function families | [Special functions](special_functions.md) |
| Form finite quotients | [Join semilattices](finite_join_semilattice.md) |
| Study finite quantum models | [Quantum and memory contracts](QUANTUM_MEMORY_NUMERICAL_CONTRACTS.md), [Dirac modes](dirac_modes.md) |

## Interpret and preserve results

A certificate records validity, arithmetic class, residuals and assumptions.
Exact, numerical and conditional results remain distinct. Persisted records
retain their original evidence, while fresh validity requires revalidation.
Read the [persistence guide](PERSISTENCE_SCHEMAS.md),
[export guide](EXPORT_AND_LATEX.md) and [stable API contract](STABLE_API.md)
when connecting calculations or exchanging results.

## Develop and reproduce

The [platform policy](SUPPORTED_PLATFORMS.md) distinguishes required dependencies
from optional runtimes. The [release guide](RELEASE_PROCESS.md) gives the
maintainer workflow for source tests, strict documentation, clean installations,
reproducible distributions and platform qualification.
