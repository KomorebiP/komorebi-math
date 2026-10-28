# Komorebi Math

[Read the documentation website](https://KomorebiP.github.io/Operation-KOMOREBI/math/index.html) · [First runs and expected output](https://KomorebiP.github.io/Operation-KOMOREBI/collection/FIRST_RUN.html) · [All Komorebi works](https://KomorebiP.github.io/Operation-KOMOREBI/)

> **Beta · Work in progress.** This library is available for research, evaluation and feedback. Its APIs, numerical behavior and documentation may change. Version 1.0.0 is still a beta release and is not ready for production. Check results for your intended use.

Komorebi Math is a Python library for finite operator, coordinate, signal and
arithmetic calculations. It turns the constructions of Realization Calculus
into reusable objects, algorithms and certificates you can inspect.

Results distinguish exact arithmetic from numerical checks and calculations
that depend on supplied assumptions. Those limits stay with a result when it
is displayed, saved or used in another calculation.

## Install

Use Python 3.11, 3.12 or 3.13. From a source checkout:

```bash
python -m pip install .
python -m komorebi_math verify stable
```

The base package needs NumPy. Install optional dependencies only when a
workflow needs them:

```bash
python -m pip install ".[exact,scientific,yaml]"
```

`exact` adds SymPy, `scientific` adds SciPy and mpmath, and `yaml` adds PyYAML.
JAX and PyTorch adapters have separate `jax` and `torch` extras. CuPy must match
the local CUDA installation. See the [user guide](docs/USER_GUIDE.md) for clean
environments, wheel installation and dependency details.

## A first calculation

A packet system decomposes a finite vector using a family of projectors. The
certificate checks idempotence, pairwise annihilation and completeness within
the specified numerical tolerance.

```python
import numpy as np
from komorebi_math.stable import PacketSystem

packets = PacketSystem(
    [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])],
    names=("left", "right"),
)
packets.certificate().require()

vector = np.array([2.0, 3.0])
coordinates = packets.analyze(vector)
np.testing.assert_array_equal(coordinates.reconstruct(), vector)
```

For exact rational coordinates, use integers and `Fraction` values:

```python
from fractions import Fraction
from komorebi_math.stable import ExactCoordinateTransform

transform = ExactCoordinateTransform(
    "A", "B", ((1, Fraction(1, 2)), (0, 1)),
    "EXAMPLE-SHEAR", "exact rational shear",
)
vector = (Fraction(2), Fraction(3))
mapped = transform.apply(vector)
assert mapped == (Fraction(7, 2), Fraction(3))
assert transform.inverse().apply(mapped) == vector
transform.certificate().require()
```

## Mathematical scope

| Area | Capabilities | Guide |
|---|---|---|
| Operators and packets | Finite realizations, coordinate analysis, support, transfer and refinement | [Unified language](docs/UNIFIED_LANGUAGE.md) |
| Coordinate calculus | Exact bases, generating functions, recurrences, tensor transport and finite-field maps | [Coordinate atlas](docs/GENERAL_COORDINATE_ATLAS.md) |
| Signals | Polyphase refinement, filters, resampling, streaming transforms and Fourier jump recovery | [Signal processing](docs/PRODUCTION_SIGNAL_PROCESSING.md) |
| Nonlinear calculus | Derivatives, packet jets, composition, sparse execution and streamed snapshots | [Nonlinear calculus](docs/NONLINEAR_CALCULUS.md) |
| Exact integration | Finite measures, polynomial moments, quadrature and typed finite expressions | [Integration](docs/FINITE_MEASURE_INTEGRATION.md) |
| Arithmetic and sequences | Modular domains, recurrence models, generating functions and local sequence comparisons | [Sequences](docs/sequences_oeis.md) |
| Numerical models | Positive quadratic-form tails, special functions, quantum matrices and fixed Dirac modes | [Worked examples](docs/WORKED_EXAMPLES.md) |

The [public API reference](docs/PUBLIC_API.md) describes the domain namespaces.
Each mathematical guide explains the algorithm's supported inputs, arithmetic
model and limits. Finite tests do not imply global convergence or unrestricted
infinite-dimensional theorems.

## API compatibility

Use one of these equivalent namespaces when your application needs the
1.x compatibility guarantees:

```python
import komorebi_math.stable as stable
import komorebi_math.v1 as v1
import komorebi_math.api.v1 as api_v1

assert stable.__all__ == v1.__all__ == api_v1.__all__
```

The stable API contains 131 reviewed symbols. Other domain APIs follow their
own documented compatibility rules. A saved certificate records a past result.
Rerun the operation to check it again. See the
[stable API policy](docs/STABLE_API.md) and
[persistence guide](docs/PERSISTENCE_SCHEMAS.md).

## Documentation and verification

Start with the [user guide](docs/USER_GUIDE.md), then use the
[worked examples](docs/WORKED_EXAMPLES.md) and
[API reference](docs/PUBLIC_API.md) for your domain.

```bash
python -m komorebi_math doctor
python -m komorebi_math api check --stable
python -m komorebi_math verify all
python -m komorebi_math release check
```

The complete `verify all` suite requires the `exact`, `scientific` and `yaml`
extras shown above. These commands check the installed package. The
[release guide](docs/RELEASE_PROCESS.md) explains how maintainers can repeat
checks on the source, distributions and supported platforms.

## License

Original software and documentation are available under the [MIT License](LICENSE.md).
Bundled third-party components retain their notices. Separately supplied OEIS
or other external data retain their own terms.
