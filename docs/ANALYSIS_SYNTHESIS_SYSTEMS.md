# Finite Analysis--Synthesis Systems

## Definition

Let

\[
A\in\mathbb{F}^{m\times n},
\qquad
S\in\mathbb{F}^{n\times m}.
\]

`AnalysisSynthesisSystem` represents the checked finite law

\[
SA=I_n.
\]

The square case \(m=n\) is an invertible coordinate transform. The redundant case \(m>n\) is supported when \(A\) has full column rank and \(S\) is a left inverse.

## Construction

```python
import numpy as np
from komorebi_math.stable import AnalysisSynthesisSystem

square = AnalysisSynthesisSystem.from_invertible(
    np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
)

redundant = AnalysisSynthesisSystem.canonical_dual(
    np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
)

square.certificate().require()
redundant.certificate().require()
```

`canonical_dual()` computes the Moore--Penrose left inverse for a full-column-rank analysis matrix. Rank-deficient input is rejected.

## Components versus projectors

For a coordinate index set \(I\), define

\[
C_I=SE_IA,
\]

where \(E_I\) is the diagonal coordinate selector. A complete coordinate partition satisfies

\[
\sum_I C_I=I_n.
\]

In the square invertible case, \(AS=I_m\), so the components are mutually annihilating projectors. In a redundant frame, generally \(AS\neq I_m\). The component law still reconstructs, but individual terms need not satisfy \(C_I^2=C_I\).

```python
import numpy as np
from komorebi_math.stable import AnalysisSynthesisSystem

redundant = AnalysisSynthesisSystem.canonical_dual(
    np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
)

components = redundant.components()
assert np.allclose(sum(components), np.eye(redundant.source_dimension))
```

Use `component()` and `components()` in new code. `projector()` and `projectors()` are compatibility aliases and do not independently certify idempotence for a redundant system.

## Transfer blocks

Given an operator \(T\) on the source space,

\[
T=\sum_{\beta,\alpha}C_{\beta}TC_{\alpha}.
\]

```python
import numpy as np
from komorebi_math.stable import AnalysisSynthesisSystem

redundant = AnalysisSynthesisSystem.canonical_dual(
    np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
)

operator = np.array([[2.0, -1.0], [3.0, 4.0]])
blocks = redundant.operator_blocks(operator, groups=((0, 2), (1,)))
blocks.certificate().require()
assert np.allclose(blocks.reconstruct(), operator)
```

## Persistence

```python
import numpy as np
from komorebi_math.stable import AnalysisSynthesisSystem

redundant = AnalysisSynthesisSystem.canonical_dual(
    np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
)

from komorebi_math.stable import dumps_portable, loads_portable

restored = loads_portable(dumps_portable(redundant))
restored.certificate().require()
```

The codec validates the left-inverse law when reading the record. A portable payload cannot silently reconstruct an invalid analysis--synthesis system.
