# Finite predictive foundations

`komorebi_math.signals.foundations` implements five exact rational matrix
constructions from *Komorebi Large Signal Model: Foundations of realization,
refinement and predictive closure*, edition 0.8.0, 9 October 2026. It is a
research module imported by its full module path. Install the `exact` optional
dependency to execute it. Importing the module does not import SymPy.

The input domain is integers and exact rational numbers, including
`fractions.Fraction` and SymPy rational matrices. Float values, Boolean values,
strings, irrational symbolic numbers and free symbols are rejected. The module
does not convert rounded measurements into exact mathematical evidence.

| Construction | API | Source | What the result establishes |
|---|---|---|---|
| Compression obstruction | `compression_closure` | Section 5.1, Proposition 7 | The exact supplied-matrix defect `PL(I-P)` and the return-path composition identity |
| Minimal closed refinement | `minimal_predictive_completion` | Section 6, Proposition 8 | The smallest operator-invariant row space containing the supplied summary, after successful exact termination |
| Projection memory | `projected_memory` | Section 8, Proposition 10 | Exact finite kernels `BD**k C`, initial forcing `BD**t`, and evaluation of the memory formula |
| Causal delay closure | `delay_closure` | Section 15, Proposition 14 | Closure of the observation map of the supplied autonomous operator at the declared delay |
| Data-span identification | `identify_data_span` | Section 16, Proposition 15 | Exact fit consistency and sampled-span closure, with unknown global coverage explicitly unestablished |

## A complete exact example

The known reference dynamics have a two-coordinate interacting block and an
independent third coordinate. Retaining the first coordinate alone omits a
return path with coefficient `1/4`.

```python
from fractions import Fraction as F
from komorebi_math.signals.foundations import (
    compression_closure, delay_closure, identify_data_span,
    minimal_predictive_completion, projected_memory,
)

L = [[F(1, 2), 1, 0], [F(1, 4), F(1, 3), 0], [0, 0, F(5, 6)]]
P = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
defect = compression_closure(L, P)
assert not defect.closed
assert defect.composition_defect[0, 0] == F(1, 4)

completion = minimal_predictive_completion([[1, 0, 0]], {"L": L})
completion.require_complete()
assert completion.basis.rows == 2
# Its row space contains the first two coordinates, not the third.

known_delay = delay_closure(L, [[1, 0, 0]], delay=2)
D = known_delay.require_transition()
assert list(D) == [0, 1, F(1, 12), F(5, 6)]

# Fit only observed windows, without passing L to the fitting routine.
fit = identify_data_span(
    [[1, F(1, 2)], [F(1, 2), F(1, 2)]],
    [[F(1, 2), F(1, 2)], [F(1, 2), F(11, 24)]],
)
assert fit.require_closed_span() == D
assert fit.global_coverage == "not_established"
assert (fit.require_fit() * fit.successor_states[:, -1])[-1] == F(61, 144)

memory = projected_memory(L, P, horizon=2)
next_state = memory.next_state(
    [[1, 0, 0], [F(1, 2), 0, 0], [F(1, 2), 0, 0]],
    [0, 0, 0],
)
assert next_state[0] == F(11, 24)
```

The executable version is `examples/finite_predictive_foundations.py`.

## Certificates and obstructions have different meanings

`CompressionClosure.certificate` certifies the exact defect calculation and
composition identity. It can be valid while `closed` is false. Calling
`require_closed()` explicitly requires zero closure defect. Changing a
diagnostic dictionary cannot change that decision.

`PredictiveCompletion.certificate` is valid only after exact invariance and
retention of the original summary are established. The default round limit is
the ambient dimension minus initial rank plus one. An explicit smaller
`max_rounds` can return a partial basis, `complete=False`, and an invalid
certificate. `dimension_chain`, `rounds` and `round_limit` report the bounded
construction. `require_complete()` rejects an incomplete result. The routine
does not establish that newly added observables can actually be measured.

`DelayClosure.certificate` is valid only when the supplied known observation
maps have equal ranks at delays `h` and `h+1`. Its transition is an ambient
extension, unique only on the realizable observation-image space. Redundant
output coordinates are allowed. A rank plateau in a finite recorded trajectory
is a different statement and is not accepted by this API.

`DataSpanIdentification.certificate` validates a fit on the sampled input span.
`require_closed_span()` also requires successors to remain in that span.
`ambient_span_complete` means the samples span the coordinate space supplied
to this computation. It does not prove that an unknown generating system has
no additional state. `global_coverage` always returns `"not_established"`.
Both true-process coverage and the correctness of supplied successors remain
unchecked assumptions. A finite fit is never promoted to universal discovery.

Projection memory evaluates exact finite kernels on supplied retained-history
vectors and an initial unresolved vector. It checks both subspace memberships.
It does not attach the proposition's conditional geometric tail bound without
the required norm contraction and history bounds. It also does not turn a
tail estimate evaluated on the true history into a stable approximate rollout.

Matrices and retained mappings are immutable snapshots. Certificate diagnostics
remain descriptive metadata. Exact residuals use Python rational values, so a
nonzero obstruction below floating-point range remains nonzero.

The implementation rejects input matrices above 65,536 entries, projection
memory above 10,000 steps or one million stored entries, and delay observation
maps, ambient delay transitions or fitted ambient operators above 65,536
entries. Refinement permits at most 1,024 declared operators and one million
entries in its worst-case stacked basis. Those are explicit implementation bounds, not bounds
on the mathematical statements. There is no claim of practical complexity for
large rational denominators.

## Coverage beyond this module

The source has 87 propositions. Five constructions here do not implement the
whole volume. Existing library packet systems, transfer operators, nonlinear
packet jets, Haar best-basis selection and Bayesian partitions already supply
related finite mathematics for Propositions 3--5 and 12. Their exact or numerical
contracts remain those of their own APIs.

The 0.8.0 source archive contains separately maintained controlled-instrument,
finite positive-lift, curved matrix/Gram, quadratic/Bernstein, continuous-moment,
structured confidence and repair reference checkers. They have not been
replaced by these five primitives or silently made stable Python APIs.
Statistical coverage assumptions, continuous observation measures,
infinite-dimensional convergence, hidden-state/dictionary discovery and
general geometry optimization remain separately stated source boundaries.

`verification/paper_coverage/signal_machine.json` records the source hashes and
the distinction between integrated finite code, existing related code,
separate reference implementations and mathematical statements without a
general implementation. The focused tests include direct full-state memory
comparisons, all 81 two-by-two matrices over `{-1,0,1}`, negative model controls,
resource limits and mutation controls. These are finite checks of the code.
