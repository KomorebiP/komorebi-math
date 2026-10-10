# Papers V–VIII: runtime coverage and finite stress machinery

This audit compares the authoritative traditional final PDFs with callable Python
machinery. The machine-readable `verification/paper_coverage/middle.json`
has 67 mechanisms, 200 resolvable API references, named test files, and explicit
boundaries. Every source PDF is pinned by SHA-256 below. A matching API name or a
successful finite computation does not implement all hypotheses or conclusions of
an infinite-dimensional theorem. No Lean build or historical GPU corpus was rerun
as part of this audit.

| Source | SHA-256 |
| --- | --- |
| V, *Multiresolution Signal Calculus* | `59110c60fcc01564a2c310dd7f5c0b49c50f89181fe84286ca9bb79d7ca251a5` |
| VI, *Transfer Semigroups and Memory* | `c175aea8db4809483d281a9712b1a988489446a6bac7380ab9c95d07610ea52a` |
| VII, *Quantum Seeds and Effective Dynamics* | `b9b90e811e5de67908b8df8410e0efc20daf620155cbe3dfb2bbed35aa32178a` |
| VIII, *Packet Operator Algorithms* | `a13bdaddb95ef19b62c50c5453268bcf11f234abcd0c6ad7e3666dccafa58730` |

The source copies are `../paper_sources/Paper_{V,VI,VII,VIII}_Traditional_Final.pdf`
relative to this checkout. Sections refer to the documents, including their
extended parts; they are not inferred from module names. `implemented` means an
executable realization exists for the finite domain described by that entry.
`partial` identifies a narrower domain, missing hypotheses, numerical rather than
exact verification, or absent components. `missing` identifies computable
machinery without a corresponding API. `theory_only` identifies stated open
extensions or criteria that a finite test cannot settle.

## What is already present

| Paper | Existing machinery | Material boundary |
| --- | --- | --- |
| V | Residue/FIR coordinates, finite banks and lifting, Haar packet trees, retained singular packets, mask and overlap errors, separated-jump recovery, nonlinear residue products, Smith charts | Lattice phases are rectangular; filter-bank checks concern finite matrices; recovery requires supplied regularity, separation and noise hypotheses. |
| VI | Matrix/packet semigroups, Duhamel and resolvent defects, Markov lumping, conditional expectations, Bayesian sufficiency and filtering, fractional powers, interval Green operators, finite Feynman–Kac, Feshbach memory and phase-type survival | Matrix and quadrature evaluations are numerical. Finite prefixes do not certify infinite tails, closure of a reconstruction range, or unbounded generator domains. |
| VII | Seed ranks, central symmetry projectors and cyclic charges, circuit path interference, Kato transport, Kraus/Choi and Lindblad channels, subspace QEC, quantum Feshbach reduction, reflection-positive finite transfer, lattice heat traces, constraint quotient defects, finite Hamiltonian variation | Finite Gram positivity and spectral-dimension curves do not establish a continuum field, Einstein action, or spin-two sector. Increasing Hilbert-space direct limits are broader than the common-space finite tower API. |
| VIII | Packet IR and cost models, structured primitives, BDLR algebra and recompression, support graphs, Krylov/preconditioning, precision programs, finite additive tree compilers, schedules, amortization and benchmark contracts | Compiler optimality applies to its finite additive model. Support inclusion permits cancellation. CPU correctness does not qualify hardware performance. |

Three convention differences deserve explicit treatment when comparing runtime
results with the PDFs. Feynman–Kac uses column-state forward matrices and backward
`Q.T - diag(V)`, with current-state killing in the discrete operator.
`QuantumFeshbachReduction.memory_kernel()` uses
`-B exp(-i t D) B*` in the derivative equation `psi_dot = -i H psi`; the paper's
positive oscillatory kernel appears in its `i dpsi/dt` convention. Bayesian
mass-only sufficiency with at least two blocks differs from E9.2's stronger
requirement that also preserves conditional distributions within every block.

## Added exact Gaussian polynomial machinery

`dynamics.hermite` implements the real rational polynomial sector of VI E7.1–E7.4.
It supplies standard Gaussian moments, polynomial inner products and variances,
the OU generator `f'' - x f'`, and exact Gaussian transfer at a rational
correlation `rho` in `[0,1]`:

```python
from fractions import Fraction
from komorebi_math.dynamics.hermite import (
    gaussian_transfer_polynomial, ornstein_uhlenbeck_generator,
    gaussian_polynomial_variance,
)

f = (0, 0, 1)  # ascending coefficients: x**2
assert gaussian_transfer_polynomial(f, Fraction(1, 2)) == (
    Fraction(3, 4), Fraction(0), Fraction(1, 4)
)
assert ornstein_uhlenbeck_generator(f) == (
    Fraction(2), Fraction(0), Fraction(-2)
)
assert gaussian_polynomial_variance(f) == 2
```

The transfer is computed from independent monomial Gaussian moments. Tests reuse
the existing probabilists' Hermite polynomial primitive to check orthogonality
and eigenaction, rather than introducing a second Hermite implementation.
`rho = exp(-t)` translates the paper convention; an arbitrary exponential is not
silently rounded into an exact rational claim. Floats, complex coefficients,
invalid correlations and resource-limit truncations are rejected. Infinite
Hermite completeness, strong reconstruction, general Gaussian `Lp` actions and
fractional infinite-series convergence remain theorem-level boundaries.

## Added independent tree-compiler oracle

`algorithms.compiler_audit` enumerates every admissible pruning under explicit
node, depth and pruning limits. It sums all split overheads and leaf costs/errors
as exact fractions of the supplied binary floats. It then independently checks
Lagrangian or quantized-budget optimality and the relevant plan metadata. It never calls
the optimized dynamic-programming recurrence and never trusts a plan's attached
certificate. An exceeded limit raises an error; it cannot return an optimum from
an incomplete enumeration.

VIII's unsupported Pareto example illustrates why the budget program matters:

```python
from komorebi_math.algorithms.compiler import TreeNode, compile_tree_budget
from komorebi_math.algorithms.compiler_audit import audit_tree_budget

a1 = TreeNode("a1", keep_cost=3, keep_error=1)
a2 = TreeNode("a2", keep_cost=1, keep_error=0)
a = TreeNode("a", keep_cost=2, keep_error=4, children=(a1, a2))
b = TreeNode("b", keep_cost=1, keep_error=0)
root = TreeNode("r", keep_cost=1, keep_error=6, children=(a, b))
plan = compile_tree_budget(root, 4)
audit = audit_tree_budget(root, plan, 4)
assert plan.leaves == ("a", "b")
assert audit.prunings_examined == 3
assert audit.certificate.valid
```

The three `(cost, error)` pairs are `(1,6)`, `(3,4)`, and `(5,1)`. Keeping
`a,b` is the least-cost budget-four solution, although no Lagrangian
penalty makes it optimal. Quantized weights are summed after rounding **each
retained leaf upward**, not by rounding their total. Exact dyadic comparison
also distinguishes costs that round to the same large floating-point objective.
The oracle certifies only the supplied enumerated model; it supplies no general
optimality claim for unbounded trees, overlapping decompositions or nonlocal
hardware costs.

## Remaining computable gaps

| Source section | Missing or narrower runtime mechanism |
| --- | --- |
| V E3 | General integer-matrix sublattice cosets and nested lattice transfer. Existing axiswise rectangular phases are a special case. |
| V E5; E8.3; Part I §13 | Uniform frequency-symbol frame bounds; the sharp integrated higher-order location formula; projective infinity/homogeneous Smith coordinates. |
| VI E2, Part I §15, E3.4, R7 | General Dyson path-simplex evaluation and canonical-jet/holomorphic contour substitution with branch and zero-free-domain contracts. Special-case exponentials/resolvents do not supply this whole calculus. |
| VI Part I 10.3, E11.6–E11.8 | Gapless fractional Holder-bound helper and general subordination measures. Current fractional-defect bounds require a common positive spectral gap. |
| VI E15.1–E15.4; R5.2 | Primary Jordan mode extraction and largest certified-weight compression; construction of a realization from an arbitrary rational matrix function. Supplied `B,D,C` models are supported. |
| VII II.3.2, II.6.7–II.6.10, II.8.9 | Nonabelian representation matrix units/fusion intertwiners; multiparameter curvature and Chern integrals; arbitrary correctable observable algebras. Central projectors, one-parameter transport and subspace KL correction are narrower. |
| VII II.9.5–II.9.6, II.9.9, II.9.12 | Uniform spectral-gap/stability-bound utilities and positive oscillatory-memory Gram construction. Pointwise self-energy evaluation is insufficient. |
| VIII Appendix E.2; Theorem 9.12 | Stagewise block-triangular forward solver and cached subtree/ancestor-only recompilation. Graph levels and replacement plus full recompilation exist. |
| VIII Standard Fact 9.1; Theorem 9.11 | General joint-eigenspace discovery; a decision-margin reporter consistent with exact dyadic comparisons near floating ties. Current margin scores use floats. |

These gaps are recorded as gaps, not filled by aliases or by success on a small
example. The infinite, continuum, domain-control and open research boundaries are
listed separately in the crosswalk.

## Validation and negative controls

The new suite has twelve tests. It compares 1,140 optimized compiler results with
complete exact enumeration on 60 deterministic generated trees, tests unsupported
Pareto points, exact integer-boundary quantization, alternative optimal ties,
forged plans/certificates, float objective ties and resource limits. Gaussian
tests independently verify moments through order 79, all pairwise Hermite inner
products through degree 18, generator/transfer eigenaction, and exact semigroup,
mean, Dirichlet, Poincare and spectral-gap identities on 40 deterministic rational
polynomials. Deliberately invalid arithmetic and degree truncation are rejected.

The related recovery, dynamics, quantum, algorithms and new machinery test run
passed **174 tests**. An expanded run covering all 31 named crosswalk test files
passed **284 tests**. Source hashes, every named test path and all 200 qualified
API references were also checked. This is reproducible finite evidence, not a
universal proof or a fresh GPU benchmark.

```sh
PYTHONPATH=src python -m pytest -q \
  tests/test_signals_recovery.py tests/test_dynamics_*.py \
  tests/test_quantum_*.py tests/test_algorithms_*.py \
  tests/test_paper_middle_machinery.py
```

To run every test file named in this crosswalk:

```sh
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
import pytest
coverage = json.loads(Path("verification/paper_coverage/middle.json").read_text())
paths = sorted({p for entry in coverage["mechanisms"] for p in entry["tests"]})
raise SystemExit(pytest.main(["-q", *paths]))
PY
```
