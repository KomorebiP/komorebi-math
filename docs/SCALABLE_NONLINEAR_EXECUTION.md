# Scalable nonlinear execution

## 1. Directional derivative contract

Every Block 10 provider implements the same finite directional interface. At a point `x` and directions `v_1, ..., v_r`, the call

```python
provider.derivative(r, x, (v_1, ..., v_r))
```

returns the ordered directional derivative

\[
D^rF(x)[v_1,\ldots,v_r].
\]

The protocol does not require construction of a dense Jacobian or higher-order tensor. Dense polynomial, sparse polynomial, finite-difference, declared callable, SymPy, JAX, and PyTorch implementations can therefore be consumed by the same packet-jet machinery.

## 2. Sparse symmetric tensors

A dense order-`r` derivative tensor has shape

```text
(target_dimension, source_dimension, ..., source_dimension)
```

with `r` source axes. For a symmetric derivative, coefficients related by a permutation of the source axes are equal. `SparseSymmetricTensor` stores canonical nondecreasing source-index tuples and expands their distinct permutations only when reconstructing dense data or contracting with directions.

```python
from fractions import Fraction
from komorebi_math.nonlinear import SparseSymmetricTensor

tensor = SparseSymmetricTensor.from_mapping(
    order=3,
    source_dimension=3,
    target_dimension=1,
    values={(0, 0, 1, 2): Fraction(7, 11)},
)
tensor.certificate().require()
```

For an entry indexed by `(o, i_1, ..., i_r)`, contraction computes the sum over the full distinct permutation orbit. Repeated indices are handled without multiplicity errors.

### Storage diagnostics

The object reports:

* `stored_entry_count`
* `expanded_nonzero_count`
* `dense_entry_count`
* `storage_ratio`

These are deterministic counts. They do not include Python object overhead, allocator behavior, cache effects, or backend-specific memory layouts.

## 3. Framework-neutral AD

`AutomaticDifferentiationProvider` accepts a JVP function with the form:

```python
def jvp(function, point, direction):
    return primal_value, directional_tangent
```

Nested JVPs provide higher-order directional derivatives. The generic adapter is useful for custom dual-number systems, local differentiable programming runtimes, or finite validation JVPs.

```python
import numpy as np
from komorebi_math.nonlinear import AutomaticDifferentiationProvider

def function(x):
    return np.array([x[0] ** 3 + x[0] * x[1]])

def central_jvp(function, point, direction):
    h = 1e-6
    return function(point), (function(point + h * direction) - function(point - h * direction)) / (2 * h)

provider = AutomaticDifferentiationProvider(function, central_jvp, 2, 1, max_order=1)
```

The accuracy and differentiability properties of a custom JVP remain the caller's responsibility. A finite-difference JVP is numerical approximation, not automatic differentiation.

### JAX and PyTorch

```python
from komorebi_math.nonlinear import JaxDerivativeProvider, TorchDerivativeProvider
```

Both adapters are lazy. Merely importing these names does not import the optional framework. The provider validates source and target dimensions and guards the declared maximum order.

Framework tracing rules still apply. A function that escapes to NumPy, mutates traced values, or performs unsupported control flow can fail in the underlying framework. Komorebi Math preserves and reports that failure rather than silently replacing the derivative.

## 4. Packet-jet plan compilation

A packet-jet family combines a provider, a base point, input packets, output packets, and a finite order. The naive address count is

\[
N_{\mathrm{out}}N_{\mathrm{in}}^r.
\]

Plan compilation performs two independent reductions:

1. **Certified support pruning.** A complete support certificate may prove some packet addresses zero.
2. **Input-group reuse.** Addresses with the same ordered input packet tuple share the same projected directions and one provider derivative call.

```python
support = packet_jet_support(family)
plan = compile_packet_jet_plan(family, support)
result = execute_packet_jet_plan(plan, family, directions)
result.certificate.require()
```

An incomplete or absent support certificate never justifies pruning. This conservative rule prevents finite probing from being promoted into an exact structural zero claim.

## 5. Execution certificates

The batch result records:

* active packet evaluations
* reconstructed derivative
* direct derivative
* reconstruction residual
* provider derivative-call count
* naive derivative-call count
* number of pruned blocks
* exactness inherited from the provider and packet system

The certificate validates the finite identity

\[
D^rF(x)[v_1,\ldots,v_r]
=
\sum_{b,a_1,\ldots,a_r}
Q_bD^rF(x)[P_{a_1}v_1,\ldots,P_{a_r}v_r]
\]

within the declared tolerance for numerical data.

## 6. CuPy execution boundary

The CuPy classes accept device-aware providers and packet data when a compatible runtime is present. `cupy_runtime_status()` returns a structured status before construction. No implicit CPU fallback is used by a class whose contract promises CuPy execution.

A matched benchmark contract can describe the case signature and timing scope. It does not itself execute a benchmark or certify a speedup.

## 7. Complexity boundaries

Sparse storage is beneficial only when derivative tensors contain structural zeros or repeated symmetric coefficients. Projection caching reduces repeated packet projections and provider calls, but the output projection and reconstruction costs remain. Snapshot capture is exponential in derivative order at fixed source dimension. These APIs expose finite structure and reuse. They do not remove the general combinatorial growth of high-order derivatives.
