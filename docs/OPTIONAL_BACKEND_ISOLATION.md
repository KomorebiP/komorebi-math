# Optional derivative backends

JAX and PyTorch adapters are loaded lazily. A package being discoverable does
not prove that its native runtime can import or provide the required derivative
operation. DLL conflicts and native initialization failures can terminate an
interpreter before Python raises an exception.

## Availability contract

For a backend \(B\), availability means

\[
A(B)=D(B)\land I(B)\land C(B),
\]

where \(D\) is package discovery, \(I\) is successful import in a bounded child
interpreter, and \(C\) is the required directional JVP capability. The adapter
uses `jax.jvp` or `torch.func.jvp` respectively.

`ad_framework_diagnostics()` reports discovery, availability, version, return
code and a bounded diagnostic. `available_ad_frameworks()` returns the resulting
Boolean states. An isolated probe must succeed before a provider attempts its
in-process import. Python-level failures become `BackendUnavailableError`.

## Diagnose the active environment

```bash
python verification/optional_ad_backend_status.py
python -m komorebi_math doctor
```

Use the same interpreter that will execute the calculation. Install the
`jax`, `torch` or `autodiff` extra only when that adapter is needed. CPU backend
availability does not imply CUDA availability.

## Reproduce the integration checks

With working CPU JAX and PyTorch runtimes:

```bash
python verification/stress_release_candidate.py --campaign optional_backends --skip-resources --output-directory qualification/optional-backends
```

All seven selected cases must execute without skips in this strict campaign.
The integration checks compare derivatives through order three or four with
analytic controls, exercise real and complex dtypes, reject nonfinite evidence
and check CPU execution when a PyTorch build advertises CUDA support. That last
case changes reported build metadata; it does not execute a GPU.

The ordinary source suite can skip unavailable optional backends. The strict
campaign instead reports a missing or unusable backend as failed qualification.
Both successful and failed records belong with the tested source identity and
dependency versions. See the [release guide](RELEASE_PROCESS.md).
