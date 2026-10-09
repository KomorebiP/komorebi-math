# Optional Automatic Differentiation Backend Isolation

## Problem

Optional automatic differentiation adapters depend on native JAX and PyTorch runtimes. Python package discovery does not prove that those runtimes can load. DLL conflicts, incompatible builds, damaged installations, and native initialization errors can terminate the interpreter before Python raises an exception.

`pytest.importorskip()` cannot contain a native abort because no Python exception is produced.

## Availability contract

For a backend \(B\), Komorebi Math now reports availability only when

\[
A(B)=D(B)\land I(B)\land C(B),
\]

where

* \(D(B)\) means the package is discoverable
* \(I(B)\) means a child interpreter imports it successfully
* \(C(B)\) means the required directional JVP capability exists

For JAX, the required capability is `jax.jvp`. For PyTorch, it is `torch.func.jvp`.

## Runtime behavior

`ad_framework_diagnostics()` launches bounded child processes and records discovery, availability, version, return code, and a bounded diagnostic. `available_ad_frameworks()` returns only the resulting Boolean states and preserves its original no-argument signature.

Provider execution remains lazy. Before an in-process provider import is attempted, the isolated probe must pass. Python-level import failures are translated into `BackendUnavailableError`.

## Test behavior

The two original JAX and two original PyTorch derivative integration cases run
in dedicated child interpreters and check derivatives through order three.
The stress suite adds one CPU case per backend, comparing mixed derivatives
through order four against independent analytic formulas. Those cases also
check real and complex dtypes and reject nonfinite primal or derivative
evidence. A seventh case verifies CPU execution when PyTorch advertises CUDA
build support. It changes only the reported build metadata and does not test
a CUDA runtime. All seven integration cases retain child-process isolation.

The adversarial release test shadows PyTorch with a package that deliberately calls `os.abort()` during import. The parent process records the child return code, marks PyTorch unavailable, and completes the relevant test files with the two PyTorch checks skipped.

## Commands

```powershell
python .\verification\optional_ad_backend_status.py
```

Run the report through the same interpreter used for the publication gate.
After the [source preparation steps](USER_GUIDE.md#command-line-workflows-and-verification),
the current source wrapper is:

```powershell
.\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

The preflight report is diagnostic. An unavailable optional backend does not invalidate the base NumPy package. Repair the external environment when JAX or PyTorch support is specifically required.

For strict CPU integration qualification, install working JAX and PyTorch
runtimes, then run:

```powershell
python verification/stress_release_candidate.py --campaign optional_backends --skip-resources --output-directory release/LOCAL_OPTIONAL_STRESS
```

This replay requires all seven selected integration cases to execute without
skips. It reports a missing or unusable backend as a failed qualification,
whereas the ordinary source suite can skip those optional cases. The recorded
Linux CPU environment and its exact versions are in the
[stress audit](STRESS_AUDIT_2026_10_08.md).
