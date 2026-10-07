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

The two JAX and two PyTorch derivative integration cases run in dedicated child interpreters. This preserves the mathematical checks through order three while preventing a native failure from terminating the parent pytest process.

The adversarial release test shadows PyTorch with a package that deliberately calls `os.abort()` during import. The parent process records the child return code, marks PyTorch unavailable, and completes the relevant test files with the two PyTorch checks skipped.

## Commands

```powershell
.\Run_Block15_Source_Validation.ps1
```

```powershell
python .\verification\optional_ad_backend_status.py
```

The preflight report is diagnostic. An unavailable optional backend does not invalidate the base NumPy package. Repair the external environment when JAX or PyTorch support is specifically required.
