# GPU Validation

GPU support is optional and excluded from the base NumPy dependency set.
Package correctness and hardware qualification are separate checks.

For a new candidate, use the artifact-bound
[Windows hardware handoff](STABLE_RELEASE_CANDIDATE.md#windows-cuda-handoff).
It installs the exact candidate wheel in a temporary environment, records
source and artifact identity, and retains both successful and failed CUDA
results. The older commands below remain useful for diagnosing the package
already installed in an interpreter, but do not supply that candidate binding.

## Historical Windows configuration

The committed `validation_windows_0_16_0a2.json` record includes the
user-executed configuration for package version `0.16.0a2`:

- Windows 11 x86-64
- Python 3.13.0
- NumPy 2.5.2
- JAX 0.11.1 on CPU
- PyTorch 2.12.1+cu126 with CUDA 12.6
- CuPy 14.1.1 for CUDA 12.x
- NVIDIA GeForce RTX 3070 Ti Laptop GPU
- 1,009 source tests passed, installed-wheel qualification passed, and 4/4 GPU
  verification checks passed

This record does not qualify the subsequent `1.0.0b4` repairs or other hardware.
New qualification must identify the source revision and installed artifact
actually tested.

## PowerShell gate

```powershell
.\Run_GPU_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe" -Output release\CURRENT_GPU_QUALIFICATION.json
```

Use the intended interpreter's full path when Conda is not active. The wrapper
runs both commands below against that interpreter's installed package:

```powershell
python -m komorebi_math release qualify --require-gpu --output .\release\CURRENT_GPU_QUALIFICATION.json --force
python -m komorebi_math verify gpu
```

The wrapper's default output filename still contains `BLOCK16` for compatibility;
the explicit output above distinguishes a new run. It does not run final
platform readiness. That separate condition is requested with
`python -m komorebi_math release readiness --final --require-gpu`.

## Checked computations

The two isolated backend workers check:

1. `torch.cuda.is_available()`
2. a cubic directional JVP at `2` on the default CUDA device, returning value
   `8` and tangent `12`
3. a CUDA matrix-vector product with expected result `[14, 38, 62, 86]`
4. at least one CuPy CUDA device
5. device-resident `PacketIR` execution through the CuPy backend
6. the expected vector `[0, 1, 4, 6]` with Euclidean residual norm at most `1e-12`

## Isolation

Native frameworks can fail below Python's exception boundary. The GPU gate therefore executes PyTorch and CuPy in separate child interpreters. A timeout, native abort, access violation, import error, or nonzero worker exit becomes a failed qualification record while the parent process remains alive.

## CuPy package rule

Install exactly one CuPy distribution appropriate for the CUDA line. For CUDA 12.x on the validated Windows configuration, that distribution is `cupy-cuda12x`. Do not install both `cupy` and `cupy-cuda12x` in the same environment.
