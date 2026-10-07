# GPU Validation

GPU support is optional and excluded from the base NumPy dependency set. Block 16 distinguishes package correctness from hardware qualification.

## Validated Windows configuration

The committed support record includes the user-executed configuration:

- Windows 11 x86-64
- Python 3.13.0
- NumPy 2.5.2
- JAX 0.11.1 on CPU
- PyTorch 2.12.1 with CUDA 12.6
- CuPy 14.1.1 for CUDA 12.x
- NVIDIA GeForce RTX 3070 Ti Laptop GPU
- 960 source tests passed with both automatic-differentiation adapters available

This is recorded evidence for that configuration, not a universal hardware claim.

## PowerShell gate

```powershell
.\Run_GPU_Validation.ps1
```

or directly:

```powershell
python -m komorebi_math release qualify --require-gpu --output .\release\BLOCK16_GPU_QUALIFICATION.json --force
```

## Checked computations

The two isolated backend workers check:

1. `torch.cuda.is_available()`
2. a cubic directional JVP on `cuda:0`
3. at least one CuPy CUDA device
4. device-resident `PacketIR` execution through the CuPy backend
5. the expected vector `[0, 1, 4, 6]` with maximum absolute residual at most `1e-12`

## Isolation

Native frameworks can fail below Python's exception boundary. The GPU gate therefore executes PyTorch and CuPy in separate child interpreters. A timeout, native abort, access violation, import error, or nonzero worker exit becomes a failed qualification record while the parent process remains alive.

## CuPy package rule

Install exactly one CuPy distribution appropriate for the CUDA line. For CUDA 12.x on the validated Windows configuration, that distribution is `cupy-cuda12x`. Do not install both `cupy` and `cupy-cuda12x` in the same environment.
