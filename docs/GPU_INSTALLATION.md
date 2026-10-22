# CUDA installation

The base Komorebi Math wheel needs no GPU runtime. CUDA execution uses optional
PyTorch and CuPy installations that must be compatible with the NVIDIA driver
and the selected Python interpreter.

Use the [official PyTorch installer](https://pytorch.org/get-started/locally/)
to choose a CUDA-enabled build for the host. Install one CuPy distribution for
the matching CUDA line using the
[official CuPy instructions](https://docs.cupy.dev/en/stable/install.html).
For a configured CUDA 12.x environment, the CuPy package is `cupy-cuda12x`.
Do not install it alongside `cupy` or a second CUDA-specific CuPy package.

After those runtimes are available, install the local Komorebi wheel through
the same interpreter:

```powershell
python -m pip install .\dist\komorebi_math-1.0.0-py3-none-any.whl
python -m pip check
python -m komorebi_math doctor
python -m komorebi_math verify gpu
```

If using an existing environment, invoke its full `python.exe` path rather
than relying on an unrelated `python` on `PATH`. Package detection alone does
not prove that a driver, device or native runtime works.

The [GPU validation guide](GPU_VALIDATION.md) describes the actual computations.
Maintainers binding a result to exact release files should follow the
[release workflow](RELEASE_PROCESS.md#stress-and-hardware-qualification).
