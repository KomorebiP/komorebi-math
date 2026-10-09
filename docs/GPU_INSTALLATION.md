# Windows CUDA Installation

The base wheel requires only NumPy. CUDA PyTorch and CuPy are optional and must match the local NVIDIA runtime.

The historical Windows `0.16.0a2` qualification environment used Python 3.13,
CUDA 12.6, PyTorch `2.12.1+cu126`, CuPy `14.1.1`, and an RTX 3070 Ti Laptop GPU.
The commands below reuse that dependency line for a new local check. They do
not supply current-version hardware evidence until the GPU gate actually
passes. Build or obtain the current `4b21` wheel before installing it, following
the [user guide](USER_GUIDE.md#local-wheel-installation).

```powershell
conda create --name komorebi-full python=3.13 -y
conda activate komorebi-full
python -m pip install --upgrade pip
python -m pip install numpy==2.5.2 jax==0.11.1 cupy-cuda12x==14.1.1 numba==0.67.0 "llvmlite>=0.49,<0.50"
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install .\release\dist\komorebi_math-1.0.0b4-4b21-py3-none-any.whl
python -m pip check
```

Only one CuPy distribution should be installed. Remove `cupy` before installing `cupy-cuda12x` when both appear in `pip freeze`.

```powershell
python -m pip uninstall -y cupy cupy-cuda12x
python -m pip install cupy-cuda12x==14.1.1
```

Then run:

```powershell
.\Run_GPU_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

JAX remains CPU-backed on native Windows in this support policy. PyTorch and CuPy provide the recorded native Windows CUDA paths.
