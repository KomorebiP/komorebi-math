# Komorebi Math Block 16 Handoff

This handoff contains the `0.16.0a2` 1.0 freeze candidate.

## Fast Windows validation

```powershell
.\Run_Block16_Source_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe" -Focused
.\Install_And_Verify_Block16.ps1 -Python "$env:CONDA_PREFIX\python.exe" -Recreate
```

Omitting `-Focused` runs the complete suite one test module per child interpreter and writes a machine-readable ledger. Use `-Monolithic` only when a single-process run is specifically needed. `Run_Block16_Batched_Validation.ps1` provides the explicit module-isolated release gate.

## GPU qualification

```powershell
.\Run_GPU_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

GPU qualification requires a working CUDA PyTorch build and exactly one compatible CuPy distribution. The base wheel remains NumPy-only.

## Stable contract

```powershell
komorebi api check --stable
komorebi release contract
komorebi release environment
komorebi verify stable
komorebi verify release
komorebi verify block16
komorebi release qualify
```

See `docs/BLOCK_16_RELEASE.md` for the mathematical and release architecture.

## Distribution build

```powershell
.\Build_Block16_Distributions.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

This builds the wheel and source distribution, then reconstructs the wheel from the source distribution and compares normalized payloads.

## Companion C crosswalk

The reviewed Companion C overlap is recorded in `verification/COMPANION_C_BLOCK16_CROSSWALK.json`. New stable promotions are limited to package-native interfaces with certificates, deterministic tests, and a persistence decision.
