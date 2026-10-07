# Komorebi Math 1.0.0b1 — Block 17 Public Beta

Block 17 is the final major engineering block before the release-candidate sequence. It does not add another mathematical domain. It freezes the public 1.0 contract, proves compatibility with the supported alpha predecessors, binds platform evidence to digest-checked records, and makes the remaining route to `1.0.0` explicit.

## Release position

```text
0.16.0a2  ->  1.0.0b1  ->  1.0.0rc1  ->  1.0.0
```

`1.0.0b1` is a public beta. The stable namespace is frozen, but the release-candidate gate remains closed until this exact beta is executed on every required platform family. The committed evidence currently includes a current Linux build-host record and authorized Windows evidence inherited from the exact `0.16.0a2` stable predecessor.

## Frozen stable surface

The following modules expose the same 131 object-identical symbols:

```python
import komorebi_math.stable as stable
import komorebi_math.v1 as v1
import komorebi_math.api.v1 as api_v1

assert stable.__all__ == v1.__all__ == api_v1.__all__
for name in stable.__all__:
    assert getattr(stable, name) is getattr(v1, name)
    assert getattr(stable, name) is getattr(api_v1, name)
```

The stable surface is exactly unchanged from `0.16.0a2`: no added names, removed names, kind changes, or signature changes.

## Block 17 contracts

Block 17 binds eight independent artifacts:

| Contract | Size | Purpose |
|---|---:|---|
| Stable API | 131 symbols | Exact names, kinds, modules, qualified names, and signatures |
| Beta API | 83 symbols | Strict additive compatibility for the nonlinear beta surface |
| Research API | 1,179 symbols | Broad executable research inventory |
| Stable semantics | 10 cases | Mathematical behavior, not only signatures |
| Portable codecs | 51 codecs | Tags, implementation types, versions, and migration paths |
| Upgrade fixtures | 7 records | Real records produced by `0.15.0a2` and `0.16.0a2` |
| Platform evidence | 2 records | Current Linux evidence and authorized predecessor Windows evidence |
| Stable documentation | 131 entries | One exact documentation entry for every stable export |

The aggregate release contract is stored at:

```text
src/komorebi_math/data/release_contract_1_0.json
```

## Installation

From the extracted handoff:

```powershell
conda activate komorebi-full
python -m pip install --no-deps --force-reinstall .\release\dist\komorebi_math-1.0.0b1-py3-none-any.whl
```

Verify the installed identity:

```powershell
python -c "import komorebi_math; print(komorebi_math.__version__, komorebi_math.__file__); assert komorebi_math.__version__ == '1.0.0b1'"
```

## Source qualification

Focused Block 17 suite:

```powershell
.\Run_Block17_Source_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe" -Focused
```

Complete isolated source suite:

```powershell
.\Run_Block17_Source_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

Monolithic pytest mode:

```powershell
.\Run_Block17_Source_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe" -Monolithic
```

## Installed-wheel qualification

```powershell
.\Install_And_Verify_Block17.ps1 -Python "$env:CONDA_PREFIX\python.exe" -Recreate
```

This creates an isolated environment, installs the wheel rather than the source tree, verifies the stable API, upgrade records, platform evidence, documentation, cumulative Block 17 suite, and dependency consistency.

## Predecessor upgrade qualification

```powershell
.\Run_Block17_Upgrade_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

The validator performs two independent paths:

```text
0.15.0a2 wheel -> 1.0.0b1 wheel -> load and certify all predecessor records
0.16.0a2 wheel -> 1.0.0b1 wheel -> load and certify all predecessor records
```

The two predecessor wheels are included under `release/predecessors`.

## CUDA qualification

On the validated Windows CUDA environment:

```powershell
.\Run_Block17_GPU_Validation.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

The gate checks CUDA PyTorch directional JVP execution and Komorebi CuPy packet execution in isolated workers. GPU speed is not part of the stable semantic promise. Device correctness and CPU/GPU numerical equivalence are.

## CLI

```powershell
komorebi api check --stable
komorebi verify upgrade
komorebi verify evidence
komorebi verify documentation
komorebi verify beta
komorebi verify block17
komorebi release contract
komorebi release upgrade
komorebi release evidence
komorebi release documentation
komorebi release readiness
```

The stricter release-candidate gate is:

```powershell
komorebi release readiness --final
```

It is expected to remain closed until `1.0.0b1` itself is qualified on Windows and Linux. Inherited predecessor evidence is sufficient for beta publication but not for `1.0.0rc1`.

## Distribution rebuild

```powershell
.\Build_Block17_Distributions.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

The distribution audit verifies safe archive paths, wheel metadata, required package data, source-distribution contents, wheel RECORD shape, and a normalized wheel-payload comparison after rebuilding from the source distribution.

## Compatibility rule

For stable object behavior, Block 17 requires:

\[
\operatorname{API}_{0.16.0a2}=\operatorname{API}_{1.0.0b1},
\]

\[
\operatorname{decode}_{1.0.0b1}(\operatorname{encode}_{v}(x))\sim x,
\qquad v\in\{0.15.0a2,0.16.0a2\},
\]

and the ten committed stable semantic cases must remain unchanged.

## What remains before 1.0.0

1. Execute the exact `1.0.0b1` source, wheel, upgrade, and GPU gates on Windows.
2. Record current-version Windows evidence in `validation_evidence_1_0.json`.
3. Require `komorebi release readiness --final` to pass.
4. Build `1.0.0rc1` without changing the 131-symbol stable surface or persistence contract.
5. Promote the successful release candidate to `1.0.0` with version and release metadata changes only.
