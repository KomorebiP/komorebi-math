# Komorebi Math 1.0.0b1 — Block 17 Release Report

## Disposition

Block 17 is complete as a **public beta qualification block**. The package version is `1.0.0b1`. The 131-symbol stable namespace is frozen and is exactly unchanged from `0.16.0a2`.

This handoff is not labeled `1.0.0rc1` or `1.0.0`. The remaining release-candidate gate is exact current-version Windows qualification of this beta.

## Scope

Block 17 adds release closure rather than another mathematical domain. It provides:

- an exact stable API freeze across `komorebi_math.stable`, `komorebi_math.v1`, and `komorebi_math.api.v1`
- executable upgrade paths from `0.15.0a2` and `0.16.0a2`
- seven digest-checked predecessor portable records
- two-tier platform evidence rules for beta and release-candidate qualification
- one documentation entry for every stable export
- dedicated source, installed-wheel, upgrade, CUDA, and distribution runners
- an explicit beta-to-RC gate that cannot be bypassed by inherited evidence

## Frozen contract inventories

| Contract | Size |
|---|---:|
| Stable API | 131 symbols |
| Beta API | 83 symbols |
| Research API | 1,179 symbols |
| Stable semantics | 10 cases |
| Portable codecs | 51 codecs |
| Upgrade fixtures | 7 records |
| Platform evidence | 2 records |
| Stable documentation | 131 entries |

Stable comparison against `0.16.0a2`:

```text
added             0
removed           0
kind changed      0
signature changed 0
```

## Final validation

| Gate | Result |
|---|---:|
| Isolated source modules | **125 / 125** |
| Source tests | **1,040 passed, 0 skipped** |
| Post-evidence focused gate | **57 passed, 0 skipped** |
| Package audit | **7 / 7** |
| Static audit | **4 / 4** |
| Release contract | **21 / 21** |
| Source cumulative verifier | **196 / 196** |
| Installed beta readiness | **8 / 8** |
| Installed release qualification | **17 / 17** |
| Installed cumulative verifier | **194 / 194** |
| Predecessor upgrade paths | **2 / 2** |
| Distribution audit | **11 / 11** |
| Rebuilt wheel normalized payload | **identical** |

The clean installed environment inherited one unrelated Linux system-package warning: `supervisor` expected `setuptools`. Komorebi Math's declared dependency was consistent, and the installed package gates passed.

## Executed platform evidence

### Current beta evidence

The Linux build host executed the complete `1.0.0b1` source qualification:

```text
Python 3.13.5
NumPy 2.3.5
1,040 passed
125 isolated test modules
0 skipped
0 failed files
```

### Authorized predecessor evidence

The retained Windows record is for the exact frozen predecessor `0.16.0a2`:

```text
Windows 11 AMD64
Python 3.13.0
NumPy 2.5.2
1,009 source tests passed
installed-wheel gate passed
CUDA GPU gate 4/4
NVIDIA GeForce RTX 3070 Ti Laptop GPU
PyTorch 2.12.1+cu126
CuPy 14.1.1
```

This Windows result is accepted for public beta because the stable API, semantics, codecs, and predecessor records are unchanged and protected by the upgrade contract. It is not accepted for release-candidate status.

## Upgrade qualification

Both real predecessor wheel paths passed:

```text
0.15.0a2 -> 1.0.0b1 -> load and verify all seven records
0.16.0a2 -> 1.0.0b1 -> load and verify all seven records
```

Each path separately created an environment, installed the predecessor, imported the predecessor, installed the beta, and loaded the complete current upgrade fixture inventory.

## Distribution identities

```text
wheel
f3af8ac92616dbf57950b3468bce09d149368ebb660b0b4850b3a473609e03d8  komorebi_math-1.0.0b1-py3-none-any.whl

sdist
53f64485842bf7529b5fade391a6be45b2e5e156fbb64b9dfdab9d16a553de99  komorebi_math-1.0.0b1.tar.gz
```

The source distribution rebuilt a wheel whose normalized payload exactly matched the published wheel.

## Remaining gate to `1.0.0rc1`

The release-candidate readiness command intentionally remains closed:

```powershell
komorebi release readiness --final
```

The only missing required platform family is current-version Windows evidence for `1.0.0b1`. Run the beta source, installed-wheel, predecessor-upgrade, and CUDA gates on Windows, commit that evidence, and rerun the final readiness gate.

No stable API, semantic contract, or persistence change should be made between this beta and the release candidate. A change to any frozen contract requires another beta.
