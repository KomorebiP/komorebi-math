# Stable API Contract

## Scope

`komorebi_math.stable`, `komorebi_math.v1`, and `komorebi_math.api.v1` expose one object-identical 1.0-candidate namespace. The complete research package remains larger and continues to evolve.

The stable contract is intentionally selective. It includes the finite foundations needed to construct operator systems, analyze and synthesize coordinates, refine packets, certify results, persist portable records, and render mathematical documents.

## Compatibility promise

For the eventual 1.x line, a symbol in the stable namespace is intended to retain:

* its exported name
* its broad kind, such as class or function
* its documented signature
* its documented finite mathematical behavior
* its portable codec identity when persistence is promised

Additive names may appear in later 1.x releases. A stable removal or incompatible signature change requires a major release or a documented deprecation cycle.

## Research namespace

An object being importable from `komorebi_math`, a domain package, or a research module does not by itself place that object under the stable contract. The committed stable manifest is authoritative.

## Verification

```powershell
komorebi api check --stable
komorebi verify stable
```

Programmatically:

```python
from komorebi_math.stable import check_stable_api, check_stable_contract

check_stable_api().certificate.require()
check_stable_contract().certificate.require()
```
