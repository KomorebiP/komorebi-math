# API compatibility 0.13

Version `0.13.0a1` adds `komorebi_math.generating` to the broader additive public API manifest.

The broader manifest covers:

* `komorebi_math`
* `komorebi_math.nonlinear`
* `komorebi_math.export`
* `komorebi_math.signals`
* `komorebi_math.generating`

The policy remains `pre-beta-additive`. Additions are permitted. Export removals and broad object-kind changes are incompatible. Signature changes are reported and become failures under strict checking.

The curated `komorebi_math.beta` namespace remains under the separate `beta-additive-strict-signatures` policy. Block 13 does not add generating-function objects to that namespace.

Structural API manifests do not certify mathematical or behavioral equivalence. Installed verification, unit tests, behavioral fixtures, and independent arithmetic audits provide separate evidence.

```powershell
komorebi api check --strict-signatures
komorebi api check --beta
```
