# API compatibility contract for 0.12

Version `0.12.0a1` retains the additive pre-beta compatibility policy and expands the committed research manifest to cover four namespaces:

- `komorebi_math`
- `komorebi_math.nonlinear`
- `komorebi_math.export`
- `komorebi_math.signals`

The committed research manifest contains 699 symbols. Additions are permitted. Removed names, changes in broad symbol kind, and strict reflected-signature changes fail the corresponding check.

```powershell
komorebi api check --strict-signatures
```

The curated `komorebi_math.beta` namespace remains unchanged at 83 symbols and retains its additive strict-signature policy.

```powershell
komorebi api check --beta
```

The 0.11 research and beta manifests remain in the source archive as historical records. Before the 0.12 manifest was adopted, the Block 12 source was checked against both 0.11 manifests with zero additions, removals, kind changes, or signature changes in their covered namespaces.

The structural manifest does not prove mathematical or behavioral equivalence. Block 12 supplements it with signal-specific unit tests, installed verification, deterministic certificates, and an independent randomized audit.
