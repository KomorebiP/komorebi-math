# Contributing

Komorebi Math is reviewable by authorized repository collaborators research software under the license in `LICENSE.md`. The license is restrictive and is not an OSI open-source license. Do not assume that public visibility grants permission to redistribute or incorporate the code elsewhere.

Before preparing a substantial change, open an issue describing the mathematical claim, implementation boundary, expected API tier, verification strategy, and compatibility impact. Small corrections may be submitted directly when repository permissions allow.

A contribution should include:

1. a finite statement of what is implemented and what is not claimed
2. tests that fail before the change and pass afterward
3. a certificate, residual, or explicit conditional boundary where applicable
4. documentation for every new public object
5. no silent expansion of `komorebi_math.stable`, `komorebi_math.v1`, or `komorebi_math.api.v1`
6. successful execution of the Block 21 publication gate

Run the complete gate on Windows with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run_Block21_Publication_Gate.ps1 -Python "$env:CONDA_PREFIX\python.exe"
```

Run it on Linux with:

```bash
PYTHON=python ./run_block21_publication_gate.sh
```
