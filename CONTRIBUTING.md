# Contributing

Komorebi Math is research software that authorized repository collaborators can review under the license in `LICENSE.md`. The license is restrictive and is not an OSI open-source license. Do not assume that public visibility grants permission to redistribute or incorporate the code elsewhere.

Before preparing a substantial change, open an issue describing the mathematical claim, implementation boundary, expected API tier, verification strategy, and compatibility impact. Small corrections may be submitted directly when repository permissions allow.

A contribution should include:

1. a finite statement of what is implemented and what is not claimed
2. tests that fail before the change and pass afterward
3. a certificate, residual, or explicit conditional boundary where applicable
4. documentation for every new public object
5. no silent expansion of `komorebi_math.stable`, `komorebi_math.v1`, or `komorebi_math.api.v1`
6. successful execution of the Block 21 publication gate

Prepare the dependencies and current distributions using
[local validation](docs/LOCAL_VALIDATION.md). Then run the complete gate on Windows
with the same interpreter used for preparation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run_Block21_Publication_Gate.ps1 -Python "python"
```

For an explicit interpreter path, replace `python` in `-Python` with that path.
Run the strict source and documentation gate on Linux with:

```bash
python verification/block21_publication_gate.py --full-tests --require-mkdocs
```

The POSIX wrapper automatically skips MkDocs when unavailable. That reduced
diagnostic run does not establish strict documentation qualification. Installed
artifacts, platform evidence, and GPU qualification are separate checks described
in the [release process](docs/RELEASE_PROCESS.md).
