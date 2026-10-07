# Block 36 release repair R1

**Author:** Komorebi P.  
**Release identifier:** `1.0.0b4-block36-r1`  
**API/package version:** `1.0.0b4`  
**Wheel build tag:** `1`

This is a packaging repair, not a new mathematical API release. All package source files are retained byte for byte from the stored Block 36 source. The previously shipped wheel and sdist contained stale versions of `coordinates/families.py` and `export/persistence.py`. Both distributions must now be rebuilt from the source and pass package-byte parity checks. The installable wheel is explicitly named `komorebi_math-1.0.0b4-1-py3-none-any.whl`.

Use `--force-reinstall` when replacing an earlier 1.0.0b4 installation. The public version is deliberately unchanged to preserve the reviewed API and manual contracts. `RELEASE_ID.json`, the wheel build tag, and current SHA-256 records identify this repair. The sdist uses the standard version-based filename and must be identified by its R1 bundle and checksum, not by its filename alone.

## Rebuild and verify

From the repository root with the declared development and mathematical dependencies installed:

```text
python verification/build_block21_distributions.py --root . --output-directory release/dist --report release/REPAIR_R1_DISTRIBUTION_BUILD.json
python verification/release_artifact_gate.py --root . --output release/REPAIR_R1_ARTIFACT_GATE.json
python verification/block21_publication_gate.py --full-tests --installed-artifacts --skip-mkdocs --output release/REPAIR_R1_PUBLICATION_GATE.json --markdown release/REPAIR_R1_PUBLICATION_GATE.md
```

Use `--require-mkdocs` instead of `--skip-mkdocs` only when actually executing the strict documentation build. The repair's installed-artifact gate uses temporary virtual environments with inherited dependencies and offline `--no-deps` installation. It verifies that `komorebi_math` is imported from each new environment, not from the repository. This is package isolation, not a fully dependency-isolated qualification. It rebuilds the sdist, checks package parity, and runs all three behavioral regressions against each installed wheel.

The quick publication gate now rejects missing, stale, malformed, or mismatched current distributions and non-executable POSIX runners. It does not claim installed-artifact behavior unless `--installed-artifacts` is requested. `RELEASE_ID.json` and the current generated evidence identify this delivery; all inherited Block 16–21 evidence remains historical.

## License and scope

Only the outdated word “alpha” was corrected to “beta” in `LICENSE.md`. All existing proprietary/private-evaluation restrictions are unchanged. This repair does not choose an open-source license, grant new rights, or certify public redistribution terms. Windows, PowerShell, other Python versions, CUDA, hosted Actions, and a strict documentation build need their own executed records. Existing PDFs and manual verification reports are retained, not newly rendered or proofread.
