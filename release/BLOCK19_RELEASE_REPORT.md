# Komorebi Math 1.0.0b3 — Block 19 release report

Status: **qualified research beta on the executed Linux/Python 3.12 host**

Block 19 adds finite-measure integration calculus at the research tier. It
does not promote integration names into the frozen stable or beta namespaces,
and `1.0.0b3` remains a beta rather than a release candidate.

## Qualification ledger

| Gate | Required result | Evidence | Status |
|---|---:|---|---|
| source integration CLI | 12/12 | final source validation | passed |
| source Block 19 CLI | 24/24 | final source validation | passed |
| frozen stable CLI | 6/6 | final source and both installed validations | passed |
| cumulative Block 15 CLI | 174/174 | final source and both installed validations | passed |
| frozen stable documentation | 1/1 (exact 131-entry contract) | final source and both installed validations | passed |
| independent mathematics | 16/16 | source, primary wheel, and fresh-sdist rebuilt wheel records | passed |
| adversarial suite | 47/47 | source, primary wheel, and fresh-sdist rebuilt wheel records | passed |
| worked example | exact `106/5` | source, primary wheel, and fresh-sdist rebuilt wheel records | passed |
| API additivity and frozen tiers | research +38; stable +0; beta +0 | `release/BLOCK19_API_ADDITIVITY.json` | passed |
| wheel integrity and complete RECORD | 0 corrupt or unrecorded members | `release/BLOCK19_DISTRIBUTION_AUDIT.json` | passed |
| wheel/source payload equality | 0 payload differences | `release/BLOCK19_DISTRIBUTION_AUDIT.json` | passed |
| fresh-sdist rebuild and isolated install | 0 normalized wheel differences; all installed gates pass | `release/BLOCK19_DISTRIBUTION_VALIDATION.json` | passed |
| clean source handoff ZIP | safe single root; fresh-extract source gates pass | `release/BLOCK19_VALIDATION_SUMMARY.json` and `SHA256SUMS_BLOCK19.txt` | passed |

The source-tree run, the primary-wheel install, and the wheel rebuilt from a
fresh sdist extraction each used the final frozen source. Installed checks ran
outside the project tree with `PYTHONPATH` cleared.

## Distribution hygiene

The final `release/dist` directory contains only the `1.0.0b3` wheel and sdist.
Earlier `1.0.0b1` and `1.0.0b2` artifacts are retained separately under
`release/predecessors`; they are never mixed into the current distribution
directory.

The root files `PACKAGE_AUDIT.json`, `PACKAGE_AUDIT.md`, `STATIC_AUDIT.json`,
and `STATIC_AUDIT.md` are historical Block 16 records for `0.16.0a2`. They are
not Block 19 evidence, are excluded from the `1.0.0b3` sdist, and must not be
interpreted as current package or static audits. They remain in the source
handoff only to preserve the historical record.

The root `SHA256SUMS.txt` file is a historical Block 17 checksum ledger. It is
not Block 19 evidence and is excluded from both the `1.0.0b3` sdist and the
clean Block 19 handoff ZIP. Current Block 19 digests are written separately to
`SHA256SUMS_BLOCK19.txt` only after final archive validation.

The clean handoff ZIP excludes build directories, virtual environments,
egg-info, bytecode, cache directories, the historical generic checksum, and non-Block-19 files from
`release/dist`. Its archive paths are relative, safe, and rooted under one
`Komorebi_Math_Complete_Handoff_1.0.0b3` directory.

## Platform boundary

The Block 19 closure ran on Linux with Python 3.12. The fixture-free adversarial
runner supplied the committed 47-test result. Windows, CUDA, pytest, and a
rendered MkDocs build were not executed for this closure and are not claimed.
Historical platform evidence is not recast as a current Block 19 result.

The aggregate `verify release`, `verify beta`, and `release qualify` commands
remain legacy Block 17 / `1.0.0b1`-hardwired gates. They are not claimed as
Block 19 or `1.0.0b3` passes. Block 19 instead runs and records the exact
`verify documentation` 1/1 contract directly alongside its current source and
isolated-install gates.

## Distribution digests

The wheel, sdist, and handoff ZIP digests are recorded externally in
`SHA256SUMS_BLOCK19.txt` after the final single-root archive is written and
fresh-extracted. Keeping the sdist's own digest outside the files embedded in
that sdist avoids a self-referential checksum claim.
