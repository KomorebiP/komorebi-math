# Komorebi Math 1.0.0

This directory contains the final Python 1.0.0 build and its validation records.

| File | Purpose |
| --- | --- |
| `komorebi_math-1.0.0-py3-none-any.whl` | Installable Python package |
| `komorebi_math-1.0.0.tar.gz` | Standard source distribution |
| `komorebi_math-1.0.0-source.zip` | Clean repository source, documentation, tests, current wheel and source distribution |
| `komorebi_math-1.0.0-qualification.zip` | Validation reports and execution evidence, including superseded failed attempts |
| `full-manual/` | Full technical reference, editable sources and editorial-review records |
| `manual/` | Separate 207-page companion guide, editable sources and proofreading checks |
| `SHA256SUMS.txt` | SHA-256 checksums for all four downloads |
| `FINAL_RELEASE_STATUS.json` | Exact source identity, validation scope and remaining publication prerequisites |

The 2,569-page full technical manual preserves the long-form reference and its extended appendices. The separate 207-page companion guide follows the 47 current reader guides and provides a shorter route through the library.

Install the downloaded wheel with:

```sh
python -m pip install komorebi_math-1.0.0-py3-none-any.whl
```

Source commit: `5f2b7ddc879c12d2bb00cbe5c487aa0bc7709222`.

Local validation passed: 2,752 source tests; all seven optional CPU cases passed separately after their skips in the source environment; 1,556 tests for each of the direct and source-rebuilt installed wheels; 84 executable source-documentation examples; 351 stress checks and 11 resource controls. Strict documentation, stable API and installed CLI checks passed. The wheel, source distribution and repository archive rebuild reproducibly; the repository archive also passed a clean-extraction replay.

These files are prepared release downloads. No GitHub or PyPI release has been published. Fresh hosted Linux/Windows qualification could not start because GitHub reported an account payment or spending-limit restriction. Real NVIDIA qualification and a qualified tagged RC predecessor remain required by the publication review. The RC integration evidence is a test of the transition, not a qualified release.
