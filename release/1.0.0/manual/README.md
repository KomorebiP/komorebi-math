# Komorebi Math: The Python Library

A professional monograph manual of the Python library, by Komorebi P.

The manual brings 47 reader guides into 27 chapters and six appendices. It includes an introduction, contents, numbered cross-references, syntax-highlighted code, two vector illustrations, API references and a glossary. Installation, compatibility and qualification procedures are collected in the appendices.

`Komorebi_Math_Manual.pdf` is the finished reading copy. `source/` contains the proofread Markdown, typesetting template, figure generator and build scripts. `review/` records the example replay, token-preservation checks and layout review.

## Build the PDF

Install Pandoc 3.1 or newer and a TeX distribution with XeLaTeX, KOMA-Script, fontspec, unicode-math, fvextra, tcolorbox, seqsplit and the standard table packages. The template uses Pagella and Heros when available, with P052/Nimbus or Windows Palatino/Arial fallbacks. Latin Modern Math supplies mathematical glyphs.

From the extracted directory:

```sh
python source/compile_manual.py
```

The vector figures are included. Regenerating them requires NumPy and Matplotlib:

```sh
python source/make_figures.py
```

## Verify the examples and preserved tokens

Install the final library wheel with its exact, scientific and YAML dependencies. Directed interval examples require mpmath 1.3.0. Run with Python assertions enabled:

```sh
python source/check_examples.py --output review/examples.json
python source/audit_preservation.py
```

The manual contains 123 Python listings: 81 executable examples and 42 explicitly identified interface sketches, contextual fragments or external-runtime examples. The 81 executable examples pass independently against the installed final library. All 207 fenced listings and all 390 mathematical spans inherited from the guides retain their original tokens. The two new illustrations show explicit finite packet and analysis-synthesis constructions.

The proofread corrects prose, navigation and descriptions of scope. It does not change the library implementation or claim additional hardware qualification. The frozen implementation reference is source commit `5f2b7ddc879c12d2bb00cbe5c487aa0bc7709222` in `KomorebiP/komorebi-math`.

Original documentation and software are distributed under the MIT License. Third-party notices and separately supplied data terms remain applicable.
