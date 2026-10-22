# Komorebi Math: Technical Manual and Reference

This 2,529-page book is the full technical reference for the Python library, based on the preserved long-form manual and its extended API, module-reference and research appendices. It contains 14 parts, 91 chapters and 15 appendices.

| File | Purpose |
| --- | --- |
| [Komorebi_Math_Technical_Manual.pdf](Komorebi_Math_Technical_Manual.pdf) | Complete technical reference |
| [Komorebi_Math_Technical_Manual_Editable.zip](Komorebi_Math_Technical_Manual_Editable.zip) | Direct download of the editable manuscript and build instructions |
| [Source part 1](Komorebi_Math_Technical_Manual_Source.zip.part01) | First part of the complete source and evidence bundle |
| [Source part 2](Komorebi_Math_Technical_Manual_Source.zip.part02) | Second part of the complete source and evidence bundle |
| [Source part 3](Komorebi_Math_Technical_Manual_Source.zip.part03) | Third part of the complete source and evidence bundle |
| [Source part 4](Komorebi_Math_Technical_Manual_Source.zip.part04) | Fourth part of the complete source and evidence bundle |
| [Source part 5](Komorebi_Math_Technical_Manual_Source.zip.part05) | Fifth part of the complete source and evidence bundle |
| [join_source.py](join_source.py) | Reassembles and verifies the complete source ZIP |
| [SOURCE_PARTS.json](SOURCE_PARTS.json) | Ordered part sizes and checksums, plus the complete archive checksum |
| [SHA256SUMS.txt](SHA256SUMS.txt) | Checksums for the PDF, parts and reassembled source archive |
| [PROOFREADING_SUMMARY.json](PROOFREADING_SUMMARY.json) | Editorial-review scope, checks and limitations |

The editable-manuscript ZIP is the direct source download. The complete source and evidence bundle also retains historical reference material, preservation checks and review records. To restore that larger archive, download all five source parts, `SOURCE_PARTS.json` and `join_source.py` into the same folder, then run:

```sh
python join_source.py
python -m zipfile -e Komorebi_Math_Technical_Manual_Source.zip manual_sources
```

The first command verifies every part and the complete archive before creating the ZIP. The second extracts the sources. Both commands use Python's standard library.

The review addresses prose, notation, navigation and the distinction between historical descriptions and the final 1.0.0 interfaces. Its recorded checks should be read at their stated scope; they do not establish the correctness of every theorem, derivation or research assertion in the reference.

The [207-page companion guide](../manual/) is a separate book assembled from the 47 current reader guides.

The software implementation remains source commit `5f2b7ddc879c12d2bb00cbe5c487aa0bc7709222`. Editing the manual does not change the implementation or extend its hardware qualification. The [release-download index](../README.md) records the software's validation and publication status.

Original documentation and software use the MIT License. Third-party notices and separately supplied data terms remain applicable.
