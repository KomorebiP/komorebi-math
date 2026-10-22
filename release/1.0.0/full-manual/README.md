# Komorebi Math: Technical Manual and Reference

This 2,569-page book is the complete technical reference for the Python library. It includes a practical reader's guide, installation instructions, first calculations with expected output, topic routes, troubleshooting guidance and the full mathematical and API reference: 14 parts, 91 chapters, 15 appendices and 1,526 API cards.

| File | Purpose |
| --- | --- |
| [Komorebi_Math_Technical_Manual.pdf](Komorebi_Math_Technical_Manual.pdf) | Complete reading copy |
| [Komorebi_Math_Technical_Manual_Editable.zip](Komorebi_Math_Technical_Manual_Editable.zip) | Direct download of the complete editable manuscript, standalone LaTeX and build instructions |
| [Source part 1](Komorebi_Math_Technical_Manual_Source.zip.part01) | Part 1 of the complete source and evidence bundle |
| [Source part 2](Komorebi_Math_Technical_Manual_Source.zip.part02) | Part 2 of the complete source and evidence bundle |
| [Source part 3](Komorebi_Math_Technical_Manual_Source.zip.part03) | Part 3 of the complete source and evidence bundle |
| [Source part 4](Komorebi_Math_Technical_Manual_Source.zip.part04) | Part 4 of the complete source and evidence bundle |
| [Source part 5](Komorebi_Math_Technical_Manual_Source.zip.part05) | Part 5 of the complete source and evidence bundle |
| [Source part 6](Komorebi_Math_Technical_Manual_Source.zip.part06) | Part 6 of the complete source and evidence bundle |
| [join_source.py](join_source.py) | Reassembles and verifies the complete source ZIP |
| [SOURCE_PARTS.json](SOURCE_PARTS.json) | Ordered part sizes and checksums, plus the full archive checksum |
| [SHA256SUMS.txt](SHA256SUMS.txt) | Checksums for the PDF, editable ZIP, source parts and complete source archive |
| [PROOFREADING_SUMMARY.json](PROOFREADING_SUMMARY.json) | Review scope, preservation checks and limitations |

The editable ZIP is the direct source download. The larger complete source bundle also contains the manuscript parts, assembly workflow, historical reference material and detailed review evidence. To restore that bundle, download all 6 source parts, `SOURCE_PARTS.json` and `join_source.py` into the same folder, then run:

```sh
python join_source.py
python -m zipfile -e Komorebi_Math_Technical_Manual_Source.zip manual_sources
```

The first command verifies every part and the complete archive before creating the ZIP. The second extracts it. Both commands use Python's standard library.

The final reader-focused review preserves all 4,229 original code listings and all delimited mathematical spans. It improves navigation and explanations and checks the new introductory Python examples. One mathematical prose hypothesis was corrected in Section 53.3, with a clarification in Appendix H: the zero large-time spectral-dimension limit requires a zero eigenvalue. Its focused review and the limits of that check are included in the source bundle. This is not a whole-book theorem audit or a replay of every example.

The [207-page companion guide](../manual/) remains a separate, shorter book assembled from 47 current reader guides.

The software implementation remains source commit `5f2b7ddc879c12d2bb00cbe5c487aa0bc7709222`. This manual revision does not change the implementation or extend hardware qualification. The [release-download index](../README.md) records validation and publication status.

Original software and documentation use the MIT License. Third-party notices and separately supplied data terms remain applicable.
