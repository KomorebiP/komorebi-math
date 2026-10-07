# Block 8 release

Version `0.8.0a1` adds portable serialization, data exports, mathematical report objects, LaTeX generation, local PDF compilation, manifested multi format bundles with normalized ZIP metadata, command line export tools, and PowerShell smoke tests.

Implemented formats:

* JSON
* optional YAML
* CSV
* TSV
* plain text
* Markdown
* standalone HTML
* complete LaTeX
* PDF compiled from LaTeX
* NumPy NPY
* compressed NumPy NPZ

The export system accepts ordinary Python data, NumPy arrays, dataclasses, exact fractions, complex numbers, certificates, computation results, and `MathDocument` objects.

The LaTeX compiler discovers local engines, disables shell escape by default, enforces timeouts, captures logs, supports repeated passes, and preserves generated `.tex` sources beside PDFs by default.

This block does not provide a remote LaTeX service. PDF output requires a supported TeX engine on the local machine.
