# Portable export and LaTeX compilation

Version `0.8.0a1` adds a general output layer. It is separate from the research manuscripts. The export package accepts ordinary Python values, NumPy arrays, dataclasses, exact fractions, certificates, computation results, and renderer independent mathematical documents.

## Supported formats

```text
json
yaml
csv
tsv
text
markdown
html
latex
pdf
npy
npz
```

YAML requires the optional `yaml` extra. PDF export requires a local LaTeX engine. The package detects `lualatex`, `xelatex`, `pdflatex`, and `latexmk`.

## Export ordinary data

```python
from fractions import Fraction
from komorebi_math.export import ExportOptions, export_data

values = [
    {"index": 0, "value": Fraction(1, 1)},
    {"index": 1, "value": Fraction(1, 2)},
]

result = export_data(
    values,
    "values.json",
    options=ExportOptions(overwrite=True),
)
result.certificate.require()
```

The same object can be exported to CSV, Markdown, HTML, LaTeX, or PDF by changing the suffix.

## Build a mathematical document

```python
from komorebi_math.export import DocumentMetadata, ExportOptions, MathDocument, export_data

document = MathDocument(
    DocumentMetadata(
        title="Exact finite calculation",
        author="Komorebi P.",
        abstract="A report generated from executable mathematics.",
    )
)
document.paragraph("Ordinary text is escaped by each renderer.")
document.equation(r"E_n = \sum_{r=0}^{n/2} c_r A_{n-2r}", numbered=True)
document.table(("n", "value"), ((0, 1), (2, -1), (4, 5)))

export_data(document, "report.tex", options=ExportOptions(overwrite=True)).certificate.require()
export_data(document, "report.pdf", options=ExportOptions(overwrite=True)).certificate.require()
```

PDF export writes the generated `.tex` file beside the PDF by default. Set `keep_tex=False` to remove it after successful compilation.

## Compile an existing LaTeX source

```python
from komorebi_math.export import LatexCompileOptions, compile_latex

result = compile_latex(
    "paper.tex",
    "build",
    options=LatexCompileOptions(
        engine="pdflatex",
        passes=2,
        keep_auxiliary=False,
    ),
)
pdf = result.require()
```

Shell escape is disabled by default. Compilation has an explicit timeout and writes a `.komorebi.log` file.

## Multi format bundle

```python
from komorebi_math.export import ExportOptions, export_bundle

bundle = export_bundle(
    {"values": [1, 1, 2, 3, 5, 8]},
    "fibonacci_bundle.zip",
    formats=("json", "markdown", "latex", "pdf"),
    options=ExportOptions(overwrite=True),
)
bundle.certificate.require()
```

The ZIP includes a SHA256 manifest. PDF can be omitted on systems without a LaTeX installation.

## Custom document class

```python
from komorebi_math.export import ExportOptions, LatexRenderOptions

options = ExportOptions(
    overwrite=True,
    latex=LatexRenderOptions(
        document_class="report",
        include_table_of_contents=True,
    ),
)
```

A custom class such as `aomart` can be supplied when it is installed in the local TeX distribution.

## Safety boundary

Ordinary text is escaped. `Equation` stores trusted mathematical LaTeX. `RawLatex` is an explicit escape hatch for trusted source only. Shell escape remains disabled unless a Python caller deliberately enables it for a direct engine.

## PowerShell smoke test

The source distribution includes `examples\Test-KomorebiExport.ps1`. It creates a test environment, installs the wheel, exercises the command line interface, exports several formats, compiles LaTeX when an engine is available, and prints SHA256 hashes.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\examples\Test-KomorebiExport.ps1 -Wheel .\komorebi_math-0.11.0a1-py3-none-any.whl -OutputDirectory .\export_smoke
```

For an offline environment that already exposes NumPy through the base Python installation:

```powershell
.\examples\Test-KomorebiExport.ps1 -Wheel .\komorebi_math-0.11.0a1-py3-none-any.whl -OutputDirectory .\export_smoke -UseSystemSitePackages -NoDependencies
```

## Reproducibility boundary

ZIP member ordering, ZIP timestamps, and the bundle manifest are normalized. JSON, CSV, Markdown, HTML, LaTeX, NPY, and NPZ content is deterministic for deterministic inputs and options. PDF bytes can still vary with the selected TeX engine, TeX distribution, fonts, and compiler version. The manifest records the exact PDF that was produced rather than claiming cross installation byte identity.
