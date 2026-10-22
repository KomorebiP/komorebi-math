# PowerShell installation and release tests

## Install the wheel

```powershell
$Wheel = ".\dist\komorebi_math-1.0.0-py3-none-any.whl"
python -m venv .venv
$Py = ".\.venv\Scripts\python.exe"
$Komorebi = ".\.venv\Scripts\komorebi.exe"
& $Py -m pip install $Wheel
```


For a fully offline installation, supply a compatible wheelhouse containing
NumPy and the other requested dependencies:

```powershell
& $Py -m pip install --no-index --find-links .\wheelhouse $Wheel
```

## Verify the package

```powershell
& $Komorebi info
& $Komorebi doctor
& $Komorebi verify stable
& $Komorebi release check
& $Komorebi api check --stable
& $Komorebi export formats
& $Komorebi latex engines
```

The complete installed suite also needs the `exact`, `scientific` and `yaml`
extras. Install them from the checkout before running it:

```powershell
& $Py -m pip install ".[exact,scientific,yaml]"
& $Komorebi verify all
```

## Export a sequence

```powershell
New-Item -ItemType Directory -Force .\out | Out-Null
& $Komorebi export sequence fibonacci --terms 30 --compile --output .\out\fibonacci.json --force
& $Komorebi export sequence fibonacci --terms 30 --compile --output .\out\fibonacci.csv --force
& $Komorebi export sequence fibonacci --terms 30 --compile --output .\out\fibonacci.md --force
& $Komorebi export sequence fibonacci --terms 30 --compile --output .\out\fibonacci.html --force
& $Komorebi export sequence fibonacci --terms 30 --compile --output .\out\fibonacci.tex --force
& $Komorebi export sequence fibonacci --terms 30 --compile --output .\out\fibonacci.pdf --force
```

The PDF call requires a local LaTeX engine. The `.tex` file is preserved beside the PDF by default.

## Recompile the generated LaTeX

```powershell
& $Komorebi latex compile .\out\fibonacci.tex --output .\out\fibonacci_recompiled.pdf
```

## Export local arithmetic

```powershell
& $Komorebi export standard-window --prime 7 --compression 2 --address 3 --count 24 --output .\out\standard_window.tex --title "Standard window" --force
& $Komorebi export euler --prime 2 --index 20 --depth 2 --output .\out\euler20.pdf --title "Dyadic Euler coordinate" --force
```

## Convert JSON to LaTeX

```powershell
@'
{
  "name": "example",
  "indices": [0, 1, 2, 3],
  "values": [1, 1, 2, 3]
}
'@ | Set-Content -Encoding UTF8 .\out\data.json

& $Komorebi export json --input .\out\data.json --output .\out\data.tex --title "JSON report" --force
& $Komorebi export json --input .\out\data.json --output .\out\data.pdf --title "JSON report" --force
```

## Build a multi format ZIP

```powershell
& $Komorebi export bundle --input .\out\data.json --output .\out\data_bundle.zip --formats "json,markdown,html,latex,pdf" --force
```

## Test the Python API from PowerShell

```powershell
@'
from pathlib import Path
from fractions import Fraction
from komorebi_math.export import DocumentMetadata, ExportOptions, MathDocument, export_data

out = Path("out")
out.mkdir(exist_ok=True)

document = MathDocument(DocumentMetadata(title="Python API report", author="Komorebi P."))
document.paragraph("This report was generated from the installed Python package.")
document.equation(r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}", numbered=True)
document.table(("n", "value"), ((0, 1), (1, Fraction(1, 2)), (2, Fraction(1, 3))))

for suffix in ("json", "md", "html", "tex", "pdf"):
    result = export_data(document, out / f"python_report.{suffix}", options=ExportOptions(overwrite=True))
    result.certificate.require()
    print(result.path, result.bytes_written)
'@ | & $Py -
```

## Source qualification

The [release guide](RELEASE_PROCESS.md) describes full source tests, strict
documentation, offline installed checks and the CPU matrix. Use its neutral
Python entry points through `$Py` so that all commands share the environment.

## Run the bundled Python example

```powershell
& $Py .\examples\export_reports.py --output .\out\full_report --engine pdflatex
```

Replace `pdflatex` with `xelatex`, `lualatex`, `latexmk`, or `auto`.
