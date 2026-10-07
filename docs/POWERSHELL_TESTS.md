# PowerShell installation and release tests

## Install the wheel

```powershell
$Wheel = ".\komorebi_math-0.11.0a1-py3-none-any.whl"
python -m venv .venv
$Py = ".\.venv\Scripts\python.exe"
$Komorebi = ".\.venv\Scripts\komorebi.exe"
& $Py -m pip install $Wheel
```


For an offline test using packages already installed in the base Python environment:

```powershell
python -m venv --system-site-packages .venv
$Py = ".\.venv\Scripts\python.exe"
$Komorebi = ".\.venv\Scripts\komorebi.exe"
& $Py -m pip install --no-deps $Wheel
```

Use `--no-deps` only when NumPy is already available through the base environment or another local wheel source.

## Verify the package

```powershell
& $Komorebi info
& $Komorebi doctor
& $Komorebi verify export
& $Komorebi verify block8
& $Komorebi verify block9
& $Komorebi verify hardening
& $Komorebi verify block10
& $Komorebi verify streaming-beta
& $Komorebi verify block11
& $Komorebi golden
& $Komorebi api check --strict-signatures
& $Komorebi api check --beta
& $Komorebi export formats
& $Komorebi latex engines
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

## Run the complete test script

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\examples\Test-KomorebiExport.ps1 -Wheel $Wheel -OutputDirectory .\export_smoke

# Offline reuse of the base environment when NumPy is already installed
.\examples\Test-KomorebiExport.ps1 -Wheel $Wheel -OutputDirectory .\export_smoke_offline -UseSystemSitePackages -NoDependencies

# Run all non PDF exports on a machine without LaTeX
.\examples\Test-KomorebiExport.ps1 -Wheel $Wheel -OutputDirectory .\export_smoke_no_pdf -SkipPdf
```

## Run the Block 9 nonlinear audit

From the source distribution or repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\examples\Test-KomorebiBlock9.ps1 -Python $Py
```

The script runs the three nonlinear examples, installed Block 9 verification, and the independent Block 9 deep audit.


## Run the Block 10 hardening audit

From the source distribution or repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\examples\Test-KomorebiBlock10.ps1 -Python $Py
```

The script runs the scalable nonlinear examples, focused and aggregate installed verification, the committed API check, and the independent Block 10 deep audit. CuPy device checks remain conditional on an installed compatible runtime and CUDA device.

## Run the bundled Python example

```powershell
& $Py .\examples\export_reports.py --output .\out\full_report --engine pdflatex
```

Replace `pdflatex` with `xelatex`, `lualatex`, `latexmk`, or `auto`.
