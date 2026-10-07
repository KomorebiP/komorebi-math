param(
    [Parameter(Mandatory = $true)]
    [string]$Wheel,
    [string]$Python = "python",
    [string]$OutputDirectory = ".\komorebi_export_test",
    [switch]$UseSystemSitePackages,
    [switch]$NoDependencies,
    [switch]$SkipPdf
)

$ErrorActionPreference = "Stop"
$Venv = Join-Path $OutputDirectory ".venv"
$Output = Join-Path $OutputDirectory "output"

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$VenvArgs = @("-m", "venv")
if ($UseSystemSitePackages) {
    $VenvArgs += "--system-site-packages"
}
$VenvArgs += $Venv
& $Python @VenvArgs
$Py = Join-Path $Venv "Scripts\python.exe"
$Komorebi = Join-Path $Venv "Scripts\komorebi.exe"

$InstallArgs = @("-m", "pip", "install")
if ($NoDependencies) {
    $InstallArgs += "--no-deps"
}
$InstallArgs += $Wheel
& $Py @InstallArgs

& $Komorebi info
& $Komorebi doctor
& $Komorebi verify export
& $Komorebi export formats
& $Komorebi latex engines

New-Item -ItemType Directory -Force -Path $Output | Out-Null

& $Komorebi export sequence fibonacci --terms 24 --compile --output (Join-Path $Output "fibonacci.json") --force
& $Komorebi export sequence fibonacci --terms 24 --compile --output (Join-Path $Output "fibonacci.csv") --force
& $Komorebi export sequence fibonacci --terms 24 --compile --output (Join-Path $Output "fibonacci.md") --force
& $Komorebi export sequence fibonacci --terms 24 --compile --output (Join-Path $Output "fibonacci.html") --force
& $Komorebi export sequence fibonacci --terms 24 --compile --output (Join-Path $Output "fibonacci.tex") --force

$Engines = & $Komorebi latex engines | ConvertFrom-Json
if (-not $SkipPdf -and ($Engines.pdflatex -or $Engines.xelatex -or $Engines.lualatex -or $Engines.latexmk)) {
    & $Komorebi export sequence fibonacci --terms 24 --compile --output (Join-Path $Output "fibonacci.pdf") --force
    & $Komorebi latex compile (Join-Path $Output "fibonacci.tex") --output (Join-Path $Output "fibonacci_recompiled.pdf")
}

@'
{
  "name": "sample data",
  "indices": [0, 1, 2, 3],
  "values": [1, 1, 2, 3]
}
'@ | Set-Content -Encoding UTF8 (Join-Path $Output "sample.json")

& $Komorebi export json --input (Join-Path $Output "sample.json") --output (Join-Path $Output "sample_report.tex") --title "Sample report" --force
& $Komorebi export bundle --input (Join-Path $Output "sample.json") --output (Join-Path $Output "sample_bundle.zip") --formats "json,markdown,latex" --force

@'
from pathlib import Path
from fractions import Fraction
from komorebi_math.export import DocumentMetadata, ExportOptions, MathDocument, export_data

output = Path(r"OUTPUT_PATH")
document = MathDocument(DocumentMetadata(title="PowerShell Python API test", author="Komorebi P."))
document.paragraph("This report was created through the Python API.")
document.equation(r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}", numbered=True)
document.table(("n", "value"), ((0, 1), (1, Fraction(1, 2)), (2, Fraction(1, 3))))
export_data(document, output / "python_api.tex", options=ExportOptions(overwrite=True)).certificate.require()
print("Python API export passed")
'@.Replace("OUTPUT_PATH", $Output.Replace("\", "\\")) | & $Py -

Get-ChildItem $Output | Select-Object Name, Length
Get-FileHash (Get-ChildItem $Output -File) -Algorithm SHA256
