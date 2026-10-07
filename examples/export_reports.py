"""Create JSON, CSV, Markdown, HTML, LaTeX, PDF, and ZIP exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from komorebi_math import EulerLocalCoordinates
from komorebi_math.export import (
    DocumentMetadata,
    ExportOptions,
    LatexCompileOptions,
    MathDocument,
    available_latex_engines,
    export_bundle,
    export_data,
)
from komorebi_math.sequences import compile_sequence, fibonacci_sequence


def build_report(term_count: int) -> MathDocument:
    """Build one mixed mathematical report."""

    sequence = fibonacci_sequence()
    compiled = compile_sequence(sequence)
    values = compiled.first(term_count)
    euler = EulerLocalCoordinates(2).truncate(index=20, depth=2)
    euler.certificate.require()

    document = MathDocument(
        DocumentMetadata(
            title="Komorebi Math export demonstration",
            author="Komorebi P.",
            abstract="A generated report containing exact sequence data, one formula, and one certified local arithmetic calculation.",
            keywords=("packet mathematics", "exact arithmetic", "Python export"),
        )
    )
    document.paragraph(f"The Fibonacci values were evaluated through the {compiled.strategy} strategy.")
    document.section("Fibonacci values")
    document.table(
        ("Index", "Value"),
        tuple((index, value) for index, value in enumerate(values)),
        caption=f"First {term_count} Fibonacci values",
    )
    document.section("Euler coordinate")
    document.equation(r"E_{20}=370371188237525", numbered=True, label="eq:euler20")
    document.table(
        ("Field", "Value"),
        (
            ("Prime", 2),
            ("Depth", 2),
            ("Truncated coordinate", euler.value),
            ("Exact value", euler.exact_value),
            ("Certified modulus", euler.modulus),
        ),
    )
    document.section("Certificate")
    document.table(
        ("Field", "Value"),
        (
            ("Valid", euler.certificate.valid),
            ("Exactness", euler.certificate.exactness.value),
            ("Message", euler.certificate.message),
        ),
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("komorebi_export_demo"))
    parser.add_argument("--terms", type=int, default=24)
    parser.add_argument(
        "--engine",
        default="auto",
        choices=("auto", "pdflatex", "xelatex", "lualatex", "latexmk"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report = build_report(args.terms)
    options = ExportOptions(
        overwrite=True,
        title=report.metadata.title,
        author=report.metadata.author,
        compiler=LatexCompileOptions(engine=args.engine),
    )

    for suffix in ("txt", "md", "html", "tex"):
        export_data(report, args.output / f"report.{suffix}", options=options).certificate.require()

    sequence_rows = tuple((index, value) for index, value in enumerate(fibonacci_sequence().first(args.terms)))
    export_data(sequence_rows, args.output / "fibonacci.csv", options=options).certificate.require()
    export_data(sequence_rows, args.output / "fibonacci.json", options=options).certificate.require()

    if any(available_latex_engines().values()):
        export_data(report, args.output / "report.pdf", options=options).certificate.require()

    bundle_formats = ["json", "markdown", "latex"]
    if any(available_latex_engines().values()):
        bundle_formats.append("pdf")
    export_bundle(
        sequence_rows,
        args.output / "fibonacci_bundle.zip",
        base_name="fibonacci",
        formats=tuple(bundle_formats),
        options=options,
    ).certificate.require()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
