"""Installed verification suite for portable exports and LaTeX output."""

from __future__ import annotations

import json
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..export import (
    DocumentMetadata,
    ExportOptions,
    MathDocument,
    available_latex_engines,
    compile_latex,
    export_bundle,
    export_data,
    latex_escape,
    render_latex,
    to_serializable,
)
from .core import VerificationCheck, VerificationReport


def _capture(name: str, function) -> VerificationCheck:
    try:
        function()
    except Exception as exc:
        return VerificationCheck(name, False, f"{type(exc).__name__}: {exc}")
    return VerificationCheck(name, True)


def run_export_verification() -> VerificationReport:
    """Run portable serialization, rendering, file, and optional PDF checks."""

    checks: list[VerificationCheck] = []

    def serialization_check() -> None:
        value = {
            "fraction": Fraction(2, 3),
            "complex": 1 + 2j,
            "array": np.arange(6).reshape(2, 3),
        }
        normalized = to_serializable(value)
        assert normalized["fraction"]["numerator"] == 2
        assert normalized["array"]["shape"] == [2, 3]
        json.dumps(normalized)

    checks.append(_capture("portable serialization", serialization_check))

    def latex_render_check() -> None:
        document = MathDocument(DocumentMetadata(title="A & B", author="X_Y"))
        document.paragraph("100% exact")
        document.equation(r"E=mc^2", numbered=True, label="eq:energy")
        document.table(("n", "value"), ((0, 1), (1, Fraction(1, 2))))
        source = render_latex(document)
        assert r"A \& B" in source
        assert r"X\_Y" in source
        assert "\\begin{equation}" in source
        assert latex_escape("a_b") == r"a\_b"

    checks.append(_capture("LaTeX rendering", latex_render_check))

    def file_exports_check() -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            options = ExportOptions(overwrite=True, title="Verification")
            value = [{"n": index, "value": index * index} for index in range(6)]
            for suffix in ("json", "csv", "tsv", "txt", "md", "html", "tex"):
                result = export_data(value, directory / f"values.{suffix}", options=options)
                result.certificate.require()
            npy = export_data(np.arange(8), directory / "values.npy", options=options)
            npz = export_data({"a": np.arange(4), "b": np.eye(2)}, directory / "values.npz", options=options)
            npy.certificate.require()
            npz.certificate.require()
            assert np.array_equal(np.load(npy.path), np.arange(8))
            with np.load(npz.path) as loaded:
                assert np.array_equal(loaded["a"], np.arange(4))

    checks.append(_capture("portable file exports", file_exports_check))

    def bundle_check() -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = export_bundle(
                {"alpha": 1, "beta": (2, 3)},
                directory / "bundle.zip",
                formats=("json", "markdown", "latex"),
                options=ExportOptions(overwrite=True),
            )
            result.certificate.require()
            assert "MANIFEST.sha256" in result.members
            assert result.manifest_path.is_file()

    checks.append(_capture("multi-format export bundle", bundle_check))

    def pdf_check() -> None:
        if not any(available_latex_engines().values()):
            return
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            document = MathDocument(DocumentMetadata(title="PDF verification"))
            document.paragraph("The generated document is compilable.")
            document.equation(r"\sum_{k=0}^{n} k = \frac{n(n+1)}{2}")
            result = export_data(document, directory / "report.pdf", options=ExportOptions(overwrite=True))
            result.certificate.require()
            assert result.path.read_bytes().startswith(b"%PDF")
            compilation = compile_latex(render_latex(document), directory, source_name="direct.tex")
            compilation.certificate.require()

    checks.append(_capture("optional LaTeX PDF compilation", pdf_check))

    return VerificationReport(tuple(checks))


__all__ = ["run_export_verification"]
