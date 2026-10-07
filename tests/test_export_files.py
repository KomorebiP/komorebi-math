from __future__ import annotations

import json
import shutil
import struct
import zlib
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from komorebi_math.exceptions import BackendUnavailableError, ValidationError
from komorebi_math.export import (
    DocumentMetadata,
    ExportOptions,
    MathDocument,
    available_latex_engines,
    compile_latex,
    export_bundle,
    export_data,
    export_many,
)


def _one_pixel_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    image = zlib.compress(b"\x00\xff\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", image) + chunk(b"IEND", b"")


def test_text_and_data_exports(tmp_path: Path) -> None:
    options = ExportOptions(overwrite=True, title="Values")
    value = [{"n": index, "value": Fraction(1, index + 1)} for index in range(5)]
    for suffix in ("json", "csv", "tsv", "txt", "md", "html", "tex"):
        result = export_data(value, tmp_path / f"values.{suffix}", options=options)
        result.certificate.require()
        assert result.path.stat().st_size > 0
    parsed = json.loads((tmp_path / "values.json").read_text())
    assert parsed[1]["value"]["denominator"] == 2
    assert "n,value" in (tmp_path / "values.csv").read_text()
    assert "\\documentclass" in (tmp_path / "values.tex").read_text()



def test_document_figure_assets_are_localized(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "pixel.png"
    source.write_bytes(_one_pixel_png())
    document = MathDocument(DocumentMetadata(title="Figure export"))
    document.figure(source, caption="One pixel")
    options = ExportOptions(overwrite=True)

    for suffix in ("md", "html", "tex"):
        result = export_data(document, tmp_path / f"figure.{suffix}", options=options)
        result.certificate.require()
        asset = tmp_path / "figure_asset_001.png"
        assert asset.is_file()
        assert asset in result.companion_files
        assert asset.name in result.path.read_text()


@pytest.mark.skipif(not any(available_latex_engines().values()), reason="LaTeX is not installed")
def test_pdf_figure_asset_compiles(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "pixel.png"
    source.write_bytes(_one_pixel_png())
    document = MathDocument(DocumentMetadata(title="Figure PDF"))
    document.figure(source, caption="One pixel", width=r"0.1\textwidth")
    result = export_data(
        document,
        tmp_path / "figure.pdf",
        options=ExportOptions(overwrite=True),
    )
    result.certificate.require()
    assert result.path.read_bytes().startswith(b"%PDF")
    assert tmp_path / "figure_asset_001.png" in result.companion_files

def test_numpy_exports(tmp_path: Path) -> None:
    array = np.arange(12).reshape(3, 4)
    npy = export_data(array, tmp_path / "array.npy")
    npz = export_data({"a": array, "b": np.eye(2)}, tmp_path / "arrays.npz")
    assert np.array_equal(np.load(npy.require()), array)
    with np.load(npz.require()) as loaded:
        assert np.array_equal(loaded["a"], array)
        assert np.array_equal(loaded["b"], np.eye(2))


def test_overwrite_policy(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    export_data({"a": 1}, path)
    with pytest.raises(ValidationError):
        export_data({"a": 2}, path)
    result = export_data({"a": 2}, path, options=ExportOptions(overwrite=True))
    result.certificate.require()


def test_export_many_and_bundle(tmp_path: Path) -> None:
    results = export_many(
        {"alpha": 1, "beta": 2},
        tmp_path / "many",
        base_name="report",
        formats=("json", "markdown", "latex"),
        options=ExportOptions(overwrite=True),
    )
    assert {result.format.value for result in results} == {"json", "markdown", "latex"}

    bundle = export_bundle(
        {"alpha": 1, "beta": 2},
        tmp_path / "report.zip",
        formats=("json", "markdown", "latex"),
        options=ExportOptions(overwrite=True),
    )
    bundle.certificate.require()
    assert bundle.manifest_path.is_file()
    with zipfile.ZipFile(bundle.path) as archive:
        assert "MANIFEST.sha256" in archive.namelist()
        assert any(name.endswith(".tex") for name in archive.namelist())



def test_export_many_latex_and_pdf_share_generated_source(tmp_path: Path) -> None:
    if not any(available_latex_engines().values()):
        pytest.skip("LaTeX is not installed")
    document = MathDocument(DocumentMetadata(title="Shared source"))
    document.equation(r"a^2+b^2=c^2")
    results = export_many(
        document,
        tmp_path,
        base_name="shared",
        formats=("latex", "pdf"),
    )
    assert tuple(result.format.value for result in results) == ("latex", "pdf")
    assert (tmp_path / "shared.tex").is_file()
    assert (tmp_path / "shared.pdf").is_file()

def test_yaml_optional_dependency(tmp_path: Path) -> None:
    if shutil.which("python") is None:
        pytest.skip("Python executable unavailable")
    try:
        result = export_data({"a": 1}, tmp_path / "value.yaml")
    except BackendUnavailableError:
        return
    result.certificate.require()
    assert "a:" in result.path.read_text()


@pytest.mark.skipif(not any(available_latex_engines().values()), reason="LaTeX is not installed")
def test_pdf_export_and_direct_compile(tmp_path: Path) -> None:
    document = MathDocument(DocumentMetadata(title="Export test"))
    document.paragraph("This PDF was generated from LaTeX.")
    document.equation(r"\sum_{k=1}^{n}k=\frac{n(n+1)}{2}")
    result = export_data(
        document,
        tmp_path / "report.pdf",
        options=ExportOptions(overwrite=True),
    )
    result.certificate.require()
    assert result.path.read_bytes().startswith(b"%PDF")
    assert (tmp_path / "report.tex").is_file()

    compilation = compile_latex((tmp_path / "report.tex"), tmp_path, source_name="unused.tex")
    compilation.certificate.require()
