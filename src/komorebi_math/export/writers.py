"""File writers for data, reports, LaTeX, and compiled PDFs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..certificates import Certificate, Exactness
from ..exceptions import BackendUnavailableError, ValidationError
from .latex import LatexCompileOptions, compile_latex
from .model import Figure, MathDocument
from .renderers import HtmlRenderOptions, LatexRenderOptions, render_html, render_latex, render_markdown, render_text
from .report import document_from_data
from .serialization import table_rows, to_serializable


class ExportFormat(StrEnum):
    """Supported on disk export formats."""

    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    TSV = "tsv"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"
    PDF = "pdf"
    NPY = "npy"
    NPZ = "npz"


_SUFFIX_FORMATS = {
    ".json": ExportFormat.JSON,
    ".yaml": ExportFormat.YAML,
    ".yml": ExportFormat.YAML,
    ".csv": ExportFormat.CSV,
    ".tsv": ExportFormat.TSV,
    ".txt": ExportFormat.TEXT,
    ".md": ExportFormat.MARKDOWN,
    ".markdown": ExportFormat.MARKDOWN,
    ".html": ExportFormat.HTML,
    ".htm": ExportFormat.HTML,
    ".tex": ExportFormat.LATEX,
    ".pdf": ExportFormat.PDF,
    ".npy": ExportFormat.NPY,
    ".npz": ExportFormat.NPZ,
}

_FORMAT_ALIASES = {
    "md": ExportFormat.MARKDOWN,
    "tex": ExportFormat.LATEX,
    "txt": ExportFormat.TEXT,
    "yml": ExportFormat.YAML,
}

_FORMAT_SUFFIXES = {
    ExportFormat.JSON: ".json",
    ExportFormat.YAML: ".yaml",
    ExportFormat.CSV: ".csv",
    ExportFormat.TSV: ".tsv",
    ExportFormat.TEXT: ".txt",
    ExportFormat.MARKDOWN: ".md",
    ExportFormat.HTML: ".html",
    ExportFormat.LATEX: ".tex",
    ExportFormat.PDF: ".pdf",
    ExportFormat.NPY: ".npy",
    ExportFormat.NPZ: ".npz",
}


@dataclass(frozen=True, slots=True)
class ExportOptions:
    """Common export behavior and renderer options."""

    title: str = "Komorebi Math Export"
    author: str | None = None
    date: str | None = None
    abstract: str | None = None
    overwrite: bool = False
    create_parents: bool = True
    preserve_types: bool = True
    json_indent: int = 2
    include_certificate: bool = True
    keep_tex: bool = True
    html: HtmlRenderOptions = field(default_factory=HtmlRenderOptions)
    latex: LatexRenderOptions = field(default_factory=LatexRenderOptions)
    compiler: LatexCompileOptions = field(default_factory=LatexCompileOptions)


@dataclass(frozen=True, slots=True)
class ExportResult:
    """One completed file export and its validation certificate."""

    path: Path
    format: ExportFormat
    bytes_written: int
    companion_files: tuple[Path, ...]
    certificate: Certificate

    @property
    def valid(self) -> bool:
        return self.certificate.valid

    def require(self) -> Path:
        """Return the primary path after requiring a valid certificate."""

        self.certificate.require()
        return self.path


@dataclass(frozen=True, slots=True)
class ExportBundleResult:
    """A ZIP bundle containing several exports and a hash manifest."""

    path: Path
    members: tuple[str, ...]
    formats: tuple[ExportFormat, ...]
    manifest_path: Path
    certificate: Certificate

    @property
    def valid(self) -> bool:
        """Return whether the bundle certificate is valid."""

        return self.certificate.valid

    def require(self) -> Path:
        """Return the bundle path after requiring a valid certificate."""

        self.certificate.require()
        return self.path


def available_export_formats() -> tuple[str, ...]:
    """Return every supported public format name."""

    return tuple(format_.value for format_ in ExportFormat)


def infer_export_format(path: str | Path) -> ExportFormat:
    """Infer an export format from a file suffix."""

    suffix = Path(path).suffix.lower()
    try:
        return _SUFFIX_FORMATS[suffix]
    except KeyError as exc:
        raise ValidationError(f"cannot infer an export format from suffix: {suffix or '<none>'}") from exc


def _resolve_format(path: Path, format_: ExportFormat | str | None) -> ExportFormat:
    if format_ is None:
        return infer_export_format(path)
    if isinstance(format_, ExportFormat):
        return format_
    normalized = str(format_).strip().lower()
    if normalized in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[normalized]
    return ExportFormat(normalized)


def _prepare_path(path: Path, options: ExportOptions) -> None:
    if path.exists() and not options.overwrite:
        raise ValidationError(f"export path already exists: {path}")
    if options.create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.exists():
        raise ValidationError(f"export parent directory does not exist: {path.parent}")


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> int:
    with tempfile.NamedTemporaryFile("w", encoding=encoding, delete=False, dir=path.parent, prefix=path.name, suffix=".tmp") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)
    return path.stat().st_size


def _atomic_write_bytes(path: Path, data: bytes) -> int:
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=path.name, suffix=".tmp") as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(path)
    path.chmod(0o644)
    return path.stat().st_size


def _document(value: Any, options: ExportOptions) -> MathDocument:
    return document_from_data(
        value,
        title=options.title,
        author=options.author,
        date=options.date,
        abstract=options.abstract,
        include_certificate=options.include_certificate,
    )


def _yaml_text(value: Any, options: ExportOptions) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise BackendUnavailableError("YAML export requires PyYAML, install komorebi-math[yaml]") from exc
    normalized = to_serializable(value, preserve_types=options.preserve_types)
    return str(yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True))


def _csv_bytes(value: Any, *, delimiter: str) -> bytes:
    headers, rows = table_rows(value)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _same_file_bytes(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(1024 * 1024)
            right_chunk = right.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _localize_figure_assets(
    document: MathDocument,
    target: Path,
    *,
    overwrite: bool,
) -> tuple[MathDocument, tuple[Path, ...]]:
    copied: list[Path] = []
    localized = MathDocument(document.metadata)
    figure_index = 0
    for block in document.blocks:
        if not isinstance(block, Figure):
            localized.add(block)
            continue
        figure_index += 1
        source = block.path.expanduser().resolve()
        if not source.is_file():
            raise ValidationError(f"figure file does not exist: {source}")
        suffix = source.suffix.lower() or ".bin"
        asset = target.parent / f"{target.stem}_asset_{figure_index:03d}{suffix}"
        if source != asset:
            if asset.exists() and not _same_file_bytes(source, asset):
                if not overwrite:
                    raise ValidationError(f"figure asset already exists with different content: {asset}")
                shutil.copy2(source, asset)
            elif not asset.exists():
                shutil.copy2(source, asset)
        copied.append(asset)
        localized.add(Figure(Path(asset.name), caption=block.caption, label=block.label, width=block.width))
    return localized, tuple(copied)


def export_data(
    value: Any,
    path: str | Path,
    *,
    format: ExportFormat | str | None = None,
    options: ExportOptions | None = None,
) -> ExportResult:
    """Export one Python or Komorebi Math value to a selected file format."""

    options = options or ExportOptions()
    target = Path(path).expanduser().resolve()
    format_ = _resolve_format(target, format)
    _prepare_path(target, options)
    companions: list[Path] = []

    if format_ is ExportFormat.JSON:
        text = json.dumps(
            to_serializable(value, preserve_types=options.preserve_types),
            indent=options.json_indent,
            sort_keys=True,
            ensure_ascii=False,
        ) + "\n"
        size = _atomic_write_text(target, text)
    elif format_ is ExportFormat.YAML:
        size = _atomic_write_text(target, _yaml_text(value, options))
    elif format_ is ExportFormat.CSV:
        size = _atomic_write_bytes(target, _csv_bytes(value, delimiter=","))
    elif format_ is ExportFormat.TSV:
        size = _atomic_write_bytes(target, _csv_bytes(value, delimiter="\t"))
    elif format_ is ExportFormat.TEXT:
        size = _atomic_write_text(target, render_text(_document(value, options)))
    elif format_ is ExportFormat.MARKDOWN:
        document, assets = _localize_figure_assets(_document(value, options), target, overwrite=options.overwrite)
        companions.extend(assets)
        size = _atomic_write_text(target, render_markdown(document))
    elif format_ is ExportFormat.HTML:
        document, assets = _localize_figure_assets(_document(value, options), target, overwrite=options.overwrite)
        companions.extend(assets)
        size = _atomic_write_text(target, render_html(document, options=options.html))
    elif format_ is ExportFormat.LATEX:
        document, assets = _localize_figure_assets(_document(value, options), target, overwrite=options.overwrite)
        companions.extend(assets)
        size = _atomic_write_text(target, render_latex(document, options=options.latex))
    elif format_ is ExportFormat.PDF:
        document, assets = _localize_figure_assets(_document(value, options), target, overwrite=options.overwrite)
        companions.extend(assets)
        tex_path = target.with_suffix(".tex")
        tex_source = render_latex(document, options=options.latex)
        tex_preexisting = tex_path.exists()
        if tex_preexisting:
            existing_source = tex_path.read_text(encoding="utf-8")
            if existing_source != tex_source:
                if not options.overwrite:
                    raise ValidationError(f"generated LaTeX path already exists with different content: {tex_path}")
                _atomic_write_text(tex_path, tex_source)
        else:
            _atomic_write_text(tex_path, tex_source)
        compilation = compile_latex(tex_path, target.parent, options=options.compiler)
        compilation.certificate.require()
        compiled_pdf = compilation.require()
        if compiled_pdf != target:
            if target.exists() and not options.overwrite:
                raise ValidationError(f"export path already exists: {target}")
            shutil.copy2(compiled_pdf, target)
            if compiled_pdf.name != target.name:
                compiled_pdf.unlink(missing_ok=True)
        if options.keep_tex or tex_preexisting:
            companions.append(tex_path)
        else:
            tex_path.unlink(missing_ok=True)
        companions.append(compilation.log_path)
        size = target.stat().st_size
    elif format_ is ExportFormat.NPY:
        array = np.asarray(value)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        size = _atomic_write_bytes(target, buffer.getvalue())
    elif format_ is ExportFormat.NPZ:
        buffer = io.BytesIO()
        if isinstance(value, Mapping):
            arrays = {str(key): np.asarray(item) for key, item in value.items()}
        else:
            arrays = {"data": np.asarray(value)}
        np.savez_compressed(buffer, **arrays)
        size = _atomic_write_bytes(target, buffer.getvalue())
    else:
        raise AssertionError(f"unhandled export format: {format_}")

    certificate = Certificate(
        valid=target.is_file() and target.stat().st_size == size and size > 0,
        exactness=Exactness.EXACT,
        assumptions_checked=(
            "output format is supported",
            "destination policy was enforced",
            "primary output exists and is nonempty",
        ),
        diagnostics={
            "path": str(target),
            "format": format_.value,
            "bytes": size,
            "companions": tuple(str(path) for path in companions),
        },
        message=f"exported {format_.value} to {target}",
    )
    return ExportResult(target, format_, size, tuple(companions), certificate)



def export_document(
    document: MathDocument,
    path: str | Path,
    *,
    format: ExportFormat | str | None = None,
    options: ExportOptions | None = None,
) -> ExportResult:
    """Export one renderer independent mathematical document."""

    return export_data(document, path, format=format, options=options)


def export_many(
    value: Any,
    directory: str | Path,
    *,
    base_name: str,
    formats: Sequence[ExportFormat | str],
    options: ExportOptions | None = None,
) -> tuple[ExportResult, ...]:
    """Export one value into several formats inside one directory."""

    options = options or ExportOptions()
    target_dir = Path(directory).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExportResult] = []
    for requested in formats:
        format_ = _resolve_format(Path(base_name + ".tmp"), requested)
        suffix = _FORMAT_SUFFIXES[format_]
        results.append(export_data(value, target_dir / f"{base_name}{suffix}", format=format_, options=options))
    return tuple(results)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(
    value: Any,
    path: str | Path,
    *,
    base_name: str = "komorebi_export",
    formats: Sequence[ExportFormat | str] = (
        ExportFormat.JSON,
        ExportFormat.MARKDOWN,
        ExportFormat.LATEX,
    ),
    options: ExportOptions | None = None,
) -> ExportBundleResult:
    """Create a ZIP archive with several exports and a SHA256 manifest."""

    options = options or ExportOptions()
    target = Path(path).expanduser().resolve()
    _prepare_path(target, options)
    with tempfile.TemporaryDirectory(prefix="komorebi_export_bundle_") as temporary:
        directory = Path(temporary)
        bundle_options = ExportOptions(
            title=options.title,
            author=options.author,
            date=options.date,
            abstract=options.abstract,
            overwrite=True,
            create_parents=True,
            preserve_types=options.preserve_types,
            json_indent=options.json_indent,
            include_certificate=options.include_certificate,
            keep_tex=options.keep_tex,
            html=options.html,
            latex=options.latex,
            compiler=options.compiler,
        )
        results = export_many(
            value,
            directory,
            base_name=base_name,
            formats=formats,
            options=bundle_options,
        )
        files: dict[str, str] = {}
        for result in results:
            files[result.path.name] = _sha256(result.path)
            for companion in result.companion_files:
                if companion.is_file() and companion.parent == directory:
                    files[companion.name] = _sha256(companion)
        manifest = directory / "MANIFEST.sha256"
        manifest.write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(files.items())) + "\n", encoding="utf-8")
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for file in sorted(directory.iterdir()):
                if file.is_file():
                    info = zipfile.ZipInfo(file.name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    archive.writestr(info, file.read_bytes())
        manifest_copy = target.with_suffix(target.suffix + ".manifest")
        manifest_copy.write_bytes(manifest.read_bytes())

    certificate = Certificate(
        valid=target.is_file() and target.stat().st_size > 0,
        exactness=Exactness.EXACT,
        assumptions_checked=(
            "every requested export completed",
            "bundle manifest was generated",
            "ZIP member order and timestamps were normalized",
        ),
        diagnostics={
            "path": str(target),
            "formats": tuple(result.format.value for result in results),
            "manifest": str(manifest_copy),
        },
        message=f"export bundle created at {target}",
    )
    members = tuple(sorted(files)) + ("MANIFEST.sha256",)
    return ExportBundleResult(
        target,
        members,
        tuple(result.format for result in results),
        manifest_copy,
        certificate,
    )


__all__ = [
    "ExportBundleResult",
    "ExportFormat",
    "ExportOptions",
    "ExportResult",
    "available_export_formats",
    "export_bundle",
    "export_data",
    "export_document",
    "export_many",
    "infer_export_format",
]
