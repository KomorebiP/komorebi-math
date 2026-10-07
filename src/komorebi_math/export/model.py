"""Document objects for portable mathematical exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadata used by Markdown, HTML, and LaTeX document renderers."""

    title: str = "Komorebi Math Export"
    author: str | None = None
    date: str | None = None
    abstract: str | None = None
    keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValidationError("document title must not be empty")


class DocumentBlock:
    """Marker base class for document blocks."""


@dataclass(frozen=True, slots=True)
class Heading(DocumentBlock):
    """One document heading."""

    text: str
    level: int = 1
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValidationError("heading text must not be empty")
        if not 1 <= self.level <= 6:
            raise ValidationError("heading level must lie between 1 and 6")


@dataclass(frozen=True, slots=True)
class Paragraph(DocumentBlock):
    """One prose paragraph whose text is escaped by each renderer."""

    text: str


@dataclass(frozen=True, slots=True)
class Equation(DocumentBlock):
    """One LaTeX mathematical expression."""

    latex: str
    label: str | None = None
    numbered: bool = False
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.latex.strip():
            raise ValidationError("equation source must not be empty")


@dataclass(frozen=True, slots=True)
class Table(DocumentBlock):
    """A rectangular table with optional caption and label."""

    headers: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    caption: str | None = None
    label: str | None = None
    alignment: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        width = len(self.headers)
        if width == 0:
            raise ValidationError("table must have at least one column")
        if any(len(row) != width for row in self.rows):
            raise ValidationError("every table row must match the header width")
        if self.alignment is not None:
            if len(self.alignment) != width:
                raise ValidationError("table alignment must match the header width")
            if any(value not in {"l", "c", "r"} for value in self.alignment):
                raise ValidationError("table alignment entries must be l, c, or r")

    @classmethod
    def from_rows(
        cls,
        headers: Sequence[str],
        rows: Iterable[Sequence[Any]],
        *,
        caption: str | None = None,
        label: str | None = None,
        alignment: Sequence[str] | None = None,
    ) -> "Table":
        """Build a validated table from ordinary Python row sequences."""

        return cls(
            tuple(str(value) for value in headers),
            tuple(tuple(row) for row in rows),
            caption=caption,
            label=label,
            alignment=None if alignment is None else tuple(alignment),
        )


@dataclass(frozen=True, slots=True)
class CodeBlock(DocumentBlock):
    """A literal code block."""

    code: str
    language: str = "text"
    caption: str | None = None


@dataclass(frozen=True, slots=True)
class ListBlock(DocumentBlock):
    """An ordered or unordered list."""

    items: tuple[str, ...]
    ordered: bool = False

    def __post_init__(self) -> None:
        if not self.items:
            raise ValidationError("list block must contain at least one item")


@dataclass(frozen=True, slots=True)
class Figure(DocumentBlock):
    """A figure file referenced by a document."""

    path: Path
    caption: str | None = None
    label: str | None = None
    width: str = r"0.9\textwidth"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.width.strip():
            raise ValidationError("figure width must not be empty")


@dataclass(frozen=True, slots=True)
class RawLatex(DocumentBlock):
    """Trusted raw LaTeX inserted only by the LaTeX renderer."""

    source: str


@dataclass(slots=True)
class MathDocument:
    """A renderer independent mathematical document."""

    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    blocks: list[DocumentBlock] = field(default_factory=list)

    def add(self, *blocks: DocumentBlock) -> "MathDocument":
        """Append blocks and return the document for fluent construction."""

        self.blocks.extend(blocks)
        return self

    def section(self, title: str, *, level: int = 1, label: str | None = None) -> "MathDocument":
        """Append one heading."""

        return self.add(Heading(title, level=level, label=label))

    def paragraph(self, text: str) -> "MathDocument":
        """Append one prose paragraph."""

        return self.add(Paragraph(text))

    def equation(
        self,
        latex: str,
        *,
        label: str | None = None,
        numbered: bool = False,
        description: str | None = None,
    ) -> "MathDocument":
        """Append one mathematical equation."""

        return self.add(Equation(latex, label=label, numbered=numbered, description=description))

    def table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[Any]],
        *,
        caption: str | None = None,
        label: str | None = None,
        alignment: Sequence[str] | None = None,
    ) -> "MathDocument":
        """Append one rectangular table."""

        return self.add(
            Table.from_rows(
                headers,
                rows,
                caption=caption,
                label=label,
                alignment=alignment,
            )
        )

    def code(self, source: str, *, language: str = "python", caption: str | None = None) -> "MathDocument":
        """Append one literal code block."""

        return self.add(CodeBlock(source, language=language, caption=caption))

    def bullet_list(self, items: Iterable[str], *, ordered: bool = False) -> "MathDocument":
        """Append one list."""

        return self.add(ListBlock(tuple(str(item) for item in items), ordered=ordered))

    def figure(
        self,
        path: str | Path,
        *,
        caption: str | None = None,
        label: str | None = None,
        width: str = r"0.9\textwidth",
    ) -> "MathDocument":
        """Append one figure reference."""

        return self.add(Figure(Path(path), caption=caption, label=label, width=width))


__all__ = [
    "CodeBlock",
    "DocumentBlock",
    "DocumentMetadata",
    "Equation",
    "Figure",
    "Heading",
    "ListBlock",
    "MathDocument",
    "Paragraph",
    "RawLatex",
    "Table",
]
