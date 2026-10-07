"""Text, Markdown, HTML, and LaTeX renderers."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .model import (
    CodeBlock,
    Equation,
    Figure,
    Heading,
    ListBlock,
    MathDocument,
    Paragraph,
    RawLatex,
    Table,
)

_LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclass(frozen=True, slots=True)
class LatexRenderOptions:
    """Configuration for generated LaTeX documents."""

    document_class: str = "article"
    class_options: tuple[str, ...] = ("11pt",)
    packages: tuple[str, ...] = (
        "amsmath",
        "amssymb",
        "booktabs",
        "longtable",
        "array",
        "graphicx",
        "hyperref",
        "geometry",
        "microtype",
    )
    include_table_of_contents: bool = False
    number_sections: bool = True
    geometry: str = "margin=1in"

    def __post_init__(self) -> None:
        name_pattern = re.compile(r"^[A-Za-z0-9_.-]+$")
        if not name_pattern.fullmatch(self.document_class):
            raise ValueError("LaTeX document class contains unsupported characters")
        if any(not name_pattern.fullmatch(package) for package in self.packages):
            raise ValueError("LaTeX package names contain unsupported characters")


@dataclass(frozen=True, slots=True)
class HtmlRenderOptions:
    """Configuration for standalone HTML exports."""

    include_mathjax: bool = False
    title: str | None = None


def latex_escape(value: Any) -> str:
    """Escape ordinary text for safe inclusion in LaTeX."""

    text = str(value)
    return "".join(_LATEX_ESCAPE.get(character, character) for character in text)


def latex_label(value: str) -> str:
    """Normalize a user label to a conservative LaTeX label."""

    normalized = re.sub(r"[^A-Za-z0-9:._-]+", "-", value.strip())
    return normalized.strip("-") or "item"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def render_markdown(document: MathDocument) -> str:
    """Render a document as Markdown with TeX equations."""

    lines: list[str] = [f"# {document.metadata.title}", ""]
    if document.metadata.author:
        lines.extend([f"**Author:** {document.metadata.author}", ""])
    if document.metadata.date:
        lines.extend([f"**Date:** {document.metadata.date}", ""])
    if document.metadata.abstract:
        lines.extend(["## Abstract", "", document.metadata.abstract, ""])
    for block in document.blocks:
        if isinstance(block, Heading):
            lines.extend([f"{'#' * min(block.level + 1, 6)} {block.text}", ""])
        elif isinstance(block, Paragraph):
            lines.extend([block.text, ""])
        elif isinstance(block, Equation):
            if block.description:
                lines.extend([block.description, ""])
            lines.extend(["$$", block.latex, "$$", ""])
        elif isinstance(block, Table):
            lines.append("| " + " | ".join(_markdown_cell(value) for value in block.headers) + " |")
            lines.append("| " + " | ".join("---" for _ in block.headers) + " |")
            lines.extend(
                "| " + " | ".join(_markdown_cell(value) for value in row) + " |" for row in block.rows
            )
            if block.caption:
                lines.extend(["", f"*{block.caption}*"])
            lines.append("")
        elif isinstance(block, CodeBlock):
            lines.extend([f"```{block.language}", block.code, "```", ""])
        elif isinstance(block, ListBlock):
            for index, item in enumerate(block.items, start=1):
                prefix = f"{index}." if block.ordered else "-"
                lines.append(f"{prefix} {item}")
            lines.append("")
        elif isinstance(block, Figure):
            caption = block.caption or block.path.name
            lines.extend([f"![{caption}]({block.path.as_posix()})", ""])
        elif isinstance(block, RawLatex):
            lines.extend(["```latex", block.source, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_text(document: MathDocument) -> str:
    """Render a document as plain UTF-8 text."""

    lines: list[str] = [document.metadata.title, "=" * len(document.metadata.title), ""]
    if document.metadata.author:
        lines.append(f"Author: {document.metadata.author}")
    if document.metadata.date:
        lines.append(f"Date: {document.metadata.date}")
    if document.metadata.author or document.metadata.date:
        lines.append("")
    if document.metadata.abstract:
        lines.extend(["Abstract", "--------", document.metadata.abstract, ""])
    for block in document.blocks:
        if isinstance(block, Heading):
            lines.extend([block.text, "-" * len(block.text), ""])
        elif isinstance(block, Paragraph):
            lines.extend([block.text, ""])
        elif isinstance(block, Equation):
            if block.description:
                lines.append(block.description)
            lines.extend([block.latex, ""])
        elif isinstance(block, Table):
            rows = [tuple(str(value) for value in block.headers)] + [
                tuple(str(value) for value in row) for row in block.rows
            ]
            widths = [max(len(row[index]) for row in rows) for index in range(len(block.headers))]
            lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])))
            lines.append("  ".join("-" * width for width in widths))
            lines.extend(
                "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
                for row in rows[1:]
            )
            if block.caption:
                lines.append(block.caption)
            lines.append("")
        elif isinstance(block, CodeBlock):
            lines.extend([block.code, ""])
        elif isinstance(block, ListBlock):
            for index, item in enumerate(block.items, start=1):
                prefix = f"{index}." if block.ordered else "*"
                lines.append(f"{prefix} {item}")
            lines.append("")
        elif isinstance(block, Figure):
            lines.extend([f"Figure: {block.path}  {block.caption or ''}".rstrip(), ""])
        elif isinstance(block, RawLatex):
            lines.extend([block.source, ""])
    return "\n".join(lines).rstrip() + "\n"


def render_html(document: MathDocument, *, options: HtmlRenderOptions | None = None) -> str:
    """Render a standalone HTML document."""

    options = options or HtmlRenderOptions()
    title = options.title or document.metadata.title
    head = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;line-height:1.55}",
        "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #bbb;padding:.45rem;text-align:left}",
        "pre{overflow:auto;background:#f6f6f6;padding:1rem}code{font-family:ui-monospace,monospace}",
        ".math{overflow:auto;text-align:center;margin:1rem 0;font-family:serif}",
        "figure{text-align:center}figure img{max-width:100%;height:auto}",
        "</style>",
    ]
    if options.include_mathjax:
        head.extend(
            [
                '<script>window.MathJax={tex:{inlineMath:[["\\(","\\)"],["$","$"]]}}</script>',
                '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>',
            ]
        )
    head.extend(["</head>", "<body>", f"<h1>{html.escape(document.metadata.title)}</h1>"])
    if document.metadata.author:
        head.append(f"<p><strong>Author:</strong> {html.escape(document.metadata.author)}</p>")
    if document.metadata.date:
        head.append(f"<p><strong>Date:</strong> {html.escape(document.metadata.date)}</p>")
    if document.metadata.abstract:
        head.extend(["<section><h2>Abstract</h2>", f"<p>{html.escape(document.metadata.abstract)}</p></section>"])

    body: list[str] = []
    for block in document.blocks:
        if isinstance(block, Heading):
            level = min(block.level + 1, 6)
            identifier = f' id="{html.escape(block.label)}"' if block.label else ""
            body.append(f"<h{level}{identifier}>{html.escape(block.text)}</h{level}>")
        elif isinstance(block, Paragraph):
            body.append(f"<p>{html.escape(block.text)}</p>")
        elif isinstance(block, Equation):
            if block.description:
                body.append(f"<p>{html.escape(block.description)}</p>")
            body.append(f'<div class="math">\\[{html.escape(block.latex)}\\]</div>')
        elif isinstance(block, Table):
            body.append("<table><thead><tr>")
            body.extend(f"<th>{html.escape(str(value))}</th>" for value in block.headers)
            body.append("</tr></thead><tbody>")
            for row in block.rows:
                body.append("<tr>")
                body.extend(f"<td>{html.escape(str(value))}</td>" for value in row)
                body.append("</tr>")
            body.append("</tbody></table>")
            if block.caption:
                body.append(f"<p><em>{html.escape(block.caption)}</em></p>")
        elif isinstance(block, CodeBlock):
            body.append(
                f'<pre><code class="language-{html.escape(block.language)}">{html.escape(block.code)}</code></pre>'
            )
        elif isinstance(block, ListBlock):
            tag = "ol" if block.ordered else "ul"
            body.append(f"<{tag}>")
            body.extend(f"<li>{html.escape(item)}</li>" for item in block.items)
            body.append(f"</{tag}>")
        elif isinstance(block, Figure):
            body.append("<figure>")
            body.append(f'<img src="{html.escape(block.path.as_posix())}" alt="{html.escape(block.caption or block.path.name)}">')
            if block.caption:
                body.append(f"<figcaption>{html.escape(block.caption)}</figcaption>")
            body.append("</figure>")
        elif isinstance(block, RawLatex):
            body.append(f"<pre><code>{html.escape(block.source)}</code></pre>")
    return "\n".join(head + body + ["</body>", "</html>", ""])


def _latex_heading(block: Heading, *, number_sections: bool) -> str:
    commands = {1: "section", 2: "subsection", 3: "subsubsection", 4: "paragraph", 5: "subparagraph", 6: "subparagraph"}
    command = commands[block.level]
    star = "" if number_sections else "*"
    label = f"\n\\label{{{latex_label(block.label)}}}" if block.label else ""
    return f"\\{command}{star}{{{latex_escape(block.text)}}}{label}"


def _latex_table(block: Table) -> str:
    alignment = "".join(block.alignment or tuple("l" for _ in block.headers))
    if len(block.rows) > 25:
        lines = [f"\\begin{{longtable}}{{{alignment}}}"]
        if block.caption:
            lines.append(f"\\caption{{{latex_escape(block.caption)}}}\\\\")
        if block.label:
            lines.append(f"\\label{{{latex_label(block.label)}}}")
        lines.extend(
            [
                r"\toprule",
                " & ".join(latex_escape(value) for value in block.headers) + r" \\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                " & ".join(latex_escape(value) for value in block.headers) + r" \\",
                r"\midrule",
                r"\endhead",
            ]
        )
        lines.extend(" & ".join(latex_escape(value) for value in row) + r" \\" for row in block.rows)
        lines.extend([r"\bottomrule", r"\end{longtable}"])
        return "\n".join(lines)

    lines = [r"\begin{table}[htbp]", r"\centering"]
    if block.caption:
        lines.append(f"\\caption{{{latex_escape(block.caption)}}}")
    if block.label:
        lines.append(f"\\label{{{latex_label(block.label)}}}")
    lines.extend(
        [
            f"\\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            " & ".join(latex_escape(value) for value in block.headers) + r" \\",
            r"\midrule",
        ]
    )
    lines.extend(" & ".join(latex_escape(value) for value in row) + r" \\" for row in block.rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _latex_code(block: CodeBlock) -> str:
    caption = f"\\textbf{{{latex_escape(block.caption)}}}\n\n" if block.caption else ""
    return caption + "\\begin{verbatim}\n" + block.code + "\n\\end{verbatim}"


def _latex_figure(block: Figure) -> str:
    lines = [r"\begin{figure}[htbp]", r"\centering"]
    lines.append(f"\\includegraphics[width={block.width}]{{\\detokenize{{{block.path.as_posix()}}}}}")
    if block.caption:
        lines.append(f"\\caption{{{latex_escape(block.caption)}}}")
    if block.label:
        lines.append(f"\\label{{{latex_label(block.label)}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def render_latex(document: MathDocument, *, options: LatexRenderOptions | None = None) -> str:
    """Render a complete compilable LaTeX document."""

    options = options or LatexRenderOptions()
    class_options = f"[{','.join(options.class_options)}]" if options.class_options else ""
    lines = [f"\\documentclass{class_options}{{{options.document_class}}}", r"\usepackage{iftex}", r"\ifPDFTeX", r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}", r"\fi"]
    for package in options.packages:
        if package == "geometry" and options.geometry:
            lines.append(f"\\usepackage[{options.geometry}]{{geometry}}")
        else:
            lines.append(f"\\usepackage{{{package}}}")
    lines.extend(
        [
            r"\hypersetup{hidelinks}",
            f"\\title{{{latex_escape(document.metadata.title)}}}",
            f"\\author{{{latex_escape(document.metadata.author or '')}}}",
            f"\\date{{{latex_escape(document.metadata.date or '')}}}",
            r"\begin{document}",
            r"\maketitle",
        ]
    )
    if document.metadata.abstract:
        lines.extend([r"\begin{abstract}", latex_escape(document.metadata.abstract), r"\end{abstract}"])
    if options.include_table_of_contents:
        lines.extend([r"\tableofcontents", r"\clearpage"])
    for block in document.blocks:
        if isinstance(block, Heading):
            lines.append(_latex_heading(block, number_sections=options.number_sections))
        elif isinstance(block, Paragraph):
            lines.append(latex_escape(block.text))
        elif isinstance(block, Equation):
            if block.description:
                lines.append(latex_escape(block.description))
            if block.numbered:
                lines.append(r"\begin{equation}")
                lines.append(block.latex)
                if block.label:
                    lines.append(f"\\label{{{latex_label(block.label)}}}")
                lines.append(r"\end{equation}")
            else:
                lines.extend([r"\[", block.latex, r"\]"])
        elif isinstance(block, Table):
            lines.append(_latex_table(block))
        elif isinstance(block, CodeBlock):
            lines.append(_latex_code(block))
        elif isinstance(block, ListBlock):
            environment = "enumerate" if block.ordered else "itemize"
            lines.append(f"\\begin{{{environment}}}")
            lines.extend(f"\\item {latex_escape(item)}" for item in block.items)
            lines.append(f"\\end{{{environment}}}")
        elif isinstance(block, Figure):
            lines.append(_latex_figure(block))
        elif isinstance(block, RawLatex):
            lines.append(block.source)
    if document.metadata.keywords:
        lines.extend(
            [
                r"\bigskip",
                r"\noindent\textbf{Keywords:} " + latex_escape(", ".join(document.metadata.keywords)),
            ]
        )
    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


__all__ = [
    "HtmlRenderOptions",
    "LatexRenderOptions",
    "latex_escape",
    "latex_label",
    "render_html",
    "render_latex",
    "render_markdown",
    "render_text",
]
