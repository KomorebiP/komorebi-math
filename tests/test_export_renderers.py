from __future__ import annotations

from fractions import Fraction

import pytest

from komorebi_math.export import (
    DocumentMetadata,
    HtmlRenderOptions,
    LatexRenderOptions,
    MathDocument,
    latex_escape,
    render_html,
    render_latex,
    render_markdown,
    render_text,
)


def build_document(row_count: int = 2) -> MathDocument:
    document = MathDocument(
        DocumentMetadata(
            title="A & B",
            author="X_Y",
            abstract="A compact report.",
        )
    )
    document.paragraph("100% of ordinary text is escaped.")
    document.section("Results")
    document.equation(r"E=mc^2", numbered=True, label="eq:energy")
    document.table(
        ("n", "value"),
        tuple((index, Fraction(1, index + 1)) for index in range(row_count)),
        caption="Exact values",
    )
    document.code("print('ok')", language="python")
    document.bullet_list(("first", "second"))
    return document


def test_latex_renderer_is_compilable_source() -> None:
    source = render_latex(build_document())
    assert r"\title{A \& B}" in source
    assert r"\author{X\_Y}" in source
    assert r"100\%" in source
    assert r"\begin{equation}" in source
    assert r"\begin{tabular}" in source
    assert source.rstrip().endswith(r"\end{document}")


def test_large_table_uses_longtable() -> None:
    source = render_latex(build_document(row_count=30))
    assert r"\begin{longtable}" in source
    assert r"\end{longtable}" in source


def test_markdown_html_and_text_renderers() -> None:
    document = build_document()
    markdown = render_markdown(document)
    html = render_html(document, options=HtmlRenderOptions(include_mathjax=True))
    text = render_text(document)
    assert "# A & B" in markdown
    assert "MathJax" in html
    assert "<table>" in html
    assert "A & B" in text


def test_latex_escape() -> None:
    assert latex_escape(r"a_b&c%") == r"a\_b\&c\%"


def test_invalid_latex_document_class_is_rejected() -> None:
    with pytest.raises(ValueError):
        LatexRenderOptions(document_class="article}\\input{bad")
