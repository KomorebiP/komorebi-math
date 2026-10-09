"""Check math preservation and local assets in an already built MkDocs site.

This documentation-only check needs the development Markdown dependencies.
It checks generated HTML and asset bytes; browser SVG execution is separate.
"""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.math: list[str] = []
        self.code: list[str] = []
        self.links: list[str] = []
        self.assets: list[str] = []
        self._math_depth = 0
        self._code_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if self._math_depth:
            self._math_depth += 1
        elif "arithmatex" in (values.get("class") or "").split():
            self._math_depth = 1
            self.math.append("")
        if tag == "code":
            self._code_depth += 1
            self.code.append("")
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"])
        if tag == "link" and values.get("rel") == "stylesheet" and values.get("href"):
            self.assets.append(values["href"])

    def handle_endtag(self, tag: str) -> None:
        if self._math_depth:
            self._math_depth -= 1
        if tag == "code" and self._code_depth:
            self._code_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._math_depth:
            self.math[-1] += data
        if self._code_depth:
            self.code[-1] += data


def inspect(text: str) -> Page:
    page = Page()
    page.feed(text)
    return page


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check(root: Path, site: Path) -> dict:
    pages = sorted(site.rglob("*.html"))
    require(bool(pages), "the built documentation site is empty")
    import markdown

    # These samples failed without math preservation: brackets made a link,
    # backslashes disappeared, and underscores entered Markdown processing.
    source = r"""Inline \(q_i^2\).

\[
\mathcal M_p[R](z)=\prod_{j\ge0}R\!\left(z^{p^j}\right).
\]

```python
literal = r"\(x_[R](z)\)"
```
"""
    sample = inspect(markdown.markdown(
        source, extensions=["fenced_code", "pymdownx.arithmatex"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    ))
    formula = r"\mathcal M_p[R](z)=\prod_{j\ge0}R\!\left(z^{p^j}\right)."
    require(len(sample.math) == 2, "inline/display math wrappers are missing")
    require(sample.math[0] == r"\(q_i^2\)", "inline math changed during Markdown processing")
    require(formula in sample.math[1], "display math changed during Markdown processing")
    require("z" not in sample.links, "math became a Markdown hyperlink")
    require(sample.code == ['literal = r"\\(x_[R](z)\\)"\n'], "fenced code was typeset or changed")

    math_pages = 0
    math_expressions = 0
    for path in pages:
        page = inspect(path.read_text(encoding="utf-8"))
        require(not any(value.startswith(("http://", "https://", "//")) for value in page.assets),
                f"external script or stylesheet in {path.relative_to(site)}")
        if page.math:
            math_pages += 1
            math_expressions += len(page.math)
            require(any(value.endswith("assets/mathjax-config.js") for value in page.assets),
                    f"math configuration missing in {path.relative_to(site)}")
            require(any(value.endswith("assets/mathjax-3.2.2/tex-svg-full.js") for value in page.assets),
                    f"local renderer missing in {path.relative_to(site)}")
    target = inspect((site / "NORMALIZED_GENERATING_FUNCTIONS/index.html").read_text(encoding="utf-8"))
    require(any(formula in value for value in target.math), "the original generating-function formula is not preserved")
    require("z" not in target.links, "the original spurious formula link remains")

    vendor = root / "docs/assets/mathjax-3.2.2"
    provenance = json.loads((vendor / "SOURCE.json").read_text(encoding="utf-8"))
    verified_assets = []
    for entry in provenance["files"]:
        relative = entry["path"]
        source_bytes = (vendor / relative).read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        require(digest == entry["sha256"], f"vendored asset hash mismatch: {relative}")
        require(source_bytes == (site / "assets/mathjax-3.2.2" / relative).read_bytes(),
                f"built asset differs from source: {relative}")
        verified_assets.append({"path": relative, "sha256": digest})
    return {"passed": True, "pages": len(pages), "math_pages": math_pages,
            "math_expressions": math_expressions, "assets": verified_assets,
            "scope": "Markdown preservation, code exclusion, local HTML assets, and pinned asset bytes. "
                     "Browser execution and numerical correctness are separate checks."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = check(args.root.resolve(), args.site_dir.resolve())
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["passed"]:
        print(f"Math documentation passed: {report['math_expressions']} expressions in {report['math_pages']} pages")
    else:
        print(f"Math documentation FAILED: {report['error']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
