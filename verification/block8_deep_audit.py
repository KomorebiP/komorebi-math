from __future__ import annotations

import json
import random
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from komorebi_math import Certificate, ComputationResult, Exactness
from komorebi_math.export import (
    DocumentMetadata,
    ExportOptions,
    MathDocument,
    available_export_formats,
    available_latex_engines,
    export_bundle,
    export_data,
    latex_escape,
    render_html,
    render_latex,
    render_markdown,
    table_rows,
    to_serializable,
)


@dataclass
class GroupResult:
    name: str
    checks: int
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def run_group(name: str, function) -> GroupResult:
    checks = 0
    failures: list[str] = []

    def check(condition: bool, detail: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(detail)

    try:
        function(check)
    except Exception as exc:
        failures.append(f"unexpected {type(exc).__name__}: {exc}")
    return GroupResult(name, checks, failures)


def serialization_group(check) -> None:
    rng = random.Random(801)
    for index in range(4000):
        rows = rng.randrange(1, 5)
        columns = rng.randrange(1, 5)
        array = np.array([[rng.randrange(-1000, 1001) for _ in range(columns)] for _ in range(rows)])
        fraction = Fraction(rng.randrange(-1000, 1001), rng.randrange(1, 1001))
        value = {
            "index": index,
            "array": array,
            "fraction": fraction,
            "complex": complex(rng.random(), rng.random()),
            "tuple": (index, index + 1),
        }
        normalized = to_serializable(value)
        check(normalized["array"]["shape"] == [rows, columns], f"array shape {index}")
        check(normalized["fraction"]["denominator"] == fraction.denominator, f"fraction {index}")
        check(json.loads(json.dumps(normalized))["index"] == index, f"json round trip {index}")


def table_group(check) -> None:
    rng = random.Random(802)
    for width in range(1, 8):
        for height in range(0, 40):
            array = np.arange(max(height, 1) * width).reshape(max(height, 1), width)
            headers, rows = table_rows(array[:height])
            check(len(headers) == width, f"table width {width},{height}")
            check(len(rows) == height, f"table height {width},{height}")
    for _ in range(3000):
        length = rng.randrange(0, 15)
        mapping = {
            "a": [rng.randrange(-50, 51) for _ in range(length)],
            "b": [rng.randrange(-50, 51) for _ in range(length)],
        }
        headers, rows = table_rows(mapping)
        check(headers == ("a", "b"), "mapping headers")
        check(len(rows) == length, "mapping length")


def rendering_group(check) -> None:
    rng = random.Random(803)
    specials = "\\&%$#_{}~^"
    for index in range(2500):
        text = "".join(rng.choice("abcXYZ012 " + specials) for _ in range(30))
        escaped = latex_escape(text)
        check(len(escaped) >= len(text), f"escape length {index}")
        document = MathDocument(DocumentMetadata(title=f"Report {index}"))
        document.paragraph(text)
        document.equation(r"x^2+y^2=z^2")
        document.table(("n", "value"), ((0, index), (1, index + 1)))
        latex = render_latex(document)
        markdown = render_markdown(document)
        html = render_html(document)
        check("\\begin{document}" in latex and "\\end{document}" in latex, f"latex shell {index}")
        check("| n | value |" in markdown, f"markdown table {index}")
        check("<table>" in html, f"html table {index}")


def file_group(check) -> None:
    rng = random.Random(804)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        options = ExportOptions(overwrite=True, title="Deep audit")
        for index in range(500):
            rows = tuple((row, rng.randrange(-1000, 1001)) for row in range(rng.randrange(1, 12)))
            json_result = export_data(rows, root / f"value_{index}.json", options=options)
            csv_result = export_data(rows, root / f"value_{index}.csv", options=options)
            tex_result = export_data(rows, root / f"value_{index}.tex", options=options)
            check(json_result.valid, f"json export {index}")
            check(csv_result.valid, f"csv export {index}")
            check(tex_result.valid, f"tex export {index}")
            check(json.loads(json_result.path.read_text())[0][0] == 0, f"json content {index}")
        for index in range(300):
            array = np.arange(index % 17 + 1)
            result = export_data(array, root / f"array_{index}.npy", options=options)
            check(np.array_equal(np.load(result.path), array), f"npy {index}")


def certificate_group(check) -> None:
    for index in range(1000):
        certificate = Certificate(
            valid=index % 2 == 0,
            exactness=Exactness.EXACT if index % 3 == 0 else Exactness.NUMERICAL,
            residuals={"residual": 1.0 / (index + 1)},
            theorem_id=f"TEST-{index}",
        )
        result = ComputationResult(index, certificate)
        normalized = to_serializable(result)
        check(normalized["value"] == index, f"result value {index}")
        check(normalized["certificate"]["theorem_id"] == f"TEST-{index}", f"certificate {index}")


def bundle_group(check) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for index in range(100):
            result = export_bundle(
                {"index": index, "values": [index, index + 1]},
                root / f"bundle_{index}.zip",
                formats=("json", "markdown", "latex"),
                options=ExportOptions(overwrite=True),
            )
            check(result.valid if hasattr(result, "valid") else result.certificate.valid, f"bundle {index}")
            check("MANIFEST.sha256" in result.members, f"bundle manifest {index}")
            check(result.manifest_path.is_file(), f"external manifest {index}")


def pdf_group(check) -> None:
    engines = available_latex_engines()
    check(set(engines) == {"pdflatex", "xelatex", "lualatex", "latexmk"}, "engine keys")
    if not any(engines.values()):
        return
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for index in range(4):
            document = MathDocument(DocumentMetadata(title=f"PDF {index}"))
            document.paragraph("Generated through a local LaTeX engine.")
            document.equation(r"\sum_{k=0}^{n} k = \frac{n(n+1)}{2}")
            document.table(("n", "value"), tuple((n, n * n) for n in range(35 if index == 3 else 5)))
            result = export_data(document, root / f"report_{index}.pdf", options=ExportOptions(overwrite=True))
            check(result.certificate.valid, f"pdf certificate {index}")
            check(result.path.read_bytes().startswith(b"%PDF"), f"pdf header {index}")
            check((root / f"report_{index}.tex").is_file(), f"tex companion {index}")


def source_group(check) -> None:
    import komorebi_math
    import komorebi_math.export as export_module

    check(komorebi_math.__version__ == "0.8.0a1", "version")
    check("pdf" in available_export_formats(), "pdf format")
    for name in (
        "MathDocument",
        "ExportOptions",
        "export_data",
        "export_bundle",
        "compile_latex",
    ):
        check(hasattr(export_module, name), f"export public {name}")
        check(hasattr(komorebi_math, name), f"top level public {name}")


GROUPS = (
    ("serialization", serialization_group),
    ("table normalization", table_group),
    ("document rendering", rendering_group),
    ("file exports", file_group),
    ("certificate exports", certificate_group),
    ("multi-format bundles", bundle_group),
    ("LaTeX PDF compilation", pdf_group),
    ("source integrity", source_group),
)


def main() -> int:
    results = [run_group(name, function) for name, function in GROUPS]
    payload = {
        "groups": [asdict(result) | {"passed": result.passed} for result in results],
        "checks": sum(result.checks for result in results),
        "failures": sum(len(result.failures) for result in results),
        "passed": all(result.passed for result in results),
    }
    output = Path(__file__).with_name("BLOCK8_DEEP_AUDIT.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Block 8 deep audit: {payload['checks']} checks, {payload['failures']} failures")
    for result in results:
        print(f"  {result.name}: {result.checks} checks, {len(result.failures)} failures")
        for failure in result.failures[:10]:
            print(f"    {failure}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
