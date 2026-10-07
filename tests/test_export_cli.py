from __future__ import annotations

import json
from pathlib import Path

from komorebi_math.cli.main import main
from komorebi_math.export import available_latex_engines


def test_export_formats_cli(capsys) -> None:
    assert main(["export", "formats"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "latex" in payload["formats"]
    assert "pdf" in payload["formats"]


def test_export_sequence_cli(tmp_path: Path, capsys) -> None:
    output = tmp_path / "fibonacci.tex"
    assert main([
        "export",
        "sequence",
        "fibonacci",
        "--terms",
        "12",
        "--compile",
        "--output",
        str(output),
        "--title",
        "Fibonacci report",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "latex"
    assert output.is_file()
    assert "Fibonacci" in output.read_text()


def test_export_json_cli(tmp_path: Path, capsys) -> None:
    source = tmp_path / "data.json"
    source.write_text(json.dumps({"x": [1, 2], "y": [3, 4]}), encoding="utf-8")
    output = tmp_path / "data.csv"
    assert main([
        "export",
        "json",
        "--input",
        str(source),
        "--output",
        str(output),
    ]) == 0
    json.loads(capsys.readouterr().out)
    assert output.read_text().splitlines()[0] == "x,y"


def test_latex_engines_cli(capsys) -> None:
    assert main(["latex", "engines"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"pdflatex", "xelatex", "lualatex", "latexmk"}


def test_latex_compile_cli(tmp_path: Path, capsys) -> None:
    if not any(available_latex_engines().values()):
        return
    source = tmp_path / "small.tex"
    source.write_text(
        "\\documentclass{article}\\begin{document}test\\end{document}",
        encoding="utf-8",
    )
    output = tmp_path / "compiled.pdf"
    assert main([
        "latex",
        "compile",
        str(source),
        "--output",
        str(output),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["certificate"]["valid"] is True
    assert output.is_file()
