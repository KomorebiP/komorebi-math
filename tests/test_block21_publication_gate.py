"""Block 21 external reproducibility and publication-gate tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from komorebi_math.compatibility import check_public_api
from verification.block21_publication_gate import main, run_gate

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from block21_public_capability_tour import build_record  # noqa: E402


def test_generic_alias_signatures_are_portable_across_supported_python_versions() -> None:
    report = check_public_api(strict_signatures=True)
    report.certificate.require()
    assert report.signature_changed == ()


def test_public_capability_tour_is_deterministic_and_exact() -> None:
    left = build_record()
    right = build_record()
    assert left == right
    assert left["passed"] is True
    assert left["capabilities"]["finite_integration"]["exact_integral"] == "106/5"
    assert left["capabilities"]["typed_expression"]["exact_integral"] == "5/4"
    assert left["capabilities"]["stable_contract"]["symbol_count"] == 131
    assert len(left["record_sha256"]) == 64


def test_quick_publication_gate_passes_without_platform_overclaim() -> None:
    report = run_gate(ROOT, skip_mkdocs=True)
    assert report.passed, report.to_dict()
    assert report.mode == "quick"
    assert report.full_test_record is None
    assert any("complete source test suite" in warning for warning in report.warnings)
    assert report.capability_record["passed"] is True
    assert len(report.to_dict()["report_sha256"]) == 64


def test_publication_gate_cli_writes_self_digesting_records(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    markdown = tmp_path / "gate.md"
    assert (
        main(
            [
                "--root",
                str(ROOT),
                "--output",
                str(output),
                "--markdown",
                str(markdown),
                "--skip-mkdocs",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert len(payload["report_sha256"]) == 64
    assert "Block 21 Publication Gate" in markdown.read_text(encoding="utf-8")


def test_publication_launchers_and_policies_are_present() -> None:
    required = (
        ".gitignore",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "REPRODUCIBILITY.md",
        "Run_Block21_Publication_Gate.ps1",
        "run_block21_publication_gate.sh",
        ".github/workflows/release.yml",
    )
    assert all((ROOT / name).is_file() for name in required)
