"""Assembly cannot silently drop a damaged or missing reviewed source group."""

import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "verification/build_paper_coverage.py"
SPEC = importlib.util.spec_from_file_location("coverage_builder", PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def fragments(root):
    directory = root / "verification/paper_coverage"
    directory.mkdir(parents=True)
    for index, name in enumerate(sorted(builder.REQUIRED_FRAGMENTS)):
        source = {"id": str(index), "title": name, "path": name, "sha256": "0" * 64}
        mechanism = {
            "key": str(index),
            "source_id": str(index),
            "section": "Section 1",
            "summary": "A bounded source scope",
            "status": "theory_only",
            "api": [],
            "tests": [],
            "boundaries": ["No executable evaluator is supplied."],
        }
        (directory / name).write_text(json.dumps({"sources": [source], "mechanisms": [mechanism]}))
    return directory


def test_missing_required_source_group_stops_assembly(tmp_path):
    directory = fragments(tmp_path)
    (directory / "core.json").unlink()
    with pytest.raises(ValueError, match=r"core\.json"):
        builder.assemble(tmp_path)


def test_empty_required_group_cannot_disappear_from_a_valid_catalog(tmp_path):
    directory = fragments(tmp_path)
    (directory / "core.json").write_text(json.dumps({"sources": [], "mechanisms": []}))
    with pytest.raises(ValueError, match="Malformed coverage fragment"):
        builder.assemble(tmp_path)


@pytest.mark.parametrize("name", ["core.json", "additional_review.json"])
def test_corrupt_or_incomplete_fragments_are_not_run_outputs(tmp_path, name):
    directory = fragments(tmp_path)
    (directory / name).write_text(json.dumps({"sources": []}))
    with pytest.raises(ValueError, match="Malformed coverage fragment"):
        builder.assemble(tmp_path)


def test_explicit_run_report_is_separate_from_inventory(tmp_path):
    directory = fragments(tmp_path)
    (directory / "stress.json").write_text(json.dumps({"kind": "finite_mathematical_stress_report"}))
    catalog = builder.assemble(tmp_path)
    assert len(catalog["sources"]) == len(builder.REQUIRED_FRAGMENTS)
    assert len(catalog["mechanisms"]) == len(builder.REQUIRED_FRAGMENTS)
    assert len(catalog["fragments"]) == len(builder.REQUIRED_FRAGMENTS)
    (directory / "core.json").write_text(json.dumps({"kind": "finite_mathematical_stress_report"}))
    with pytest.raises(ValueError, match="Malformed coverage fragment"):
        builder.assemble(tmp_path)
