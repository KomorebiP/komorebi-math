"""A valid inventory must not turn missing mathematics into approval."""

import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from komorebi_math.audit.coverage import SCHEMA, load_coverage_catalog, validate_coverage_catalog


def inventory():
    return {
        "schema": SCHEMA,
        "sources": [
            {
                "id": "source",
                "title": "Finite construction",
                "path": "source.txt",
                "sha256": hashlib.sha256(b"source").hexdigest(),
            }
        ],
        "mechanisms": [
            {
                "key": "finite",
                "source_id": "source",
                "section": "Section 2",
                "summary": "A bounded exact construction",
                "status": "partial",
                "api": ["komorebi_math.arithmetic.collatz.CollatzWord"],
                "tests": ["tests/test_collatz_words.py"],
                "boundaries": ["Finite words do not prove universal descent."],
            }
        ],
    }


def test_live_links_and_original_bytes_with_partial_boundary(tmp_path):
    catalog = inventory()
    (tmp_path / "source.txt").write_bytes(b"source")
    root = Path(__file__).resolve().parents[1]
    report = validate_coverage_catalog(catalog, repository_root=root, paper_source_root=tmp_path)
    assert report.valid and report.checked_apis == 1
    data = report.to_dict()
    assert data["source_bytes_checked"] and data["test_paths_checked"]
    assert not data["all_catalogued_mechanisms_implemented"]
    assert data["unresolved_mechanisms"] == ["finite"]
    (tmp_path / "source.txt").write_bytes(b"different edition")
    changed = validate_coverage_catalog(catalog, paper_source_root=tmp_path)
    assert not changed.valid and not changed.source_bytes_checked
    assert "original source hash differs" in changed.errors[0]


def test_archive_member_hash_is_checked_without_execution(tmp_path):
    catalog = inventory()
    catalog["sources"][0].update(archive="source.zip", path="manuscript/section.txt")
    with zipfile.ZipFile(tmp_path / "source.zip", "w") as archive:
        archive.writestr("manuscript/section.txt", b"source")
        archive.writestr("not_executed.py", "raise RuntimeError('not a source verifier')")
    report = validate_coverage_catalog(catalog, paper_source_root=tmp_path)
    assert report.valid and report.source_bytes_checked
    catalog["sources"][0]["sha256"] = "0" * 64
    assert not validate_coverage_catalog(catalog, paper_source_root=tmp_path).valid


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["sources"].append(deepcopy(d["sources"][0])),
        lambda d: d["mechanisms"].append(deepcopy(d["mechanisms"][0])),
        lambda d: d["mechanisms"][0].update(source_id="unrecorded"),
        lambda d: d["mechanisms"][0].update(status="approved"),
        lambda d: d["mechanisms"][0].update(boundaries=[]),
        lambda d: d["mechanisms"][0].update(status="implemented", tests=[]),
        lambda d: d["mechanisms"][0].update(api=["os.system"]),
        lambda d: d["mechanisms"][0].update(tests=["../outside.py"]),
        lambda d: d["sources"][0].update(path="/private/source.txt"),
        lambda d: d["sources"][0].update(path="C:/outside.txt"),
        lambda d: d["sources"][0].update(path="C:outside.txt"),
        lambda d: d["sources"][0].update(sha256="old edition"),
    ],
)
def test_forged_or_incomplete_metadata_never_validates(mutation):
    catalog = inventory()
    mutation(catalog)
    report = validate_coverage_catalog(catalog, resolve_apis=False)
    assert not report.valid and report.errors
    assert not report.to_dict()["all_catalogued_mechanisms_implemented"]


def test_a_live_api_and_repository_test_must_really_exist():
    catalog = inventory()
    catalog["mechanisms"][0]["api"] = ["komorebi_math.arithmetic.collatz.NotAnAPI"]
    assert not validate_coverage_catalog(catalog).valid
    catalog = inventory()
    catalog["mechanisms"][0]["tests"] = ["tests/not_a_test.py"]
    report = validate_coverage_catalog(catalog, repository_root=Path(__file__).resolve().parents[1])
    assert not report.valid and not report.test_paths_checked


def test_metadata_only_validation_cannot_approve_declared_implementation():
    catalog = inventory()
    catalog["mechanisms"][0].update(status="implemented", api=["komorebi_math.no_such_module.no_such_api"])
    report = validate_coverage_catalog(catalog, resolve_apis=False)
    assert report.valid
    assert not report.api_links_checked
    assert not report.to_dict()["all_catalogued_mechanisms_implemented"]
    assert not validate_coverage_catalog(catalog).valid


def test_nonfinite_and_opaque_catalogs_cannot_be_evidence():
    catalog = inventory()
    catalog["unrecorded_number"] = float("nan")
    assert not validate_coverage_catalog(catalog).valid
    catalog["unrecorded_number"] = object()
    assert not validate_coverage_catalog(catalog).valid


def test_report_retains_its_owned_state_after_caller_mutation():
    catalog = inventory()
    report = validate_coverage_catalog(catalog)
    before = json.dumps(report.to_dict(), sort_keys=True)
    catalog["mechanisms"][0]["boundaries"].clear()
    catalog["mechanisms"][0]["status"] = "implemented"
    assert json.dumps(report.to_dict(), sort_keys=True) == before


def test_shipped_inventory_has_all_twelve_series_sources_and_live_links():
    catalog = load_coverage_catalog()
    names = {source["path"] for source in catalog["sources"]}
    for roman in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"):
        assert f"Paper_{roman}_Traditional_Final.pdf" in names
    root = Path(__file__).resolve().parents[1]
    report = validate_coverage_catalog(catalog, repository_root=root)
    assert report.valid, report.errors
    assert report.api_links_checked and report.test_paths_checked
    assert not report.to_dict()["all_catalogued_mechanisms_implemented"]
    assert dict(report.status_counts)["missing"] > 0
    catalog["mechanisms"].clear()
    assert load_coverage_catalog()["mechanisms"]
