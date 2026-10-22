"""Exact documentation coverage for the frozen stable API."""

from __future__ import annotations

from komorebi_math import __version__

import json
import inspect
from pathlib import Path

import pytest

from komorebi_math.exceptions import ValidationError
from komorebi_math.release import (
    StableDocumentationContract,
    check_stable_documentation,
    create_stable_documentation_contract,
    load_stable_documentation,
)
from komorebi_math.release import documentation as documentation_module


def test_stable_documentation_covers_every_stable_symbol_exactly_once() -> None:
    contract = load_stable_documentation()
    assert contract.package_version == __version__
    assert len(contract.symbols) == 131
    assert len({entry.name for entry in contract.symbols}) == 131
    assert all(entry.summary for entry in contract.symbols)
    assert all(entry.reference == "docs/STABLE_API.md" for entry in contract.symbols)


def test_stable_documentation_matches_live_namespace() -> None:
    reference = load_stable_documentation()
    current = create_stable_documentation_contract()
    assert reference == current
    assert reference.digest == current.digest
    report = check_stable_documentation()
    report.certificate.require()
    assert report.missing == report.extra == report.changed == report.deprecated == ()


def test_stable_data_summaries_are_semantic_and_platform_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_getdoc = inspect.getdoc

    def platform_variant(value: object) -> str | None:
        if isinstance(value, Path):
            return "Path subclass for Windows systems."
        return original_getdoc(value)

    monkeypatch.setattr(documentation_module.inspect, "getdoc", platform_variant)
    contract = documentation_module.create_stable_documentation_contract()
    assert contract == load_stable_documentation()
    summaries = {entry.name: entry.summary for entry in contract.symbols}
    assert summaries["DEFAULT_PORTABLE_CODEC_CONTRACT"] == (
        "Path to the committed portable-codec compatibility contract."
    )
    assert summaries["DEFAULT_STABLE_SEMANTICS_FIXTURE"] == (
        "Path to the committed stable-semantics fixture."
    )
    assert summaries["PolyphaseCoordinateRecord"] == (
        "Union of the stable one-dimensional polyphase coordinate record types."
    )


def test_stable_documentation_detects_changed_summary(tmp_path: Path) -> None:
    payload = load_stable_documentation().to_dict()
    payload["symbols"][0]["summary"] += " changed"
    path = tmp_path / "documentation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = check_stable_documentation(path)
    assert not report.passed
    assert report.changed


def test_stable_documentation_rejects_unknown_fields() -> None:
    payload = load_stable_documentation().to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="fields do not match"):
        StableDocumentationContract.from_dict(payload)
