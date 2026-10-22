"""Curated Companion C integration without unreviewed API expansion."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CROSSWALK = ROOT / "verification" / "companion_c_crosswalk.json"


def _resolve(qualified_name: str):
    module_name, attribute = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def test_companion_crosswalk_is_closed_and_resolvable() -> None:
    payload = json.loads(CROSSWALK.read_text(encoding="utf-8"))
    assert payload["schema"] == "komorebi-math.companion-crosswalk"
    assert payload["schema_version"] == 1
    assert payload["package_version"] == "0.16.0a2"
    assert len(payload["source_archive"]["sha256"]) == 64
    assert len(payload["entries"]) == 8
    for entry in payload["entries"]:
        assert entry["companion_measure"]
        assert entry["reference_residual"] >= 0
        assert entry["package_targets"]
        assert entry["block16_status"]
        for target in entry["package_targets"]:
            assert _resolve(target) is not None


def test_companion_stable_promotions_are_limited_to_reviewed_overlap() -> None:
    payload = json.loads(CROSSWALK.read_text(encoding="utf-8"))
    stable_entries = [
        entry
        for entry in payload["entries"]
        if entry["block16_status"] != "research API retained, not promoted during freeze"
    ]
    assert stable_entries
    assert all(
        all(target.startswith("komorebi_math.stable.") for target in entry["package_targets"])
        for entry in stable_entries
    )
