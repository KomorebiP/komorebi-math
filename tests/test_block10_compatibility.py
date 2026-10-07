from __future__ import annotations

import dataclasses
import inspect
import json
import warnings

import pytest

from komorebi_math.compatibility import (
    ApiCompatibilityReport,
    KomorebiDeprecationWarning,
    PublicApiManifest,
    PublicApiSymbol,
    compare_public_api,
    check_public_api,
    create_public_api_manifest,
    deprecated,
    load_public_api_manifest,
    save_public_api_manifest,
)
from komorebi_math.exceptions import ValidationError


def test_current_public_api_manifest_is_deterministic() -> None:
    first = create_public_api_manifest()
    second = create_public_api_manifest()
    assert first == second
    assert first.digest == second.digest
    assert tuple(sorted(first.symbols)) == first.symbols
    assert any(symbol.name == "PacketJet" for symbol in first.symbols)


def test_public_api_manifest_file_roundtrip(tmp_path) -> None:
    manifest = create_public_api_manifest()
    path = tmp_path / "api.json"
    save_public_api_manifest(manifest, path)
    assert load_public_api_manifest(path) == manifest
    with pytest.raises(ValidationError, match="refusing to overwrite"):
        save_public_api_manifest(manifest, path)


def test_default_manifest_signatures_are_relocation_independent() -> None:
    load_signature = str(inspect.signature(load_public_api_manifest))
    check_signature = str(inspect.signature(check_public_api))

    assert "public_api_0_10.json" not in load_signature
    assert "public_api_0_10.json" not in check_signature
    assert "None" in load_signature
    assert "None" in check_signature


def test_additive_api_change_is_compatible() -> None:
    reference = PublicApiManifest(
        package_version="0.9",
        namespaces=("example",),
        symbols=(PublicApiSymbol("example", "a", "function", "example", "a", "()"),),
    )
    current = PublicApiManifest(
        package_version="0.10",
        namespaces=("example",),
        symbols=(
            PublicApiSymbol("example", "a", "function", "example", "a", "()"),
            PublicApiSymbol("example", "b", "class", "example", "b", "()"),
        ),
    )
    report = compare_public_api(reference, current)
    assert isinstance(report, ApiCompatibilityReport)
    report.certificate.require()
    assert report.added == (("example", "b"),)


def test_removal_kind_change_and_strict_signature_change_fail() -> None:
    reference = PublicApiManifest(
        package_version="0.9",
        namespaces=("example",),
        symbols=(
            PublicApiSymbol("example", "a", "function", "example", "a", "(x)"),
            PublicApiSymbol("example", "b", "class", "example", "b", "()"),
        ),
    )
    removed = PublicApiManifest(
        package_version="0.10",
        namespaces=("example",),
        symbols=(PublicApiSymbol("example", "a", "function", "example", "a", "(x)"),),
    )
    assert not compare_public_api(reference, removed).compatible

    changed_kind = PublicApiManifest(
        package_version="0.10",
        namespaces=("example",),
        symbols=(
            PublicApiSymbol("example", "a", "data", "example", "a", None),
            PublicApiSymbol("example", "b", "class", "example", "b", "()"),
        ),
    )
    assert not compare_public_api(reference, changed_kind).compatible

    changed_signature = PublicApiManifest(
        package_version="0.10",
        namespaces=("example",),
        symbols=(
            PublicApiSymbol("example", "a", "function", "example", "a", "(x, y=0)"),
            PublicApiSymbol("example", "b", "class", "example", "b", "()"),
        ),
    )
    assert compare_public_api(reference, changed_signature).compatible
    assert not compare_public_api(reference, changed_signature, strict_signatures=True).compatible


def test_manifest_validation_rejects_duplicates_and_bad_fields(tmp_path) -> None:
    symbol = PublicApiSymbol("example", "a", "function", "example", "a", "()")
    with pytest.raises(ValidationError, match="duplicate"):
        PublicApiManifest("0.1", ("example",), (symbol, symbol))

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    with pytest.raises(ValidationError, match="fields"):
        load_public_api_manifest(path)


def test_deprecation_decorator_preserves_call_and_metadata() -> None:
    @deprecated(since="0.10", replacement="new_function", remove_in="1.0")
    def old_function(value: int) -> int:
        return value + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert old_function(4) == 5
    assert len(caught) == 1
    assert issubclass(caught[0].category, KomorebiDeprecationWarning)
    assert "new_function" in str(caught[0].message)
    record = getattr(old_function, "__komorebi_deprecation__")
    assert dataclasses.asdict(record)["since"] == "0.10"
