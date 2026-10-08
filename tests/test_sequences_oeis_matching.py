"""Offset alignment, incomplete evidence, and strict local OEIS input parsing."""
from __future__ import annotations

import gzip
import json
import pytest

from komorebi_math.sequences import (
    LocalOEISCorpus, OEISPrefixMatch, OEISRecord, compare_oeis_prefix,
    find_local_oeis_matches, load_local_oeis, parse_oeis_internal, parse_oeis_stripped,
)
from komorebi_math.exceptions import ValidationError


def test_declared_offsets_align_indices_without_assuming_zero() -> None:
    record = OEISRecord("A000045", (1, 1, 2, 3, 5, 8), 1, "Fibonacci")
    match = compare_oeis_prefix((2, 3, 5), record, query_offset=3, minimum_terms=3)
    assert match.passed
    assert match.index_alignment_verified
    assert match.compared_terms == 3
    mismatch = compare_oeis_prefix((2, 99, 5), record, query_offset=3, minimum_terms=3)
    assert not mismatch.passed
    assert mismatch.first_mismatch_position == 1
    assert mismatch.first_mismatch_sequence_index == 4


def test_unknown_reference_offset_does_not_grant_verified_index_alignment() -> None:
    record = OEISRecord("A000045", (0, 1, 1, 2, 3, 5))
    positional = compare_oeis_prefix(record.terms, record)
    assert positional.passed
    assert not positional.index_alignment_verified
    explicit = compare_oeis_prefix(record.terms, record, query_offset=0)
    assert explicit.status == "unknown-reference-offset"
    assert not explicit.passed


def test_exhausted_reference_cannot_verify_query_tail() -> None:
    record = OEISRecord("A000045", (0, 1, 1, 2, 3, 5), 0)
    match = compare_oeis_prefix((0, 1, 1, 2, 3, 5, 999), record)
    assert match.status == "reference-exhausted-prefix"
    assert match.agrees_on_overlap
    assert not match.complete_query_match
    assert not match.passed
    assert find_local_oeis_matches(match.project_terms, [record]) == (match,)


def test_agreeing_minimum_then_known_mismatch_is_excluded() -> None:
    record = OEISRecord("A000045", (0, 1, 1, 2, 3, 5, 8), 0)
    query = (0, 1, 1, 2, 3, 5, 99)
    assert find_local_oeis_matches(query, [record]) == ()
    match = compare_oeis_prefix(query, record)
    assert match.matched_terms == 6
    assert match.compared_terms == 7
    assert match.first_mismatch_position == 6


def test_match_result_cannot_be_forged_by_supplying_a_status() -> None:
    record = OEISRecord("A000045", (0, 1, 1, 2, 3, 5), 0)
    with pytest.raises(TypeError):
        OEISPrefixMatch(record, (9,) * 6, status="exact-prefix")
    assert not OEISPrefixMatch(record, (9,) * 6).passed


def test_negative_offset_and_query_before_reference_are_distinct() -> None:
    record = OEISRecord("A000045", (4, 5, 6, 7), -2)
    assert compare_oeis_prefix((5, 6), record, query_offset=-1, minimum_terms=2).passed
    before = compare_oeis_prefix((4, 5), record, query_offset=-3, minimum_terms=2)
    assert before.status == "query-start-before-reference"
    assert not before.passed


def test_empty_and_insufficient_queries_do_not_pass() -> None:
    record = OEISRecord("A000045", (0, 1, 1, 2, 3, 5), 0)
    assert not compare_oeis_prefix((), record).passed
    assert not compare_oeis_prefix((0, 1), record).passed
    with pytest.raises(ValidationError):
        find_local_oeis_matches((0, 1), [record])


def test_internal_fields_use_signed_rows_and_documented_continuation_order() -> None:
    text = "%T A000045 2,3,5\n%S A000045 0,1,1,\n%W A000045 -2,-3,-5\n%V A000045 0,-1,-1,\n%N A000045 Signed control\n%O A000045 -2,3\n"
    corpus = parse_oeis_internal(text)
    assert corpus.records[0].terms == (0, -1, -1, -2, -3, -5)
    assert corpus.records[0].offset == -2
    assert corpus.records[0].name == "Signed control"


def test_stripped_snapshot_keeps_unknown_offsets_and_reads_names() -> None:
    corpus = parse_oeis_stripped("# dated local snapshot\nA000045 ,0,1,1,2,3,5,\n", names_text="A000045 Fibonacci\n")
    assert corpus.records[0].offset is None
    assert corpus.records[0].name == "Fibonacci"
    assert corpus.records[0].terms == (0, 1, 1, 2, 3, 5)


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_record_terms_and_offsets_reject_lossy_coercion(bad) -> None:
    with pytest.raises(ValidationError):
        OEISRecord("A000045", (bad,))
    with pytest.raises(ValidationError):
        OEISRecord("A000045", (1,), bad)
    with pytest.raises(ValidationError):
        compare_oeis_prefix((0, 1), OEISRecord("A000045", (0, 1)), minimum_terms=bad)


@pytest.mark.parametrize("text", ["%S A000045 0,1.5,2", "%S A000045 0,,2", "%O A000045 offset 0,1"])
def test_malformed_internal_numeric_fields_are_rejected(text) -> None:
    with pytest.raises(ValidationError):
        parse_oeis_internal(text)


def test_duplicate_identifiers_are_rejected() -> None:
    record = OEISRecord("A000045", (0, 1))
    with pytest.raises(ValidationError, match="duplicate"):
        LocalOEISCorpus((record, record))
    with pytest.raises(ValidationError, match="duplicate"):
        parse_oeis_stripped("A000045 ,0,1,\nA000045 ,1,2,\n")


def test_local_gzip_loader_and_json_provenance_roundtrip(tmp_path) -> None:
    path = tmp_path / "snapshot.seq.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("%S A000045 0,1,1,2,3,5\n%O A000045 0,2\n")
    assert load_local_oeis(path).records[0].offset == 0
    json_path = tmp_path / "snapshot.json"
    payload = {"provenance": {"official_export_commit": "historical-local-control"},
               "sequences": [{"id": "A000045", "terms": [0, 1, 1, 2, 3, 5], "offset": 0}]}
    json_path.write_text(json.dumps(payload))
    corpus = load_local_oeis(json_path, format="json")
    assert corpus.provenance["official_export_commit"] == "historical-local-control"
    assert compare_oeis_prefix(corpus.records[0].terms, corpus.records[0], query_offset=0).passed


def test_json_loader_does_not_truncate_noninteger_terms(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"sequences": [{"id": "A000045", "terms": [1.5]}]}))
    with pytest.raises(ValidationError):
        load_local_oeis(path, format="json")
