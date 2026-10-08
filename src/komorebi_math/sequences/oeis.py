"""Offline, offset-aware finite comparisons against user-supplied OEIS data.

No corpus download or external request occurs. Positional prefix agreement and
verified index alignment are separate. Exhausted references cannot certify the
unobserved query tail, and finite agreement never proves infinite identity.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from .._number_theory import require_integer
from ..exceptions import ValidationError

_ID = re.compile(r"A\d{6}\Z")
_LINE = re.compile(r"%([A-Z])\s+(A\d{6})(?:\s+(.*))?\Z")
_INT = re.compile(r"[-+]?\d+\Z")


def _terms(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(require_integer(value, name="sequence term") for value in values)


def _positive(value: int, name: str) -> int:
    number = require_integer(value, name=name)
    if number < 1:
        raise ValidationError(f"{name} must be positive")
    return number


@dataclass(frozen=True, slots=True)
class OEISRecord:
    """A local finite OEIS record. Offset None means the indexing is unknown."""
    identifier: str
    terms: tuple[int, ...]
    offset: int | None = None
    name: str = ""
    source: str = "user-supplied local data"

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or _ID.fullmatch(self.identifier) is None:
            raise ValidationError("OEIS identifier must have form A000000")
        object.__setattr__(self, "terms", _terms(self.terms))
        if self.offset is not None:
            object.__setattr__(self, "offset", require_integer(self.offset, name="OEIS offset"))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True, slots=True)
class LocalOEISCorpus:
    """Local records and historical source metadata, without a freshness claim."""
    records: tuple[OEISRecord, ...]
    source: str = "local corpus"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, OEISRecord) for record in records):
            raise ValidationError("corpus entries must be OEISRecord objects")
        if len({record.identifier for record in records}) != len(records):
            raise ValidationError("local corpus contains duplicate OEIS identifiers")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True, slots=True)
class OEISPrefixMatch:
    """A recomputed finite comparison, including incomplete-reference evidence."""
    record: OEISRecord
    project_terms: tuple[int, ...]
    query_offset: int | None = None
    minimum_terms: int = 6
    compared_terms: int = field(init=False)
    matched_terms: int = field(init=False)
    first_mismatch_position: int | None = field(init=False)
    status: str = field(init=False)
    index_alignment_verified: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.record, OEISRecord):
            raise ValidationError("record must be an OEISRecord")
        project = _terms(self.project_terms)
        minimum = _positive(self.minimum_terms, "minimum_terms")
        offset = None if self.query_offset is None else require_integer(self.query_offset, name="query offset")
        aligned = offset is not None and self.record.offset is not None
        start = 0 if not aligned else offset - self.record.offset
        mismatch = None
        compared = matched = 0
        if offset is not None and self.record.offset is None:
            status = "unknown-reference-offset"
        elif start < 0:
            status = "query-start-before-reference"
        else:
            reference = self.record.terms[start:]
            compared = min(len(project), len(reference))
            for position in range(compared):
                if project[position] != reference[position]:
                    mismatch = position
                    break
                matched += 1
            if not compared:
                status = "no-overlap"
            elif mismatch is not None:
                status = "mismatch"
            elif compared < minimum:
                status = "insufficient-terms"
            elif compared < len(project):
                status = "reference-exhausted-prefix"
            else:
                status = "exact-prefix"
        object.__setattr__(self, "project_terms", project)
        object.__setattr__(self, "query_offset", offset)
        object.__setattr__(self, "minimum_terms", minimum)
        object.__setattr__(self, "index_alignment_verified", aligned and start >= 0)
        object.__setattr__(self, "compared_terms", compared)
        object.__setattr__(self, "matched_terms", matched)
        object.__setattr__(self, "first_mismatch_position", mismatch)
        object.__setattr__(self, "status", status)

    @property
    def agrees_on_overlap(self) -> bool:
        """Whether every available compared term agrees and meets the minimum."""
        return self.compared_terms >= self.minimum_terms and self.matched_terms == self.compared_terms

    @property
    def complete_query_match(self) -> bool:
        """Whether all supplied query terms were compared and agreed."""
        return self.agrees_on_overlap and self.compared_terms == len(self.project_terms)

    @property
    def passed(self) -> bool:
        """A complete finite query check; never a theorem of sequence identity."""
        return self.complete_query_match

    @property
    def first_mismatch_sequence_index(self) -> int | None:
        if self.first_mismatch_position is None or not self.index_alignment_verified:
            return None
        return self.query_offset + self.first_mismatch_position


def compare_oeis_prefix(terms: Iterable[int], record: OEISRecord, *,
                        query_offset: int | None = None, minimum_terms: int = 6) -> OEISPrefixMatch:
    """Compare one exact integer query; a declared offset aligns reference indices."""
    return OEISPrefixMatch(record, _terms(terms), query_offset, minimum_terms)


def find_local_oeis_matches(terms: Iterable[int], corpus: LocalOEISCorpus | Iterable[OEISRecord], *,
                           query_offset: int | None = None, minimum_terms: int = 6) -> tuple[OEISPrefixMatch, ...]:
    """Find finite agreeing windows; mismatches after the minimum are excluded.

    Short references can be returned as incomplete candidates. Inspect
    ``complete_query_match`` before treating the entire query as checked.
    """
    query = _terms(terms)
    minimum = _positive(minimum_terms, "minimum_terms")
    if len(query) < minimum:
        raise ValidationError("query must contain at least minimum_terms terms")
    records = corpus.records if isinstance(corpus, LocalOEISCorpus) else tuple(corpus)
    matches = tuple(compare_oeis_prefix(query, record, query_offset=query_offset, minimum_terms=minimum)
                    for record in records)
    return tuple(sorted((match for match in matches if match.agrees_on_overlap),
                        key=lambda match: (-match.compared_terms, match.record.identifier)))


def _parse_terms(payload: str) -> tuple[int, ...]:
    text = payload.strip().strip(",")
    if not text:
        return ()
    tokens = tuple(token.strip() for token in text.split(","))
    if any(_INT.fullmatch(token) is None for token in tokens):
        raise ValidationError("OEIS terms must be comma-delimited exact integers")
    return tuple(int(token) for token in tokens)


def parse_oeis_internal(text: str, *, source: str = "local internal-format text") -> LocalOEISCorpus:
    """Parse %S/%T/%U or signed %V/%W/%X terms, %N names, and %O offsets.

    Continuation fields are assembled in their documented field order rather
    than their appearance order. Missing offsets remain unknown.
    """
    rows: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        match = _LINE.fullmatch(line.strip())
        if match is None:
            continue
        tag, identifier, payload = match.groups()
        payload = payload or ""
        row = rows.setdefault(identifier, {"name": "", "offset": None, "fields": {}})
        if tag in "STUVWX":
            row["fields"].setdefault(tag, []).extend(_parse_terms(payload))
        elif tag == "N":
            row["name"] = payload.strip()
        elif tag == "O":
            token = payload.split(",", 1)[0].strip()
            if _INT.fullmatch(token) is None:
                raise ValidationError("OEIS offset must begin with an exact integer")
            row["offset"] = int(token)
    records = []
    for identifier, row in rows.items():
        fields = row["fields"]
        tags = "VWX" if any(tag in fields for tag in "VWX") else "STU"
        values = tuple(term for tag in tags for term in fields.get(tag, ()))
        records.append(OEISRecord(identifier, values, row["offset"], row["name"], source))
    return LocalOEISCorpus(tuple(records), source)


def parse_oeis_stripped(text: str, *, names_text: str = "", source: str = "local stripped text") -> LocalOEISCorpus:
    """Parse local stripped/names exports; stripped rows do not supply offsets."""
    names = {}
    for line in names_text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if parts and _ID.fullmatch(parts[0]):
            names[parts[0]] = parts[1] if len(parts) > 1 else ""
    records = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or _ID.fullmatch(parts[0]) is None:
            raise ValidationError("invalid OEIS stripped row")
        identifier, payload = parts
        records.append(OEISRecord(identifier, _parse_terms(payload), None, names.get(identifier, ""), source))
    return LocalOEISCorpus(tuple(records), source)


def load_local_oeis(path: str | Path, *, format: str = "internal", names_path: str | Path | None = None) -> LocalOEISCorpus:
    """Load an explicit local internal, stripped, or JSON snapshot, including gzip.

    JSON snapshots use the recovered archive schema: ``sequences`` records with
    ``id``, ``terms``, optional ``offset``/``name``, and optional ``provenance``.
    """
    if format not in {"internal", "stripped", "json"}:
        raise ValidationError("format must be internal, stripped, or json")
    location = Path(path)
    reader = gzip.open if location.suffix == ".gz" else open
    with reader(location, "rt", encoding="utf-8") as handle:
        text = handle.read()
    source = str(location)
    if format == "internal":
        return parse_oeis_internal(text, source=source)
    if format == "stripped":
        names = ""
        if names_path is not None:
            names_location = Path(names_path)
            names_reader = gzip.open if names_location.suffix == ".gz" else open
            with names_reader(names_location, "rt", encoding="utf-8") as handle:
                names = handle.read()
        return parse_oeis_stripped(text, names_text=names, source=source)
    try:
        payload = json.loads(text)
        rows = payload["sequences"]
        if not isinstance(rows, list) or not isinstance(payload.get("provenance", {}), dict):
            raise ValidationError("invalid local OEIS JSON snapshot schema")
        records = tuple(OEISRecord(row["id"], row["terms"], row.get("offset"), row.get("name", ""), source)
                        for row in rows)
        return LocalOEISCorpus(records, source, payload.get("provenance", {}))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValidationError("invalid local OEIS JSON snapshot schema") from exc


__all__ = ["OEISRecord", "LocalOEISCorpus", "OEISPrefixMatch", "compare_oeis_prefix",
           "find_local_oeis_matches", "parse_oeis_internal", "parse_oeis_stripped", "load_local_oeis"]
