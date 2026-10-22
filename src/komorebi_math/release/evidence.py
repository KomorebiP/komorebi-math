"""Digest-checked executed-platform evidence for the Komorebi Math 1.0 beta."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .._version import __version__
from ..certificates import Certificate, Exactness
from ..exceptions import ValidationError
from .upgrade import load_upgrade_contract

VALIDATION_EVIDENCE_SCHEMA = "komorebi-math.validation-evidence"
VALIDATION_EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_VALIDATION_EVIDENCE = (
    Path(__file__).resolve().parents[1] / "data" / "validation_evidence_1_0.json"
)


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ValidationEvidenceRecord:
    """One executed or explicitly inherited platform qualification record."""

    identifier: str
    platform_key: str
    platform: str
    package_version: str
    python: str
    numpy: str
    source: str
    source_sha256: str
    inherited_via_upgrade_contract: bool
    results: Mapping[str, Any]
    optional: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (
                self.identifier,
                self.platform_key,
                self.platform,
                self.package_version,
                self.python,
                self.numpy,
                self.source,
            )
        ):
            raise ValidationError("validation evidence identity fields must be nonempty")
        if len(self.source_sha256) != 64:
            raise ValidationError("validation evidence source digest must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "platform_key": self.platform_key,
            "platform": self.platform,
            "package_version": self.package_version,
            "python": self.python,
            "numpy": self.numpy,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "inherited_via_upgrade_contract": self.inherited_via_upgrade_contract,
            "results": dict(self.results),
            "optional": dict(self.optional),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValidationEvidenceRecord":
        required = {
            "identifier",
            "platform_key",
            "platform",
            "package_version",
            "python",
            "numpy",
            "source",
            "source_sha256",
            "inherited_via_upgrade_contract",
            "results",
            "optional",
        }
        if set(value) != required:
            raise ValidationError("validation evidence record fields do not match the schema")
        if not isinstance(value["results"], Mapping) or not isinstance(value["optional"], Mapping):
            raise ValidationError("validation evidence results and optional fields must be objects")
        return cls(
            identifier=str(value["identifier"]),
            platform_key=str(value["platform_key"]),
            platform=str(value["platform"]),
            package_version=str(value["package_version"]),
            python=str(value["python"]),
            numpy=str(value["numpy"]),
            source=str(value["source"]),
            source_sha256=str(value["source_sha256"]),
            inherited_via_upgrade_contract=bool(value["inherited_via_upgrade_contract"]),
            results=dict(value["results"]),
            optional=dict(value["optional"]),
        )


@dataclass(frozen=True, slots=True)
class ValidationEvidence:
    """Closed platform-evidence inventory for one target release."""

    target_version: str
    required_platforms: tuple[str, ...]
    records: tuple[ValidationEvidenceRecord, ...]
    schema: str = VALIDATION_EVIDENCE_SCHEMA
    schema_version: int = VALIDATION_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != VALIDATION_EVIDENCE_SCHEMA:
            raise ValidationError("unknown validation evidence schema")
        if self.schema_version != VALIDATION_EVIDENCE_SCHEMA_VERSION:
            raise ValidationError("unsupported validation evidence schema version")
        if not self.required_platforms or len(self.required_platforms) != len(
            set(self.required_platforms)
        ):
            raise ValidationError("required platform keys must be nonempty and unique")
        identifiers = [record.identifier for record in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValidationError("validation evidence identifiers must be unique")

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "target_version": self.target_version,
            "required_platforms": list(self.required_platforms),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValidationEvidence":
        required = {
            "schema",
            "schema_version",
            "target_version",
            "required_platforms",
            "records",
        }
        if set(value) != required:
            raise ValidationError("validation evidence fields do not match the schema")
        if not isinstance(value["required_platforms"], list):
            raise ValidationError("required platforms must be an array")
        if not isinstance(value["records"], list):
            raise ValidationError("validation evidence records must be an array")
        if any(not isinstance(record, Mapping) for record in value["records"]):
            raise ValidationError("validation evidence records must contain objects")
        return cls(
            target_version=str(value["target_version"]),
            required_platforms=tuple(str(item) for item in value["required_platforms"]),
            records=tuple(
                sorted(ValidationEvidenceRecord.from_dict(record) for record in value["records"])
            ),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class ValidationEvidenceReport:
    """Qualification result under beta or release-candidate evidence rules."""

    evidence: ValidationEvidence
    covered_platforms: tuple[str, ...]
    current_platforms: tuple[str, ...]
    inherited_records: tuple[str, ...]
    missing_platforms: tuple[str, ...]
    missing_current_platforms: tuple[str, ...]
    require_current_all_platforms: bool
    certificate: Certificate

    @property
    def passed(self) -> bool:
        return self.certificate.valid

    @property
    def release_candidate_ready(self) -> bool:
        return self.passed and not self.missing_current_platforms

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence.to_dict(),
            "evidence_digest": self.evidence.digest,
            "covered_platforms": list(self.covered_platforms),
            "current_platforms": list(self.current_platforms),
            "inherited_records": list(self.inherited_records),
            "missing_platforms": list(self.missing_platforms),
            "missing_current_platforms": list(self.missing_current_platforms),
            "require_current_all_platforms": self.require_current_all_platforms,
            "release_candidate_ready": self.release_candidate_ready,
            "passed": self.passed,
            "certificate": self.certificate.to_dict(),
        }


def load_validation_evidence(path: str | Path | None = None) -> ValidationEvidence:
    """Load the committed platform-evidence contract."""

    source = DEFAULT_VALIDATION_EVIDENCE if path is None else Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"could not load validation evidence {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError("validation evidence is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValidationError("validation evidence root must be an object")
    return ValidationEvidence.from_dict(raw)


def _check_validation_evidence(
    path: str | Path | None = None,
    *,
    require_current_all_platforms: bool = False,
    reference_version: str | None = None,
) -> ValidationEvidenceReport:
    """Validate evidence and optionally require current-version execution everywhere.

    Beta qualification may inherit a frozen predecessor result when the stable
    API, semantics, persistence codecs, and portable records are protected by
    the committed upgrade contract. Release-candidate qualification does not.
    """

    expected_version = __version__ if reference_version is None else reference_version
    evidence = load_validation_evidence(path)
    predecessor_versions = set(load_upgrade_contract().predecessor_versions)
    covered = tuple(sorted({record.platform_key for record in evidence.records}))
    current = tuple(
        sorted(
            {
                record.platform_key
                for record in evidence.records
                if record.package_version == expected_version
            }
        )
    )
    missing = tuple(sorted(set(evidence.required_platforms) - set(covered)))
    missing_current = tuple(sorted(set(evidence.required_platforms) - set(current)))
    inherited = tuple(
        sorted(
            record.identifier
            for record in evidence.records
            if record.inherited_via_upgrade_contract
        )
    )
    invalid_records: list[str] = []
    invalid_sources: list[str] = []
    data_directory = Path(__file__).resolve().parents[1] / "data"
    for record in evidence.records:
        if record.results.get("passed") is not True:
            invalid_records.append(record.identifier)
        current_record = record.package_version == expected_version
        if current_record and record.inherited_via_upgrade_contract:
            invalid_records.append(record.identifier)
        if not current_record and (
            not record.inherited_via_upgrade_contract
            or record.package_version not in predecessor_versions
        ):
            invalid_records.append(record.identifier)
        source = data_directory / record.source
        if not source.is_file() or _file_digest(source) != record.source_sha256:
            invalid_sources.append(record.identifier)

    valid = (
        evidence.target_version == expected_version
        and not missing
        and bool(current)
        and not invalid_records
        and not invalid_sources
        and (not require_current_all_platforms or not missing_current)
    )
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={
            "missing_platforms": float(len(missing)),
            "missing_current_platforms": float(len(missing_current)),
            "invalid_records": float(len(set(invalid_records))),
            "invalid_sources": float(len(invalid_sources)),
        },
        assumptions_checked=(
            "current-version execution exists",
            "required platform families are covered",
            "inherited evidence is authorized by the upgrade contract",
            "evidence sources are digest checked",
        ),
        assumptions_unchecked=(
            ()
            if require_current_all_platforms
            else ("release candidate requires current-version evidence on every platform family",)
        ),
        diagnostics={
            "covered": list(covered),
            "current": list(current),
            "missing_current": list(missing_current),
            "invalid": sorted(set(invalid_records)),
            "source_failures": invalid_sources,
        },
        theorem_id="KM-EVIDENCE-017",
        message=(
            "1.0 beta validation evidence passed"
            if valid
            else "1.0 beta validation evidence failed"
        ),
    )
    return ValidationEvidenceReport(
        evidence=evidence,
        covered_platforms=covered,
        current_platforms=current,
        inherited_records=inherited,
        missing_platforms=missing,
        missing_current_platforms=missing_current,
        require_current_all_platforms=bool(require_current_all_platforms),
        certificate=certificate,
    )


def check_validation_evidence(
    path: str | Path | None = None,
    *,
    require_current_all_platforms: bool = False,
) -> ValidationEvidenceReport:
    """Validate the current runtime's embedded platform evidence.

    This compatibility API retains its historical contract. Current exact
    candidate promotion is checked separately by the repository promotion gate.
    """
    return _check_validation_evidence(path, require_current_all_platforms=require_current_all_platforms)


def _check_historical_validation_evidence() -> ValidationEvidenceReport:
    """Validate preserved history against its original target, not this runtime."""
    from dataclasses import replace
    evidence = load_validation_evidence()
    report = _check_validation_evidence(reference_version=evidence.target_version)
    certificate = replace(report.certificate,
        assumptions_checked=("preserved historical target and platform coverage", "digest-checked historical source records"),
        assumptions_unchecked=("current candidate CPU/GPU execution requires the external promotion gate",),
        diagnostics={**report.certificate.diagnostics, "historical_target": evidence.target_version,
                     "current_runtime": __version__, "current_candidate_qualification": False},
        message="historical platform record integrity passed" if report.passed else "historical platform record integrity failed")
    return replace(report, certificate=certificate)


__all__ = [
    "DEFAULT_VALIDATION_EVIDENCE",
    "VALIDATION_EVIDENCE_SCHEMA",
    "VALIDATION_EVIDENCE_SCHEMA_VERSION",
    "ValidationEvidence",
    "ValidationEvidenceRecord",
    "ValidationEvidenceReport",
    "check_validation_evidence",
    "load_validation_evidence",
]
