"""Aggregate structural, semantic, and persistence compatibility for 1.0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .certificates import Certificate, Exactness
from .compatibility import ApiCompatibilityReport, check_stable_api
from .persistence_contract import (
    PortableCodecCompatibilityReport,
    check_portable_codec_contract,
)
from .stable_semantics import StableSemanticReport, run_stable_semantics_fixture


@dataclass(frozen=True, slots=True)
class StableContractReport:
    """Stable 1.0 compatibility report."""

    api: ApiCompatibilityReport
    semantics: StableSemanticReport
    persistence: PortableCodecCompatibilityReport
    certificate: Certificate

    @property
    def compatible(self) -> bool:
        return self.certificate.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "api": self.api.to_dict(),
            "semantics": self.semantics.to_dict(),
            "persistence": self.persistence.to_dict(),
            "certificate": self.certificate.to_dict(),
        }


def check_stable_contract(
    *,
    api_manifest: str | Path | None = None,
    semantic_fixture: str | Path | None = None,
    persistence_contract: str | Path | None = None,
) -> StableContractReport:
    """Run every compatibility layer required by the frozen 1.0 surface."""

    api = check_stable_api(api_manifest)
    semantics = run_stable_semantics_fixture(semantic_fixture)
    persistence = check_portable_codec_contract(persistence_contract)
    valid = api.compatible and semantics.passed and persistence.compatible
    certificate = Certificate(
        valid=valid,
        exactness=Exactness.EXACT,
        residuals={
            "api_failures": float(not api.compatible),
            "semantic_failures": float(not semantics.passed),
            "persistence_failures": float(not persistence.compatible),
        },
        assumptions_checked=(
            "stable exports and signatures",
            "committed stable mathematical behavior",
            "portable codec identity and migration paths",
        ),
        assumptions_unchecked=(
            "performance stability across hardware",
            "compatibility of research and experimental namespaces",
        ),
        diagnostics={
            "api_reference_digest": api.reference_digest,
            "api_current_digest": api.current_digest,
            "semantic_fixture_digest": semantics.fixture_digest,
            "portable_reference_version": persistence.reference_version,
            "portable_current_version": persistence.current_version,
        },
        theorem_id="KM-STABLE-CONTRACT-016",
        message=(
            "the stable 1.0 compatibility contract passed"
            if valid
            else "the stable 1.0 compatibility contract failed"
        ),
    )
    return StableContractReport(api, semantics, persistence, certificate)


__all__ = ["StableContractReport", "check_stable_contract"]
