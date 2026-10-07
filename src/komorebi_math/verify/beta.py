"""Installed verification for the Komorebi Math 1.0 public beta."""
from ..release import check_beta_readiness,check_stable_documentation,check_upgrade_contract,check_validation_evidence
from .core import VerificationReport,_capture
def run_upgrade_verification():return VerificationReport((_capture("cross-version stable API and persistence upgrade",lambda:check_upgrade_contract().certificate.require()),))
def run_evidence_verification():return VerificationReport((_capture("executed beta platform evidence",lambda:check_validation_evidence().certificate.require()),))
def run_documentation_verification():return VerificationReport((_capture("frozen stable documentation coverage",lambda:check_stable_documentation().certificate.require()),))
def run_beta_readiness_verification():return VerificationReport((_capture("1.0 public-beta readiness",lambda:check_beta_readiness().certificate.require()),))
__all__=["run_upgrade_verification","run_evidence_verification","run_documentation_verification","run_beta_readiness_verification"]
