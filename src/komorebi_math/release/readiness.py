"""Aggregate public-beta and release-candidate readiness gates."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any
from .._version import __version__
from ..certificates import Certificate,Exactness
from ..stable_contract import check_stable_contract
from ..support import load_support_matrix,runtime_environment
from .contract import QualificationCheck,check_release_contract,_version_stage
from .upgrade import check_upgrade_contract
from .evidence import check_validation_evidence
from .documentation import check_stable_documentation
from .gpu import qualify_gpu
BETA=re.compile(r"^1\.0\.0b[1-9]\d*$")
@dataclass(frozen=True,slots=True)
class BetaReadinessReport:
    version:str;mode:str;checks:tuple[QualificationCheck,...];runtime:dict[str,Any];gpu:dict[str,Any]|None;certificate:Certificate
    @property
    def passed(self):return self.certificate.valid
    def to_dict(self):return {"version":self.version,"mode":self.mode,"passed":self.passed,"checks":[x.to_dict() for x in self.checks],"runtime":self.runtime,"gpu":self.gpu,"certificate":self.certificate.to_dict()}
def _capture(name,fn):
    try:fn();return QualificationCheck(name,True)
    except Exception as e:return QualificationCheck(name,False,f"{type(e).__name__}: {e}")
def check_beta_readiness(*,include_gpu=False,require_gpu=False,require_current_platforms=False):
    stage=_version_stage(__version__)
    # The historical name and call signature remain compatible. A promoted
    # package cannot opt back into beta's inherited-platform/optional-GPU policy.
    if stage in ("release-candidate","final"):
        require_current_platforms=True
        require_gpu=True
    if require_gpu:include_gpu=True
    mode="final" if stage=="final" else "release-candidate" if require_current_platforms else "beta";checks=[QualificationCheck("PEP 440 1.0 release version",stage is not None,f"observed {__version__}"),_capture("frozen stable compatibility contract",lambda:check_stable_contract().certificate.require()),_capture("beta release contract",lambda:check_release_contract().certificate.require()),_capture("cross-version upgrade contract",lambda:check_upgrade_contract().certificate.require()),_capture("stable documentation contract",lambda:check_stable_documentation().certificate.require()),_capture("executed platform evidence",lambda:check_validation_evidence(require_current_all_platforms=require_current_platforms).certificate.require())]
    def aliases():
        from .. import stable,v1
        from ..api import v1 as av1
        assert stable.__all__==v1.__all__==av1.__all__
        for n in stable.__all__:assert getattr(stable,n) is getattr(v1,n) is getattr(av1,n)
    checks.append(_capture("object-identical v1 aliases",aliases))
    def support():
        m=load_support_matrix();assert m.release_line.startswith("1.0.0b1");assert tuple(m.python["declared"])==("3.11","3.12","3.13")
    checks.append(_capture("beta support policy",support));gpu=None
    if include_gpu:
        g=qualify_gpu();gpu=g.to_dict();checks.append(QualificationCheck("CUDA hardware qualification",g.passed if require_gpu else True,g.certificate.message))
    failed=[x.name for x in checks if not x.passed];unchecked=[]
    if not include_gpu:unchecked.append("current-host CUDA hardware was not requested")
    if not require_current_platforms:unchecked.append("release candidate requires current-version evidence on every platform family")
    unchecked.append("exact candidate promotion additionally requires the external release_promotion.py gate")
    cert=Certificate(valid=not failed,exactness=Exactness.EXACT,residuals={"failed_checks":float(len(failed))},assumptions_checked=tuple(x.name for x in checks if x.passed),assumptions_unchecked=tuple(unchecked),diagnostics={"version":__version__,"mode":mode,"failures":failed},theorem_id="KM-READINESS-017",message=f"1.0 {mode} readiness gate passed" if not failed else f"1.0 {mode} readiness gate failed")
    return BetaReadinessReport(__version__,mode,tuple(checks),runtime_environment(probe_ad=False),gpu,cert)
__all__=["BetaReadinessReport","check_beta_readiness"]
