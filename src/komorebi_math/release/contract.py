"""Machine-readable contract for the current Komorebi Math 1.0 beta."""
from __future__ import annotations
import json,platform,sys
from collections.abc import Mapping,Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any
from .._version import __version__
from ..certificates import Certificate,Exactness
from ..cli.contracts import VERIFICATION_SUITE_NAMES
from ..compatibility import check_beta_api,check_public_api,check_stable_api
from ..exceptions import ValidationError
from ..persistence_contract import check_portable_codec_contract
from ..stable_semantics import run_stable_semantics_fixture
from ..support import load_support_matrix,runtime_environment
RELEASE_CONTRACT_SCHEMA="komorebi-math.release-contract";RELEASE_CONTRACT_SCHEMA_VERSION=2
DEFAULT_RELEASE_CONTRACT=Path(__file__).resolve().parents[1]/"data"/"release_contract_1_0.json"
class ApiTier(StrEnum):STABLE="stable";BETA="beta";RESEARCH="research";INTERNAL="internal"
def _digest(p:Path)->str:return sha256(p.read_bytes()).hexdigest()
def _mapping(v,n):
    if not isinstance(v,Mapping):raise ValidationError(f"{n} must be an object")
    return v
def _sequence(v,n):
    if isinstance(v,(str,bytes)) or not isinstance(v,Sequence):raise ValidationError(f"{n} must be an array")
    return v
@dataclass(frozen=True,slots=True)
class ContractArtifact:
    filename:str;sha256:str;count:int
    def __post_init__(self):
        if not self.filename or len(self.sha256)!=64 or self.count<0:raise ValidationError("invalid release contract artifact")
    def to_dict(self):return {"filename":self.filename,"sha256":self.sha256,"count":self.count}
    @classmethod
    def from_dict(cls,v):
        if set(v)!={"filename","sha256","count"}:raise ValidationError("release contract artifact fields do not match the schema")
        return cls(str(v["filename"]),str(v["sha256"]),int(v["count"]))
@dataclass(frozen=True,slots=True)
class ReleaseContract:
    package_version:str;predecessor_version:str;release_line:str;stage:str;python_requires:str;declared_python:tuple[str,...];required_platforms:tuple[str,...];stable_api:ContractArtifact;beta_api:ContractArtifact;research_api:ContractArtifact;semantics:ContractArtifact;persistence:ContractArtifact;upgrade:ContractArtifact;evidence:ContractArtifact;documentation:ContractArtifact;stable_aliases:tuple[str,...];required_cli_suites:tuple[str,...];schema:str=RELEASE_CONTRACT_SCHEMA;schema_version:int=RELEASE_CONTRACT_SCHEMA_VERSION
    def __post_init__(self):
        if self.schema!=RELEASE_CONTRACT_SCHEMA or self.schema_version!=RELEASE_CONTRACT_SCHEMA_VERSION:raise ValidationError("unsupported release contract schema")
        if self.stage!="beta":raise ValidationError("release stage must be beta")
    @property
    def digest(self):return sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(",",":")).encode()).hexdigest()
    def to_dict(self):return {"schema":self.schema,"schema_version":self.schema_version,"package_version":self.package_version,"predecessor_version":self.predecessor_version,"release_line":self.release_line,"stage":self.stage,"runtime":{"python_requires":self.python_requires,"declared_python":list(self.declared_python),"required_platforms":list(self.required_platforms)},"api":{"stable":self.stable_api.to_dict(),"beta":self.beta_api.to_dict(),"research":self.research_api.to_dict(),"stable_aliases":list(self.stable_aliases)},"semantics":self.semantics.to_dict(),"persistence":self.persistence.to_dict(),"upgrade":self.upgrade.to_dict(),"evidence":self.evidence.to_dict(),"documentation":self.documentation.to_dict(),"cli":{"required_verification_suites":list(self.required_cli_suites)}}
    @classmethod
    def from_dict(cls,v):
        req={"schema","schema_version","package_version","predecessor_version","release_line","stage","runtime","api","semantics","persistence","upgrade","evidence","documentation","cli"}
        if set(v)!=req:raise ValidationError("release contract fields do not match the schema")
        r=_mapping(v["runtime"],"runtime");a=_mapping(v["api"],"api");c=_mapping(v["cli"],"cli")
        return cls(str(v["package_version"]),str(v["predecessor_version"]),str(v["release_line"]),str(v["stage"]),str(r["python_requires"]),tuple(map(str,_sequence(r["declared_python"],"Python versions"))),tuple(map(str,_sequence(r["required_platforms"],"platforms"))),ContractArtifact.from_dict(_mapping(a["stable"],"stable")),ContractArtifact.from_dict(_mapping(a["beta"],"beta")),ContractArtifact.from_dict(_mapping(a["research"],"research")),ContractArtifact.from_dict(_mapping(v["semantics"],"semantics")),ContractArtifact.from_dict(_mapping(v["persistence"],"persistence")),ContractArtifact.from_dict(_mapping(v["upgrade"],"upgrade")),ContractArtifact.from_dict(_mapping(v["evidence"],"evidence")),ContractArtifact.from_dict(_mapping(v["documentation"],"documentation")),tuple(map(str,_sequence(a["stable_aliases"],"aliases"))),tuple(map(str,_sequence(c["required_verification_suites"],"CLI suites"))),str(v["schema"]),int(v["schema_version"]))
@dataclass(frozen=True,slots=True)
class QualificationCheck:
    name:str;passed:bool;detail:str=""
    def to_dict(self):return {"name":self.name,"passed":self.passed,"detail":self.detail}
@dataclass(frozen=True,slots=True)
class ReleaseContractReport:
    contract:ReleaseContract;checks:tuple[QualificationCheck,...];runtime:Mapping[str,Any];certificate:Certificate
    @property
    def passed(self):return self.certificate.valid
    def to_dict(self):return {"contract":self.contract.to_dict(),"contract_digest":self.contract.digest,"passed":self.passed,"checks":[x.to_dict() for x in self.checks],"runtime":dict(self.runtime),"certificate":self.certificate.to_dict()}
def load_release_contract(path=None):
    p=DEFAULT_RELEASE_CONTRACT if path is None else Path(path)
    try:return ReleaseContract.from_dict(json.loads(p.read_text()))
    except Exception as e:raise ValidationError(f"could not read release contract {p}") from e
def _capture(name,fn):
    try:fn();return QualificationCheck(name,True)
    except Exception as e:return QualificationCheck(name,False,f"{type(e).__name__}: {e}")
def _artifact(a,key):
    p=Path(__file__).resolve().parents[1]/"data"/a.filename
    if not p.is_file() or _digest(p)!=a.sha256:raise ValidationError(f"release artifact changed: {a.filename}")
    v=json.loads(p.read_text()).get(key)
    if not isinstance(v,list) or len(v)!=a.count:raise ValidationError(f"release artifact count changed: {a.filename}")
def check_release_contract(path=None):
    from .upgrade import check_upgrade_contract
    from .evidence import check_validation_evidence
    from .documentation import check_stable_documentation
    c=load_release_contract(path);checks=[QualificationCheck("release version identity",__version__==c.package_version,f"observed {__version__}; expected {c.package_version}"),QualificationCheck("release stage identity",c.stage=="beta" and c.package_version.startswith("1.0.0b"),f"stage={c.stage}; version={c.package_version}; line={c.release_line}")]
    for name,a,key in (("stable API artifact identity",c.stable_api,"symbols"),("beta API artifact identity",c.beta_api,"symbols"),("research API artifact identity",c.research_api,"symbols"),("semantic fixture identity",c.semantics,"cases"),("portable codec contract identity",c.persistence,"entries"),("upgrade contract identity",c.upgrade,"fixtures"),("validation evidence identity",c.evidence,"records"),("stable documentation identity",c.documentation,"symbols")):checks.append(_capture(name,lambda a=a,key=key:_artifact(a,key)))
    checks += [_capture("stable API structural compatibility",lambda:check_stable_api().certificate.require()),_capture("beta API structural compatibility",lambda:check_beta_api().certificate.require()),_capture("research API snapshot compatibility",lambda:check_public_api(strict_signatures=True).certificate.require()),_capture("stable semantic compatibility",lambda:run_stable_semantics_fixture().certificate.require()),_capture("portable codec compatibility",lambda:check_portable_codec_contract().certificate.require()),_capture("cross-version upgrade compatibility",lambda:check_upgrade_contract().certificate.require()),_capture("executed beta platform evidence",lambda:check_validation_evidence().certificate.require()),_capture("stable documentation compatibility",lambda:check_stable_documentation().certificate.require())]
    def aliases():
        s=import_module("komorebi_math.stable");v=import_module("komorebi_math.v1");a=import_module("komorebi_math.api.v1");assert s.__all__==v.__all__==a.__all__
        for n in s.__all__:assert getattr(s,n) is getattr(v,n) is getattr(a,n)
    checks.append(_capture("versioned stable aliases",aliases))
    checks.append(QualificationCheck("frozen CLI verification routes",set(c.required_cli_suites)<=set(VERIFICATION_SUITE_NAMES),""))
    def support():
        m=load_support_matrix();assert m.release_line==c.release_line;assert tuple(m.python["declared"])==c.declared_python
    checks.append(_capture("declared support policy",support));failed=[x.name for x in checks if not x.passed]
    cert=Certificate(valid=not failed,exactness=Exactness.EXACT,residuals={"failed_checks":float(len(failed))},assumptions_checked=tuple(x.name for x in checks if x.passed),assumptions_unchecked=("current-version execution on all platforms is enforced by RC readiness","hardware performance is not semantic compatibility"),diagnostics={"package_version":c.package_version,"contract_digest":c.digest,"failures":failed},theorem_id="KM-RELEASE-CONTRACT-021",message="current public-beta release contract passed" if not failed else "current public-beta release contract failed")
    return ReleaseContractReport(c,tuple(checks),current_runtime_record(),cert)
def current_runtime_record():return {"package_version":__version__,"python":platform.python_version(),"python_executable":sys.executable,"platform":platform.platform(),"environment":runtime_environment()}
__all__=["ApiTier","ContractArtifact","DEFAULT_RELEASE_CONTRACT","QualificationCheck","RELEASE_CONTRACT_SCHEMA","RELEASE_CONTRACT_SCHEMA_VERSION","ReleaseContract","ReleaseContractReport","check_release_contract","current_runtime_record","load_release_contract"]
