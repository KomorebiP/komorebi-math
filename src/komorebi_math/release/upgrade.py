"""Cross-version upgrade contract for the Komorebi Math 1.0 line."""
from __future__ import annotations
import json, inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from .._version import __version__
from ..certificates import Certificate, Exactness
from ..compatibility import create_stable_api_manifest, load_public_api_manifest, compare_public_api
from ..exceptions import ValidationError
from ..export.persistence import load_portable, portable_digest
from ..persistence_contract import load_portable_codec_contract, compare_portable_codec_contract
from ..stable_semantics import load_stable_semantics_fixture

SCHEMA="komorebi-math.upgrade-contract"; SCHEMA_VERSION=1
DATA=Path(__file__).resolve().parents[1]/"data"
DEFAULT_UPGRADE_CONTRACT=DATA/"upgrade_contract_1_0.json"

def _digest(path: Path)->str:return sha256(path.read_bytes()).hexdigest()
def _mapping(v:Any,n:str)->Mapping[str,Any]:
    if not isinstance(v,Mapping): raise ValidationError(f"{n} must be an object")
    return v
def _sequence(v:Any,n:str)->Sequence[Any]:
    if isinstance(v,(str,bytes)) or not isinstance(v,Sequence): raise ValidationError(f"{n} must be an array")
    return v

def _qtype(v:Any)->str:return f"{type(v).__module__}.{type(v).__qualname__}"

@dataclass(frozen=True,slots=True,order=True)
class UpgradeArtifact:
    version:str; filename:str; sha256:str; count:int
    def __post_init__(self):
        if not self.version or not self.filename or len(self.sha256)!=64 or self.count<0: raise ValidationError("invalid upgrade artifact")
    def to_dict(self):return {"version":self.version,"filename":self.filename,"sha256":self.sha256,"count":self.count}
    @classmethod
    def from_dict(cls,v):
        if set(v)!={"version","filename","sha256","count"}: raise ValidationError("upgrade artifact fields do not match the schema")
        return cls(str(v["version"]),str(v["filename"]),str(v["sha256"]),int(v["count"]))

@dataclass(frozen=True,slots=True,order=True)
class UpgradeFixture:
    name:str; producer_version:str; filename:str; sha256:str; payload_sha256:str; expected_type:str
    def __post_init__(self):
        if not all((self.name,self.producer_version,self.filename,self.expected_type)): raise ValidationError("invalid upgrade fixture")
        if len(self.sha256)!=64 or len(self.payload_sha256)!=64: raise ValidationError("invalid upgrade fixture digest")
    def to_dict(self):return {k:getattr(self,k) for k in ("name","producer_version","filename","sha256","payload_sha256","expected_type")}
    @classmethod
    def from_dict(cls,v):
        keys={"name","producer_version","filename","sha256","payload_sha256","expected_type"}
        if set(v)!=keys: raise ValidationError("upgrade fixture fields do not match the schema")
        return cls(**{k:str(v[k]) for k in keys})

@dataclass(frozen=True,slots=True)
class UpgradeContract:
    target_version:str; predecessor_versions:tuple[str,...]; stable_api:tuple[UpgradeArtifact,...]; stable_semantics:UpgradeArtifact; portable_codecs:UpgradeArtifact; fixtures:tuple[UpgradeFixture,...]; schema:str=SCHEMA; schema_version:int=SCHEMA_VERSION
    @property
    def digest(self):return sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(",",":")).encode()).hexdigest()
    def to_dict(self):return {"schema":self.schema,"schema_version":self.schema_version,"target_version":self.target_version,"predecessor_versions":list(self.predecessor_versions),"baselines":{"stable_api":[x.to_dict() for x in self.stable_api],"stable_semantics":self.stable_semantics.to_dict(),"portable_codecs":self.portable_codecs.to_dict()},"fixtures":[x.to_dict() for x in self.fixtures]}
    @classmethod
    def from_dict(cls,v):
        req={"schema","schema_version","target_version","predecessor_versions","baselines","fixtures"}
        if set(v)!=req: raise ValidationError("upgrade contract fields do not match the schema")
        b=_mapping(v["baselines"],"baselines")
        if set(b)!={"stable_api","stable_semantics","portable_codecs"}: raise ValidationError("upgrade baseline fields do not match the schema")
        result=cls(str(v["target_version"]),tuple(str(x) for x in _sequence(v["predecessor_versions"],"predecessors")),tuple(sorted(UpgradeArtifact.from_dict(_mapping(x,"artifact")) for x in _sequence(b["stable_api"],"stable API"))),UpgradeArtifact.from_dict(_mapping(b["stable_semantics"],"semantics")),UpgradeArtifact.from_dict(_mapping(b["portable_codecs"],"codecs")),tuple(sorted(UpgradeFixture.from_dict(_mapping(x,"fixture")) for x in _sequence(v["fixtures"],"fixtures"))),str(v["schema"]),int(v["schema_version"]))
        if result.schema!=SCHEMA or result.schema_version!=SCHEMA_VERSION: raise ValidationError("unsupported upgrade contract schema")
        return result

@dataclass(frozen=True,slots=True)
class UpgradeCheck:
    name:str; passed:bool; detail:str=""
    def to_dict(self):return {"name":self.name,"passed":self.passed,"detail":self.detail}
@dataclass(frozen=True,slots=True)
class UpgradeContractReport:
    contract:UpgradeContract; checks:tuple[UpgradeCheck,...]; loaded_fixtures:tuple[str,...]; certificate:Certificate
    @property
    def passed(self):return self.certificate.valid
    def to_dict(self):return {"contract":self.contract.to_dict(),"contract_digest":self.contract.digest,"passed":self.passed,"loaded_fixtures":list(self.loaded_fixtures),"checks":[x.to_dict() for x in self.checks],"certificate":self.certificate.to_dict()}

def load_upgrade_contract(path=None):
    p=DEFAULT_UPGRADE_CONTRACT if path is None else Path(path)
    try:v=json.loads(p.read_text())
    except (OSError,json.JSONDecodeError) as e: raise ValidationError(f"could not load upgrade contract {p}") from e
    return UpgradeContract.from_dict(_mapping(v,"upgrade contract"))
def _capture(name,fn):
    try:fn();return UpgradeCheck(name,True)
    except Exception as e:return UpgradeCheck(name,False,f"{type(e).__name__}: {e}")

def check_upgrade_contract(path=None):
    c=load_upgrade_contract(path); checks=[]; loaded=[]; current=create_stable_api_manifest()
    checks.append(UpgradeCheck("upgrade target version identity",c.target_version==__version__,f"observed {__version__}; expected {c.target_version}"))
    for a in c.stable_api:
        def check(a=a):
            p=DATA/a.filename
            if not p.is_file() or _digest(p)!=a.sha256: raise ValidationError(f"stable baseline changed: {a.filename}")
            old=load_public_api_manifest(p); r=compare_public_api(old,current,strict_signatures=True)
            if a.version!="0.15.0a2":
                if r.added or r.removed or r.kind_changed or r.signature_changed: raise ValidationError("0.16 stable freeze changed")
            else:
                changed={(ns,n) for ns,n,*_ in r.signature_changed}; allowed={("komorebi_math.stable","create_stable_api_manifest")}
                if r.removed or r.kind_changed or changed-allowed: raise ValidationError("unsupported 0.15 stable migration")
        checks.append(_capture(f"stable API upgrade from {a.version}",check))
    def semantics():
        a=c.stable_semantics;p=DATA/a.filename
        if _digest(p)!=a.sha256: raise ValidationError("semantic baseline digest changed")
        _,old,od=load_stable_semantics_fixture(p);_,new,nd=load_stable_semantics_fixture()
        if len(old["cases"])!=a.count or od!=nd or old!=new: raise ValidationError("stable semantics changed")
    checks.append(_capture("stable semantic continuity from 0.16.0a2",semantics))
    def codecs():
        a=c.portable_codecs;p=DATA/a.filename
        if _digest(p)!=a.sha256: raise ValidationError("codec baseline digest changed")
        old=load_portable_codec_contract(p)
        if len(old.entries)!=a.count: raise ValidationError("codec count changed")
        compare_portable_codec_contract(old).certificate.require()
    checks.append(_capture("portable codec continuity from 0.16.0a2",codecs))
    for f in c.fixtures:
        def check(f=f):
            p=DATA/f.filename
            if not p.is_file() or _digest(p)!=f.sha256: raise ValidationError("fixture digest changed")
            raw=json.loads(p.read_text())
            if raw.get("producer_version")!=f.producer_version or raw.get("payload_sha256")!=f.payload_sha256: raise ValidationError("fixture envelope changed")
            value=load_portable(p)
            if portable_digest(value)!=f.payload_sha256 or _qtype(value)!=f.expected_type: raise ValidationError("fixture reconstruction changed")
            cert=getattr(value,"certificate",None)
            if callable(cert):
                try:inspect.signature(cert).bind()
                except (TypeError,ValueError):pass
                else:cert().require()
            loaded.append(f.name)
        checks.append(_capture(f"portable fixture {f.name}",check))
    failed=[x.name for x in checks if not x.passed]
    cert=Certificate(valid=not failed,exactness=Exactness.EXACT,residuals={"failed_checks":float(len(failed))},assumptions_checked=("stable API predecessor manifests","closed stable semantics","portable codec continuity","digest-checked predecessor records"),assumptions_unchecked=("undocumented implementation objects","pickle payloads","third-party codecs"),diagnostics={"target":c.target_version,"predecessors":list(c.predecessor_versions),"loaded":len(loaded),"failures":failed},theorem_id="KM-UPGRADE-017",message="1.0 beta upgrade contract passed" if not failed else "1.0 beta upgrade contract failed")
    return UpgradeContractReport(c,tuple(checks),tuple(loaded),cert)

__all__=["DEFAULT_UPGRADE_CONTRACT","UpgradeArtifact","UpgradeFixture","UpgradeContract","UpgradeCheck","UpgradeContractReport","load_upgrade_contract","check_upgrade_contract"]
