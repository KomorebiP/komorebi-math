from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/data/komorebi_ep005/Komorebi_Math_R4_Block9_Episode005/komorebi-math')
sys.path.insert(0,str(ROOT/'src'))
from komorebi_math.coordinates import AnalysisSynthesisSystem, AnalysisSynthesisGauge
from komorebi_math.nonlinear import TransportGraph, TransportEdge
from komorebi_math.exceptions import ValidationError

rng=np.random.default_rng(20260922)

frame_cases=0
frame_metric_fail=0
frame_max_rel=0.0
frame_max_semi=0.0
frame_max_comp=0.0
frame_euclid_ratios=[]

def full_rank(rows, cols):
    while True:
        a=rng.normal(size=(rows,cols))
        if np.linalg.matrix_rank(a)==cols:
            return a

def oblique_left_inverse(A, strength=1.0):
    pinv=np.linalg.pinv(A)
    P=A@pinv
    L=rng.normal(size=(A.shape[1],A.shape[0]))*strength
    return pinv + L@(np.eye(A.shape[0])-P)

for _ in range(1500):
    n=int(rng.integers(1,5))
    m1=int(rng.integers(n,n+5)); m2=int(rng.integers(n,n+5)); m3=int(rng.integers(n,n+5))
    A1=full_rank(m1,n); A2=full_rank(m2,n); A3=full_rank(m3,n)
    S1=oblique_left_inverse(A1,10**rng.uniform(-1,1))
    S2=oblique_left_inverse(A2,10**rng.uniform(-1,1))
    S3=oblique_left_inverse(A3,10**rng.uniform(-1,1))
    X1=AnalysisSynthesisSystem(A1,S1,source='x',tolerance=1e-9)
    X2=AnalysisSynthesisSystem(A2,S2,source='x',tolerance=1e-9)
    X3=AnalysisSynthesisSystem(A3,S3,source='x',tolerance=1e-9)
    g1,g2,g3=map(AnalysisSynthesisGauge,(X1,X2,X3))
    T12=g1.transition_to(g2); T23=g2.transition_to(g3); T13=g1.transition_to(g3)
    M1=g1.coefficient_metric(); M2=g2.coefficient_metric()
    lhs=T12.T@M2@T12
    denom=max(1.0,np.linalg.norm(M1,'fro'))
    rel=float(np.linalg.norm(lhs-M1,'fro')/denom)
    frame_max_rel=max(frame_max_rel,rel)
    if rel>1e-8: frame_metric_fail+=1
    c=rng.normal(size=m1)
    semi=abs(g2.coefficient_seminorm(T12@c)-g1.coefficient_seminorm(c))
    semi/=max(1.0,g1.coefficient_seminorm(c))
    frame_max_semi=max(frame_max_semi,float(semi))
    comp=float(np.linalg.norm(T23@T12-T13,'fro')/max(1.0,np.linalg.norm(T13,'fro')))
    frame_max_comp=max(frame_max_comp,comp)
    pc=g1.canonical_coefficients(c)
    if np.linalg.norm(pc)>1e-14:
        frame_euclid_ratios.append(float(np.linalg.norm(T12@pc)/np.linalg.norm(pc)))
    frame_cases += 1

# Deliberate coefficient-coordinate reparameterizations of one fixed system.
A=np.array([[1.,0.],[0.,1.],[1.,1.]])
S=np.linalg.pinv(A)
base=AnalysisSynthesisGauge(AnalysisSynthesisSystem(A,S,source='x',tolerance=1e-10))
vector=np.array([3.,-4.,5.])
reparam=[]
for exponent in range(0,13,2):
    t=10.0**exponent
    G=np.diag([t,1.0,1.0/t]); Gi=np.linalg.inv(G)
    target=AnalysisSynthesisGauge(AnalysisSynthesisSystem(G@A,S@Gi,source='x',tolerance=1e-10))
    cp=G@vector
    reparam.append({
        't':t,
        'kappa_G':float(np.linalg.cond(G)),
        'euclidean_ratio':float(np.linalg.norm(cp)/np.linalg.norm(vector)),
        'quotient_seminorm_relative_error':float(abs(target.coefficient_seminorm(cp)-base.coefficient_seminorm(vector))/base.coefficient_seminorm(vector)),
    })

# Metric-covariant holonomy with randomized SPD metrics and GL coordinate changes.
holo_cases=0
holo_rejected=0
holo_compare=0
holo_max_rel=0.0
metric_cond_bound_violations=0
metric_cond_ratio_max=0.0
for _ in range(1500):
    n=int(rng.integers(2,5))
    R=rng.normal(size=(n,n))
    Q=R.T@R + 0.5*np.eye(n)
    E=rng.normal(size=(n,n))*1e-5
    H=np.eye(n)+E
    graph=TransportGraph(('v',),(TransportEdge('v','v',H,'loop'),),tolerance=1e-10)
    # singular values spread over up to 10^3 to stay mostly resolvable after metric transport
    U,_=np.linalg.qr(rng.normal(size=(n,n)))
    V,_=np.linalg.qr(rng.normal(size=(n,n)))
    logs=rng.uniform(-1.5,1.5,size=n)
    G=U@np.diag(10.0**logs)@V.T
    kG=float(np.linalg.cond(G,2))
    try:
        before=graph.metric_gluing_certificate(Q)
        Qp=graph.transport_root_metric(Q,G)
        after_graph=graph.gauge_transform({'v':G},gauge_regime='full_gl')
        after=after_graph.metric_gluing_certificate(Qp)
    except ValidationError:
        holo_rejected += 1
        continue
    a=before.residuals['loop[0]']; b=after.residuals['loop[0]']
    rel=abs(a-b)/max(1e-30,a,b)
    holo_max_rel=max(holo_max_rel,float(rel)); holo_compare += 1
    kQ=float(np.linalg.cond(Q,2)); kQp=float(np.linalg.cond(Qp,2))
    bound=kQ*(kG**2)
    if kQp > bound*(1+1e-9): metric_cond_bound_violations += 1
    metric_cond_ratio_max=max(metric_cond_ratio_max,kQp/max(1e-30,bound))
    holo_cases += 1

out={
 'frame_random':{
   'cases':frame_cases,
   'metric_pullback_failures_gt_1e-8':frame_metric_fail,
   'max_metric_pullback_relative_error':frame_max_rel,
   'max_quotient_seminorm_relative_error':frame_max_semi,
   'max_transition_composition_relative_error':frame_max_comp,
   'euclidean_transition_ratio_min':min(frame_euclid_ratios),
   'euclidean_transition_ratio_max':max(frame_euclid_ratios),
 },
 'deliberate_reparameterization':reparam,
 'holonomy_metric_covariance':{
   'attempts':1500,
   'accepted':holo_compare,
   'rejected_metric_resolution':holo_rejected,
   'max_residual_relative_error':holo_max_rel,
   'metric_condition_bound_violations':metric_cond_bound_violations,
   'max_ratio_kappa_Qprime_over_kappaQ_kappaG2':metric_cond_ratio_max,
 }
}
print(json.dumps(out,indent=2))
Path('/mnt/data/komorebi_ep005/episode005_metric_stress.json').write_text(json.dumps(out,indent=2))
