"""Permanent Block 1 regressions plus bounded arithmetic/representation checks.

The 15 case IDs preserve the prior audit. A corrected constructor is permitted
only the documented ValidationError, never an arbitrary error or import failure.
"""
from __future__ import annotations

from fractions import Fraction
from math import inf, nextafter
import numpy as np
import pytest
import sympy as sp

from komorebi_math import PacketSystem, PacketOperator, OperatorMorphism, PacketMorphism, SeedMorphism, SeedSystem
from komorebi_math.backends import get_backend
from komorebi_math.config import Settings, configure, get_settings, settings_context
from komorebi_math.exceptions import ValidationError
from komorebi_math.refinement.towers import PacketRefinement
from komorebi_math.nonlinear.trees import majorant_convergence, quadratic_catalan_majorant

CASES = (
    'tiny_invalid_packet', 'tiny_oblique_hilbert_claim', 'tiny_unsupported_operator',
    'tiny_false_intertwiner', 'tiny_packet_morphism_leakage', 'tiny_false_seed_image',
    'tiny_false_refinement', 'oblique_adjoint_loses_packet_support',
    'operator_claims_invalid_packet_axioms_checked',
    'refinement_claims_invalid_packet_axioms_checked', 'morphism_claims_invalid_packets_checked',
    'infinite_tolerance_accepts_invalid_packet', 'numpy_zero_tolerance_false_zero',
    'majorant_default_tolerance_accepts_no_fixed_point',
    'majorant_explicit_tolerance_accepts_no_fixed_point',
)

def diagonal_packets():
    return PacketSystem([sp.diag(1, 0), sp.diag(0, 1)], backend='sympy', names=('a', 'b'))

@pytest.mark.parametrize('case', CASES)
def test_original_soundness_regressions(case):
    e = sp.Rational(1, 10**400)
    p = diagonal_packets(); I=sp.eye(2); Z=sp.zeros(2); E=sp.Matrix([[0, e], [0, 0]])
    bad = PacketSystem([sp.Matrix([[2]])], backend='sympy', validate=False)
    assert not bad.certificate().valid
    if case == 'tiny_invalid_packet':
        c = PacketSystem([sp.Matrix([[1+e]])], backend='sympy', validate=False).certificate()
        assert not c.valid
        assert c.diagnostics['residual_decisions']['completeness']['status'] == 'nonzero'
    elif case == 'tiny_oblique_hilbert_claim':
        q=sp.Matrix([[1,e],[0,0]])
        ob=PacketSystem([q,I-q],backend='sympy')
        assert ob.certificate().valid
        assert not ob.hilbert_certificate().valid
    elif case == 'tiny_unsupported_operator':
        assert not PacketOperator(p,p,((E,Z),(Z,Z))).certificate().valid
    elif case == 'tiny_false_intertwiner':
        assert not OperatorMorphism.create({'A':sp.zeros(1)},{'A':sp.Matrix([[e]])},sp.eye(1),backend='sympy').certificate().valid
    elif case == 'tiny_packet_morphism_leakage':
        assert not PacketMorphism(p,p,I+E,{'a':('a',),'b':('b',)}).certificate().valid
    elif case == 'tiny_false_seed_image':
        source=SeedSystem({'A':sp.eye(1)},[sp.Matrix([1])],backend='sympy')
        assert not SeedMorphism(source,source,sp.eye(1),{'A':'A'},[sp.Matrix([1+e])]).certificate().valid
    elif case == 'tiny_false_refinement':
        q=sp.Matrix([[1,e],[0,0]]); ob=PacketSystem([q,I-q],backend='sympy',names=('a','b'))
        with pytest.raises(ValidationError, match='parent packet or refinement checks'):
            PacketRefinement(p,ob,(0,1))
    elif case == 'oblique_adjoint_loses_packet_support':
        q=sp.Matrix([[1,1],[0,0]]); ob=PacketSystem([q,I-q],backend='sympy',names=('a','b'))
        T=sp.Matrix([[1,2],[3,4]]);op=PacketOperator.from_operator(T,ob);adj=op.adjoint()
        assert adj.certificate().valid
        assert adj.reconstruct() == T.conjugate().T
        assert adj.source is ob and adj.target is ob
    elif case == 'operator_claims_invalid_packet_axioms_checked':
        c=PacketOperator.from_operator(sp.zeros(1),bad).certificate()
        assert not c.valid and not c.diagnostics['source_certificate']['valid']
    elif case == 'refinement_claims_invalid_packet_axioms_checked':
        with pytest.raises(ValidationError, match='parent packet or refinement checks'):
            PacketRefinement(bad,bad,(0,))
    elif case == 'morphism_claims_invalid_packets_checked':
        assert not PacketMorphism(bad,bad,sp.zeros(1),{'P0':('P0',)}).certificate().valid
    elif case == 'infinite_tolerance_accepts_invalid_packet':
        with pytest.raises(ValidationError, match='finite and nonnegative'):
            PacketSystem([np.array([[2.]])],tolerance=inf)
    elif case == 'numpy_zero_tolerance_false_zero':
        assert not get_backend('numpy').is_zero(np.array([[1e-200]]),tolerance=0.0)
    else:
        explicit = 'explicit' in case
        a, r, tol = (0.26,0.49,0.02) if explicit else (0.2500000000001,0.4999999,1e-12)
        assert 1 - 4*Fraction(a) < 0
        c=majorant_convergence(a,{2:1.0},r,tolerance=tol)
        assert not c.certificate.valid
        assert c.certificate.diagnostics['guard_status']=='FAILED_GUARDS'
        with pytest.raises(ValidationError, match='scalar majorant guards fail'):
            c.picard_tail_bound(1)

@pytest.mark.parametrize('exponent', [20, 200, 400, 1000])
@pytest.mark.parametrize('sign', [-1, 1])
def test_exact_rational_residual_scale_and_sign(exponent, sign):
    q=sp.Rational(sign, 10**exponent)
    c=PacketSystem([sp.Matrix([[1+q]])],backend='sympy',validate=False).certificate()
    assert not c.valid
    assert Fraction(sign,10**exponent) != 0
    assert c.diagnostics['residual_decisions']['completeness']['status']=='nonzero'
    assert c.diagnostics['residual_decisions']['completeness']['witness']['expression']


def test_symbolic_unknown_is_not_success_or_exception():
    x=sp.Symbol('x')
    c=PacketSystem([sp.Matrix([[x]])],backend='sympy',validate=False).certificate()
    assert not c.valid and c.assumptions_unchecked
    assert c.diagnostics['residual_decisions']['completeness']['status']=='unknown'
    with pytest.raises(ValidationError): c.require()


def test_symbolic_identity_can_be_established_without_float_conversion():
    x=sp.Symbol('x',real=True)
    v=sp.sin(x)**2+sp.cos(x)**2
    c=PacketSystem([sp.Matrix([[v]])],backend='sympy',validate=False).certificate()
    assert c.valid
    assert all(d['status']=='zero' for d in c.diagnostics['residual_decisions'].values())

@pytest.mark.parametrize('entry', [sp.oo, -sp.oo, sp.zoo, sp.nan])
def test_exact_nonfinite_is_never_certified(entry):
    c=PacketSystem([sp.Matrix([[entry]])],backend='sympy',validate=False).certificate()
    assert not c.valid

@pytest.mark.parametrize('tol',[inf,-inf,float('nan'),-1.0])
@pytest.mark.parametrize('entrypoint',['packet','settings','configure','context','operator','numpy'])
def test_invalid_tolerance_rejected_at_repaired_entrypoints(tol,entrypoint):
    previous=get_settings()
    with pytest.raises(ValidationError, match='finite and nonnegative'):
        if entrypoint=='packet':PacketSystem([np.eye(1)],tolerance=tol,validate=False)
        elif entrypoint=='settings':Settings(tolerance=tol)
        elif entrypoint=='configure':configure(tolerance=tol)
        elif entrypoint=='context':
            with settings_context(tolerance=tol):pass
        elif entrypoint=='operator':OperatorMorphism.create({'A':sp.eye(1)},{'A':sp.eye(1)},sp.eye(1),backend='sympy',tolerance=tol)
        else:get_backend('numpy').is_zero(np.zeros((1,1)),tolerance=tol)
    assert get_settings()==previous

@pytest.mark.parametrize('value',[5e-324,1e-300,1e-200,1.0,1e200,1e308,1e-200j,complex(1e200,-1e200)])
def test_numpy_zero_and_norm_extreme_nonzero_values(value):
    b=get_backend('numpy');a=np.array([[value]])
    assert not b.is_zero(a,tolerance=0.0)
    assert b.norm(a)>0

@pytest.mark.parametrize('value',[float('nan'),inf,-inf,complex(1,float('nan'))])
def test_numpy_nonfinite_rejected(value):
    assert not get_backend('numpy').is_zero(np.array([[value]]),tolerance=1e100)


def test_numpy_scaled_tolerance_uses_frobenius_not_max_entry():
    b=get_backend('numpy');a=np.array([[0.8e-200,0.8e-200]])
    assert not b.is_zero(a,tolerance=1e-200)
    assert b.is_zero(a,tolerance=1.2e-200)
    assert b.is_zero(np.zeros((0,2)),tolerance=0.0)

@pytest.mark.parametrize('kind',['operator','morphism','refinement','hilbert'])
def test_fresh_certificate_rechecks_mutated_parent(kind):
    p=PacketSystem([sp.eye(1)],backend='sympy')
    if kind=='operator':obj=PacketOperator.from_operator(sp.zeros(1),p)
    elif kind=='morphism':obj=PacketMorphism(p,p,sp.zeros(1),{'P0':('P0',)})
    elif kind=='refinement':obj=PacketRefinement(p,p,(0,))
    else:obj=p
    checker=obj.hilbert_certificate if kind=='hilbert' else obj.certificate
    old=checker();assert old.valid
    p.projectors[0][0,0]=2
    assert not checker().valid
    # The old object is a historical observation, not a self-invalidating proof.
    assert old.valid

@pytest.mark.parametrize('coefficient',[1,sp.I,sp.Rational(2,3)+sp.I,sp.Rational(1,10**400),sp.Integer(10)**400])
def test_rectangular_complex_oblique_adjoint_and_involution(coefficient):
    P=sp.Matrix([[1,coefficient],[0,0]])
    source=PacketSystem([P,sp.eye(2)-P],backend='sympy',names=('a','b'))
    Q=sp.Matrix([[1,0,sp.I],[0,1,1],[0,0,0]])
    target=PacketSystem([Q,sp.eye(3)-Q],backend='sympy',names=('c','d'))
    T=sp.Matrix([[1,sp.I],[2,3],[-sp.I,4]])
    op=PacketOperator.from_operator(T,source,target)
    adj=op.adjoint();assert adj.certificate().valid
    assert adj.source is target and adj.target is source
    assert all(sp.expand(x)==0 for x in adj.reconstruct()-T.conjugate().T)
    twice=adj.adjoint()
    assert twice.certificate().valid
    for actual_row, expected_row in zip(twice.blocks, op.blocks, strict=True):
        for actual, expected in zip(actual_row, expected_row, strict=True):
            assert all(sp.expand(x)==0 for x in actual-expected)


def test_adjoint_reverses_composition_in_original_oblique_systems():
    P=sp.Matrix([[1,1+sp.I],[0,0]])
    p=PacketSystem([P,sp.eye(2)-P],backend='sympy')
    A=PacketOperator.from_operator(sp.Matrix([[1,2],[0,sp.I]]),p)
    B=PacketOperator.from_operator(sp.Matrix([[0,1],[2,3]]),p)
    left=A.compose(B).adjoint();right=B.adjoint().compose(A.adjoint())
    assert left.certificate().valid and right.certificate().valid
    for lrow, rrow in zip(left.blocks, right.blocks, strict=True):
        for lblock, rblock in zip(lrow, rrow, strict=True):
            assert all(sp.expand(x)==0 for x in lblock-rblock)


def test_majorant_guard_grid_against_integer_inequalities():
    # 2160 dyadic polynomial systems; integer arithmetic supplies an independent
    # guard oracle, not float comparison or reuse of the production helper.
    count=0
    for ai in range(9):
        for bi in range(5):
            for ci in range(4):
                for ri in range(1,13):
                    expected=(4096*ai+32*bi*ri**2+ci*ri**3<=4096*ri
                              and 64*bi*ri+3*ci*ri**2<4096)
                    result=majorant_convergence(ai/16,{2:bi/8,3:ci/16},ri/16,tolerance=1e10)
                    assert result.certificate.valid is expected
                    count+=1
    assert count==2160

@pytest.mark.parametrize('tiny',[5e-324,1e-300,1e-200])
def test_self_map_rounding_equality_does_not_prove_guard(tiny):
    # Floating a+b rounds back to one, but the exact binary-rational sum is >1.
    assert 1.0+tiny==1.0
    result=majorant_convergence(1.0,{2:tiny},1.0)
    assert not result.certificate.valid
    assert result.certificate.residuals['ball_image_excess']>0

@pytest.mark.parametrize('depth',[0,1,2,10,100,1000])
def test_tail_is_upward_rounded_even_underflow(depth):
    result=majorant_convergence(0.125,{2:1.0},0.25)
    actual=result.picard_tail_bound(depth)
    exact=Fraction(1,2)**depth*Fraction(1,8)/(1-Fraction(1,2))
    assert Fraction(actual)>=exact
    if exact>0:assert actual>0


def test_tail_rechecks_mutated_scalar_norm_map():
    result=majorant_convergence(0.125,{2:1.0},0.25)
    assert result.certificate.valid
    result.operation_norms[2]=100.0
    with pytest.raises(ValidationError,match='scalar majorant guards fail'):
        result.picard_tail_bound(1)


def test_exact_boundary_and_strict_contraction():
    assert majorant_convergence(0.1875,{2:1.0},0.25).certificate.valid
    r=majorant_convergence(0.25,{2:1.0},0.5)
    assert not r.certificate.valid
    assert r.certificate.diagnostics['self_map_proved']
    assert not r.certificate.diagnostics['strict_contraction_proved']

@pytest.mark.parametrize('a,b',[(0.1,0.5),(0.125,1.0),(1e-200,0.1),(0.0,1e308),(0.1,0.0),(0.0,0.0),(nextafter(.25,0),1.0)])
def test_catalan_candidate_has_verified_guards(a,b):
    r=quadratic_catalan_majorant(a,b)
    assert r.certificate.valid
    aq,bq,rq=map(Fraction,(a,b,r.radius))
    assert aq+bq*rq*rq<=rq and 2*bq*rq<1


def test_majorant_work_budgets_fail_closed():
    with pytest.raises(ValidationError,match='guard work budget'):
        majorant_convergence(0.1,{10**9:1.0},0.3)
    r=majorant_convergence(.125,{2:1.0},.25)
    with pytest.raises(ValidationError,match='tail work budget'):
        r.picard_tail_bound(10**9)


def test_large_exact_guard_diagnostics_use_lossless_hex():
    result=majorant_convergence(0.01,{1000:0.5},0.3)
    assert result.certificate.valid
    encoded=result.certificate.diagnostics["exact_image"]
    assert isinstance(encoded,dict) and encoded["format"]=="hex-rational"
    actual=Fraction(int(encoded["numerator"],16),int(encoded["denominator"],16))
    assert actual==Fraction(0.01)+Fraction(0.5)*Fraction(0.3)**1000


def test_proved_symbolic_zero_has_zero_diagnostic():
    x=sp.Symbol("x",real=True)
    value=sp.sin(x)**2+sp.cos(x)**2
    result=PacketSystem([sp.Matrix([[value]])],backend="sympy",validate=False).certificate()
    assert result.valid and all(v==0.0 for v in result.residuals.values())
