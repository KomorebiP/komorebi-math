"""Public-path contracts for the Block 3 numerical/exact boundary audit.

False declared callbacks exercise rejection of a finite comparison, not a claim
that the library can prove arbitrary callbacks correct. Integer expectations use
Python integers. No current production module is imported from the predecessor.
"""
from dataclasses import replace
from fractions import Fraction
import numpy as np
import pytest
import sympy as sp
from komorebi_math import Exactness, PacketSystem
from komorebi_math._residuals import value_audit
from komorebi_math.exceptions import ValidationError
from komorebi_math.nonlinear.correction import HilbertObstructionProjector
from komorebi_math.nonlinear.derivatives import CallableDerivativeProvider, PolynomialDerivativeProvider
from komorebi_math.nonlinear.execution import execute_packet_jets
from komorebi_math.nonlinear.jets import PacketJetFamily
from komorebi_math.nonlinear.obstruction import ObstructionSquare
from komorebi_math.nonlinear.snapshots import snapshot_packet_jet, PacketJetCoefficient
from komorebi_math.nonlinear.sparse import SparseSymmetricTensor, SparsePolynomialDerivativeProvider
from komorebi_math.nonlinear.streaming import capture_packet_jet_stream, open_packet_jet_snapshot_stream, save_packet_jet_snapshot_stream
from komorebi_math.nonlinear.trees import PolynomialFixedPointProblem, OrderedRootedTree, packet_tree_reconstruction

EPS=sp.Rational(1,10**400)

def family(coefficient=1, *, exact=True, dimension=1, order=1, constant=False):
    exactness=Exactness.EXACT if exact else Exactness.NUMERICAL
    basis=[]
    for i in range(dimension):
        matrix=sp.zeros(dimension) if exact else np.zeros((dimension,dimension))
        matrix[i,i]=1;basis.append(matrix)
    packets=PacketSystem(basis,backend='sympy' if exact else 'numpy',names=tuple(f'p{i}' for i in range(dimension)),tolerance=0)
    def derivative(point,directions):
        if constant:return [coefficient]*dimension
        values=[]
        for i in range(dimension):
            product=coefficient
            for vector in directions:product=product*vector[i]
            values.append(product)
        return values
    provider=CallableDerivativeProvider(lambda x:[0]*dimension,{order:derivative},dimension,dimension,exactness=exactness)
    return PacketJetFamily(provider,[0]*dimension,packets,packets,order)

def jet(coefficient=1, **kwargs):
    f=family(coefficient,**kwargs)
    return f.jet('p0',('p0',)*f.order)

@pytest.mark.parametrize('value',[1e-200,5e-324,1e-100])
def test_hilbert_nonzero_obstruction_at_zero_tolerance(value):
    result=HilbertObstructionProjector([[0]],tolerance=0).solve([value])
    assert not result.vanishes
    assert result.least_norm_correction is None
    assert result.obstruction_norm>0
    assert result.certificate.valid

@pytest.mark.parametrize('value',[1e-12,1e-20,1e-200])
def test_hilbert_pseudoinverse_uses_same_retained_rank(value):
    model=HilbertObstructionProjector([[value]],tolerance=1e-10)
    assert model.rank==0
    assert np.array_equal(model.pseudoinverse,np.zeros((1,1)))
    assert np.array_equal(model.matrix@model.pseudoinverse,model.image_projector)

@pytest.mark.parametrize('q',[2**32,2**40,2**62])
@pytest.mark.parametrize('storage',['native','object'])
def test_sparse_exact_contraction_never_uses_wrapping_integer_products(q,storage):
    vectors=[np.array([q],dtype=np.int64)]*2
    if storage=='object':vectors=[np.array([np.int64(q)],dtype=object)]*2
    tensor=SparseSymmetricTensor.from_mapping(2,1,1,{(0,0,0):np.int64(3)})
    assert tensor.contract(vectors,exact=True)[0]==3*q*q

@pytest.mark.parametrize('q',[2**32,2**40,2**62])
def test_snapshot_exact_contraction_keeps_integer_factors_exact(q):
    snap=snapshot_packet_jet(jet(1,order=2))
    assert snap.evaluate([np.array([q],dtype=np.int64)]*2)[0]==q*q

@pytest.mark.parametrize('q',[2**32,2**40])
def test_stream_exact_contraction_keeps_integer_factors_exact(q,tmp_path):
    snap=snapshot_packet_jet(jet(1,order=2))
    # The stream codec supports Python integers, not arbitrary SymPy objects.
    snap=replace(snap,point=tuple(int(v) for v in snap.point),
                 coefficients=tuple(PacketJetCoefficient(c.indices,tuple(int(v) for v in c.value)) for c in snap.coefficients))
    save_packet_jet_snapshot_stream(snap,tmp_path/'data')
    reader=open_packet_jet_snapshot_stream(tmp_path/'data',verify_chunks=True)
    result=reader.evaluate([np.array([q],dtype=np.int64)]*2)
    assert result.value[0]==q*q
    assert result.certificate.valid

@pytest.mark.parametrize('coefficient',[EPS,sp.Rational(1,10**20),sp.Rational(-1,10**20)])
@pytest.mark.parametrize('factory',['constructor','from_dense_provider'])
def test_exact_sparse_provider_never_prunes_using_numerical_tolerance(coefficient,factory):
    tensors={1:np.array([[coefficient]],dtype=object)}
    if factory=='constructor':
        provider=SparsePolynomialDerivativeProvider(tensors,exactness=Exactness.EXACT)
    else:
        dense=PolynomialDerivativeProvider(tensors,exactness=Exactness.EXACT)
        provider=SparsePolynomialDerivativeProvider.from_dense_provider(dense,tolerance=1e-10)
    assert provider.stored_entry_count==1
    assert provider.value([1])[0]==coefficient
    assert provider.derivative(1,[0],[[1]])[0]==coefficient

@pytest.mark.parametrize('coefficient',[sp.Symbol('x',real=True),sp.Integer(10)**400])
def test_symbolic_snapshot_does_not_require_float_diagnostic(coefficient):
    source=jet(coefficient)
    snap=snapshot_packet_jet(source)
    assert snap.nnz==1
    assert sp.simplify(snap.evaluate([[2]])[0]-2*coefficient)==0
    assert snap.comparison_certificate(source,[[2]]).valid

@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_snapshot_rejects_nonfinite_evaluated_coefficients(value):
    with pytest.raises(ValidationError):snapshot_packet_jet(jet(value,exact=False))

@pytest.mark.parametrize('residual',[EPS,sp.Symbol('x',real=True)])
def test_snapshot_comparison_does_not_accept_tiny_or_unknown_exact_disagreement(residual):
    source=jet(0)
    snap=snapshot_packet_jet(source)
    altered=replace(snap,coefficients=(PacketJetCoefficient((0,),(residual,)),))
    assert not altered.comparison_certificate(source,[[1]]).valid

@pytest.mark.parametrize('residual',[EPS,sp.Symbol('x',real=True)])
def test_planned_reconstruction_checks_residual_not_float_norm(residual):
    f=family(residual,dimension=2,constant=True)
    result=execute_packet_jets(f,[[1,1]],tolerance=0)
    assert not result.certificate.valid

@pytest.mark.parametrize('residual',[EPS,sp.Symbol('x',real=True),sp.Rational(1,10**20)])
def test_dense_polynomial_symmetry_rejects_nonsymmetric_exact_tensors(residual):
    t=np.zeros((1,2,2),object);t[0,0,1]=residual
    provider=PolynomialDerivativeProvider({2:t},exactness=Exactness.EXACT)
    assert not provider.symmetry_certificate().valid

@pytest.mark.parametrize('residual',[1e-200,5e-324])
def test_dense_polynomial_symmetry_zero_tolerance_is_not_squaring_underflow(residual):
    t=np.zeros((1,2,2));t[0,0,1]=residual
    provider=PolynomialDerivativeProvider({2:t},tolerance=0)
    assert not provider.symmetry_certificate().valid

@pytest.mark.parametrize('residual',[EPS,sp.Symbol('x',real=True)])
def test_obstruction_square_exact_check_is_not_a_float_norm(residual):
    square=ObstructionSquare(sp.Matrix([[1]]),sp.Matrix([[1]]),sp.Matrix([[1]]),sp.Matrix([[1+residual]]),backend='sympy')
    assert not square.certificate().valid

@pytest.mark.parametrize('value',[-1,float('inf'),float('nan')])
@pytest.mark.parametrize('api',['execution','snapshot_compare','obstruction_square','dense_provider'])
def test_selected_tolerances_reject_invalid_values(value,api):
    with pytest.raises(ValidationError):
        if api=='execution':execute_packet_jets(family(1),[[1]],tolerance=value)
        elif api=='snapshot_compare':
            j=jet();snapshot_packet_jet(j).comparison_certificate(j,[[1]],tolerance=value)
        elif api=='obstruction_square':ObstructionSquare([[1]],[[1]],[[1]],[[1]],tolerance=value).certificate()
        else:PolynomialDerivativeProvider({1:np.ones((1,1))},tolerance=value)

@pytest.mark.parametrize('coefficient',[EPS,sp.Symbol('x',real=True)])
def test_nonmultilinear_tiny_tree_callback_is_not_certified(coefficient):
    # B(0,0) must vanish for multilinearity; this deliberately false callback
    # lets the finite identity checker expose a discrepancy without global proof.
    problem=PolynomialFixedPointProblem(sp.Matrix([0]),{2:lambda args:sp.Matrix([coefficient])},exactness=Exactness.EXACT,tolerance=0)
    assert not problem.tree_expansion(1).certificate.valid

@pytest.mark.parametrize('bad',[sp.nan,sp.oo])
def test_exact_residual_scan_does_not_skip_nonfinite_after_nonzero(bad):
    audit=value_audit(exact=True,tolerance=0)
    audit.check_value('x',[1,bad])
    assert audit.decisions['x']['status']=='nonfinite'

@pytest.mark.parametrize('coefficient',[sp.Rational(2,3),-7,EPS])
def test_exact_snapshot_and_plan_positive_controls(coefficient):
    f=family(coefficient,order=2)
    j=f.jet('p0',('p0','p0'))
    snap=snapshot_packet_jet(j)
    assert snap.evaluate([[2],[3]])[0]==6*coefficient
    assert snap.comparison_certificate(j,[[2],[3]]).valid
    assert execute_packet_jets(f,[[2],[3]],tolerance=0).certificate.valid

@pytest.mark.parametrize('matrix,expected_rank',[([[1.,0.],[0.,0.]],1),([[2.,0.],[0.,3.]],2),([[0.,0.],[0.,0.]],0)])
def test_hilbert_positive_controls(matrix,expected_rank):
    model=HilbertObstructionProjector(matrix,tolerance=1e-10)
    assert model.rank==expected_rank
    assert np.allclose(model.matrix@model.pseudoinverse,model.image_projector,atol=1e-14)
    result=model.solve([0,0]);assert result.vanishes and result.certificate.valid

def test_symbolic_tree_identity_positive_control():
    x=sp.Symbol('x',real=True)
    problem=PolynomialFixedPointProblem(sp.Matrix([x]),{2:lambda args:sp.Matrix([args[0][0]*args[1][0]])},exactness=Exactness.EXACT,tolerance=0)
    assert problem.tree_expansion(3).certificate.valid


def test_exact_sparse_storage_symbolic_positive_control():
    x=sp.Symbol('x',real=True)
    provider=SparsePolynomialDerivativeProvider({1:np.array([[x]],object)},exactness=Exactness.EXACT,tolerance=0)
    assert provider.storage_certificate(([1],),([1],)).valid

@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_hilbert_rejects_nonfinite_matrix(value):
    with pytest.raises(ValidationError):HilbertObstructionProjector([[value]])

@pytest.mark.parametrize('shape',[(0,0),(0,2),(2,0)])
def test_hilbert_empty_dimensional_models(shape):
    model=HilbertObstructionProjector(np.zeros(shape),tolerance=0)
    assert model.pseudoinverse.shape==shape[::-1]
    result=model.solve(np.zeros(shape[0]));assert result.vanishes and result.certificate.valid

@pytest.mark.parametrize('large,small,tol',[(1.,1e-12,1e-10),(1e6,1e-8,1e-10),(1e-3,1e-12,1e-10)])
def test_hilbert_mixed_scale_retained_rank_contract(large,small,tol):
    model=HilbertObstructionProjector(np.diag([large,small]),tolerance=tol)
    assert model.rank==1
    assert np.allclose(model.pseudoinverse,np.diag([1/large,0]),rtol=1e-15,atol=0)
    assert np.allclose(model.matrix@model.pseudoinverse,model.image_projector,rtol=0,atol=1e-15)
    assert not model.solve([0,1]).vanishes

@pytest.mark.parametrize('value',[1e-200,5e-324,1e-20])
def test_numerical_capture_keeps_nonzero_at_zero_tolerance(value):
    source=jet(value,exact=False)
    snapshot=snapshot_packet_jet(source,tolerance=0)
    assert snapshot.nnz==1
    assert snapshot.evaluate([[1]])[0]==value

@pytest.mark.parametrize('value',[sp.nan,sp.oo])
def test_symbolic_capture_rejects_known_nonfinite_even_after_nonzero(value):
    f=family(1,dimension=2)
    f.provider.derivatives={1:lambda x,d:[1,value]}
    with pytest.raises(ValidationError):snapshot_packet_jet(f.jet('p0',('p0',)))

@pytest.mark.parametrize('tol',[-1,np.inf,np.nan])
def test_sparse_certificate_rejects_invalid_tolerance(tol):
    tensor=SparseSymmetricTensor.from_mapping(1,1,1,{(0,0):1})
    with pytest.raises(ValidationError):tensor.certificate(tolerance=tol)

@pytest.mark.parametrize('residual',[EPS,sp.Symbol('x',real=True),float('nan')])
def test_sparse_comparison_rejects_wrong_dense_provider(residual,monkeypatch):
    provider=SparsePolynomialDerivativeProvider({1:np.array([[0]],object)},exactness=Exactness.EXACT,tolerance=0)
    wrong=PolynomialDerivativeProvider({1:np.array([[residual]],object)},exactness=Exactness.EXACT,tolerance=0)
    monkeypatch.setattr(provider,'to_dense_provider',lambda:wrong)
    assert not provider.storage_certificate(([1],),([1],)).valid

@pytest.mark.parametrize('coefficient',[EPS,sp.Symbol('x',real=True)])
def test_packet_tree_false_multilinear_callback_rejected(coefficient):
    packets=PacketSystem([sp.diag(1,0),sp.diag(0,1)],backend='sympy',names=('a','b'),tolerance=0)
    problem=PolynomialFixedPointProblem(sp.zeros(2,1),{2:lambda args:sp.Matrix([coefficient,coefficient])},exactness=Exactness.EXACT,tolerance=0)
    tree=OrderedRootedTree((OrderedRootedTree.leaf(),OrderedRootedTree.leaf()))
    assert not packet_tree_reconstruction(problem,tree,packets,tolerance=0).certificate.valid

@pytest.mark.parametrize('order',[0,1,2,3])
def test_symmetry_certificate_never_ignores_nan_at_any_order(order):
    tensor=np.full((1,)+(1,)*order,np.nan)
    p=PolynomialDerivativeProvider({order:tensor},source_dimension=1,target_dimension=1)
    assert not p.symmetry_certificate().valid


def test_snapshot_comparison_does_not_promote_numerical_jet_to_exact():
    j=jet(1,exact=False)
    s=snapshot_packet_jet(j)
    s=replace(s,exactness=Exactness.EXACT)
    assert s.comparison_certificate(j,[[1]]).exactness is Exactness.NUMERICAL


def test_no_probe_storage_comparison_declares_absent_evidence():
    p=SparsePolynomialDerivativeProvider({1:np.ones((1,1))})
    cert=p.storage_certificate()
    assert cert.diagnostics['value_probes']==0
    assert any('no value' in h for h in cert.assumptions_unchecked)
