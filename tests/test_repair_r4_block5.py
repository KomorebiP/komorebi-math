from __future__ import annotations
import numpy as np
import sympy as sp
import pytest

from komorebi_math import Exactness, PacketSystem
from komorebi_math.nonlinear.derivatives import CallableDerivativeProvider
from komorebi_math.nonlinear.jets import PacketJetFamily, packet_jet_support, packet_taylor
from komorebi_math.nonlinear.execution import execute_packet_jets
from komorebi_math.nonlinear.composition import packet_faa_di_bruno
from komorebi_math.nonlinear.graded import FiniteAbelianGroup, GradedPacketOperation
from komorebi_math.nonlinear.coordinates import NonlinearPacketMap
from komorebi_math.nonlinear.trees import PolynomialFixedPointProblem, OrderedRootedTree, packet_tree_reconstruction
from komorebi_math.nonlinear.transport import MovingPacketSystem, KatoTransport
from komorebi_math.deformation.companions import finite_frobenius_separator


def bad_packets():
    return PacketSystem([sp.Matrix([[2]])], names=('0',), backend='sympy', validate=False, tolerance=0)


def good_packets():
    return PacketSystem([sp.eye(1)], names=('0',), backend='sympy', tolerance=0)


def zero_provider(max_order=2):
    derivatives = {n: (lambda x, directions: sp.zeros(1, 1)) for n in range(1, max_order + 1)}
    return CallableDerivativeProvider(lambda x: sp.zeros(1,1), derivatives, 1, 1, exactness=Exactness.EXACT)


def assert_bad_parent_recorded(cert):
    assert cert.valid is False
    parent_diags = [v for k,v in cert.diagnostics.items() if k.endswith('_packet_certificate')]
    assert parent_diags and any(item['valid'] is False for item in parent_diags)
    assert any('packet axioms' in item for item in cert.assumptions_checked)


def test_jet_reconstruction_rechecks_parent_packet_axioms():
    bad=bad_packets(); f=PacketJetFamily(zero_provider(),sp.zeros(1,1),bad,bad,1)
    assert_bad_parent_recorded(f.reconstruction_certificate((sp.ones(1,1),)))


def test_jet_symmetry_rechecks_parent_packet_axioms():
    bad=bad_packets(); j=PacketJetFamily(zero_provider(),sp.zeros(1,1),bad,bad,1).jet(0,(0,))
    assert_bad_parent_recorded(j.symmetry_certificate((sp.ones(1,1),)))


def test_jet_support_rechecks_parent_packet_axioms():
    bad=bad_packets(); f=PacketJetFamily(zero_provider(),sp.zeros(1,1),bad,bad,1)
    assert_bad_parent_recorded(packet_jet_support(f).certificate)


def test_packet_execution_rechecks_parent_packet_axioms():
    bad=bad_packets(); f=PacketJetFamily(zero_provider(),sp.zeros(1,1),bad,bad,1)
    assert_bad_parent_recorded(execute_packet_jets(f,(sp.ones(1,1),)).certificate)


def test_packet_taylor_rechecks_parent_packet_axioms_even_order_zero():
    bad=bad_packets(); r=packet_taylor(zero_provider(),sp.zeros(1,1),sp.ones(1,1),bad,bad,0)
    assert_bad_parent_recorded(r.certificate)


def test_packet_faa_di_bruno_rechecks_all_parent_packet_axioms():
    bad=bad_packets(); z=zero_provider()
    r=packet_faa_di_bruno(z,z,sp.zeros(1,1),(sp.ones(1,1),),bad,bad,bad,input_packets=(0,),output_packet=0)
    assert_bad_parent_recorded(r.certificate)
    assert {'source_packet_certificate','intermediate_packet_certificate','target_packet_certificate'} <= set(r.certificate.diagnostics)


def test_graded_certificate_rechecks_packet_axioms():
    bad=bad_packets(); op=GradedPacketOperation(FiniteAbelianGroup.cyclic(1),bad,1,lambda u:sp.zeros(1,1),exactness=Exactness.EXACT,tolerance=0)
    assert_bad_parent_recorded(op.grading_certificate().certificate)


def test_graded_convolution_rechecks_packet_axioms():
    bad=bad_packets(); op=GradedPacketOperation(FiniteAbelianGroup.cyclic(1),bad,1,lambda u:sp.zeros(1,1),exactness=Exactness.EXACT,tolerance=0)
    assert_bad_parent_recorded(op.packet_convolution((sp.ones(1,1),),'0').certificate)


def test_nonlinear_coordinate_certificate_rechecks_parent_packet_axioms():
    bad=bad_packets(); m=NonlinearPacketMap(lambda x:sp.zeros(1,1),bad,bad)
    assert_bad_parent_recorded(m.certificate(sp.ones(1,1)))


def test_fixed_point_residual_marks_invalid_coordinate_system_invalid():
    bad=bad_packets(); m=NonlinearPacketMap(lambda x:sp.zeros(1,1),bad,bad)
    coords=bad.analyze(sp.zeros(1,1))
    assert_bad_parent_recorded(m.fixed_point_residual(coords).certificate)


def test_packet_tree_rechecks_parent_packet_axioms():
    bad=bad_packets(); problem=PolynomialFixedPointProblem(sp.zeros(1,1),{2:lambda args:sp.zeros(1,1)},exactness=Exactness.EXACT,tolerance=0)
    tree=OrderedRootedTree((OrderedRootedTree.leaf(),OrderedRootedTree.leaf()))
    assert_bad_parent_recorded(packet_tree_reconstruction(problem,tree,bad,tolerance=0).certificate)


def false_derivative_system():
    P0=np.diag([1.,0.]); P1=np.diag([0.,1.]); K=np.array([[0.,-1.],[1.,0.]])
    D0=K@P0-P0@K; D1=K@P1-P1@K
    return MovingPacketSystem(lambda t:(P0,P1),lambda t:(D0,D1),tolerance=1e-9)


def test_generator_does_not_claim_differentiability_was_checked():
    system=false_derivative_system()
    assert not system.derivative_certificate(0.0,step=1e-5).certificate.valid
    cert=system.generator_certificate(0.0)
    assert cert.valid
    assert 'declared differentiable packet family' not in cert.assumptions_checked
    assert any('actual time derivatives' in a for a in cert.assumptions_unchecked)


def test_kato_transport_keeps_derivative_correctness_external():
    cert=KatoTransport(false_derivative_system()).evolve(0.0,2*np.pi,steps=256).certificate
    assert cert.valid
    assert any('actual time derivatives' in a for a in cert.assumptions_unchecked)


def test_kato_refinement_propagates_derivative_prerequisite():
    cert=KatoTransport(false_derivative_system()).refinement_certificate(0.0,2*np.pi,coarse_steps=8).certificate
    assert cert.valid
    assert any('actual time derivatives' in a for a in cert.assumptions_unchecked)


def test_valid_packet_controls_still_certify():
    good=good_packets(); z=zero_provider()
    family=PacketJetFamily(z,sp.zeros(1,1),good,good,1)
    assert family.reconstruction_certificate((sp.ones(1,1),)).valid
    assert packet_jet_support(family).certificate.valid
    assert execute_packet_jets(family,(sp.ones(1,1),)).certificate.valid
    assert NonlinearPacketMap(lambda x:sp.zeros(1,1),good,good).certificate(sp.ones(1,1)).valid


def test_external_chebotarev_premise_remains_explicit_control():
    result=finite_frobenius_separator({'p2':(1,0),'p3':(0,1)})
    assert result.certificate.valid
    assert any('Chebotarev' in a for a in result.certificate.assumptions_unchecked)
