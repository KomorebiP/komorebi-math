"""Exact-decision and prime-field regressions. Finite tests, not a formal proof."""
from fractions import Fraction
from itertools import combinations, permutations, product
from math import inf, nan
import random

import numpy as np
import pytest
import sympy as sp

from komorebi_math.certificates import Exactness
from komorebi_math.exceptions import ValidationError
from komorebi_math.core.packets import PacketSystem
from komorebi_math.core.operators import PacketOperator
from komorebi_math.core.transfer import block_from_paths
from komorebi_math.nonlinear.coordinates import NonlinearPacketMap
from komorebi_math.nonlinear.derivatives import CallableDerivativeProvider
from komorebi_math.nonlinear.holonomy import TransportEdge, TransportGraph
from komorebi_math.nonlinear.jets import PacketJetFamily, packet_jet_support
from komorebi_math.nonlinear.composition import faa_di_bruno, packet_faa_di_bruno
from komorebi_math.nonlinear.graded import (FiniteAbelianGroup, GradedPacketOperation,
    maurer_cartan_packets, higher_maurer_cartan_packets, divided_square_packets)
from komorebi_math.refinement.defects import LinearRefinementDefect, multilevel_defect
from komorebi_math.refinement.weighted import WeightedTransferGraph
from komorebi_math.deformation.complexes import FiniteCochainComplex, row_rank_mod

P = 4294967311
Q = 2**32
EPS = sp.Rational(1, 10**400)


def loop_graph(value):
    return TransportGraph(['v'], [TransportEdge('v', 'v', sp.Matrix([[value]]))], backend='sympy')


def two_packets():
    return PacketSystem([sp.diag(1, 0), sp.diag(0, 1)], names=['0', '1'], backend='sympy')


def test_overflow_invalid_cochain_rejected():
    assert Q*Q % P == 225
    with pytest.raises(ValidationError, match='compose to zero'):
        FiniteCochainComplex({0: 1, 1: 1, 2: 1}, {0: [[Q]], 1: [[Q]]}, prime=P)


def test_overflow_cochain_certificate_rechecks_integer_product():
    c = FiniteCochainComplex({0: 1, 1: 1, 2: 1}, {0: [[0]], 1: [[0]]}, prime=P)
    c.differentials[0][0, 0] = Q
    c.differentials[1][0, 0] = Q
    assert not c.certificate().valid
    assert c.certificate().residuals['d1_d0'] == 225


def test_overflow_valid_complex_accepted():
    # d1*d0 = Q^2-225 = 0 mod P. The literal int64 path loses Q^2.
    c = FiniteCochainComplex({0: 1, 1: 2, 2: 1},
                            {0: [[Q], [1]], 1: [[Q, -225]]}, prime=P)
    assert c.certificate().valid
    assert c.cohomology(1).cohomology_dimension == 0


def test_overflow_rank_one_not_two():
    a = [[2120775645, 3160953416], [499721574, 2242295589]]
    assert (a[0][0]*a[1][1]-a[0][1]*a[1][0]) % P == 0
    assert row_rank_mod(a, P) == 1


def test_overflow_packet_cohomology_accepted():
    # Idempotent with row sum 1 mod P, and its complementary projector.
    a = [[16, Q], [16, Q]]
    b = [[1-16, -Q], [-16, 1-Q]]
    c = FiniteCochainComplex({0: 2}, {}, prime=P)
    split = c.packet_cohomology({0: {'a': a, 'b': b}})
    assert split['a'][0].cohomology_dimension == 1
    assert split['b'][0].cohomology_dimension == 1


def test_tiny_holonomy_is_not_trivial():
    cert = loop_graph(1+EPS).gluing_certificate()
    assert not cert.valid


def test_tiny_holonomy_intertwiner_fails():
    cert = loop_graph(1+EPS).intertwine(loop_graph(1), sp.eye(1)).certificate
    assert not cert.valid


def test_tiny_nonlinear_coordinate_mismatch_rejected():
    p = PacketSystem([sp.Matrix([[1+EPS]])], backend='sympy', validate=False)
    f = NonlinearPacketMap(lambda x: x, p, p)
    assert not f.certificate(sp.ones(1, 1)).valid


def asymmetric_provider(scale=EPS):
    return CallableDerivativeProvider(lambda x: sp.zeros(1, 1),
        {2: lambda x, uv: sp.Matrix([scale*uv[0][0]*uv[1][1]])},
        2, 1, exactness=Exactness.EXACT)


def test_tiny_derivative_symmetry_rejected():
    d = asymmetric_provider()
    assert not d.validation_certificate(sp.zeros(2, 1), (sp.Matrix([1,0]), sp.Matrix([0,1]))).valid


def test_tiny_jet_symmetry_rejected():
    s = PacketSystem([sp.eye(2)], backend='sympy')
    t = PacketSystem([sp.eye(1)], backend='sympy')
    j = PacketJetFamily(asymmetric_provider(), sp.zeros(2, 1), s, t, 2).jet(0, (0, 0))
    assert not j.symmetry_certificate((sp.Matrix([1,0]), sp.Matrix([0,1]))).valid


def test_tiny_jet_reconstruction_rejected():
    s = two_packets()
    t = PacketSystem([sp.eye(1)], backend='sympy')
    # Not a derivative: finite reconstruction comparison must expose it.
    d = CallableDerivativeProvider(lambda x: sp.zeros(1,1),
        {1: lambda x, u: sp.Matrix([EPS*u[0][0]*u[0][1]])}, 2, 1, exactness=Exactness.EXACT)
    assert not PacketJetFamily(d, sp.zeros(2,1), s, t, 1).reconstruction_certificate((sp.ones(2,1),)).valid


def scalar_provider(multiplier):
    return CallableDerivativeProvider(lambda x: multiplier*x,
        {1: lambda x, u: multiplier*u[0]}, 1, 1, exactness=Exactness.EXACT)


def test_tiny_composition_comparison_rejected():
    r = faa_di_bruno(scalar_provider(1), scalar_provider(1), sp.zeros(1,1),
        (sp.ones(1,1),), direct_provider=scalar_provider(1+EPS))
    assert not r.certificate.valid


def test_tiny_packet_composition_mismatch_rejected():
    s = PacketSystem([sp.eye(1)], backend='sympy')
    m = PacketSystem([sp.Matrix([[1+EPS]])], backend='sympy', validate=False)
    r = packet_faa_di_bruno(scalar_provider(1), scalar_provider(1), sp.zeros(1,1),
        (sp.ones(1,1),), s, m, s, input_packets=(0,), output_packet=0)
    assert not r.certificate.valid


def test_tiny_forbidden_grade_rejected():
    p = two_packets()
    op = GradedPacketOperation(FiniteAbelianGroup.cyclic(2), p, 1,
          lambda u: sp.Matrix([[0,EPS],[EPS,0]])*u[0], exactness=Exactness.EXACT)
    assert not op.grading_certificate().certificate.valid
    assert not op.packet_convolution((sp.ones(2,1),), '0').certificate.valid


@pytest.mark.parametrize('which', ['ordinary', 'higher', 'divided'])
def test_tiny_graded_reconstruction_rejected(which):
    g = FiniteAbelianGroup.cyclic(2)
    components = {'0': sp.Matrix([1]), '1': sp.Matrix([2])}
    d = lambda x: EPS*x.applyfunc(lambda a: a*a)
    zero = lambda x, y: sp.zeros(1, 1)
    if which == 'ordinary':
        r = maurer_cartan_packets(g, components, d, zero, exactness=Exactness.EXACT)
    elif which == 'higher':
        r = higher_maurer_cartan_packets(g, components, {1: lambda u: d(u[0])}, exactness=Exactness.EXACT)
    else:
        r = divided_square_packets(g, components, d, zero, lambda x: sp.zeros(1,1), exactness=Exactness.EXACT)
    assert not r.certificate.valid


def test_symbolic_unknown_holonomy_is_unresolved():
    x = sp.Symbol('x', real=True)
    cert = loop_graph(1+x).gluing_certificate()
    assert not cert.valid and cert.assumptions_unchecked
    assert any(v['status']=='unknown' for v in cert.diagnostics['residual_decisions'].values())


def test_unknown_jet_support_not_proved_nonzero():
    x = sp.Symbol('x', real=True)
    p = PacketSystem([sp.eye(1)], backend='sympy')
    f = PacketJetFamily(scalar_provider(x), sp.zeros(1,1), p, p, 1)
    support = packet_jet_support(f)
    assert not support.nonzero and not support.zero
    assert len(support.untested) == 1 and not support.complete


@pytest.mark.parametrize('value', [-1, inf, nan])
def test_holonomy_tolerance_rejected(value):
    with pytest.raises(ValidationError, match='finite and nonnegative'):
        TransportGraph(['v'], [], backend='sympy', tolerance=value)


def determinant(a):
    n = len(a)
    total = 0
    for pi in permutations(range(n)):
        inversions = sum(pi[i] > pi[j] for i in range(n) for j in range(i+1,n))
        term = (-1)**inversions
        for i in range(n): term *= int(a[i][pi[i]])
        total += term
    return total


def minor_rank(a, p):
    m, n = a.shape
    for k in range(min(m,n), 0, -1):
        for rs in combinations(range(m),k):
            for cs in combinations(range(n),k):
                if determinant([[a[i,j] for j in cs] for i in rs]) % p:
                    return k
    return 0


def test_modular_ranks_against_independent_minors():
    rng = random.Random(20260921)
    count = 0
    for p in [2,3,5,7,101,P]:
        for m in range(4):
            for n in range(4):
                for sample in range(3):
                    a = np.array([[rng.randrange(-2*p,2*p) for j in range(n)] for i in range(m)],dtype=object).reshape(m,n)
                    assert row_rank_mod(a,p)==minor_rank(a,p)
                    count += 1
    assert count == 288


@pytest.mark.parametrize('p',[2,3,5,101,P])
def test_complex_and_packet_identities_integer_oracle(p):
    # Consecutive maps are outer factors with literal orthogonal middle rows.
    a = [[p-1],[1]]
    b = [[1,1]]
    assert sum(b[0][i]*a[i][0] for i in range(2)) % p == 0
    c=FiniteCochainComplex({0:1,1:2,2:1},{0:a,1:b},prime=p)
    assert c.certificate().valid
    assert [c.cohomology(i).cohomology_dimension for i in range(3)] == [0,0,0]


@pytest.mark.parametrize('v',[sp.Rational(1,7), EPS, sp.Integer(10)**400])
def test_positive_holonomy_gauge_control(v):
    g=TransportGraph(['a','b'],[TransportEdge('a','b',sp.Matrix([[v]])),
        TransportEdge('b','a',sp.Matrix([[1/v]]))],backend='sympy')
    assert g.gluing_certificate().valid
    h=g.gauge_transform({'a':sp.Matrix([[3]]),'b':sp.Matrix([[5]])})
    assert h.gluing_certificate().valid


def test_positive_symbolic_refinement_identities():
    x=sp.Symbol('x',real=True)
    d=LinearRefinementDefect(sp.eye(1),sp.Matrix([[x]]),sp.Matrix([[x+1]]),backend='sympy')
    assert d.power(3).certificate.valid
    assert d.polynomial([1,2,3]).certificate.valid
    assert d.resolvent(x+2).certificate.valid
    assert d.exponential(sp.Rational(1,2)).certificate.valid
    assert multilevel_defect([sp.eye(1),sp.eye(1)],
        [sp.Matrix([[x]]),sp.Matrix([[x+1]]),sp.Matrix([[x+2]])],backend='sympy').certificate.valid


def test_positive_symbolic_paths_and_acyclic_inverse():
    x=sp.Symbol('x',nonzero=True,real=True)
    p=two_packets()
    op=PacketOperator.from_operator(sp.Matrix([[0,x],[0,0]]),p)
    assert block_from_paths(op,source=1,target=0,length=1).certificate.valid
    w=WeightedTransferGraph(p,sp.diag(x,x+1),[],backend='sympy')
    assert w.exact_inverse_if_acyclic().certificate.valid


def test_exact_symbolic_composition_positive():
    x=sp.Symbol('x',real=True)
    r=faa_di_bruno(scalar_provider(x),scalar_provider(2),sp.zeros(1,1),
        (sp.ones(1,1),),direct_provider=scalar_provider(2*x))
    assert r.certificate.valid
    assert 'global correctness of declared derivative providers' in r.certificate.assumptions_unchecked


# Extra controls below exercise the new private decision helper and bounded
# modular computation. They are not included in the predecessor failure count.
from komorebi_math._residuals import value_audit
from komorebi_math.exceptions import DimensionError


@pytest.mark.parametrize("value,shape", [
    (sp.Integer(0), [1, 1]), ([0, 0], [2, 1]),
    (sp.zeros(2, 3), [2, 3]), (sp.zeros(0, 2), [0, 2]),
    (sp.zeros(2, 0), [2, 0]), (Fraction(0), [1, 1]),
])
def test_value_audit_exact_zero_shapes(value, shape):
    a = value_audit(exact=True, tolerance=0)
    assert a.check_value("zero", value) == 0
    assert a.valid and a.decisions["zero"]["shape"] == shape


@pytest.mark.parametrize("value,status", [
    (EPS, "nonzero"), (-EPS, "nonzero"),
    (sp.Integer(10)**400, "nonzero"),
    (sp.I*EPS, "nonzero"),
    (sp.Symbol("u", real=True), "unknown"),
    (sp.nan, "nonfinite"), (sp.oo, "nonfinite"), (sp.zoo, "nonfinite"),
])
def test_value_audit_exact_rejection_status(value, status):
    a = value_audit(exact=True, tolerance=1.0)
    a.check_value("residual", value)
    assert not a.valid and a.decisions["residual"]["status"] == status
    assert bool(a.unresolved) == (status == "unknown")


def test_value_audit_exact_algebraic_identity():
    x = sp.Symbol("u", real=True)
    a = value_audit(exact=True, tolerance=0)
    a.check_value("identity", sp.sin(x)**2 + sp.cos(x)**2 - 1)
    assert a.valid


def test_value_audit_numerical_scope_not_exact_proof():
    a = value_audit(exact=False, tolerance=1e-8)
    a.check_value("small", [1e-10])
    assert a.valid
    assert a.decisions["small"]["arithmetic"] == "numerical_tolerance"
    assert a.decisions["small"]["status"] == "within_tolerance"


def test_value_audit_zero_tolerance_does_not_underflow():
    a = value_audit(exact=False, tolerance=0)
    a.check_value("small", [1e-300])
    assert not a.valid


def test_value_audit_shape_and_duplicate_labels_rejected():
    a = value_audit(exact=True, tolerance=0)
    with pytest.raises(DimensionError, match="scalar, vector, or matrix"):
        a.check_value("tensor", np.zeros((2, 2, 2)))
    a.check_value("repeat", 0)
    with pytest.raises(ValidationError, match="duplicate"):
        a.check_value("repeat", 1)


def test_unknown_support_can_be_resolved_by_later_nonzero_witness():
    x = sp.Symbol("u", real=True)
    s = PacketSystem([sp.eye(2)], backend="sympy")
    t = PacketSystem([sp.eye(1)], backend="sympy")
    provider = CallableDerivativeProvider(lambda u: sp.zeros(1, 1),
        {1: lambda point, args: sp.Matrix([x*args[0][0] + args[0][1]])},
        2, 1, exactness=Exactness.EXACT)
    support = packet_jet_support(PacketJetFamily(provider, sp.zeros(2, 1), s, t, 1))
    assert support.complete and len(support.nonzero) == 1
    assert support.evaluations == 2


def test_jet_reconstruction_does_not_claim_global_multilinearity_checked():
    p = PacketSystem([sp.eye(1)], backend="sympy")
    family = PacketJetFamily(scalar_provider(2), sp.zeros(1, 1), p, p, 1)
    cert = family.reconstruction_certificate((sp.ones(1, 1),))
    assert cert.valid
    assert "multilinearity" not in cert.assumptions_checked
    assert any("multilinearity" in a for a in cert.assumptions_unchecked)


def test_exhaustive_two_by_two_field_ranks():
    count = 0
    for p in (2, 3, 5):
        for entries in product(range(p), repeat=4):
            a, b, c, d = entries
            expected = 0 if not any(entries) else (2 if (a*d - b*c) % p else 1)
            assert row_rank_mod([[a, b], [c, d]], p) == expected
            count += 1
    assert count == 722


@pytest.mark.parametrize("bad", [Fraction(1, 2), sp.Rational(1, 2), inf, nan, 1j])
def test_field_entries_reject_nonintegral_or_nonfinite(bad):
    with pytest.raises(ValidationError, match="integral entries"):
        row_rank_mod([[bad]], 5)


def test_modular_rank_accepts_wide_and_negative_integer_representatives():
    huge = 10**400
    a = [[huge, -huge], [huge + P, -huge - P]]
    assert row_rank_mod(a, P) == 1


def test_modular_cochain_stores_python_integer_objects():
    c = FiniteCochainComplex({0: 1, 1: 1}, {0: [[Q]]}, prime=P)
    value = c.differential(0)
    assert value.dtype == object and type(value[0, 0]) is int
