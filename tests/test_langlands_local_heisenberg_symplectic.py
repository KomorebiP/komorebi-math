import numpy as np

from komorebi_math.langlands.local import (
    AlternatingForm,
    HeisenbergModel,
    ProjectiveIntertwinerEnvelope,
    QuadraticRefinement,
    SymplecticBridge,
)


def test_heisenberg_group_law_trace_and_selector() -> None:
    model = HeisenbergModel(3, 1)
    left = model.element((1,), (2,), 1)
    right = model.element((2,), (1,), 2)
    model.group_law_certificate(left, right).require()
    model.trace_support_certificate().require()
    model.selector_certificate().require()
    assert np.linalg.matrix_rank(model.translation_selector()) == 1


def test_symplectic_radical_and_bridge() -> None:
    form = AlternatingForm([[0, 1], [-1, 0]], 5)
    form.certificate().require()
    assert form.nondegenerate
    assert form.radical_basis() == ()
    bridge = SymplecticBridge.construct(form, form, np.eye(2, dtype=int))
    bridge.certificate.require()


def test_dyadic_quadratic_refinements_have_opposite_signs() -> None:
    form = AlternatingForm([[0, 1], [1, 0]], 2)
    q0 = QuadraticRefinement(form, (0, 0))
    q1 = QuadraticRefinement(form, (1, 1))
    q0.polar_certificate().require()
    q1.polar_certificate().require()
    assert q0.normalized_gauss_sign() == 1
    assert q1.normalized_gauss_sign() == -1


def test_projective_envelope_recovers_standard_form() -> None:
    model = HeisenbergModel(3, 1)
    labels = ((0, 0), (1, 0), (0, 1))
    operators = (
        np.eye(3),
        model.translation_matrix((1,)),
        model.modulation_matrix((1,)),
    )
    recovered = ProjectiveIntertwinerEnvelope(prime=3, labels=labels, operators=operators).recover()
    recovered.certificate.require()
    assert recovered.form.rank == 2
