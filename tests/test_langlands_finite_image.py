import numpy as np

from komorebi_math.langlands.finite_image import (
    AutomorphicInductionPlan,
    CliffordStep,
    CliffordTower,
    InducedRepresentation,
    LinearCharacter,
    MonomialAction,
    VirtualCharacterResolution,
    VirtualCharacterTerm,
    cyclic_group,
    mackey_tensor_packets,
    symmetric_group,
)


def test_cyclic_induced_representation_and_orbits() -> None:
    group = cyclic_group(4)
    subgroup = group.subgroup((0, 2))
    character = LinearCharacter(subgroup, {0: 1, 2: -1})
    representation = InducedRepresentation(group, character)
    representation.certificate().require()
    assert representation.dimension == 2
    action = MonomialAction(representation)
    symmetric = action.symmetric_power_packets(3)
    action.packet_certificate(symmetric, expected_dimension=4).require()
    exterior = action.exterior_power_packets(2)
    action.packet_certificate(exterior, expected_dimension=1).require()
    tensor = action.tensor_packets(action)
    action.packet_certificate(tensor, expected_dimension=4).require()


def test_mackey_tensor_dimension() -> None:
    group = cyclic_group(6)
    left_subgroup = group.subgroup((0, 2, 4))
    right_subgroup = group.subgroup((0, 3))
    left = LinearCharacter(left_subgroup, {0: 1, 2: np.exp(2j * np.pi / 3), 4: np.exp(4j * np.pi / 3)})
    right = LinearCharacter(right_subgroup, {0: 1, 3: -1})
    decomposition = mackey_tensor_packets(group, left, right)
    decomposition.certificate.require()
    assert decomposition.total_dimension == left_subgroup.index * right_subgroup.index


def test_symmetric_group_subgroup_and_normality() -> None:
    group = symmetric_group(3)
    identity = (0, 1, 2)
    cycle = (1, 2, 0)
    alternating = group.generated_subgroup((cycle,), name="A3")
    assert alternating.order == 3
    assert alternating.is_normal()
    assert group.left_coset_representatives(alternating)[0] == identity


def test_clifford_word_depth_and_plan() -> None:
    tower = CliffordTower(
        (
            CliffordStep(24, 12, 3, 3, "E", attached_descent=False),
            CliffordStep(12, 4, 3, 1, "I"),
            CliffordStep(4, 2, 1, 1, "E"),
        )
    )
    report = tower.report()
    report.certificate.require()
    assert report.word == "EIE"
    assert report.depth == 1
    assert report.unresolved_extension_steps == (0,)
    plan = tower.automorphic_plan()
    assert not plan.certificate.valid
    assert plan.certificate.assumptions_unchecked


def test_clifford_plan_complete_when_descent_supplied() -> None:
    tower = CliffordTower(
        (
            CliffordStep(12, 6, 2, 2, "E", attached_descent=True),
            CliffordStep(6, 3, 2, 1, "I"),
        )
    )
    assert tower.automorphic_plan().certificate.valid


def test_virtual_character_boundary_and_induction_plan() -> None:
    resolution = VirtualCharacterResolution(
        (
            VirtualCharacterTerm(2, "chi1"),
            VirtualCharacterTerm(-1, "chi2"),
        )
    )
    assert resolution.has_negative_terms
    assert "^-1" in resolution.l_function_expression()
    group = cyclic_group(4)
    subgroup = group.subgroup((0, 2))
    character = LinearCharacter(subgroup, {0: 1, 2: -1})
    representation = InducedRepresentation(group, character)
    plan = AutomorphicInductionPlan(representation, "F", "E", "chi")
    result = plan.operations()
    assert result.certificate.exactness.value == "conditional"
    assert result.certificate.assumptions_unchecked
