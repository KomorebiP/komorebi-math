from fractions import Fraction

from komorebi_math.coordinates import (
    LinearRealization,
    companion_matrix,
    cyclic_krylov_matrix,
    euler_fibonacci_coordinates,
    intertwiner_space,
    observability_matrix,
    realization_defect,
    recover_initial_state,
    signed_secant_euler_numbers,
    similarity_from_cyclic_vectors,
)
from komorebi_math.coordinates.exact import (
    inverse_matrix,
    matrix_multiply,
    matrix_vector_multiply,
)


def test_fibonacci_observability_and_euler_obstruction() -> None:
    companion = companion_matrix((1, 1))
    output = (1, 0)
    assert observability_matrix(companion, output, 2) == ((1, 0), (0, 1))
    first_two = recover_initial_state(signed_secant_euler_numbers(2), companion, output)
    assert first_two.exact and first_two.unique and first_two.state == (1, 0)
    longer = recover_initial_state(signed_secant_euler_numbers(8), companion, output)
    assert not longer.exact
    assert longer.state == (1, 0)
    assert longer.residual[2] == -2


def test_intertwiner_space_and_similarity() -> None:
    source = companion_matrix((1, 1))
    change = ((Fraction(1), Fraction(2)), (Fraction(3), Fraction(5)))
    target = matrix_multiply(change, matrix_multiply(source, inverse_matrix(change)))
    space = intertwiner_space(source, target)
    assert space.dimension == 2
    assert any(matrix_multiply(target, matrix) == matrix_multiply(matrix, source) for matrix in space.basis)
    source_vector = (1, 0)
    target_vector = matrix_vector_multiply(change, source_vector)
    recovered = similarity_from_cyclic_vectors(source, target, source_vector, target_vector)
    assert recovered == change


def test_realization_defect_is_generated_by_source_dynamics() -> None:
    companion = companion_matrix((2, -1, 3))
    realization = LinearRealization(companion, (1, 2, 4), (1, 0, 0), "source")
    defect = realization_defect(realization, (1, 1))
    values = realization.prefix(20)
    direct = tuple(values[n + 2] - values[n + 1] - values[n] for n in range(18))
    assert defect.prefix(18) == direct


def test_cyclic_krylov_matrix() -> None:
    companion = companion_matrix((1, 1))
    assert cyclic_krylov_matrix(companion, (1, 0)) == ((1, 0), (0, 1))
