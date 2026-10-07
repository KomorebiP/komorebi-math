import numpy as np
import pytest
import sympy as sp

from komorebi_math.deformation import (
    CompatibleCarrier,
    FiniteCochainComplex,
    FrobeniusSpine,
    PresentedModule,
    finite_frobenius_separator,
    nested_subspace_occupancy,
)
from komorebi_math.exceptions import ValidationError


def test_finite_complex_and_cohomology() -> None:
    complex_ = FiniteCochainComplex(
        {0: 2, 1: 3, 2: 1},
        {
            0: [[1, 0], [0, 0], [0, 0]],
            1: [[0, 1, 0]],
        },
        prime=5,
    )
    complex_.certificate().require()
    assert complex_.cohomology(0).cohomology_dimension == 1
    assert complex_.cohomology(1).cohomology_dimension == 1
    assert complex_.cohomology(2).cohomology_dimension == 0


def test_packet_cohomology_splits() -> None:
    complex_ = FiniteCochainComplex(
        {0: 2, 1: 2},
        {0: [[1, 0], [0, 0]]},
        prime=3,
    )
    p0 = np.diag([1, 0])
    p1 = np.diag([0, 1])
    result = complex_.packet_cohomology({0: {"a": p0, "b": p1}, 1: {"a": p0, "b": p1}})
    assert result["a"][0].cohomology_dimension == 0
    assert result["b"][0].cohomology_dimension == 1
    assert result["a"][1].cohomology_dimension == 0
    assert result["b"][1].cohomology_dimension == 1


def test_presented_module_fitting_fibers() -> None:
    x, y = sp.symbols("x y")
    module = PresentedModule([[x, 0], [0, y]], variables=(x, y))
    fitting = module.fitting_generators(0)
    assert fitting.generators == (x * y,)
    origin = module.fiber({x: 0, y: 0})
    origin.certificate.require()
    assert origin.occupied and origin.fiber_dimension == 2
    generic = module.fiber({x: 1, y: 1})
    generic.certificate.require()
    assert not generic.occupied
    axis = module.fiber({x: 0, y: 1})
    assert axis.occupied and axis.fiber_dimension == 1


def test_presented_module_prime_field_fiber() -> None:
    x = sp.symbols("x")
    module = PresentedModule([[x, 2]])
    result = module.fiber({x: 0}, prime=2)
    result.certificate.require()
    assert result.occupied


def test_nested_finite_occupancy() -> None:
    e1 = np.eye(3)
    e2 = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    e3 = np.array([[1], [0], [0]], dtype=float)
    report = nested_subspace_occupancy((e1, e2, e3))
    report.certificate.require()
    assert report.dimensions == (3, 2, 1)
    assert report.intersection_dimension == 1


def test_frobenius_spine_separator_and_carrier() -> None:
    spine = FrobeniusSpine({2: (1, -3, 2), 3: (1, -4, 3)})
    assert spine.rank == 2
    separator = finite_frobenius_separator({2: (1, 1), 3: (1, -1), 5: (2, 0)})
    separator.certificate.require()
    assert separator.separating
    assert separator.recover((5, 1)) == (3, 2)
    carrier = CompatibleCarrier(("rho1", "rho2"), (4, 2))
    assert carrier.select((3, 1)) == {"rho1": 3, "rho2": 1}


def test_frobenius_separator_certificate_rejects_rank_deficient_table() -> None:
    result = finite_frobenius_separator({2: (1, 1), 3: (2, 2), 5: (3, 3)})
    assert result.rank == 1
    assert result.constituent_count == 2
    assert not result.separating
    assert not result.certificate.valid
    assert result.certificate.diagnostics["rank_deficiency"] == 1
    with pytest.raises(ValidationError, match="do not separate"):
        result.certificate.require()
    with pytest.raises(ValidationError, match="does not separate"):
        result.recover((5,))


def test_frobenius_separator_certificate_rejects_underdetermined_table() -> None:
    result = finite_frobenius_separator({2: (1, 2, 3)})
    assert result.rank == 1
    assert result.constituent_count == 3
    assert not result.separating
    assert not result.certificate.valid
    assert result.certificate.diagnostics["rank_deficiency"] == 2
    with pytest.raises(ValidationError, match="do not separate"):
        result.certificate.require()
