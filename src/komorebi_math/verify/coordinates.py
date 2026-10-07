"""Installed-package verification for the exact coordinate calculus."""

from __future__ import annotations

import json
from fractions import Fraction
from tempfile import TemporaryDirectory
from typing import Callable

from ..coordinates import (
    CoordinateBudget,
    CoordinateTransform,
    LinearRealization,
    analyze_recurrence_coordinates,
    bezout_defect_identity,
    companion_matrix,
    connection_transform,
    coordinate_change_between_recurrences,
    derivative_operator_in_family,
    euler_fibonacci_coordinates,
    euler_polynomial_family,
    euler_to_fibonacci_transform,
    exponential_basis,
    fibonacci_factorial_basis,
    fibonacci_polynomial_family,
    intertwiner_space,
    inverse_matrix,
    matrix_multiply,
    normalization_change_transform,
    operator_change_of_coordinates,
    recover_initial_state,
    save_coordinate_record,
    save_transform,
    load_coordinate_record,
    load_transform,
    signed_secant_euler_numbers,
    variation_of_constants_prefix,
)
from ..exceptions import ResourceLimitError, ValidationError
from ..export import dumps_portable, loads_portable
from .core import VerificationCheck, VerificationReport


def _capture(name: str, function: Callable[[], None]) -> VerificationCheck:
    try:
        function()
        return VerificationCheck(name, True)
    except Exception as exc:
        return VerificationCheck(name, False, f"{type(exc).__name__}: {exc}")


def run_coordinates_verification() -> VerificationReport:
    """Run lightweight exact checks for the installed coordinate subsystem."""

    checks: list[VerificationCheck] = []

    def exact_transform() -> None:
        transform = CoordinateTransform(
            "source",
            "target",
            ((1, 2), (3, 5)),
            "CC.C1",
            "exact verification transform",
        )
        inverse = transform.inverse()
        assert transform.then(inverse).matrix == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
        transform.certificate().require()

    checks.append(_capture("exact transform composition", exact_transform))

    def normalized_change() -> None:
        source = exponential_basis()
        target = fibonacci_factorial_basis()
        forward = normalization_change_transform(source, target, 12)
        backward = normalization_change_transform(target, source, 12)
        values = tuple(Fraction((-1) ** index * (index + 1), index + 2) for index in range(13))
        assert backward.apply(forward.apply(values)) == values

    checks.append(_capture("normalized basis coordinate round trip", normalized_change))

    def recurrence_roundtrip() -> None:
        values = tuple(Fraction(value) for value in (3, -2, 8, 5, 11, -7, 19, 23))
        record = analyze_recurrence_coordinates(values, (2, -1, 1), name="verification")
        record.certificate().require()
        assert record.reconstruct() == values
        assert variation_of_constants_prefix(record) == values

    checks.append(_capture("recurrence defect reconstruction", recurrence_roundtrip))

    def recurrence_change() -> None:
        values = tuple(Fraction(value) for value in range(1, 13))
        source = analyze_recurrence_coordinates(values, (1, 1), name="Fibonacci")
        target = analyze_recurrence_coordinates(values, (2, -1, 3), name="target")
        transform = coordinate_change_between_recurrences((1, 1), (2, -1, 3), len(values))
        assert transform.apply(source.coordinates()) == target.coordinates()
        assert transform.inverse().apply(target.coordinates()) == source.coordinates()

    checks.append(_capture("recurrence system coordinate change", recurrence_change))

    def euler_fibonacci() -> None:
        record = euler_fibonacci_coordinates(24)
        expected = tuple(Fraction(value) for value in signed_secant_euler_numbers(24))
        assert record.reconstruct() == expected
        assert variation_of_constants_prefix(record) == expected
        assert record.initial == (Fraction(1), Fraction(0))
        assert not record.homogeneous

    checks.append(_capture("Euler to Fibonacci forced companion coordinates", euler_fibonacci))

    def polynomial_connection() -> None:
        transform = euler_to_fibonacci_transform(12)
        inverse = transform.inverse()
        transform.certificate().require()
        assert transform.then(inverse).roundtrip_valid()
        direct = connection_transform(euler_polynomial_family(), fibonacci_polynomial_family(), 12)
        assert transform.matrix == direct.matrix

    checks.append(_capture("Euler to Fibonacci polynomial connection", polynomial_connection))

    def operator_transport() -> None:
        degree = 10
        transform = euler_to_fibonacci_transform(degree)
        euler_derivative = derivative_operator_in_family(euler_polynomial_family(), degree)
        fibonacci_derivative = derivative_operator_in_family(fibonacci_polynomial_family(), degree)
        assert operator_change_of_coordinates(euler_derivative, transform) == fibonacci_derivative

    checks.append(_capture("operator conjugation", operator_transport))

    def realization_recovery() -> None:
        transition = companion_matrix((1, 1))
        realization = LinearRealization(transition, (3, 4), (1, 0), "Fibonacci companion")
        prefix = realization.prefix(10)
        recovered = recover_initial_state(prefix, transition, (1, 0))
        recovered.certificate().require()
        assert recovered.unique and recovered.state == (Fraction(3), Fraction(4))

    checks.append(_capture("finite observability and state recovery", realization_recovery))

    def intertwiners() -> None:
        source = companion_matrix((1, 1))
        change = ((Fraction(1), Fraction(2)), (Fraction(3), Fraction(5)))
        target = matrix_multiply(change, matrix_multiply(source, inverse_matrix(change)))
        space = intertwiner_space(source, target)
        space.certificate(source, target).require()
        assert space.dimension == 2

    checks.append(_capture("exact intertwiner space", intertwiners))

    def bezout_identity() -> None:
        values = tuple(Fraction((index * index + 3 * index - 2) % 17 - 8) for index in range(30))
        result = bezout_defect_identity(values, (-1, -1, 1), (-1, 0, 1))
        result.certificate().require()
        assert result.reconstructed == result.expected

    checks.append(_capture("Bezout defect fusion", bezout_identity))

    def portable_records() -> None:
        transform = euler_to_fibonacci_transform(7)
        record = euler_fibonacci_coordinates(18)
        assert loads_portable(dumps_portable(transform)) == transform
        assert loads_portable(dumps_portable(record)) == record
        with TemporaryDirectory() as directory:
            transform_path = save_transform(f"{directory}/transform.json", transform)
            record_path = save_coordinate_record(f"{directory}/record.json", record)
            assert load_transform(transform_path) == transform
            assert load_coordinate_record(record_path) == record
            envelope = json.loads(transform_path.read_text(encoding="utf-8"))
            envelope["payload_sha256"] = "0" * 64
            transform_path.write_text(json.dumps(envelope), encoding="utf-8")
            try:
                load_transform(transform_path)
            except ValidationError:
                pass
            else:
                raise AssertionError("tampered transform was accepted")

    checks.append(_capture("portable coordinate records and tamper rejection", portable_records))

    def resource_contract() -> None:
        budget = CoordinateBudget(maximum_matrix_dimension=4, maximum_polynomial_degree=3, maximum_horizon=4, maximum_exact_entries=16)
        try:
            euler_to_fibonacci_transform(4, budget=budget)
        except ResourceLimitError:
            return
        raise AssertionError("coordinate budget did not reject an oversized transform")

    checks.append(_capture("coordinate resource contract", resource_contract))

    def arbitrary_basis_atlas() -> None:
        from ..coordinates import CoordinateAtlas, CoordinateSystem, basis_from_matrix, standard_basis

        standard = standard_basis(3, name="standard")
        shear = basis_from_matrix(
            "shear",
            ((1, 1, 0), (0, 1, 1), (0, 0, 1)),
        )
        scale = basis_from_matrix(
            "scale",
            ((2, 0, 0), (0, 3, 0), (0, 0, 5)),
        )
        atlas = CoordinateAtlas("verification atlas")
        for basis in (standard, shear, scale):
            atlas.register_system(CoordinateSystem(basis.name, basis.dimension, "basis"))
        standard_to_shear = standard.connection_to(shear)
        shear_to_scale = shear.connection_to(scale)
        standard_to_scale = standard.connection_to(scale)
        atlas.register_transform(standard_to_shear, include_inverse=True)
        atlas.register_transform(shear_to_scale, include_inverse=True)
        atlas.register_transform(standard_to_scale, include_inverse=True)
        check = atlas.compare_paths(("standard", "scale"), ("standard", "shear", "scale"))
        check.certificate().require()
        vector = (Fraction(7), Fraction(-3), Fraction(11))
        assert atlas.resolve("standard", "scale").transform.apply(vector) == scale.analyze(vector)

    checks.append(_capture("arbitrary finite bases and coordinate atlas", arbitrary_basis_atlas))

    def combinatorial_coordinates() -> None:
        from ..coordinates import (
            binomial_transform,
            falling_factorial_to_monomial_transform,
            forward_difference_transform,
            monomial_to_falling_factorial_transform,
            newton_evaluation_transform,
        )

        degree = 9
        forward = monomial_to_falling_factorial_transform(degree)
        backward = falling_factorial_to_monomial_transform(degree)
        assert backward.then(forward).roundtrip_valid()
        values = tuple(Fraction(index * index) for index in range(8))
        differences = forward_difference_transform(8).apply(values)
        assert newton_evaluation_transform(8).apply(differences) == values
        binomial = binomial_transform(8)
        assert binomial.inverse().apply(binomial.apply(values)) == values

    checks.append(_capture("Stirling, binomial, and difference coordinates", combinatorial_coordinates))

    def classical_polynomial_coordinates() -> None:
        from ..coordinates import connection_transform, get_polynomial_family

        source = get_polynomial_family("Legendre")
        target = get_polynomial_family("Chebyshev T")
        transform = connection_transform(source, target, 10)
        transform.certificate().require()
        assert transform.inverse().apply(transform.apply(tuple(range(11)))) == tuple(
            Fraction(value) for value in range(11)
        )

    checks.append(_capture("classical polynomial-family atlas", classical_polynomial_coordinates))

    def affine_and_formal_coordinates() -> None:
        from ..coordinates import (
            BarycentricCoordinateSystem,
            FormalCoordinateChange,
            affine_basis_transform,
        )

        chart = affine_basis_transform((2, -1), ((1, 1), (1, -1)))
        point = (Fraction(3, 2), Fraction(-5, 3))
        assert chart.inverse().apply(chart.apply(point)) == point
        simplex = BarycentricCoordinateSystem("triangle", ((0, 0), (2, 0), (0, 3)))
        weights = (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2))
        assert simplex.from_cartesian(simplex.to_cartesian(weights)) == weights
        formal = FormalCoordinateChange("verification jet", (0, 1, 1, -2, 3), "u", "z")
        formal.certificate().require()
        polynomial = (3, -1, 4, 1, 5)
        assert formal.pushforward_coefficients(formal.pullback_coefficients(polynomial)) == tuple(
            Fraction(value) for value in polynomial
        )

    checks.append(_capture("affine, barycentric, and formal coordinates", affine_and_formal_coordinates))

    def spectral_coordinates() -> None:
        import numpy as np

        from ..coordinates import (
            discrete_cosine_transform,
            discrete_fourier_transform,
            haar_wavelet_transform,
            walsh_hadamard_transform,
        )

        values = np.arange(8, dtype=float) - 2.5
        for transform in (
            discrete_fourier_transform(8),
            discrete_cosine_transform(8),
            walsh_hadamard_transform(8),
            haar_wavelet_transform(8),
        ):
            transform.certificate().require()
            assert np.allclose(transform.inverse().apply(transform.apply(values)), values, atol=1e-10)

    checks.append(_capture("Fourier, cosine, Hadamard, and Haar coordinates", spectral_coordinates))

    def packet_coordinates() -> None:
        from ..coordinates import (
            chinese_remainder_transform,
            digit_reversal_transform,
            residue_class_transform,
        )

        values = tuple(range(12))
        residue = residue_class_transform(12, 4)
        assert residue.inverse().apply(residue.apply(values)) == values
        reversal = digit_reversal_transform(8, 2)
        assert reversal.apply(tuple(range(8))) == (0, 4, 2, 6, 1, 5, 3, 7)
        crt = chinese_remainder_transform((3, 4, 5))
        indices = tuple(range(60))
        assert crt.inverse().apply(crt.apply(indices)) == indices

    checks.append(_capture("packet, residue, digit, and CRT coordinates", packet_coordinates))

    def tensor_and_operator_transport() -> None:
        from ..coordinates import (
            coordinate_pairing,
            pairing_invariant,
            tensor_coordinate_transform,
            transport_operator,
            transport_tensor_components,
        )

        transform = CoordinateTransform("old", "new", ((2, 1), (1, 1)), "CC.T")
        vector = (Fraction(3), Fraction(-2))
        covector = (Fraction(5), Fraction(7))
        assert coordinate_pairing(covector, vector) == 1
        assert pairing_invariant(covector, vector, transform)
        operator = ((1, 2), (3, 4))
        transported = transport_operator(operator, transform)
        assert len(transported) == 2
        tensor = tensor_coordinate_transform(transform, ("contravariant", "covariant"))
        components = (1, 2, 3, 4)
        assert tensor.apply(components) == transport_tensor_components(
            components, transform, ("contravariant", "covariant")
        )

    checks.append(_capture("tensor, covector, form, and operator transport", tensor_and_operator_transport))

    def recurrence_model_atlas() -> None:
        from ..coordinates import (
            get_recurrence_model,
            recurrence_model_catalog,
            sequence_coordinates_between_models,
        )

        assert len(recurrence_model_catalog()) >= 6
        source = get_recurrence_model("Pell")
        target = get_recurrence_model("Tribonacci")
        values = source.sequence(16)
        source_record, target_record, transform = sequence_coordinates_between_models(
            values, source, target
        )
        assert transform.apply(source_record.coordinates()) == target_record.coordinates()

    checks.append(_capture("named and custom recurrence-model atlas", recurrence_model_atlas))

    def state_space_coordinates() -> None:
        from ..coordinates import (
            certify_state_coordinate_change,
            controllability_coordinate_transform,
            observability_coordinate_transform,
            siso_state_space,
        )

        model = siso_state_space(
            ((0, 1, 0), (0, 0, 1), (2, -1, 3)),
            (0, 0, 1),
            (1, 2, -1),
            feedthrough=2,
            name="verification model",
        )
        controllability = controllability_coordinate_transform(model)
        certify_state_coordinate_change(model, controllability).certificate().require()
        observability = observability_coordinate_transform(model)
        state = (Fraction(2), Fraction(-3), Fraction(5))
        assert observability.inverse().apply(observability.apply(state)) == state

    checks.append(_capture("controllability and observability coordinates", state_space_coordinates))

    def finite_field_coordinates() -> None:
        from ..coordinates import binomial_transform, reduce_exact_transform_mod_prime

        transform = reduce_exact_transform_mod_prime(binomial_transform(9), 7)
        transform.certificate().require()
        values = tuple(range(9))
        assert transform.inverse().apply(transform.apply(values)) == tuple(value % 7 for value in values)

    checks.append(_capture("finite-field coordinate transforms", finite_field_coordinates))

    def polynomial_calculus_coordinates() -> None:
        from ..coordinates import (
            derivative_coordinate_operator,
            finite_difference_coordinate_operator,
            get_polynomial_family,
            transport_polynomial_operator,
        )

        monomials = get_polynomial_family("monomial")
        legendre = get_polynomial_family("legendre")
        derivative = derivative_coordinate_operator(monomials, 8)
        transported = transport_polynomial_operator(derivative, monomials, legendre)
        assert transported.matrix == derivative_coordinate_operator(legendre, 8).matrix
        falling = get_polynomial_family("falling")
        difference = finite_difference_coordinate_operator(falling, 8)
        basis_vector = tuple(Fraction(int(index == 5)) for index in range(9))
        assert difference.apply(basis_vector)[4] == 5

    checks.append(_capture("coordinate differential and difference calculus", polynomial_calculus_coordinates))

    def general_portable_records() -> None:
        from ..coordinates import (
            CoordinateAtlas,
            CoordinateSystem,
            FormalCoordinateChange,
            basis_from_matrix,
            get_recurrence_model,
            save_coordinate_object,
            load_coordinate_object,
        )

        atlas = CoordinateAtlas("portable atlas")
        standard = basis_from_matrix("standard", ((1, 0), (0, 1)))
        shear = basis_from_matrix("shear", ((1, 1), (0, 1)))
        for basis in (standard, shear):
            atlas.register_system(CoordinateSystem(basis.name, basis.dimension, "basis"))
        atlas.register_transform(standard.connection_to(shear), include_inverse=True)
        records = (
            atlas,
            standard,
            FormalCoordinateChange("portable jet", (0, 1, 1), "u", "z"),
            get_recurrence_model("Padovan"),
        )
        for value in records:
            decoded = loads_portable(dumps_portable(value))
            if isinstance(value, CoordinateAtlas):
                assert decoded.systems == value.systems and decoded.transforms == value.transforms
            else:
                assert decoded == value
        with TemporaryDirectory() as directory:
            path = save_coordinate_object(f"{directory}/basis.json", standard)
            assert load_coordinate_object(path) == standard

    checks.append(_capture("general portable coordinate records", general_portable_records))

    return VerificationReport(tuple(checks))


__all__ = ["run_coordinates_verification"]
