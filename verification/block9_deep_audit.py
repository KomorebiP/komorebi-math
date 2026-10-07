from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

import komorebi_math
from komorebi_math import Exactness, PacketSystem
from komorebi_math.dynamics import matrix_exponential
from komorebi_math.nonlinear import (
    CallableDerivativeProvider,
    FilteredCorrectionProblem,
    FilteredVectorSpace,
    FiniteAbelianGroup,
    FiniteFilteredCochainComplex,
    KatoTransport,
    MovingPacketSystem,
    NonlinearSchurElimination,
    PacketJetFamily,
    PolynomialDerivativeProvider,
    PolynomialFixedPointProblem,
    catalan_number,
    faa_di_bruno,
    higher_maurer_cartan_packets,
    majorant_convergence,
    ordered_tree_count,
    ordered_trees_with_leaves,
    set_partitions,
    stirling_second,
)


@dataclass
class GroupResult:
    name: str
    checks: int
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def run_group(name: str, function) -> GroupResult:
    checks = 0
    failures: list[str] = []

    def check(condition: bool, detail: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(detail)

    try:
        function(check)
    except Exception as exc:
        failures.append(f"unexpected {type(exc).__name__}: {exc}")
    return GroupResult(name, checks, failures)


def combinatorics_group(check) -> None:
    bell = [1]
    for n in range(1, 9):
        bell.append(sum(math.comb(n - 1, k) * bell[k] for k in range(n)))
    for n, expected in enumerate(bell):
        partitions = set_partitions(n)
        check(len(partitions) == expected, f"Bell number n={n}")
        check(len(set(partitions)) == expected, f"partition uniqueness n={n}")

    recurrence = {(0, 0): 1}
    for n in range(1, 13):
        for k in range(1, n + 1):
            recurrence[n, k] = recurrence.get((n - 1, k - 1), 0) + k * recurrence.get((n - 1, k), 0)
            check(stirling_second(n, k) == recurrence[n, k], f"Stirling n={n} k={k}")

    for leaves in range(1, 10):
        expected = math.comb(2 * (leaves - 1), leaves - 1) // leaves
        check(catalan_number(leaves - 1) == expected, f"Catalan closed form leaves={leaves}")
        check(len(ordered_trees_with_leaves(leaves, (2,), max_trees=1_000_000)) == expected, f"binary trees leaves={leaves}")

    count = 1
    for height in range(7):
        if height:
            count = 1 + count**2
        check(ordered_tree_count(height, (2,)) == count, f"height recurrence h={height}")


def _polynomial_provider(coefficients: list[int]) -> PolynomialDerivativeProvider:
    tensors: dict[int, np.ndarray] = {}
    for order, coefficient in enumerate(coefficients):
        shape = (1,) * (order + 1)
        tensors[order] = np.full(shape, math.factorial(order) * coefficient, dtype=object)
    return PolynomialDerivativeProvider(tensors, exactness=Exactness.EXACT)


def _poly_multiply(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def _poly_compose(outer: list[int], inner: list[int]) -> list[int]:
    result = [0]
    power = [1]
    for coefficient in outer:
        if len(result) < len(power):
            result.extend([0] * (len(power) - len(result)))
        for index, value in enumerate(power):
            result[index] += coefficient * value
        power = _poly_multiply(power, inner)
    return result


def _directional_derivative(coefficients: list[int], order: int, point: int, directions: tuple[int, ...]) -> int:
    derivative = 0
    for degree in range(order, len(coefficients)):
        derivative += coefficients[degree] * math.factorial(degree) // math.factorial(degree - order) * point ** (degree - order)
    return derivative * math.prod(directions)


def composition_group(check) -> None:
    rng = random.Random(901)
    for case in range(1200):
        order = rng.randrange(1, 5)
        outer_degree = rng.randrange(order, 5)
        inner_degree = rng.randrange(order, 5)
        outer_coefficients = [rng.randrange(-3, 4) for _ in range(outer_degree + 1)]
        inner_coefficients = [rng.randrange(-3, 4) for _ in range(inner_degree + 1)]
        outer_coefficients[-1] = rng.choice((-3, -2, -1, 1, 2, 3))
        inner_coefficients[-1] = rng.choice((-3, -2, -1, 1, 2, 3))
        point = rng.randrange(-2, 3)
        directions = tuple(rng.choice((-2, -1, 1, 2)) for _ in range(order))
        outer = _polynomial_provider(outer_coefficients)
        inner = _polynomial_provider(inner_coefficients)
        result = faa_di_bruno(
            outer,
            inner,
            np.array([point], dtype=object),
            tuple(np.array([direction], dtype=object) for direction in directions),
        )
        direct = _poly_compose(outer_coefficients, inner_coefficients)
        expected = _directional_derivative(direct, order, point, directions)
        check(result.certificate.valid, f"composition certificate case={case}")
        check(result.value[0] == expected, f"composition value case={case}")
        check(result.raw_term_count == len(set_partitions(order)), f"composition terms case={case}")


def packet_jet_group(check) -> None:
    rng = random.Random(902)
    dimension = 3
    packets = PacketSystem(
        tuple(np.diag([1.0 if index == packet else 0.0 for index in range(dimension)]) for packet in range(dimension)),
        names=tuple(str(index) for index in range(dimension)),
    )
    for case in range(400):
        order = rng.randrange(1, 4)
        tensors: dict[int, np.ndarray] = {}
        for degree in range(1, order + 1):
            tensor = np.zeros((dimension,) + (dimension,) * degree)
            for output in range(dimension):
                tensor[(output,) + (output,) * degree] = rng.randrange(-4, 5)
            tensors[degree] = tensor
        provider = PolynomialDerivativeProvider(tensors, source_dimension=dimension, target_dimension=dimension)
        point = np.array([rng.uniform(-1, 1) for _ in range(dimension)])
        directions = tuple(np.array([rng.uniform(-1, 1) for _ in range(dimension)]) for _ in range(order))
        family = PacketJetFamily(provider, point, packets, packets, order)
        certificate = family.reconstruction_certificate(directions)
        check(certificate.valid, f"jet reconstruction case={case}")
        direct = provider.derivative(order, point, directions)
        reconstructed = sum(
            family.jet(output, inputs).evaluate(directions)
            for output in packets.names
            for inputs in product(packets.names, repeat=order)
        )
        check(np.allclose(reconstructed, direct, atol=1e-10), f"jet direct sum case={case}")


def trees_group(check) -> None:
    rng = random.Random(903)
    for case in range(1000):
        seed = rng.uniform(-0.12, 0.12)
        quadratic = rng.uniform(-0.8, 0.8)
        cubic = rng.uniform(-0.3, 0.3)
        operations = {
            2: lambda values, q=quadratic: np.array([q * values[0][0] * values[1][0]]),
            3: lambda values, c=cubic: np.array([c * values[0][0] * values[1][0] * values[2][0]]),
        }
        problem = PolynomialFixedPointProblem(np.array([seed]), operations)
        depth = rng.randrange(0, 4)
        expansion = problem.tree_expansion(depth, max_trees=1_000_000)
        value = 0.0
        for _ in range(depth):
            value = seed + quadratic * value**2 + cubic * value**3
        check(expansion.certificate.valid, f"tree certificate case={case}")
        check(abs(float(expansion.value[0]) - value) < 1e-12, f"tree Picard case={case}")

    for seed_index in range(1, 10):
        for bound_index in range(1, 10):
            seed = seed_index / 100
            bound = bound_index / 20
            radius = 0.3
            result = majorant_convergence(seed, {2: bound}, radius)
            expected = seed + bound * radius**2 <= radius + 1e-12 and 2 * bound * radius < 1
            check(result.certificate.valid == expected, f"majorant a={seed} b={bound}")


def graded_group(check) -> None:
    rng = random.Random(904)
    for modulus in range(2, 9):
        group = FiniteAbelianGroup.cyclic(modulus)
        check(group.certificate().valid, f"group certificate m={modulus}")
        for case in range(120):
            components = {
                label: np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
                for label in group.labels
            }
            operations = {
                1: lambda values: 2.0 * values[0],
                2: lambda values: values[0] * values[1],
                3: lambda values: 0.25 * values[0] * values[1] * values[2],
            }
            result = higher_maurer_cartan_packets(group, components, operations)
            check(result.certificate.valid, f"graded certificate m={modulus} case={case}")
            check(np.allclose(result.reconstructed, result.direct, atol=1e-10), f"graded reconstruction m={modulus} case={case}")


def elimination_group(check) -> None:
    rng = random.Random(905)
    for case in range(1200):
        a = rng.choice((-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0))
        b, c, d, e = (rng.uniform(-2, 2) for _ in range(4))
        A, B, C, D, E = (rng.uniform(-2, 2) for _ in range(5))

        retained = CallableDerivativeProvider(
            lambda point, A=A, B=B, C=C, D=D, E=E: np.array([
                A * point[0] + B * point[1] + C * point[0] ** 2 + D * point[0] * point[1] + E * point[1] ** 2
            ]),
            {
                1: lambda point, directions, A=A, B=B, C=C, D=D, E=E: np.array([
                    (A + 2 * C * point[0] + D * point[1]) * directions[0][0]
                    + (B + D * point[0] + 2 * E * point[1]) * directions[0][1]
                ]),
                2: lambda point, directions, C=C, D=D, E=E: np.array([
                    2 * C * directions[0][0] * directions[1][0]
                    + D * (directions[0][0] * directions[1][1] + directions[0][1] * directions[1][0])
                    + 2 * E * directions[0][1] * directions[1][1]
                ]),
            },
            source_dimension=2,
            target_dimension=1,
        )
        hidden = CallableDerivativeProvider(
            lambda point, a=a, b=b, c=c, d=d, e=e: np.array([
                a * point[1] + b * point[0] + c * point[0] ** 2 + d * point[0] * point[1] + e * point[1] ** 2
            ]),
            {
                1: lambda point, directions, a=a, b=b, c=c, d=d, e=e: np.array([
                    (b + 2 * c * point[0] + d * point[1]) * directions[0][0]
                    + (a + d * point[0] + 2 * e * point[1]) * directions[0][1]
                ]),
                2: lambda point, directions, c=c, d=d, e=e: np.array([
                    2 * c * directions[0][0] * directions[1][0]
                    + d * (directions[0][0] * directions[1][1] + directions[0][1] * directions[1][0])
                    + 2 * e * directions[0][1] * directions[1][1]
                ]),
            },
            source_dimension=2,
            target_dimension=1,
        )
        reduction = NonlinearSchurElimination(retained, hidden, 1, 1, [0.0], [0.0], tolerance=1e-11)
        first = reduction.first_derivatives()
        second = reduction.second_derivative(np.array([1.0]), np.array([1.0]))
        slope = -b / a
        curvature = -(2 * c + 2 * d * slope + 2 * e * slope**2) / a
        expected_first = A + B * slope
        expected_second = 2 * C + 2 * D * slope + 2 * E * slope**2 + B * curvature
        check(first.certificate.valid, f"Schur first certificate case={case}")
        check(second.certificate.valid, f"Schur second certificate case={case}")
        check(abs(complex(first.reduced_derivative[0, 0]).real - expected_first) < 1e-10, f"Schur first case={case}")
        check(abs(complex(np.asarray(second.reduced_second_derivative).reshape(-1)[0]).real - expected_second) < 1e-9, f"Schur second case={case}")


def _weighted_space(weights: tuple[int, ...], prime: int, terminal: int) -> FilteredVectorSpace:
    dimension = len(weights)
    levels: dict[int, np.ndarray] = {}
    identity = np.eye(dimension, dtype=int)
    for depth in range(terminal + 1):
        columns = [identity[:, index] for index, weight in enumerate(weights) if weight >= depth]
        levels[depth] = np.column_stack(columns) if columns else np.zeros((dimension, 0), dtype=int)
    return FilteredVectorSpace(dimension, levels, prime=prime)


def filtered_group(check) -> None:
    rng = random.Random(906)
    primes = (2, 3, 5, 7)
    for case in range(1600):
        prime = rng.choice(primes)
        source_dimension = rng.randrange(1, 5)
        target_dimension = rng.randrange(1, 5)
        source_weights = tuple(rng.randrange(0, 4) for _ in range(source_dimension))
        target_weights = tuple(rng.randrange(0, 4) for _ in range(target_dimension))
        terminal = max((*source_weights, *target_weights)) + 1
        source = _weighted_space(source_weights, prime, terminal)
        target = _weighted_space(target_weights, prime, terminal)
        matrix = np.zeros((target_dimension, source_dimension), dtype=int)
        for row, target_weight in enumerate(target_weights):
            for column, source_weight in enumerate(source_weights):
                if target_weight >= source_weight:
                    matrix[row, column] = rng.randrange(prime)
        complex_ = FiniteFilteredCochainComplex(
            {0: source_dimension, 1: target_dimension},
            {0: matrix},
            {0: source, 1: target},
            prime=prime,
        )
        result = complex_.spectral_sequence()
        rank = int(np.linalg.matrix_rank(matrix.astype(float))) if prime > 7 else None
        check(result.certificate.valid, f"spectral convergence case={case}")
        check(result.total_cohomology_dimensions == result.limiting_dimensions, f"spectral dimensions case={case}")
        check(all(page.certificate.valid for page in result.pages), f"spectral pages case={case}")
        # Graded dimensions must partition each ambient dimension independently of the differential.
        check(sum(source.graded_dimension(p) for p in range(terminal)) == source_dimension, f"source graded sum case={case}")
        check(sum(target.graded_dimension(p) for p in range(terminal)) == target_dimension, f"target graded sum case={case}")
        _ = rank

    for prime in primes:
        space = FilteredVectorSpace.from_weights([0, 1, 2], prime=prime)
        for depth in range(3):
            defect = np.zeros(3, dtype=int)
            defect[depth] = 1
            problem = FilteredCorrectionProblem(space, space, np.eye(3, dtype=int), defect, depth)
            result = problem.solve(enumerate_limit=prime**3 + 1)
            check(result.certificate.valid, f"filtered correction p={prime} depth={depth}")
            check(result.vanishes, f"filtered obstruction vanishes p={prime} depth={depth}")
            for representative in result.correction_representatives:
                corrected = (defect + np.asarray(representative)) % prime
                check(space.contains(corrected, depth + 1), f"filtered correction improves p={prime} depth={depth}")


def transport_group(check) -> None:
    rng = random.Random(907)
    base = np.diag([1, 0]).astype(complex)
    for case in range(250):
        speed = rng.uniform(-2.0, 2.0)
        generator = speed * np.array([[0, -1], [1, 0]], dtype=complex)

        def packets(time: float, generator=generator):
            rotation = matrix_exponential(time * generator)
            first = rotation @ base @ rotation.conj().T
            return first, np.eye(2) - first

        def derivatives(time: float, generator=generator):
            return tuple(generator @ value - value @ generator for value in packets(time))

        system = MovingPacketSystem(packets, derivatives)
        time = rng.uniform(-1.0, 1.0)
        derivative = system.derivative_certificate(time, step=2e-5)
        check(derivative.certificate.valid, f"transport derivative case={case}")
        start = rng.uniform(-0.5, 0.5)
        stop = start + rng.uniform(0.05, 1.0)
        result = KatoTransport(system).evolve(start, stop, steps=64)
        exact = matrix_exponential((stop - start) * generator)
        check(result.certificate.valid, f"transport certificate case={case}")
        check(np.linalg.norm(result.value - exact, ord=2) < 1e-10, f"transport exact rotation case={case}")


def source_group(check) -> None:
    import komorebi_math.nonlinear as nonlinear
    import komorebi_math.quantum as quantum

    check(komorebi_math.__version__ == "0.9.0a1", "version")
    check(quantum.KatoTransport is nonlinear.KatoTransport, "shared Kato class")
    check(quantum.MovingPacketSystem is nonlinear.MovingPacketSystem, "shared moving packet class")
    for name in (
        "PacketJetFamily",
        "faa_di_bruno",
        "PolynomialFixedPointProblem",
        "NonlinearSchurElimination",
        "FiniteFilteredCochainComplex",
        "KatoTransport",
    ):
        check(hasattr(nonlinear, name), f"nonlinear public {name}")
        check(hasattr(komorebi_math, name), f"top-level public {name}")


GROUPS = (
    ("combinatorial identities", combinatorics_group),
    ("polynomial Faà di Bruno", composition_group),
    ("packet jet reconstruction", packet_jet_group),
    ("rooted tree expansions", trees_group),
    ("graded packet equations", graded_group),
    ("nonlinear elimination", elimination_group),
    ("filtered lifting and spectral sequences", filtered_group),
    ("smooth packet transport", transport_group),
    ("source integrity", source_group),
)


def main() -> int:
    results = [run_group(name, function) for name, function in GROUPS]
    payload = {
        "groups": [asdict(result) | {"passed": result.passed} for result in results],
        "checks": sum(result.checks for result in results),
        "failures": sum(len(result.failures) for result in results),
        "passed": all(result.passed for result in results),
    }
    output = Path(__file__).with_name("BLOCK9_DEEP_AUDIT.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Block 9 deep audit: {payload['checks']} checks, {payload['failures']} failures")
    for result in results:
        print(f"  {result.name}: {result.checks} checks, {len(result.failures)} failures")
        for failure in result.failures[:10]:
            print(f"    {failure}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
