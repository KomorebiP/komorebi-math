"""Independent finite oracles for installed mathematical machinery.

The registry is deliberately a tuple of factories. Additional domain checks
can be appended by callers without replacing or narrowing these suites.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from itertools import permutations, product
from math import comb, factorial

from ..certificates import Exactness
from .runner import CheckOutcome, StressCheck, StressContext, _integer


def _digest(values) -> str:
    return hashlib.sha256(
        json.dumps(values, default=str, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _core(context: StressContext) -> CheckOutcome:
    import numpy as np

    from ..core import OperatorGrammar, OperatorRealization, PacketOperator, PacketSystem

    cases = []
    for _ in range(context.parameters["samples"]):
        size = context.rng.randrange(2, 7)
        split = context.rng.randrange(1, size)
        projectors = [
            np.diag([int(i < split) for i in range(size)]),
            np.diag([int(i >= split) for i in range(size)]),
        ]
        packets = PacketSystem(projectors, backend="numpy")
        data = [[context.rng.randrange(-9, 10) for _ in range(size)] for _ in range(size)]
        matrix = np.asarray(data, dtype=float)
        blocked = PacketOperator.from_operator(matrix, packets)
        # Direct entry-wise multiplication, independent of packet composition.
        squared = [
            [sum(data[i][k] * data[k][j] for k in range(size)) for j in range(size)] for i in range(size)
        ]
        if not np.array_equal(blocked.power(2).reconstruct(), squared):
            return CheckOutcome(
                False, {"matrix": data}, "block powers disagree with direct scalar multiplication"
            )
        blocked.certificate().require()
        packets.certificate().require()
        cases.append({"dimension": size, "split": split, "matrix": data})
    bad_packet = PacketSystem([2 * np.eye(2)], backend="numpy", validate=False)
    grammar = OperatorGrammar(["X"], {"T": ("X", "X")})
    grammar.relation("idempotent", [(1, grammar.word("T", "T")), (-1, grammar.word("T"))])
    realization = OperatorRealization(
        grammar, {"X": 2}, {"T": 2 * np.eye(2)}, backend="numpy", validate_relations=False
    )
    rejected = not bad_packet.certificate().valid and not realization.relation_certificate().valid
    return CheckOutcome(
        rejected, {"cases": cases, "negative_controls": ["2I is not idempotent", "T=2I violates T^2=T"]}
    )


def _arithmetic(context: StressContext) -> CheckOutcome:
    import numpy as np

    from ..arithmetic.hurwitz import hurwitz_product
    from ..arithmetic.recurrence import LinearRecurrence, matrix_power_mod
    from ..sequences.families import kbonacci_closed_form, reciprocal_coefficients
    from ..sequences.transforms import binomial_transform_values, inverse_binomial_transform_values

    cases = []
    for _ in range(context.parameters["samples"]):
        values = tuple(context.rng.randrange(-(2**120), 2**120) for _ in range(12))
        transformed = tuple(sum(comb(n, k) * values[k] for k in range(n + 1)) for n in range(12))
        if (
            binomial_transform_values(values) != transformed
            or inverse_binomial_transform_values(transformed) != values
        ):
            return CheckOutcome(False, {"values": values}, "binomial oracle mismatch")
        order = context.rng.randrange(1, 6)
        coefficients = tuple(context.rng.randrange(-7, 8) for _ in range(order))
        initial = tuple(context.rng.randrange(-(2**120), 2**120) for _ in range(order))
        direct = list(initial)
        for n in range(24 - order):
            direct.append(sum(coefficients[k] * direct[n + k] for k in range(order)))
        recurrence = LinearRecurrence(coefficients, initial)
        modulus = context.rng.choice((7, 2**63 + 159, 10**100 + 7))
        if any(
            recurrence.term(n) != direct[n] or recurrence.term_mod(n, modulus) != direct[n] % modulus
            for n in range(24)
        ):
            return CheckOutcome(
                False,
                {"coefficients": coefficients, "initial": initial, "modulus": modulus},
                "recurrence oracle mismatch",
            )
        a = tuple(F(context.rng.randrange(-10, 11), context.rng.randrange(1, 8)) for _ in range(7))
        b = tuple(F(context.rng.randrange(-10, 11), context.rng.randrange(1, 8)) for _ in range(7))
        expected = tuple(
            factorial(n)
            * sum(
                (
                    a[k] / factorial(k) * b[n - k] / factorial(n - k)
                    for k in range(n + 1)
                    if k < 7 and n - k < 7
                ),
                F(0),
            )
            for n in range(13)
        )
        if hurwitz_product(a, b, degree=12) != expected:
            return CheckOutcome(False, {"a": a, "b": b}, "Hurwitz/ordinary Taylor oracle mismatch")
        matrix = [[0, 2**120 + 3], [-(2**90 + 7), 1]]
        repeated = [[1, 0], [0, 1]]
        for _power in range(7):
            repeated = [
                [sum(repeated[i][k] * matrix[k][j] for k in range(2)) % modulus for j in range(2)]
                for i in range(2)
            ]
        if matrix_power_mod(np.array(matrix, dtype=object), 7, modulus).tolist() != repeated:
            return CheckOutcome(
                False, {"matrix": matrix, "modulus": modulus}, "large-integer matrix oracle mismatch"
            )
        cases.append(
            {
                "values": values,
                "coefficients": coefficients,
                "initial": initial,
                "modulus": modulus,
                "hurwitz_a": a,
                "hurwitz_b": b,
                "outputs_sha256": _digest([direct, transformed, expected, repeated]),
            }
        )
    # Direct rolling sums and ordinary rational series inversion are independent
    # of the archive's closed form and compressed Hurwitz algorithm.
    family_count = 0
    for order in range(2, 9):
        direct = [0] * (order - 1) + [1]
        while len(direct) < 50:
            direct.append(sum(direct[-order:]))
        if any(kbonacci_closed_form(order, n) != value for n, value in enumerate(direct)):
            return CheckOutcome(False, {"order": order}, "k-bonacci closed form mismatch")
        family_count += 50
    for spacing in range(1, 6):
        degree = spacing * 6
        denominator = [
            F((-1) ** (n // spacing), factorial(n)) if n % spacing == 0 else F(0) for n in range(degree + 1)
        ]
        reciprocal = [F(1)]
        for n in range(1, degree + 1):
            reciprocal.append(-sum((denominator[k] * reciprocal[n - k] for k in range(1, n + 1)), F(0)))
        expected = tuple(reciprocal[spacing * n] * factorial(spacing * n) for n in range(7))
        if reciprocal_coefficients(spacing, 7) != expected:
            return CheckOutcome(False, {"spacing": spacing}, "reciprocal normalization mismatch")
        family_count += 7
    return CheckOutcome(True, {"cases": cases, "fixed_sequence_comparisons": family_count})


def _special(context: StressContext) -> CheckOutcome:
    from ..special_functions import bernoulli_packet_integer, q_racah_lattice, q_racah_scalar

    def poch(value, q, length):
        result = F(1)
        for index in range(length):
            result *= 1 - value * q**index
        return result

    a, b, g, d, q = F(2, 7), F(3, 5), F(4, 9), F(5, 8), F(2, 3)
    cases = []
    for _ in range(context.parameters["samples"]):
        n, y = context.rng.randrange(7), context.rng.randrange(7)
        expected = F(0)
        for k in range(n + 1):
            numerator = (
                poch(q ** (-n), q, k)
                * poch(a * b * q ** (n + 1), q, k)
                * poch(q ** (-y), q, k)
                * poch(g * d * q ** (y + 1), q, k)
            )
            denominator = poch(a * q, q, k) * poch(b * d * q, q, k) * poch(g * q, q, k) * poch(q, q, k)
            expected += numerator * q**k / denominator
        actual = q_racah_scalar(n, y, a, b, g, d, q)
        if actual != expected:
            return CheckOutcome(False, {"n": n, "y": y}, "full terminating hypergeometric oracle mismatch")
        cases.append({"n": n, "y": y, "value": actual})
    q, g = F(1, 2), F(4)
    lattice = [q_racah_lattice(y, g, 1, q) for y in range(2)]
    degenerate = [[q_racah_scalar(n, y, q, q, g, 1, q) for y in range(2)] for n in range(2)]
    if lattice != [3, 3] or degenerate != [[1, 1], [1, 1]]:
        return CheckOutcome(
            False, {"lattice": lattice, "table": degenerate}, "known non-orthogonality witness changed"
        )
    polynomials = (
        lambda x: F(1),
        lambda x: x - F(1, 2),
        lambda x: x * x - x + F(1, 6),
        lambda x: x**3 - F(3, 2) * x * x + F(1, 2) * x,
        lambda x: x**4 - 2 * x**3 + x * x - F(1, 30),
    )
    bernoulli_count = 0
    for degree, polynomial in enumerate(polynomials):
        for modulus in (1, 2, 5):
            for residue in range(modulus):
                translation = F(2, 3)
                expected = F(modulus) ** (degree - 1) * polynomial((translation + residue) / modulus)
                if bernoulli_packet_integer(degree, translation, modulus, residue) != expected:
                    return CheckOutcome(
                        False,
                        {"degree": degree, "modulus": modulus, "residue": residue},
                        "Bernoulli polynomial oracle mismatch",
                    )
                bernoulli_count += 1
    return CheckOutcome(
        True,
        {
            "q_racah_cases": cases,
            "q_racah_parameters": [a, b, F(4, 9), d, F(2, 3)],
            "non_orthogonal_lattice": lattice,
            "rank_one_table": degenerate,
            "bernoulli_polynomial_comparisons": bernoulli_count,
        },
    )


def _coordinates(context: StressContext) -> CheckOutcome:
    from ..coordinates.exact import (
        as_matrix,
        determinant,
        inverse_matrix,
        matrix_vector_multiply,
        solve_linear_system,
    )
    from ..exceptions import ValidationError

    cases = []
    for _ in range(context.parameters["samples"]):
        size = context.rng.randrange(1, 5)
        rows = [
            [F(context.rng.randrange(-9, 10), context.rng.randrange(1, 6)) for _ in range(size)]
            for _ in range(size)
        ]
        matrix = as_matrix(rows)
        expected = sum(
            (
                (-1) ** sum(p[i] > p[j] for i in range(size) for j in range(i + 1, size))
                * _product(matrix[i][p[i]] for i in range(size))
                for p in permutations(range(size))
            ),
            F(0),
        )
        if determinant(matrix) != expected:
            return CheckOutcome(False, {"matrix": rows}, "Leibniz determinant oracle mismatch")
        if expected:
            inverse = inverse_matrix(matrix)
            # Scalar summation is distinct from the production matrix multiply.
            if any(
                sum((matrix[i][k] * inverse[k][j] for k in range(size)), F(0)) != int(i == j)
                for i, j in product(range(size), repeat=2)
            ):
                return CheckOutcome(False, {"matrix": rows}, "inverse scalar-product witness failed")
        cases.append({"matrix": rows, "determinant": expected})
    singular = as_matrix(((1, 2), (2, 4)))
    try:
        inverse_matrix(singular)
    except ValidationError:
        pass
    else:
        return CheckOutcome(False, {}, "singular matrix was invertible")
    inconsistent = solve_linear_system(singular, (1, 3))
    consistent = solve_linear_system(singular, (1, 2))
    accepted = (
        not inconsistent.consistent
        and consistent.consistent
        and matrix_vector_multiply(singular, consistent.solution) == (1, 2)
    )
    return CheckOutcome(
        accepted,
        {"cases": cases, "singular_matrix": singular, "consistent_rhs": [1, 2], "inconsistent_rhs": [1, 3]},
    )


def _product(values):
    answer = F(1)
    for value in values:
        answer *= value
    return answer


def _partitions(size, prefix=()):
    if len(prefix) == size:
        yield prefix
    else:
        for label in range(max(prefix, default=-1) + 2):
            yield from _partitions(size, (*prefix, label))


def _finite_joins(context: StressContext) -> CheckOutcome:
    from ..core.semilattice import FiniteJoinSemilattice

    cases = []
    partitions = tuple(_partitions(4))
    for name, operation in (("chain", max), ("boolean", lambda a, b: a | b)):
        table = tuple(tuple(operation(a, b) for b in range(4)) for a in range(4))
        lattice = FiniteJoinSemilattice(table)
        relations = [((a, b),) for a, b in product(range(4), repeat=2)]
        relations.extend(
            tuple((context.rng.randrange(4), context.rng.randrange(4)) for _ in range(2))
            for _ in range(context.parameters["samples"])
        )
        for requested in relations:
            admissible = [
                labels
                for labels in partitions
                if all(labels[a] == labels[b] for a, b in requested)
                and all(
                    labels[table[a][c]] == labels[table[b][c]]
                    for a, b, c in product(range(4), repeat=3)
                    if labels[a] == labels[b]
                )
            ]
            expected = tuple(
                min(b for b in range(4) if all(labels[a] == labels[b] for labels in admissible))
                for a in range(4)
            )
            actual = lattice.generated_congruence(requested)
            if actual.partition != expected:
                return CheckOutcome(
                    False,
                    {"table": table, "relations": requested},
                    "exhaustive-partition least congruence mismatch",
                )
            actual.certificate().require()
            cases.append({"table": name, "relations": requested, "partition": actual.partition})
    return CheckOutcome(True, {"cases": cases, "candidate_partitions_per_case": len(partitions)})


def _measurements(locations, amplitudes, period, degree=3, bound=5):
    """Integrate constant segments, independently of the jump moment formula."""
    import numpy as np

    ordered = sorted(zip(locations, amplitudes, strict=True))
    edges = [0.0, *(float(location) for location, _ in ordered), float(period)]
    levels = [0.0]
    for _, amplitude in ordered:
        levels.append(levels[-1] + float(amplitude))
    widths = np.diff(edges)
    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    levels = np.asarray(levels)
    levels -= float(np.dot(widths, levels) / period)
    # Integral of exp(-i*omega*x) over a constant segment, written with sinc
    # to avoid subtracting nearly equal endpoint exponentials at close edges.
    return {
        n: complex(
            np.sum(
                levels
                * widths
                / period
                * np.exp(-2j * np.pi * n * centers / period)
                * np.sinc(n * widths / period)
            )
        )
        for n in range(degree + 1, degree + 1 + 2 * bound)
    }


def _signal(context: StressContext) -> CheckOutcome:
    import numpy as np

    from ..exceptions import AssumptionError, ValidationError
    from ..signals.fourier_recovery import PeriodicJumpPacket, recover_fourier_jumps

    cases = []
    for _ in range(context.parameters["samples"]):
        period = context.rng.choice((1.0, 5.0, 2 * np.pi))
        shift = context.rng.random() * 0.08
        locations = np.array([0.11 + shift, 0.37 + shift, 0.72 + shift]) * period
        amplitudes = np.array([2.0, -0.75, -1.25])
        measured = _measurements(locations, amplitudes, period)
        result = recover_fourier_jumps(
            measured, background_degree=3, maximum_jumps=5, known_order=3, period=period
        )
        error = max(
            float(np.max(np.abs(result.packet.locations - locations))),
            float(np.max(np.abs(result.packet.amplitudes - amplitudes))),
        )
        if not result.fit_certificate.valid or error > 1e-9 or result.resolution_status != "conditional":
            return CheckOutcome(
                False,
                {"period": period, "locations": locations.tolist(), "parameter_error": error},
                "resolved model failed",
            )
        cases.append(
            {
                "period": period,
                "locations": locations.tolist(),
                "amplitudes": amplitudes.tolist(),
                "parameter_error": error,
            }
        )
    locations, amplitudes = [1.0, 1.0 + 5e-8, 3.84], [2.0, -0.7, -1.3]
    measured = _measurements(locations, amplitudes, 5.0)
    candidate = recover_fourier_jumps(measured, background_degree=3, maximum_jumps=5, period=5.0)
    true_packet = PeriodicJumpPacket(locations, amplitudes, period=5.0)
    midpoint = 1.0 + 2.5e-8
    point_error = abs(float(true_packet.evaluate(midpoint) - candidate.packet.evaluate(midpoint)))
    known_rejected = False
    try:
        recover_fourier_jumps(measured, background_degree=3, maximum_jumps=5, known_order=3, period=5.0)
    except AssumptionError:
        known_rejected = True
    guarded = False
    try:
        candidate.require_recovery()
    except ValidationError:
        guarded = True
    bad = dict(measured)
    bad[4] = float("nan")
    nonfinite_rejected = False
    try:
        recover_fourier_jumps(bad, background_degree=3, maximum_jumps=5, period=5.0)
    except ValidationError:
        nonfinite_rejected = True
    accepted = (
        candidate.order == 2
        and candidate.fit_certificate.valid
        and not candidate.certificate.valid
        and candidate.resolution_status == "candidate_only"
        and point_error > 0.69
        and guarded
        and known_rejected
        and nonfinite_rejected
    )
    return CheckOutcome(
        accepted,
        {
            "resolved_cases": cases,
            "close_edge_locations": locations,
            "close_edge_amplitudes": amplitudes,
            "candidate_order": candidate.order,
            "midpoint": midpoint,
            "pointwise_error": point_error,
            "fit_residuals": dict(candidate.fit_certificate.residuals),
            "known_order_rejected": known_rejected,
            "nonfinite_input_rejected": nonfinite_rejected,
        },
    )


def _quantum(context: StressContext) -> CheckOutcome:
    import numpy as np

    from ..dynamics import RationalMemoryRealization
    from ..quantum import QuantumFeshbachReduction, constraint_closure, spectral_dimension
    from ..quantum.dirac import dirac_propagator, dirac_symbol

    cases = []
    for _ in range(context.parameters["samples"]):
        momentum = [context.rng.uniform(-3, 3) for _ in range(3)]
        mass, time = context.rng.uniform(0, 2), context.rng.uniform(-1, 1)
        hamiltonian = dirac_symbol(momentum, mass)
        eigenvalues, vectors = np.linalg.eigh(hamiltonian)
        expected = (vectors * np.exp(-1j * time * eigenvalues)) @ vectors.conj().T
        error = float(np.max(np.abs(dirac_propagator(momentum, time, mass) - expected)))
        if error > 1e-10:
            return CheckOutcome(
                False,
                {"momentum": momentum, "mass": mass, "time": time, "error": error},
                "independent spectral evolution mismatch",
            )
        cases.append({"momentum": momentum, "mass": mass, "time": time, "error": error})
    reduction = QuantumFeshbachReduction(np.array([[0.0, 1.0], [1.0, 0.0]]), (0,))
    kernel = reduction.memory_kernel()
    memory = RationalMemoryRealization(kernel.output, kernel.hidden, kernel.input)
    memory_errors = []
    for time in (0.0, 0.4, 1.7):
        actual = memory.retained_trajectory(
            -1j * reduction.reduction.retained_block, np.array([1.0, 0.0]), time
        )
        memory_errors.append(abs(complex(actual[0]) - np.cos(time)))
    x, z = np.array([[0.0, 1.0], [1.0, 0.0]]), np.diag([1.0, -1.0])
    nonclosure = [
        not constraint_closure((scale * x, scale * z)).certificate.valid for scale in (1e-100, 1.0, 1e100)
    ]
    dimension = spectral_dimension(np.eye(3), 1e6)
    accepted = (
        max(memory_errors) < 1e-10
        and kernel.at(0)[0, 0] == -1
        and all(nonclosure)
        and abs(dimension - 2e6) < 1e-8
    )
    return CheckOutcome(
        bool(accepted),
        {
            "dirac_cases": cases,
            "memory_errors": memory_errors,
            "constraint_nonclosure": nonclosure,
            "large_time_spectral_dimension": dimension,
        },
    )


BUILTIN_SUITES = (
    "core",
    "arithmetic",
    "special_functions",
    "coordinates",
    "finite_joins",
    "signals",
    "quantum",
)


def builtin_checks(*, suites=None, samples: int = 8) -> tuple[StressCheck, ...]:
    """Build bounded base-dependency suites; samples is between 1 and 128."""
    samples = _integer(samples, "samples", 128)
    if not samples:
        raise ValueError("samples must be positive")
    chosen = BUILTIN_SUITES if suites is None else tuple(suites)
    if not chosen or any(name not in BUILTIN_SUITES for name in chosen):
        raise ValueError(f"suites must be nonempty and chosen from {BUILTIN_SUITES}")
    if len(set(chosen)) != len(chosen):
        raise ValueError("duplicate suites are not allowed")
    specifications = {
        "core": (
            _core,
            Exactness.NUMERICAL,
            {
                "random_cases": samples,
                "dimensions": [2, 6],
                "integer_entries": [-9, 9],
                "power": 2,
                "negative_controls": 2,
            },
            1e-10,
        ),
        "arithmetic": (
            _arithmetic,
            Exactness.EXACT,
            {
                "random_cases": samples,
                "integer_input_bits": 120,
                "recurrence_orders": [1, 5],
                "terms": 24,
                "transform_terms": 12,
                "hurwitz_degree": 12,
                "matrix_power": 7,
                "moduli": [7, 2**63 + 159, 10**100 + 7],
                "kbonacci_orders": [2, 8],
                "kbonacci_indices": [0, 49],
                "reciprocal_spacings": [1, 5],
                "reciprocal_indices": [0, 6],
            },
            None,
        ),
        "special_functions": (
            _special,
            Exactness.EXACT,
            {
                "random_q_racah_cases": samples,
                "degrees": [0, 6],
                "sites": [0, 6],
                "bernoulli_degrees": [0, 4],
                "bernoulli_moduli": [1, 2, 5],
                "degenerate_lattice_witness": "C11",
            },
            None,
        ),
        "coordinates": (
            _coordinates,
            Exactness.EXACT,
            {
                "random_cases": samples,
                "dimensions": [1, 4],
                "numerators": [-9, 9],
                "denominators": [1, 5],
                "singular_consistency_controls": 3,
            },
            None,
        ),
        "finite_joins": (
            _finite_joins,
            Exactness.EXACT,
            {
                "tables": ["4-element chain", "4-element Boolean join"],
                "all_single_relations": 16,
                "sampled_double_relations_per_table": samples,
                "partitions_enumerated_per_case": 15,
            },
            None,
        ),
        "signals": (
            _signal,
            Exactness.NUMERICAL,
            {
                "resolved_cases": samples,
                "periods": [1.0, 5.0, "2pi"],
                "true_jumps": 3,
                "high_modes": [4, 13],
                "maximum_jumps": 5,
                "close_separation": 5e-8,
                "nonfinite_input_controls": 1,
            },
            1e-9,
        ),
        "quantum": (
            _quantum,
            Exactness.NUMERICAL,
            {
                "random_dirac_cases": samples,
                "spinor_dimension": 4,
                "momenta": [-3.0, 3.0],
                "mass": [0.0, 2.0],
                "times": [-1.0, 1.0],
                "memory_times": [0.0, 0.4, 1.7],
                "constraint_scales": [1e-100, 1.0, 1e100],
                "spectral_time": 1e6,
            },
            1e-8,
        ),
    }
    result = []
    for suite in chosen:
        function, exactness, bounds, tolerance = specifications[suite]
        result.append(
            StressCheck(
                f"{suite}.independent-controls-v1",
                suite,
                function,
                exactness,
                bounds,
                parameters={"samples": samples},
                tolerance=tolerance,
                assumptions=("finite supplied models and the stated arithmetic/backend contracts",),
            )
        )
    return tuple(result)
