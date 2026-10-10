"""Bounded independent checks for additional finite paper machinery.

These records concern supplied finite models, not completion of the source
papers or unrestricted mathematical claims. Two checks need the SymPy exact
extra; absence is an explicit skip, which cannot qualify a run as passed.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction as F
from itertools import permutations, product
from math import factorial, lcm

from ..certificates import Exactness
from .runner import CheckOutcome, SkipCheck, StressCheck, StressContext


def _sympy():
    try:
        import sympy
    except ImportError as exc:
        raise SkipCheck("this rational-field paper check requires the installed SymPy exact extra") from exc
    return sympy


def _rational_rows(matrix):
    return tuple(
        tuple(F(int(matrix[i, j].p), int(matrix[i, j].q)) for j in range(matrix.cols))
        for i in range(matrix.rows)
    )


def _brute_kernel_dimension(matrix, columns, prime):
    count = sum(
        all(sum(int(a) * b for a, b in zip(row, vector, strict=True)) % prime == 0 for row in matrix)
        for vector in product(range(prime), repeat=columns)
    )
    dimension = 0
    while count > 1:
        if count % prime:
            raise ArithmeticError("kernel cardinality was not a prime power")
        count //= prime
        dimension += 1
    return dimension


def _cones(context: StressContext):
    import numpy as np

    from ..core.paper_machinery import finite_mapping_cone
    from ..deformation.complexes import FiniteCochainComplex
    from ..exceptions import ValidationError

    cases = []
    for _ in range(context.parameters["samples"]):
        prime = context.rng.choice((2, 3, 5))
        differential = [[context.rng.randrange(prime) for _ in range(2)] for _ in range(2)]
        complex_ = FiniteCochainComplex({0: 2, 1: 2}, {0: differential}, prime=prime)
        cone = finite_mapping_cone(
            complex_, complex_, {0: np.eye(2, dtype=object), 1: np.eye(2, dtype=object)}
        )
        dimensions = {}
        for degree, ambient in cone.dimensions.items():
            outgoing, incoming = cone.differential(degree).tolist(), cone.differential(degree - 1).tolist()
            kernel = _brute_kernel_dimension(outgoing, ambient, prime)
            previous = cone.dimensions.get(degree - 1, 0)
            rank = previous - _brute_kernel_dimension(incoming, previous, prime)
            dimensions[degree] = kernel - rank
            if dimensions[degree] != 0 or cone.cohomology(degree).cohomology_dimension != 0:
                return CheckOutcome(
                    False,
                    {"prime": prime, "differential": differential},
                    "identity cone is not acyclic by exhaustive vector enumeration",
                )
        lower = cone.differential(-1).tolist()[2:]
        if lower != [[(-value) % prime for value in row] for row in differential]:
            return CheckOutcome(
                False,
                {"prime": prime, "differential": differential},
                "mapping-cone source differential has the wrong sign",
            )
        cone.certificate().require()
        cases.append(
            {
                "prime": prime,
                "differential": differential,
                "cohomology_by_degree": {str(k): v for k, v in dimensions.items()},
            }
        )
    basic = FiniteCochainComplex({0: 1, 1: 1}, {0: [[1]]}, prime=7)
    rejected = False
    try:
        finite_mapping_cone(basic, basic, {0: [[1]], 1: [[2]]})
    except ValidationError:
        rejected = True
    return CheckOutcome(rejected, {"cases": cases, "noncommuting_square_rejected": rejected})


def _rational_packets(context: StressContext):
    sp = _sympy()
    from ..core.paper_machinery import rational_primary_packets, rational_spectral_projectors
    from ..exceptions import ValidationError

    x = sp.Symbol("x")
    diagonal = sp.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 2]])
    cases = []
    for _ in range(context.parameters["samples"]):
        a, b, c = [sp.Rational(context.rng.randrange(-3, 4), context.rng.randrange(1, 4)) for _ in range(3)]
        similarity = sp.Matrix([[1, a, b], [0, 1, c], [0, 0, 1]])
        operator = similarity * diagonal * similarity.inv()
        packets = rational_spectral_projectors(operator, (x**2, x - 2))
        expected = tuple(
            similarity * projector * similarity.inv() for projector in (sp.diag(1, 1, 0), sp.diag(0, 0, 1))
        )
        if tuple(packets.projectors) != expected:
            return CheckOutcome(
                False,
                {"similarity": _rational_rows(similarity)},
                "Jordan primary subspaces disagree with direct conjugation",
            )
        primary = rational_primary_packets(operator)
        if {_rational_rows(p) for p in primary.projectors} != {_rational_rows(p) for p in expected}:
            return CheckOutcome(
                False,
                {"operator": _rational_rows(operator)},
                "primary factor discovery changed the known subspaces",
            )
        packets.certificate().require()
        cases.append(
            {
                "similarity": _rational_rows(similarity),
                "operator": _rational_rows(operator),
                "projectors": tuple(_rational_rows(p) for p in packets.projectors),
            }
        )
    rejected = False
    try:
        rational_spectral_projectors(diagonal, (x, x * (x - 2)))
    except ValidationError:
        rejected = True
    return CheckOutcome(rejected, {"cases": cases, "overlapping_factors_rejected": rejected})


def _compiler(context: StressContext):
    from ..algorithms.compiler import TreeNode, compile_tree_budget, compile_tree_lagrangian
    from ..algorithms.compiler_audit import audit_tree_budget, audit_tree_lagrangian, enumerate_tree_prunings
    from ..exceptions import ValidationError

    cases = []
    for index in range(context.parameters["samples"]):
        recorded = []

        def subtree(depth, name, recorded=recorded):
            data = tuple(F(context.rng.randrange(12), 8) for _ in range(3))
            recorded.append({"name": name, "dyadic_cost_error_overhead": data})
            children = () if not depth else tuple(subtree(depth - 1, name + str(j)) for j in range(2))
            return TreeNode(name, *(float(value) for value in data), children)

        root = subtree(2, f"r{index}")
        space = enumerate_tree_prunings(root, max_prunings=5, max_nodes=7)
        for penalty in (F(0), F(1, 4), F(1), F(4)):
            plan = compile_tree_lagrangian(root, float(penalty))
            audit = audit_tree_lagrangian(root, plan, float(penalty), max_prunings=5, max_nodes=7)
            if not audit.certificate.valid:
                return CheckOutcome(
                    False,
                    {"tree": recorded, "penalty": penalty},
                    "compiler differs from exhaustive exact pruning oracle",
                )
        for budget in (0, 1, 3, 6):
            plan = compile_tree_budget(root, budget, error_unit=0.25)
            if not audit_tree_budget(
                root, plan, budget, error_unit=0.25, max_prunings=5, max_nodes=7
            ).certificate.valid:
                return CheckOutcome(
                    False,
                    {"tree": recorded, "budget": budget},
                    "budget compiler differs from exhaustive exact oracle",
                )
        cases.append({"tree": recorded, "complete_prunings": len(space)})
    branch = TreeNode("a", 2, 4, 0, (TreeNode("a1", 3, 1), TreeNode("a2", 1, 0)))
    root = TreeNode("r", 1, 6, 0, (branch, TreeNode("b", 1, 0)))
    middle = compile_tree_budget(root, 4)
    unsupported = (
        middle is not None
        and middle.leaves == ("a", "b")
        and all(
            compile_tree_lagrangian(root, float(penalty)).leaves != middle.leaves
            for penalty in (F(0), F(1, 2), F(4, 5), F(1), F(2), F(100))
        )
    )
    forged_rejected = (
        middle is not None
        and not audit_tree_budget(root, replace(middle, cost=middle.cost + 1), 4).certificate.valid
    )
    boundary = TreeNode("boundary", 0, float(F(2**52 + 1, 2**52)))
    boundary_rejected = (
        compile_tree_budget(boundary, 1) is None and audit_tree_budget(boundary, None, 1).certificate.valid
    )
    cap_rejected = False
    try:
        enumerate_tree_prunings(root, max_prunings=2)
    except ValidationError:
        cap_rejected = True
    return CheckOutcome(
        bool(unsupported and forged_rejected and boundary_rejected and cap_rejected),
        {
            "cases": cases,
            "unsupported_budget_optimum": unsupported,
            "forged_totals_rejected": forged_rejected,
            "strict_integer_boundary": boundary_rejected,
            "partial_enumeration_rejected": cap_rejected,
        },
    )


def _hermite_coefficients(n):
    coefficients = [F(0)] * (n + 1)
    for k in range(n // 2 + 1):
        coefficients[n - 2 * k] = F((-1) ** k * factorial(n), 2**k * factorial(k) * factorial(n - 2 * k))
    return tuple(coefficients)


def _trim(values):
    values = tuple(values)
    while len(values) > 1 and values[-1] == 0:
        values = values[:-1]
    return values


def _hermite(context: StressContext):
    from ..dynamics.hermite import (
        gaussian_moment,
        gaussian_polynomial_inner_product,
        gaussian_transfer_polynomial,
        ornstein_uhlenbeck_generator,
    )
    from ..exceptions import ValidationError

    for order in range(41):
        expected = 0 if order % 2 else 1
        if not order % 2:
            for odd in range(1, order, 2):
                expected *= odd
        if gaussian_moment(order) != expected:
            return CheckOutcome(
                False, {"moment_order": order}, "Gaussian moments disagree with odd-factor products"
            )
    for n in range(11):
        polynomial = _hermite_coefficients(n)
        for m in range(11):
            if gaussian_polynomial_inner_product(polynomial, _hermite_coefficients(m)) != (
                factorial(n) if n == m else 0
            ):
                return CheckOutcome(False, {"n": n, "m": m}, "Hermite Gaussian orthogonality failed")
        if ornstein_uhlenbeck_generator(polynomial) != _trim(-n * c for c in polynomial):
            return CheckOutcome(False, {"n": n}, "Hermite generator eigenvalue failed")
        for correlation in (F(0), F(1, 5), F(1, 2), F(1)):
            if gaussian_transfer_polynomial(polynomial, correlation) != _trim(
                correlation**n * c for c in polynomial
            ):
                return CheckOutcome(
                    False, {"n": n, "correlation": correlation}, "Hermite transfer eigenvalue failed"
                )
    cases = []
    for _ in range(context.parameters["samples"]):
        polynomial = tuple(F(context.rng.randrange(-7, 8), context.rng.randrange(1, 6)) for _ in range(9))
        first, second = F(context.rng.randrange(6), 5), F(context.rng.randrange(6), 5)
        composed = gaussian_transfer_polynomial(gaussian_transfer_polynomial(polynomial, first), second)
        if composed != gaussian_transfer_polynomial(polynomial, first * second):
            return CheckOutcome(
                False,
                {"polynomial": polynomial, "correlations": [first, second]},
                "finite Gaussian semigroup identity failed",
            )
        cases.append({"polynomial": polynomial, "correlations": [first, second]})
    cap_rejected = False
    try:
        gaussian_transfer_polynomial((1, 2, 3), F(1, 2), max_degree=1)
    except ValidationError:
        cap_rejected = True
    return CheckOutcome(
        cap_rejected,
        {
            "cases": cases,
            "moment_orders": [0, 40],
            "Hermite_orthogonality_pairs": 121,
            "degree_truncation_rejected": cap_rejected,
        },
    )


def _lifting_stencils(context: StressContext):
    from ..analytic.appell import congruential_inverse
    from ..arithmetic.packet_differences import certify_packet_difference, rational_mixed_difference
    from ..arithmetic.polynomial_lifting import IntegerPolynomialSystem, SparseIntegerPolynomial

    system = IntegerPolynomialSystem((SparseIntegerPolynomial(2, {(2, 0): 1, (0, 1): -1}),))
    cases = []
    for _ in range(context.parameters["samples"]):
        prime, depth = context.rng.choice((2, 3, 5)), context.rng.choice((1, 2))
        scale = prime**depth
        x = context.rng.randrange(scale)
        root = (x, x * x % scale)
        fiber = system.lift_fiber(root, prime=prime, depth=depth)
        # Direct polynomial substitution, independent of the Jacobian solver.
        expected = set()
        for a, b in product(range(prime), repeat=2):
            point = (root[0] + scale * a, root[1] + scale * b)
            if (point[0] ** 2 - point[1]) % (scale * prime) == 0:
                expected.add(point)
        if set(fiber.lifts()) != expected or fiber.count != len(expected):
            return CheckOutcome(
                False,
                {"prime": prime, "depth": depth, "root": root},
                "one-digit polynomial fibers disagree with direct substitution",
            )
        cases.append({"prime": prime, "depth": depth, "root": root, "lifts": sorted(expected)})
    first = IntegerPolynomialSystem((SparseIntegerPolynomial(1, {(1,): 1}),))
    second = IntegerPolynomialSystem((SparseIntegerPolynomial(1, {(1,): 1, (0,): -2}),))
    joint = first.stack(second).lift_fiber((0,), prime=2, depth=1)
    incompatible = (
        first.lift_fiber((0,), prime=2, depth=1).count == second.lift_fiber((0,), prime=2, depth=1).count == 1
        and not joint.solvable
    )
    singular = IntegerPolynomialSystem((SparseIntegerPolynomial(1, {(2,): 1, (0,): -5}),))
    first_lifts = singular.lift_fiber((1,), prime=2, depth=1).lifts()
    later_obstructed = len(first_lifts) == 2 and all(
        not singular.lift_fiber(root, prime=2, depth=2).solvable for root in first_lifts
    )
    coefficients = tuple(F((-1) ** n * (n**3 + 1), 2 * n + 1) for n in range(20))
    stencil_cases = []
    for _ in range(context.parameters["samples"]):
        row, order = context.rng.randrange(3), context.rng.randrange(5)
        steps = tuple(context.rng.randrange(4) for _ in range(order))
        expected = sum(
            (
                (-1) ** (order - sum(bits))
                * coefficients[row + sum(bit * step for bit, step in zip(bits, steps, strict=True))]
                for bits in product((0, 1), repeat=order)
            ),
            F(0),
        )
        actual = rational_mixed_difference(coefficients, row, steps)
        if actual != expected:
            return CheckOutcome(
                False, {"row": row, "steps": steps}, "mixed stencil differs from independent subset expansion"
            )
        stencil_cases.append({"row": row, "steps": steps, "difference": actual})
    archive = congruential_inverse(4, 1, 8)
    passing = certify_packet_difference(archive, 0, (2, 2), prime=2, valuation_bound=5)
    failing = certify_packet_difference(archive, 0, (2, 2), prime=2, valuation_bound=6)
    return CheckOutcome(
        bool(
            incompatible and later_obstructed and passing.certificate.valid and not failing.certificate.valid
        ),
        {
            "lift_cases": cases,
            "stencil_cases": stencil_cases,
            "incompatible_shared_digit": incompatible,
            "singular_later_obstruction": later_obstructed,
            "valuation_5_passes": passing.certificate.valid,
            "valuation_6_rejected": not failing.certificate.valid,
        },
    )


def _poly_multiply(left, right):
    result = [F(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def _det_identity_minus_u(matrix):
    size = len(matrix)
    answer = [F(0)] * (size + 1)
    for order in permutations(range(size)):
        sign = (-1) ** sum(order[i] > order[j] for i in range(size) for j in range(i + 1, size))
        term = [F(sign)]
        for i, j in enumerate(order):
            term = _poly_multiply(term, [F(i == j), -matrix[i][j]])
        for i, value in enumerate(term):
            answer[i] += value
    return _trim(answer)


def _tame(context: StressContext):
    from ..coordinates.exact import as_matrix, matrix_rank
    from ..langlands.local.tame import FrobeniusOrbit
    from ..langlands.local.tame_tensor import tame_tensor_character_packets

    cases = []
    for _ in range(context.parameters["samples"]):
        q, n, m = context.rng.choice((2, 3, 4)), context.rng.randrange(1, 4), context.rng.randrange(1, 4)
        left, right = FrobeniusOrbit(q, n, 1), FrobeniusOrbit(q, m, -1)
        result = tame_tensor_character_packets(left, right, left_scalar=2, right_scalar=F(1, 2))
        period = lcm(n, m)
        modulus = q**period - 1
        invariant = [
            i * m + j
            for i, j in product(range(n), range(m))
            if (
                left.exponent * q**i * (modulus // left.modulus)
                + right.exponent * q**j * (modulus // right.modulus)
            )
            % modulus
            == 0
        ]
        tensor = [[F(0)] * (n * m) for _ in range(n * m)]
        for i, j in product(range(n), range(m)):
            tensor[((i + 1) % n) * m + (j + 1) % m][i * m + j] = (F(2) if i == n - 1 else F(1)) * (
                F(1, 2) if j == m - 1 else F(1)
            )
        restricted = [[tensor[i][j] for j in invariant] for i in invariant]
        polynomial = _det_identity_minus_u(restricted)
        fixed = len(invariant) - matrix_rank(
            as_matrix(
                [[entry - int(i == j) for j, entry in enumerate(row)] for i, row in enumerate(restricted)]
            )
        )
        cycles = [pair for packet in result.packets for pair in packet.cycle_pairs]
        accepted = (
            len(cycles) == n * m
            and set(cycles) == set(product(range(n), range(m)))
            and result.inertia_fixed_dimension == len(invariant)
            and result.weil_fixed_dimension == fixed
            and result.local_polynomial_coefficients == polynomial
        )
        if not accepted:
            return CheckOutcome(
                False,
                {"q": q, "degrees": [n, m], "matrix_polynomial": polynomial},
                "tame tensor cycle invariants disagree with the full rational tensor matrix",
            )
        result.certificate.require()
        cases.append(
            {
                "q": q,
                "degrees": [n, m],
                "invariant_tensor_indices": invariant,
                "restricted_tensor": restricted,
                "local_polynomial": polynomial,
                "Weil_fixed_dimension": fixed,
            }
        )
    known = tame_tensor_character_packets(FrobeniusOrbit(2, 2, 1), FrobeniusOrbit(2, 2, 2))
    witness = (
        known.inertia_fixed_dimension == 2
        and known.weil_fixed_dimension == 1
        and known.local_polynomial_coefficients == (1, 0, -1)
    )
    return CheckOutcome(witness, {"cases": cases, "reducible_interaction_witness": witness})


def _direct_collatz(seed, length):
    states, values = [seed], []
    for _ in range(length):
        current, valuation = 3 * seed + 1, 0
        while current % 2 == 0:
            current //= 2
            valuation += 1
        seed = current
        values.append(valuation)
        states.append(seed)
    return tuple(values), tuple(states)


def _collatz(context: StressContext):
    from ..arithmetic.collatz import CollatzWord, first_odd_descent

    comparisons = 0
    for length in range(4):
        for valuations in product(range(1, 5), repeat=length):
            word = CollatzWord(valuations)
            for seed in range(1, 256, 2):
                observed, states = _direct_collatz(seed, length)
                realized = observed == valuations
                if word.realizes(seed) != realized or (
                    realized
                    and (
                        word.apply(seed) != states[-1]
                        or word.orbit(seed) != states
                        or word.descends(seed) != (states[-1] < seed)
                    )
                ):
                    return CheckOutcome(
                        False,
                        {"seed": seed, "valuations": valuations},
                        "Collatz cylinder disagrees with direct repeated division",
                    )
                comparisons += 1
    cases = []
    for _ in range(context.parameters["samples"]):
        seed = 2 * context.rng.randrange(2**31) + 1
        valuations, states = _direct_collatz(seed, 6)
        word = CollatzWord(valuations)
        if word.orbit(seed) != states or word.affine_value(seed) != states[-1]:
            return CheckOutcome(
                False, {"seed": seed, "valuations": valuations}, "six-step exact affine replay failed"
            )
        cases.append({"seed": seed, "valuations": valuations, "states": states})
    descents = []
    for seed in range(1, 256, 2):
        record = first_odd_descent(seed, max_steps=128)
        observed, states = _direct_collatz(seed, record.word.length)
        if observed != record.word.valuations or states != record.states:
            return CheckOutcome(False, {"seed": seed}, "first-descent record differs from direct iteration")
        if record.status == "descent" and (states[-1] >= seed or any(x < seed for x in states[:-1])):
            return CheckOutcome(False, {"seed": seed}, "reported first descent was not first")
        descents.append({"seed": seed, "status": record.status, "steps": record.word.length})
    bounded = first_odd_descent(27, max_steps=0)
    rejected = bounded.status == "inconclusive" and not bounded.certificate.valid
    return CheckOutcome(
        rejected,
        {
            "word_seed_comparisons": comparisons,
            "six_step_cases": cases,
            "first_descent_cases": descents,
            "zero_budget_inconclusive": rejected,
        },
        "finite cylinders and replay only; no universal descent conclusion",
    )


def _signal_foundations(context: StressContext):
    sp = _sympy()
    from ..exceptions import ValidationError
    from ..signals.foundations import (
        compression_closure,
        delay_closure,
        identify_data_span,
        minimal_predictive_completion,
        projected_memory,
    )

    summary = sp.Matrix([[1, 0]])
    for entries in product((-1, 0, 1), repeat=4):
        matrix = sp.Matrix(2, 2, entries)
        result = minimal_predictive_completion(summary, {"L": matrix})
        expected = 1 if entries[1] == 0 else 2
        if (
            result.require_complete().basis.rows != expected
            or result.basis * matrix != result.reduced_operators["L"] * result.basis
        ):
            return CheckOutcome(
                False,
                {"matrix": _rational_rows(matrix)},
                "observable completion disagrees with the direct two-dimensional row-space criterion",
            )
    matrix = sp.Matrix([[F(1, 2), 1, 0], [F(1, 4), F(1, 3), 0], [0, 0, F(5, 6)]])
    projection = sp.diag(1, 0, 0)
    compressed = compression_closure(matrix, projection)
    compressed.certificate.diagnostics["closed"] = True
    guarded = False
    try:
        compressed.require_closed()
    except ValidationError:
        guarded = True
    delay = delay_closure(matrix, [[1, 0, 0]], delay=2)
    expected_transition = sp.Matrix([[0, 1], [F(1, 12), F(5, 6)]])
    fit = identify_data_span([[1], [0]], [[0], [1]])
    false_global = fit.fit_consistent and not fit.span_closed and fit.global_coverage == "not_established"
    cases = []
    for _ in range(context.parameters["samples"]):
        operator = sp.Matrix(
            2, 2, [F(context.rng.randrange(-4, 5), context.rng.randrange(1, 5)) for _ in range(4)]
        )
        initial = sp.Matrix([2, -3])
        p = sp.diag(1, 0)
        memory = projected_memory(operator, p, horizon=5)
        state, history = initial, []
        for _time in range(6):
            history.append(p * state)
            if memory.next_state(history, (sp.eye(2) - p) * initial) != p * operator * state:
                return CheckOutcome(
                    False,
                    {"operator": _rational_rows(operator)},
                    "projected memory disagrees with full-state rational iteration",
                )
            state = operator * state
        cases.append({"operator": _rational_rows(operator), "initial": [2, -3], "horizon": 5})
    accepted = (
        guarded
        and not compressed.closed
        and compressed.composition_defect == sp.diag(F(1, 4), 0, 0)
        and delay.require_transition() == expected_transition
        and false_global
    )
    return CheckOutcome(
        bool(accepted),
        {
            "all_small_transition_matrices": 81,
            "memory_cases": cases,
            "return_path_defect": _rational_rows(compressed.composition_defect),
            "metadata_cannot_promote_closure": guarded,
            "delay_transition": _rational_rows(expected_transition),
            "fit_does_not_establish_global_coverage": false_global,
        },
    )


def _transpose(matrix):
    return tuple(zip(*matrix, strict=True))


def _multiply(left, right):
    return tuple(
        tuple(sum((a * b for a, b in zip(row, column, strict=True)), F(0)) for column in _transpose(right))
        for row in left
    )


def _subtract(left, right):
    return tuple(
        tuple(a - b for a, b in zip(first, second, strict=True))
        for first, second in zip(left, right, strict=True)
    )


def _small_inverse(matrix):
    # Explicit adjugate formulas are independent of the production elimination.
    if len(matrix) == 1:
        return ((1 / matrix[0][0],),)
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    return ((d / determinant, -b / determinant), (-c / determinant, a / determinant))


def _gaussian(context: StressContext):
    from ..dynamics.gaussian_verifier import (
        FactorGaussianModel,
        check_gaussian_tolerance,
        evaluate_gaussian_gain,
        exact_gaussian_posterior,
        gaussian_packet_energy,
        outward_dyadic,
        verify_gaussian_gain,
    )
    from ..exceptions import ValidationError

    cases = []
    for _ in range(context.parameters["samples"]):
        n, p = context.rng.randrange(2, 4), context.rng.randrange(1, 3)
        prior = tuple(tuple(F(context.rng.randrange(-2, 3), 3) for _ in range(2)) for _ in range(n))
        noise = tuple((F(context.rng.randrange(-2, 3), 2),) for _ in range(p))
        h = tuple(tuple(F(context.rng.randrange(-2, 3)) for _ in range(n)) for _ in range(p))
        mean = tuple(F(context.rng.randrange(-2, 3), 2) for _ in range(n))
        observed = tuple(F(context.rng.randrange(-2, 3), 3) for _ in range(p))
        sigma = F(context.rng.randrange(1, 4), 3)
        gain = tuple(tuple(F(context.rng.randrange(-2, 3), 3) for _ in range(p)) for _ in range(n))
        model = FactorGaussianModel(
            prior_factor=prior,
            observation_operator=h,
            noise_factor=noise,
            noise_floor=sigma,
            prior_mean=mean,
            observation=observed,
        )
        covariance = _multiply(prior, _transpose(prior))
        noise_gram = _multiply(noise, _transpose(noise))
        noise_covariance = tuple(
            tuple(value + sigma * int(i == j) for j, value in enumerate(row))
            for i, row in enumerate(noise_gram)
        )
        innovation_gram = _multiply(_multiply(h, covariance), _transpose(h))
        innovation_covariance = tuple(
            tuple(a + b for a, b in zip(first, second, strict=True))
            for first, second in zip(innovation_gram, noise_covariance, strict=True)
        )
        cross = _multiply(covariance, _transpose(h))
        exact_gain = _multiply(cross, _small_inverse(innovation_covariance))
        schur = _subtract(covariance, _multiply(_multiply(exact_gain, h), covariance))
        target = exact_gaussian_posterior(model)
        if target.gain != exact_gain or target.covariance != schur or target.score != 0:
            return CheckOutcome(
                False,
                {"prior": prior, "h": h, "sigma": sigma},
                "exact posterior disagrees with explicit adjugate gain and Schur complement",
            )
        delta = _subtract(exact_gain, gain)
        extra = _multiply(_multiply(delta, innovation_covariance), _transpose(delta))
        candidate_covariance = tuple(
            tuple(a + b for a, b in zip(first, second, strict=True))
            for first, second in zip(schur, extra, strict=True)
        )
        candidate = evaluate_gaussian_gain(model, gain)
        residual = _subtract(cross, _multiply(gain, innovation_covariance))
        energy = sum((value * value for row in residual for value in row), F(0))
        if candidate.covariance != candidate_covariance or candidate.residual_energy != energy:
            return CheckOutcome(
                False,
                {"gain": gain, "covariance": candidate_covariance},
                "Joseph candidate disagrees with Schur correction and scalar residual energy",
            )
        cases.append(
            {
                "prior_factor": prior,
                "observation_operator": h,
                "noise_factor": noise,
                "noise_floor": sigma,
                "prior_mean": mean,
                "observation": observed,
                "candidate_gain": gain,
                "Schur_covariance": schur,
                "candidate_residual_energy": energy,
            }
        )
    singular = FactorGaussianModel(
        prior_factor=[[1], [0]],
        observation_operator=[[1, 1]],
        noise_factor=[[]],
        noise_floor=1,
        prior_mean=[0, 0],
        observation=[1],
    )
    gain = [[F(1, 3)], [0]]
    evaluated = evaluate_gaussian_gain(singular, gain)
    evaluated.certificate.diagnostics["score"] = F(0)
    coarse = verify_gaussian_gain(singular, gain, tolerance=F(1, 2), precision=0)
    fine = verify_gaussian_gain(singular, gain, tolerance=F(1, 2), precision=8, output_radius=F(1, 100))
    fail_closed = (
        evaluated.score == F(2, 9)
        and not coarse.certificate.valid
        and fine.certificate.valid
        and not check_gaussian_tolerance(singular, gain, F(1, 10)).valid
    )
    dyadics = []
    for value in (F(-7, 3), F(-1, 1025), F(0), F(2, 9), F(5, 9)):
        for precision in (0, 1, 8, 17):
            interval = outward_dyadic(value, precision)
            scaled = value * 2**precision
            lower = F(scaled.__floor__(), 2**precision)
            upper = F(-(-scaled).__floor__(), 2**precision)
            if interval.lower != lower or interval.upper != upper or not lower <= value <= upper:
                return CheckOutcome(
                    False,
                    {"value": value, "precision": precision},
                    "outward dyadic endpoints disagree with independent rational floor/ceiling",
                )
            dyadics.append({"value": value, "precision": precision, "lower": lower, "upper": upper})
    packet_model = FactorGaussianModel(
        prior_factor=[[1], [1]],
        observation_operator=[[1, 0], [0, 1]],
        noise_factor=[[], []],
        noise_floor=1,
        prior_mean=[0, 0],
        observation=[1, -1],
    )
    packet_gain = [[F(1, 3), 0], [F(1, 5), F(-1, 7)]]
    q = [[[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]], [[F(1, 2), F(-1, 2)], [F(-1, 2), F(1, 2)]]]
    d = [[[1, 0], [0, 0]], [[0, 0], [0, 1]]]
    residual = evaluate_gaussian_gain(packet_model, packet_gain).residual
    packet = gaussian_packet_energy(packet_model, packet_gain, output_packets=q, input_packets=d)
    expected = tuple(
        tuple(
            sum(
                (
                    sum(
                        (
                            output[i][k] * residual[k][ell] * inner[ell][j]
                            for k, ell in product(range(2), repeat=2)
                        ),
                        F(0),
                    )
                )
                ** 2
                for i, j in product(range(2), repeat=2)
            )
            for inner in d
        )
        for output in q
    )
    incomplete_rejected = False
    try:
        gaussian_packet_energy(packet_model, packet_gain, output_packets=q[:1], input_packets=d)
    except ValidationError:
        incomplete_rejected = True
    accepted = (
        fail_closed
        and packet.energies == expected
        and packet.total == sum((v * v for row in residual for v in row), F(0))
        and incomplete_rejected
    )
    return CheckOutcome(
        bool(accepted),
        {
            "cases": cases,
            "coarse_dyadic_score_rejected": not coarse.certificate.valid,
            "fine_dyadic_score_accepted": fine.certificate.valid,
            "dyadic_endpoints": dyadics,
            "packet_energies": expected,
            "incomplete_packets_rejected": incomplete_rejected,
        },
    )


def _series_multiply(left, right, modulus, count):
    return tuple(
        sum(left[k] * right[n - k] for k in range(n + 1) if k < len(left) and n - k < len(right)) % modulus
        for n in range(count)
    )


def _series_inverse(polynomial, modulus, count):
    inverse = [pow(polynomial[0], -1, modulus)]
    for n in range(1, count):
        inverse.append(
            -inverse[0]
            * sum(polynomial[k] * inverse[n - k] for k in range(1, min(n + 1, len(polynomial))))
            % modulus
        )
    return tuple(inverse)


def _direct_mahler(a, b, prime, modulus, count):
    result = [1]
    inverse = pow(b[0], -1, modulus)
    for n in range(1, count):
        right = sum(
            a[k] * result[(n - k) // prime] for k in range(min(n + 1, len(a))) if (n - k) % prime == 0
        )
        tail = sum(b[k] * result[n - k] for k in range(1, min(n + 1, len(b))))
        result.append((right - tail) * inverse % modulus)
    return tuple(result)


def _decode_cartier(rep, state, count):
    f = _direct_mahler(rep.numerator, rep.denominator, rep.prime, rep.modulus, count)
    inverse = _series_inverse(rep.denominator, rep.modulus, count)
    power = (1,) + (0,) * (count - 1)
    result = [0] * count
    for polynomial in state:
        term = _series_multiply(
            _series_multiply(polynomial, power, rep.modulus, count), f, rep.modulus, count
        )
        result = [(a + b) % rep.modulus for a, b in zip(result, term, strict=True)]
        power = _series_multiply(power, inverse, rep.modulus, count)
    return tuple(result)


def _cartier(context: StressContext):
    from ..generating.cartier_prime_power import compile_prime_power_cartier_representation

    cases = []
    for _ in range(context.parameters["samples"]):
        prime, exponent = context.rng.choice((2, 3, 5)), context.rng.randrange(1, 4)
        modulus = prime**exponent
        a = (
            1 + modulus * context.rng.randrange(-3, 4),
            context.rng.randrange(-3, 4),
            context.rng.randrange(-3, 4),
        )
        b = (
            1 + modulus * context.rng.randrange(-3, 4),
            context.rng.randrange(-3, 4),
            context.rng.randrange(-3, 4),
        )
        rep = compile_prime_power_cartier_representation(a, b, prime, exponent)
        expected = _direct_mahler(a, b, prime, modulus, 64)
        if rep.sequence(64) != expected:
            return CheckOutcome(
                False,
                {"numerator_lift": a, "denominator_lift": b, "prime": prime, "exponent": exponent},
                "Cartier residues differ from direct Mahler coefficient substitution",
            )
        state = tuple(
            tuple(context.rng.randrange(-modulus, modulus + 1) for _ in range(rep.degree_bound + 1))
            for _ in range(rep.layer_bound + 1)
        )
        expanded = _decode_cartier(rep, state, prime * 8)
        for digit in range(prime):
            if _decode_cartier(rep, rep.transition(state, digit), 8) != expanded[digit::prime]:
                return CheckOutcome(
                    False,
                    {"numerator_lift": a, "denominator_lift": b, "state": state, "digit": digit},
                    "Cartier state section differs from independent rational-series inversion",
                )
        cases.append(
            {
                "prime": prime,
                "exponent": exponent,
                "numerator_lift": a,
                "denominator_lift": b,
                "arbitrary_state": state,
                "expected_prefix": expected,
            }
        )
    partition = compile_prime_power_cartier_representation((1,), (1, -1), 2, 3)
    defect = partition.frobenius_defect == (0, 7, 5) and partition.frobenius_defect != (0, 3, 1)
    known = partition.sequence(16) == (1, 1, 2, 2, 4, 4, 6, 6, 2, 2, 6, 6, 4, 4, 2, 2)
    # A telescoping mask gives F(z)=(1-z)^-3, so a huge-index value has an
    # independent finite binomial coefficient oracle.
    mask = [1]
    for _ in range(3):
        expanded = [0] * (len(mask) + 2)
        for i, value in enumerate(mask):
            for digit in range(3):
                expanded[i + digit] += value
        mask = expanded
    collapsed = compile_prime_power_cartier_representation(mask, (1,), 3, 3)
    huge_index = 10**80 + 12345
    huge_expected = ((huge_index + 1) * (huge_index + 2) // 2) % 27
    huge_ok = collapsed.evaluate(huge_index) == huge_expected
    return CheckOutcome(
        defect and known and huge_ok,
        {
            "cases": cases,
            "binary_partition_prefix": partition.sequence(16),
            "extra_padic_digit_retained": defect,
            "huge_index": huge_index,
            "closed_form_residue": huge_expected,
            "huge_index_matches": huge_ok,
        },
    )


def paper_checks(samples: int = 8) -> tuple[StressCheck, ...]:
    """Ten finite suites for the added paper modules, before special functions."""
    if type(samples) is not int or not 1 <= samples <= 128:
        raise ValueError("samples must be an integer between 1 and 128")
    specifications = (
        (
            "paper.core.mapping-cones",
            _cones,
            {
                "random_differentials": samples,
                "primes": [2, 3, 5],
                "source_dimensions": [2, 2],
                "cone_dimensions": [2, 4, 2],
                "exhaustive_vector_bound": 625,
            },
        ),
        (
            "paper.core.rational-packets",
            _rational_packets,
            {
                "similarities": samples,
                "dimension": 3,
                "Jordan_primary_degrees": [2, 1],
                "numerators": [-3, 3],
                "denominators": [1, 3],
                "required_extra": "SymPy",
            },
        ),
        (
            "paper.compiler.exact-audits",
            _compiler,
            {
                "random_trees": samples,
                "tree_depth": 2,
                "nodes": 7,
                "complete_prunings": 5,
                "dyadic_denominator": 8,
                "penalties": ["0", "1/4", "1", "4"],
                "budgets": [0, 1, 3, 6],
                "integer_boundary_controls": 1,
            },
        ),
        (
            "paper.gaussian.hermite-transfer",
            _hermite,
            {
                "moment_orders": [0, 40],
                "Hermite_degrees": [0, 10],
                "random_polynomials": samples,
                "polynomial_degree": 8,
                "correlation_denominator": 5,
            },
        ),
        (
            "paper.arithmetic.lifting-stencils",
            _lifting_stencils,
            {
                "sampled_lift_fibers": samples,
                "primes": [2, 3, 5],
                "depths": [1, 2],
                "variables": 2,
                "stencil_cases": samples,
                "stencil_orders": [0, 4],
                "steps": [0, 3],
                "prefix_length": 20,
                "shared_digit_and_later_obstruction_controls": 2,
            },
        ),
        (
            "paper.langlands.tame-tensor",
            _tame,
            {
                "sampled_tensors": samples,
                "q": [2, 3, 4],
                "input_degrees": [1, 3],
                "tensor_dimension_maximum": 9,
                "closing_scalars": ["2", "1/2"],
            },
        ),
        (
            "paper.collatz.words",
            _collatz,
            {
                "all_word_lengths": [0, 3],
                "valuation_entries": [1, 4],
                "odd_seeds": [1, 255],
                "sampled_six_step_seeds": samples,
                "sampled_seed_bits": 32,
                "first_descent_max_steps": 128,
                "zero_budget_control_seed": 27,
            },
        ),
        (
            "paper.signals.predictive-foundations",
            _signal_foundations,
            {
                "all_transition_entries": [-1, 0, 1],
                "all_2_by_2_matrices": 81,
                "sampled_memory_matrices": samples,
                "memory_horizon": 5,
                "worked_delay_dimension": 3,
                "required_extra": "SymPy",
            },
        ),
        (
            "paper.bayesian.gaussian-verifier",
            _gaussian,
            {
                "rational_models": samples,
                "state_dimensions": [2, 3],
                "observation_dimensions": [1, 2],
                "prior_factor_columns": 2,
                "dyadic_precisions": [0, 1, 8, 17],
                "rotated_packet_dimension": 2,
            },
        ),
        (
            "paper.generating.prime-power-cartier",
            _cartier,
            {
                "sampled_masks_and_states": samples,
                "primes": [2, 3, 5],
                "precision_exponents": [1, 3],
                "mask_degree_maximum": 2,
                "compared_prefix": 64,
                "section_prefix": 8,
                "noncanonical_constant_lift_multiples": [-3, 3],
                "huge_index": 10**80 + 12345,
            },
        ),
    )
    return tuple(
        StressCheck(
            identifier,
            "paper_machinery",
            function,
            Exactness.EXACT,
            bounds,
            parameters={"samples": samples},
            assumptions=("finite supplied algebraic models; no source-paper universal conclusion",),
        )
        for identifier, function, bounds in specifications
    )


__all__ = ["paper_checks"]
