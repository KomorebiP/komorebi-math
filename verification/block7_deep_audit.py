from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from komorebi_math.deformation import (
    FiniteCochainComplex,
    PresentedModule,
    VisibilityTree,
    finite_frobenius_separator,
    nilpotent_chart_count,
    normal_crossing_count,
    residue_forest,
)
from komorebi_math.langlands.completion import AutomorphicCompletionReport, CompletionGate, GateStatus
from komorebi_math.langlands.finite_image import (
    CliffordStep,
    CliffordTower,
    InducedRepresentation,
    LinearCharacter,
    MonomialAction,
    cyclic_group,
    mackey_tensor_packets,
)
from komorebi_math.langlands.global_ import FiniteHeckePackets, PatchingSupportReport, SupportComparison
from komorebi_math.langlands.local import (
    AlternatingForm,
    FiniteField,
    FrobeniusOrbit,
    HeisenbergModel,
    ProjectiveIntertwinerEnvelope,
    QuadraticRefinement,
    SpecialMonodromyBlock,
    TameWeilPacket,
    gauss_sum,
    regular_orbit_count,
    tame_tensor_refinement,
)
from komorebi_math.langlands.ramification import ProjectiveMultiplicityPacket, RamificationCliffordTower, RamificationStage


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


def deformation_counts(check) -> None:
    for prime in (2, 3, 5):
        for depth in range(1, 5):
            modulus = prime**depth
            for exponent in range(1, 5):
                brute = sum(pow(value, exponent, modulus) == 0 for value in range(modulus))
                check(brute == nilpotent_chart_count(prime, exponent, depth), f"nilpotent count p={prime} m={depth} e={exponent}")
            brute_crossing = sum((left * right) % modulus == 0 for left in range(modulus) for right in range(modulus))
            check(brute_crossing == normal_crossing_count(prime, depth), f"crossing count p={prime} m={depth}")
    rng = random.Random(701)
    for _ in range(500):
        values = tuple(rng.randrange(-1000, 1001) for _ in range(7))
        tree = VisibilityTree(values, prime=rng.choice((2, 3, 5, 7)))
        check(tree.ultrametric_certificate(), f"visibility ultrametric {values}")
    for prime in (2, 3):
        for depth in range(2, 6):
            modulus = prime**depth
            all_values = tuple(range(modulus))
            forest = residue_forest({m: all_values for m in range(1, depth + 1)}, prime=prime)
            report = forest.report(branch_limit=modulus + 1)
            check(report.certificate.valid, f"full residue forest p={prime} depth={depth}")
            check(len(report.branches) == modulus, f"full residue branch count p={prime} depth={depth}")


def complexes_support(check) -> None:
    for prime in (2, 3, 5, 7):
        for a in range(5):
            for b in range(5):
                # d0 has rank min(a,b), d1 is zero, so cohomology is explicit.
                d0 = np.zeros((b, a), dtype=int)
                for index in range(min(a, b)):
                    d0[index, index] = 1
                complex_ = FiniteCochainComplex({0: a, 1: b}, {0: d0}, prime=prime)
                check(complex_.cohomology(0).cohomology_dimension == max(a - b, 0), f"H0 p={prime} a={a} b={b}")
                check(complex_.cohomology(1).cohomology_dimension == max(b - a, 0), f"H1 p={prime} a={a} b={b}")
    import sympy as sp

    x, y = sp.symbols("x y")
    module = PresentedModule([[x, 0], [0, y]])
    for left in range(-5, 6):
        for right in range(-5, 6):
            result = module.fiber({x: left, y: right})
            expected = int(left == 0) + int(right == 0)
            check(result.fiber_dimension == expected, f"fiber ({left},{right})")
            check(result.certificate.valid, f"Fitting ({left},{right})")
    rng = random.Random(702)
    for dimension in range(1, 7):
        while True:
            matrix = [[rng.randrange(-5, 6) for _ in range(dimension)] for _ in range(dimension)]
            separator = finite_frobenius_separator({index: row for index, row in enumerate(matrix)})
            if separator.separating:
                break
        multiplicities = tuple(rng.randrange(-3, 4) for _ in range(dimension))
        observed = tuple(sum(matrix[row][col] * multiplicities[col] for col in range(dimension)) for row in range(dimension))
        check(tuple(separator.recover(observed)) == multiplicities, f"separator dimension {dimension}")


def finite_fields(check) -> None:
    fields = (
        FiniteField(2),
        FiniteField(3),
        FiniteField(5),
        FiniteField(2, 2, (1, 1, 1)),
        FiniteField(2, 3, (1, 1, 0, 1)),
        FiniteField(3, 2, (1, 0, 1)),
    )
    for field in fields:
        for a in range(field.order):
            check(field.add(a, 0) == a, f"add identity q={field.order} a={a}")
            check(field.multiply(a, 1) == a, f"mul identity q={field.order} a={a}")
            if a:
                check(field.multiply(a, field.inverse(a)) == 1, f"inverse q={field.order} a={a}")
            for b in range(field.order):
                check(field.add(a, b) == field.add(b, a), f"add comm q={field.order}")
                check(field.multiply(a, b) == field.multiply(b, a), f"mul comm q={field.order}")
                for c in range(field.order):
                    check(
                        field.multiply(a, field.add(b, c)) == field.add(field.multiply(a, b), field.multiply(a, c)),
                        f"distributive q={field.order}",
                    )
        for a in range(field.order):
            for b in range(field.order):
                check(field.trace(field.add(a, b)) == (field.trace(a) + field.trace(b)) % field.prime, f"trace linear q={field.order}")
                check(field.norm(field.multiply(a, b)) == (field.norm(a) * field.norm(b)) % field.prime, f"norm multiplicative q={field.order}")
        for exponent in range(1, min(field.order - 1, 8)):
            check(gauss_sum(field, exponent).certificate.valid, f"Gauss q={field.order} exponent={exponent}")


def tame_packets(check) -> None:
    for q in (2, 3, 4, 5):
        for degree in range(1, 6):
            modulus = q**degree - 1
            regular = sum(FrobeniusOrbit(q, degree, exponent).regular for exponent in range(modulus))
            check(regular_orbit_count(q, degree) * degree == regular, f"orbit count q={q} n={degree}")
            for exponent in range(min(modulus, 50)):
                orbit = FrobeniusOrbit(q, degree, exponent)
                check(orbit.certificate().valid, f"orbit closure q={q} n={degree} a={exponent}")
                if orbit.regular and degree <= 5:
                    check(TameWeilPacket(orbit).relation_certificate().valid, f"tame matrix q={q} n={degree} a={exponent}")
    for left in range(1, 16):
        for right in range(1, 16):
            result = tame_tensor_refinement(left, right)
            check(result.certificate.valid, f"tame tensor {left},{right}")
    for length in range(1, 10):
        check(SpecialMonodromyBlock(length, 3.0).certificate().valid, f"monodromy length {length}")


def heisenberg_symplectic(check) -> None:
    rng = random.Random(703)
    for prime in (2, 3, 5, 7):
        model = HeisenbergModel(prime, 1)
        check(model.trace_support_certificate().valid, f"trace p={prime}")
        check(model.selector_certificate().valid, f"selector p={prime}")
        elements = [model.element((v,), (w,), z) for v in range(prime) for w in range(prime) for z in range(prime)]
        for _ in range(500):
            left = rng.choice(elements)
            right = rng.choice(elements)
            check(model.group_law_certificate(left, right).valid, f"law p={prime}")
    for linear in product((0, 1), repeat=2):
        form = AlternatingForm([[0, 1], [1, 0]], 2)
        refinement = QuadraticRefinement(form, linear)
        check(refinement.polar_certificate().valid, f"quadratic linear={linear}")
        check(abs(refinement.gauss_sum()) == 2, f"quadratic magnitude linear={linear}")
    model = HeisenbergModel(5, 1)
    recovered = ProjectiveIntertwinerEnvelope(
        prime=5,
        labels=((0, 0), (1, 0), (0, 1)),
        operators=(np.eye(5), model.translation_matrix((1,)), model.modulation_matrix((1,))),
    ).recover()
    check(recovered.certificate.valid, "projective envelope p=5")
    check(recovered.form.rank == 2, "projective envelope rank")


def finite_image(check) -> None:
    for order in range(1, 13):
        group = cyclic_group(order)
        for divisor in range(1, order + 1):
            if order % divisor:
                continue
            generator = order // divisor if divisor > 1 else 0
            subgroup = group.generated_subgroup((generator,))
            check(subgroup.order == divisor, f"cyclic subgroup order n={order} d={divisor}")
            values = {element: np.exp(2j * np.pi * element / order) ** 0 for element in subgroup.elements}
            character = LinearCharacter(subgroup, values)
            representation = InducedRepresentation(group, character)
            check(representation.certificate().valid, f"induced law n={order} d={divisor}")
            action = MonomialAction(representation)
            for degree in range(0, 4):
                symmetric = action.symmetric_power_packets(degree)
                expected = 1 if degree == 0 else __import__('math').comb(representation.dimension + degree - 1, degree)
                check(action.packet_certificate(symmetric, expected_dimension=expected).valid, f"sym packets n={order} d={divisor} m={degree}")
            exterior = action.exterior_power_packets(min(2, representation.dimension))
            expected_ext = __import__('math').comb(representation.dimension, min(2, representation.dimension))
            check(action.packet_certificate(exterior, expected_dimension=expected_ext).valid, f"ext packets n={order} d={divisor}")
            check(mackey_tensor_packets(group, character, character).certificate.valid, f"Mackey n={order} d={divisor}")
    rng = random.Random(704)
    for _ in range(500):
        steps = []
        parent_order = 2 ** rng.randrange(2, 8)
        parent_degree = 2 ** rng.randrange(0, 4)
        for _ in range(rng.randrange(1, 5)):
            if parent_order <= 1:
                break
            child_order = parent_order // 2
            if parent_degree > 1 and rng.random() < 0.5:
                mode = "I"
                child_degree = parent_degree // 2
                if child_degree < 1:
                    mode = "E"
                    child_degree = parent_degree
            else:
                mode = "E"
                child_degree = parent_degree
            steps.append(CliffordStep(parent_order, child_order, parent_degree, child_degree, mode, attached_descent=rng.random() < 0.5))
            parent_order = child_order
            parent_degree = child_degree
        if steps:
            tower = CliffordTower(steps)
            report = tower.report()
            check(report.depth == sum(step.mode.value == "E" and step.parent_degree > 1 for step in steps), "Clifford depth")


def global_completion(check) -> None:
    rng = random.Random(705)
    for dimension in range(1, 20):
        labels = [rng.randrange(4) for _ in range(dimension)]
        projectors = {
            str(label): np.diag([1 if value == label else 0 for value in labels])
            for label in sorted(set(labels))
        }
        packets = FiniteHeckePackets(projectors)
        check(packets.certificate().valid, f"Hecke packets dimension {dimension}")
        check(sum(packet.dimension for packet in packets.packets()) == dimension, f"Hecke rank sum {dimension}")
    gates = tuple(CompletionGate)
    for mask in range(1 << len(gates)):
        report = AutomorphicCompletionReport({
            gate: GateStatus(bool(mask & (1 << index)), gate.value)
            for index, gate in enumerate(gates)
        })
        expected = mask == (1 << len(gates)) - 1
        check(report.complete == expected, f"completion mask {mask}")
        check(report.result().certificate.valid == expected, f"completion certificate {mask}")
    components = ("a", "b", "c")
    for required_mask in range(8):
        required = {components[index] for index in range(3) if required_mask & (1 << index)}
        for image_mask in range(8):
            image = {components[index] for index in range(3) if image_mask & (1 << index)}
            result = SupportComparison.construct(required, components, image)
            check(result.certificate.valid == required.issubset(image), f"support masks {required_mask},{image_mask}")
    for values in product((False, True), repeat=5):
        report = PatchingSupportReport(*values)
        check(report.complete == all(values), f"patching {values}")
        check(report.certificate().valid == all(values), f"patching cert {values}")
    form = AlternatingForm([[0, 1], [1, 0]], 2)
    packet = ProjectiveMultiplicityPacket(2, 2, form, quadratic_refinement=QuadraticRefinement(form))
    tower = RamificationCliffordTower((RamificationStage(0, 4, 2, 2, 1, packet),))
    check(tower.report().certificate.valid, "ramification tower")


def source_integrity(check) -> None:
    root = Path(__file__).resolve().parents[1]
    modules = tuple((root / "src" / "komorebi_math" / "deformation").glob("*.py"))
    modules += tuple((root / "src" / "komorebi_math" / "langlands").rglob("*.py"))
    for path in modules:
        text = path.read_text(encoding="utf-8")
        check("/mnt/data" not in text, f"no runtime path {path.name}")
        check("TODO" not in text, f"no TODO {path.name}")


def main() -> int:
    groups = [
        run_group("deformation_counts", deformation_counts),
        run_group("complexes_support", complexes_support),
        run_group("finite_fields", finite_fields),
        run_group("tame_packets", tame_packets),
        run_group("heisenberg_symplectic", heisenberg_symplectic),
        run_group("finite_image", finite_image),
        run_group("global_completion", global_completion),
        run_group("source_integrity", source_integrity),
    ]
    payload = {
        "passed": all(group.passed for group in groups),
        "total_checks": sum(group.checks for group in groups),
        "total_failures": sum(len(group.failures) for group in groups),
        "groups": [asdict(group) | {"passed": group.passed} for group in groups],
    }
    output = Path(__file__).with_name("BLOCK7_DEEP_AUDIT.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
