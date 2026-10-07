from __future__ import annotations

import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import combinations_with_replacement, permutations, product
from pathlib import Path
from typing import Any, Callable

import numpy as np

import komorebi_math
from komorebi_math import Exactness, PacketSystem
from komorebi_math.compatibility import (
    check_public_api,
    create_public_api_manifest,
    load_public_api_manifest,
)
from komorebi_math.exceptions import BackendUnavailableError, ValidationError
from komorebi_math.export import dumps_portable, loads_portable, portable_digest
from komorebi_math.nonlinear import (
    AutomaticDifferentiationProvider,
    JaxDerivativeProvider,
    PacketJetFamily,
    PolynomialDerivativeProvider,
    SparsePolynomialDerivativeProvider,
    SparseSymmetricTensor,
    TorchDerivativeProvider,
    available_ad_frameworks,
    compile_packet_jet_plan,
    cupy_runtime_status,
    execute_packet_jet_plan,
    nonlinear_cupy_available,
    packet_jet_support,
    snapshot_packet_jet,
)


@dataclass
class GroupResult:
    name: str
    checks: int
    failures: list[str]
    skipped: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def run_group(name: str, function: Callable[[Callable[[bool, str], None], list[str]], None]) -> GroupResult:
    checks = 0
    failures: list[str] = []
    skipped: list[str] = []

    def check(condition: bool, detail: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(detail)

    try:
        function(check, skipped)
    except Exception as exc:
        failures.append(f"unexpected {type(exc).__name__}: {exc}")
    return GroupResult(name, checks, failures, skipped)


def _coefficient(rng: random.Random) -> Fraction:
    numerator = rng.randrange(-5, 6)
    if numerator == 0:
        numerator = 1
    return Fraction(numerator, rng.randrange(1, 8))


def _sparse_tensor(
    rng: random.Random,
    order: int,
    source_dimension: int,
    target_dimension: int,
    *,
    density: float = 0.35,
) -> SparseSymmetricTensor:
    values: dict[tuple[int, ...], Fraction] = {}
    orbits = ((),) if order == 0 else combinations_with_replacement(range(source_dimension), order)
    orbit_tuple = tuple(orbits)
    for output in range(target_dimension):
        for inputs in orbit_tuple:
            if rng.random() < density:
                values[(output, *inputs)] = _coefficient(rng)
    return SparseSymmetricTensor.from_mapping(order, source_dimension, target_dimension, values)


def _dense_contract(tensor: np.ndarray, vectors: tuple[np.ndarray, ...]) -> np.ndarray:
    order = tensor.ndim - 1
    target_dimension = tensor.shape[0]
    source_dimension = vectors[0].shape[0] if vectors else 1
    output = np.zeros(target_dimension, dtype=object)
    for target in range(target_dimension):
        total: Any = Fraction(0)
        for indices in product(range(source_dimension), repeat=order):
            term: Any = tensor[(target, *indices)]
            for position, index in enumerate(indices):
                term *= vectors[position][index]
            total += term
        output[target] = total
    return output


def sparse_tensor_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    rng = random.Random(1001)
    for case in range(900):
        order = rng.randrange(0, 5)
        source_dimension = rng.randrange(1, 5)
        target_dimension = rng.randrange(1, 4)
        sparse = _sparse_tensor(rng, order, source_dimension, target_dimension)
        dense = sparse.to_dense()
        restored = SparseSymmetricTensor.from_dense(
            dense,
            source_dimension=source_dimension,
            tolerance=0.0,
        )
        directions = tuple(
            np.array([_coefficient(rng) for _ in range(source_dimension)], dtype=object)
            for _ in range(order)
        )
        expected = _dense_contract(dense, directions)
        observed = sparse.contract(directions, exact=True)
        check(sparse.certificate().valid, f"sparse certificate case={case}")
        check(restored == sparse, f"dense roundtrip case={case}")
        check(np.array_equal(observed, expected), f"exact contraction case={case}")
        check(sparse.expanded_nonzero_count <= sparse.dense_entry_count, f"expanded count case={case}")
        check(0.0 <= sparse.storage_ratio <= 1.0, f"storage ratio case={case}")

    for order in range(1, 6):
        source_dimension = 3
        target_dimension = 2
        values = {
            (output, *indices): Fraction(output + order + 1, sum(indices) + 1)
            for output in range(target_dimension)
            for indices in combinations_with_replacement(range(source_dimension), order)
        }
        sparse = SparseSymmetricTensor.from_mapping(order, source_dimension, target_dimension, values)
        dense = sparse.to_dense()
        for output in range(target_dimension):
            for indices in combinations_with_replacement(range(source_dimension), order):
                orbit = set(permutations(indices))
                reference = dense[(output, *indices)]
                check(
                    all(dense[(output, *permutation)] == reference for permutation in orbit),
                    f"symmetric orbit order={order} output={output} inputs={indices}",
                )


def sparse_provider_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    rng = random.Random(1002)
    for case in range(500):
        source_dimension = rng.randrange(1, 4)
        target_dimension = rng.randrange(1, 3)
        degree = rng.randrange(1, 5)
        sparse_tensors = {
            order: _sparse_tensor(
                rng,
                order,
                source_dimension,
                target_dimension,
                density=0.45,
            )
            for order in range(degree + 1)
        }
        dense = PolynomialDerivativeProvider(
            {
                order: tensor.to_dense(dtype=object)
                for order, tensor in sparse_tensors.items()
            },
            source_dimension=source_dimension,
            target_dimension=target_dimension,
            exactness=Exactness.EXACT,
        )
        sparse = SparsePolynomialDerivativeProvider(
            sparse_tensors,
            source_dimension=source_dimension,
            target_dimension=target_dimension,
            exactness=Exactness.EXACT,
        )
        point = np.array([_coefficient(rng) for _ in range(source_dimension)], dtype=object)
        directions = tuple(
            np.array([_coefficient(rng) for _ in range(source_dimension)], dtype=object)
            for _ in range(degree)
        )
        check(np.array_equal(sparse.value(point), dense.value(point)), f"provider value case={case}")
        check(sparse.storage_certificate((point,), directions).valid, f"provider certificate case={case}")
        check(sparse.to_dense_provider().degree == dense.degree, f"provider degree case={case}")
        for order in range(1, degree + 1):
            observed = sparse.derivative(order, point, directions[:order])
            expected = dense.derivative(order, point, directions[:order])
            check(np.array_equal(observed, expected), f"provider derivative case={case} order={order}")


def _coordinate_packets(dimension: int) -> PacketSystem:
    return PacketSystem(
        tuple(
            np.diag([1.0 if coordinate == packet else 0.0 for coordinate in range(dimension)])
            for packet in range(dimension)
        ),
        names=tuple(f"p{packet}" for packet in range(dimension)),
    )


def _random_symmetric_dense(
    rng: random.Random,
    order: int,
    source_dimension: int,
    target_dimension: int,
) -> np.ndarray:
    sparse = _sparse_tensor(
        rng,
        order,
        source_dimension,
        target_dimension,
        density=0.50,
    )
    return np.asarray(sparse.to_dense(), dtype=float)


def execution_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    rng = random.Random(1003)
    for case in range(650):
        dimension = rng.randrange(2, 4)
        order = rng.randrange(1, 4)
        tensor = _random_symmetric_dense(rng, order, dimension, dimension)
        provider = PolynomialDerivativeProvider(
            {order: tensor},
            source_dimension=dimension,
            target_dimension=dimension,
        )
        packets = _coordinate_packets(dimension)
        point = np.array([rng.uniform(-1, 1) for _ in range(dimension)])
        family = PacketJetFamily(provider, point, packets, packets, order)
        directions = tuple(
            np.array([rng.uniform(-1, 1) for _ in range(dimension)])
            for _ in range(order)
        )
        support = packet_jet_support(family)
        plan = compile_packet_jet_plan(family, support)
        result = execute_packet_jet_plan(plan, family, directions)
        check(support.complete and support.certificate.valid, f"support complete case={case}")
        check(plan.certificate.valid, f"plan certificate case={case}")
        check(result.certificate.valid, f"execution certificate case={case}")
        check(np.allclose(result.reconstructed, result.direct, atol=1e-10), f"reconstruction case={case}")
        check(plan.active_address_count + plan.pruned_address_count == plan.full_address_count, f"address partition case={case}")
        check(result.derivative_calls == plan.derivative_call_count, f"call count case={case}")
        check(plan.derivative_call_count <= plan.naive_derivative_call_count, f"call reduction case={case}")

    # Incomplete support is conservative: no untested block is pruned.
    dimension = 3
    order = 3
    tensor = np.ones((dimension,) + (dimension,) * order)
    provider = PolynomialDerivativeProvider({order: tensor}, source_dimension=dimension, target_dimension=dimension)
    packets = _coordinate_packets(dimension)
    family = PacketJetFamily(provider, np.zeros(dimension), packets, packets, order)
    partial = packet_jet_support(family, max_evaluations=1)
    plan = compile_packet_jet_plan(family, partial)
    check(not partial.complete, "partial support reports incompleteness")
    check(plan.pruned_address_count == 0, "partial support does not prune untested blocks")
    check(plan.active_address_count == plan.full_address_count, "partial support retains full address set")


def snapshot_persistence_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    rng = random.Random(1004)
    for case in range(450):
        dimension = rng.randrange(1, 4)
        order = rng.randrange(1, 4)
        tensor = _random_symmetric_dense(rng, order, dimension, dimension)
        provider = PolynomialDerivativeProvider(
            {order: tensor},
            source_dimension=dimension,
            target_dimension=dimension,
        )
        packets = _coordinate_packets(dimension)
        family = PacketJetFamily(provider, np.zeros(dimension), packets, packets, order)
        output = rng.randrange(dimension)
        inputs = tuple(rng.randrange(dimension) for _ in range(order))
        jet = family.jet(output, inputs)
        snapshot = snapshot_packet_jet(jet)
        directions = tuple(
            np.array([rng.uniform(-1, 1) for _ in range(dimension)])
            for _ in range(order)
        )
        encoded = dumps_portable(snapshot, indent=None)
        restored = loads_portable(encoded)
        check(snapshot.capture_certificate.valid, f"snapshot capture case={case}")
        check(np.allclose(snapshot.evaluate(directions), jet.evaluate(directions), atol=1e-10), f"snapshot evaluation case={case}")
        check(restored == snapshot, f"snapshot roundtrip case={case}")
        check(dumps_portable(restored, indent=None) == encoded, f"snapshot canonical encoding case={case}")
        check(restored.comparison_certificate(jet, directions).valid, f"snapshot comparison case={case}")


def persistence_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    rng = random.Random(1005)
    for case in range(700):
        values = {
            "case": case,
            "fraction": Fraction(rng.randrange(-50, 51), rng.randrange(1, 20)),
            "complex": complex(rng.uniform(-2, 2), rng.uniform(-2, 2)),
            "bytes": bytes(rng.randrange(256) for _ in range(rng.randrange(0, 12))),
            "tuple": tuple(rng.randrange(-20, 21) for _ in range(rng.randrange(0, 8))),
            "array": np.array(
                [rng.uniform(-2, 2) for _ in range(6)],
                dtype=float,
            ).reshape(2, 3),
            "set": {rng.randrange(-10, 11) for _ in range(5)},
        }
        encoded = dumps_portable(values, indent=None)
        decoded = loads_portable(encoded)
        check(dumps_portable(decoded, indent=None) == encoded, f"portable roundtrip case={case}")
        check(portable_digest(values) == portable_digest(decoded), f"portable digest case={case}")

    check(portable_digest({"a": 1, "b": 2}) == portable_digest({"b": 2, "a": 1}), "mapping order invariance")

    envelope = json.loads(dumps_portable({"value": 4}))
    envelope["payload"]["data"][0][1] = 5
    try:
        loads_portable(json.dumps(envelope))
    except ValidationError:
        check(True, "tamper rejection")
    else:
        check(False, "tamper rejection")

    for payload, label in (
        ('{"schema": NaN}', "nonfinite JSON rejection"),
        ('{"schema": "first", "schema": "second"}', "duplicate key rejection"),
    ):
        try:
            loads_portable(payload)
        except ValidationError:
            check(True, label)
        else:
            check(False, label)

    malformed = json.loads(dumps_portable(Fraction(1, 2)))
    malformed["payload"]["data"]["denominator"] = 0
    canonical = json.dumps(malformed["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    malformed["payload_sha256"] = sha256(canonical.encode()).hexdigest()
    try:
        loads_portable(json.dumps(malformed))
    except ValidationError:
        check(True, "malformed Fraction rejection")
    else:
        check(False, "malformed Fraction rejection")


def autodiff_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    rng = random.Random(1006)

    def function(value: np.ndarray) -> np.ndarray:
        return np.array([value[0] ** 3 + value[0] * value[1] + value[1] ** 2])

    def central_jvp(function_: Callable[[np.ndarray], np.ndarray], point: np.ndarray, direction: np.ndarray):
        step = 1e-6
        return function_(point), (function_(point + step * direction) - function_(point - step * direction)) / (2 * step)

    generic = AutomaticDifferentiationProvider(function, central_jvp, 2, 1, max_order=1)
    for case in range(250):
        point = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
        direction = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
        expected = (
            (3 * point[0] ** 2 + point[1]) * direction[0]
            + (point[0] + 2 * point[1]) * direction[1]
        )
        observed = generic.derivative(1, point, (direction,))[0]
        check(abs(observed - expected) < 1e-8, f"generic JVP case={case}")

    availability = available_ad_frameworks()
    check(set(availability) == {"jax", "torch"}, "AD availability keys")

    if availability["jax"]:
        import jax.numpy as jnp

        provider = JaxDerivativeProvider(
            lambda x: jnp.array([x[0] ** 4 + x[0] * x[1] ** 2]),
            2,
            1,
            max_order=3,
        )
        for case in range(40):
            point = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            u = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            v = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            w = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            expected = 24 * point[0] * u[0] * v[0] * w[0] + 2 * (
                u[0] * v[1] * w[1]
                + v[0] * u[1] * w[1]
                + w[0] * u[1] * v[1]
            )
            observed = provider.derivative(3, point, (u, v, w))[0]
            check(np.allclose(observed, expected, rtol=4e-5, atol=4e-6), f"JAX third derivative case={case}")
    else:
        skipped.append("JAX adapter checks skipped because JAX is not installed")

    if availability["torch"]:
        import torch

        provider = TorchDerivativeProvider(
            lambda x: torch.stack((x[0] ** 4 + x[0] * x[1] ** 2,)),
            2,
            1,
            max_order=3,
        )
        for case in range(40):
            point = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            u = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            v = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            w = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            expected = 24 * point[0] * u[0] * v[0] * w[0] + 2 * (
                u[0] * v[1] * w[1]
                + v[0] * u[1] * w[1]
                + w[0] * u[1] * v[1]
            )
            observed = provider.derivative(3, point, (u, v, w))[0]
            check(np.allclose(observed, expected, rtol=1e-10, atol=1e-10), f"Torch third derivative case={case}")
    else:
        skipped.append("PyTorch adapter checks skipped because PyTorch is not installed")


def compatibility_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    del skipped
    reference = load_public_api_manifest()
    first = create_public_api_manifest()
    second = create_public_api_manifest()
    report = check_public_api(strict_signatures=True)
    check(komorebi_math.__version__ == "0.10.0a1", "package version")
    check(first == second, "manifest determinism")
    check(first.digest == reference.digest, "manifest digest")
    check(first.package_version == "0.10.0a1", "manifest package version")
    check(len(first.symbols) >= 450, "manifest symbol count")
    check(report.compatible, "strict compatibility")
    check(not report.added, "no manifest additions")
    check(not report.removed, "no manifest removals")
    check(not report.kind_changed, "no manifest kind changes")
    check(not report.signature_changed, "no manifest signature changes")
    for namespace in reference.namespaces:
        names = [symbol.name for symbol in reference.symbols if symbol.namespace == namespace]
        check(len(names) == len(set(names)), f"unique manifest symbols namespace={namespace}")
        check(names == sorted(names), f"sorted manifest symbols namespace={namespace}")


def backend_boundary_group(check: Callable[[bool, str], None], skipped: list[str]) -> None:
    status = cupy_runtime_status()
    check(status.installed is nonlinear_cupy_available(), "CuPy installation status agreement")
    check(status.device_count >= 0, "nonnegative CuPy device count")
    if status.runtime_available:
        check(status.installed, "available runtime implies installed CuPy")
        check(status.device_count > 0, "available runtime reports a device")
        skipped.append("No performance claim is made by the boundary audit")
    else:
        check(status.device_count == 0, "unavailable runtime reports zero devices")
        check(bool(status.error), "unavailable runtime reports an error")
        try:
            from komorebi_math.nonlinear import CuPyPolynomialDerivativeProvider

            CuPyPolynomialDerivativeProvider({1: np.eye(1).reshape(1, 1)})
        except BackendUnavailableError:
            check(True, "CuPy constructor fails explicitly without a runtime")
        else:
            check(False, "CuPy constructor fails explicitly without a runtime")
        skipped.append("Device-backed CuPy correctness and performance were not tested")


GROUPS = (
    ("sparse symmetric tensor orbits", sparse_tensor_group),
    ("sparse polynomial provider equivalence", sparse_provider_group),
    ("projection-cached packet execution", execution_group),
    ("packet-jet snapshots and persistence", snapshot_persistence_group),
    ("portable schema integrity", persistence_group),
    ("directional automatic differentiation", autodiff_group),
    ("public API compatibility", compatibility_group),
    ("optional backend boundary", backend_boundary_group),
)


def main() -> int:
    results = [run_group(name, function) for name, function in GROUPS]
    payload = {
        "version": komorebi_math.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "groups": [asdict(result) | {"passed": result.passed} for result in results],
        "checks": sum(result.checks for result in results),
        "failures": sum(len(result.failures) for result in results),
        "skipped": sum(len(result.skipped) for result in results),
        "passed": all(result.passed for result in results),
    }
    output = Path(__file__).with_name("BLOCK10_DEEP_AUDIT.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Block 10 deep audit: {payload['checks']} checks, "
        f"{payload['failures']} failures, {payload['skipped']} boundary notes"
    )
    for result in results:
        print(
            f"  {result.name}: {result.checks} checks, "
            f"{len(result.failures)} failures, {len(result.skipped)} notes"
        )
        for failure in result.failures[:10]:
            print(f"    {failure}")
        for note in result.skipped:
            print(f"    NOTE: {note}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
