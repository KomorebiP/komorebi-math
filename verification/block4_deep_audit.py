"""Deterministic independent checks for the Block 4 signal and dynamics release."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from komorebi_math.certificates import Exactness
from komorebi_math.dynamics import (
    BayesianPartition,
    BayesianUpdate,
    HiddenMarkovFilter,
    ConditionalExpectationTower,
    DirichletGreen1D,
    FiniteFeynmanKac,
    FeshbachReduction,
    MarkovLumping,
    MatrixSemigroup,
    approximate_likelihood_partition,
    coarsest_sufficient_partition,
    matrix_exponential,
    MemoryKernel,
    PhaseTypeSurvival,
    SemigroupRefinement,
    fractional_defect,
    fractional_naturality_certificate,
    fractional_power,
    memory_kernel_error_bound,
    relative_entropy,
    relative_entropy_decomposition,
)
from komorebi_math.signals import (
    HaarPacketTree,
    PeriodicFilterBank,
    SingularPacket1D,
    TruncatedPowerTerm,
    best_basis,
    convolve_signal,
    convolution_matrix,
    finite_difference,
    gamma_from_impedance,
    impedance_from_gamma,
    lattice_polyphase_analyze,
    mixed_radix_address,
    nonlinear_shift_filter,
    packet_product,
    periodic_five_three_lifting,
    phase_mobius,
    polyphase_analyze,
    polyphase_convolution,
    polyphase_filter_kernels,
    recover_separated_jumps,
    recover_vector_jumps_subgaussian,
    retained_packet_reconstruction,
    scalar_region_lp_error,
    split_mixed_radix_address,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "komorebi_math"


class Audit:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.failures: list[dict[str, Any]] = []

    def check(self, group: str, condition: bool, detail: str = "") -> None:
        self.counts[group] += 1
        if not bool(condition):
            self.failures.append({"group": group, "detail": detail or "condition failed"})

    def close(self) -> dict[str, Any]:
        total = sum(self.counts.values())
        return {
            "passed": not self.failures,
            "total_checks": total,
            "passed_checks": total - len(self.failures),
            "failed_checks": len(self.failures),
            "groups": dict(sorted(self.counts.items())),
            "failures": self.failures[:100],
        }


def random_lumpable_generator(rng: np.random.Generator, block_sizes: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    labels = tuple(block for block, size in enumerate(block_sizes) for _ in range(size))
    blocks = [np.flatnonzero(np.asarray(labels) == index) for index in range(len(block_sizes))]
    dimension = len(labels)
    generator = np.zeros((dimension, dimension), dtype=float)
    coarse_rates = rng.uniform(0.02, 0.35, size=(len(blocks), len(blocks)))
    np.fill_diagonal(coarse_rates, 0.0)
    for source_block, source_states in enumerate(blocks):
        for source in source_states:
            for target_block, target_states in enumerate(blocks):
                if target_block == source_block:
                    continue
                total = coarse_rates[target_block, source_block]
                weights = rng.uniform(0.1, 1.0, size=target_states.size)
                weights = weights / weights.sum()
                generator[target_states, source] = total * weights
            other = source_states[source_states != source]
            if other.size:
                internal_total = float(rng.uniform(0.0, 0.4))
                weights = rng.uniform(0.1, 1.0, size=other.size)
                generator[other, source] = internal_total * weights / weights.sum()
            generator[source, source] = -float(np.sum(generator[:, source]))
    return generator, labels


def enumerate_pruning_cost(costs: dict[str, float], depth: int, path: str = "") -> float:
    keep = costs[path]
    if len(path) == depth:
        return keep
    split = enumerate_pruning_cost(costs, depth, path + "0") + enumerate_pruning_cost(costs, depth, path + "1")
    return min(keep, split)


def run() -> dict[str, Any]:
    audit = Audit()
    rng = np.random.default_rng(20260815)

    # Polyphase and lattice coordinate identities.
    for _ in range(350):
        length = int(rng.integers(1, 100))
        order = int(rng.integers(1, 12))
        values = rng.normal(size=length)
        coordinates = polyphase_analyze(values, order)
        audit.check("polyphase", np.array_equal(coordinates.reconstruct(), values), f"length={length}, order={order}")
        projected = np.zeros_like(values)
        for residue, channel in enumerate(coordinates.channels):
            projected[residue::order] = channel
        audit.check("polyphase", np.array_equal(projected, values), "manual interleave")

    for _ in range(180):
        shape = tuple(int(rng.integers(1, 9)) for _ in range(int(rng.integers(1, 4))))
        strides = tuple(int(rng.integers(1, 5)) for _ in shape)
        values = rng.integers(-50, 50, size=shape)
        coordinates = lattice_polyphase_analyze(values, strides)
        audit.check("lattice_polyphase", np.array_equal(coordinates.reconstruct(), values), f"shape={shape}")

    for _ in range(260):
        radices = tuple(int(rng.integers(2, 8)) for _ in range(int(rng.integers(1, 6))))
        digits = tuple(int(rng.integers(0, radix)) for radix in radices)
        address = mixed_radix_address(digits, radices)
        audit.check("mixed_radix", split_mixed_radix_address(address, radices) == digits)

    # Convolution, filter banks, lifting, and packet trees.
    modes = ("full", "same", "valid", "circular")
    for _ in range(350):
        length = int(rng.integers(1, 45))
        kernel_length = int(rng.integers(1, 12))
        order = int(rng.integers(1, 8))
        mode = modes[int(rng.integers(0, len(modes)))]
        signal = rng.normal(size=length)
        kernel = rng.normal(size=kernel_length)
        direct = convolve_signal(signal, kernel, mode=mode)
        packet = polyphase_convolution(signal, kernel, order, mode=mode)
        audit.check("convolution", packet.certificate.valid, packet.certificate.message)
        audit.check("convolution", np.allclose(packet.value, direct, atol=2e-11, rtol=2e-11), mode)

    identity_bank = PeriodicFilterBank(([1.0],), ([1.0],), 1)
    for length in range(1, 101):
        values = rng.normal(size=length)
        audit.check("filterbanks", identity_bank.perfect_reconstruction_certificate(length).valid)
        audit.check("filterbanks", np.allclose(identity_bank.synthesize(identity_bank.analyze(values)), values))

    lifting = periodic_five_three_lifting()
    for _ in range(160):
        length = 2 * int(rng.integers(1, 80))
        values = rng.normal(size=length)
        audit.check("filterbanks", lifting.certificate(values).valid)

    for _ in range(160):
        depth = int(rng.integers(0, 5))
        length = (2**depth) * int(rng.integers(1, 10))
        values = rng.normal(size=length)
        tree = HaarPacketTree.analyze(values, depth)
        audit.check("wavelets", tree.certificate().valid)
        costs = {path: float(rng.uniform(0.0, 5.0)) for path in tree.nodes}
        result = best_basis(tree, costs)
        audit.check("wavelets", np.isclose(result.cost, enumerate_pruning_cost(costs, depth)))
        audit.check("wavelets", np.allclose(result.reconstruct(tree), values))

    # Singular packet reconstruction and recovery.
    for _ in range(220):
        size = int(rng.integers(4, 50))
        smooth = rng.normal(size=size)
        packet = rng.normal(size=size)
        signal = smooth + packet
        q = rng.normal(size=(size, size)) / max(size, 1)
        estimate = packet + 0.01 * rng.normal(size=size)
        result = retained_packet_reconstruction(signal, estimate, q, true_smooth=smooth, true_packet=packet)
        audit.check("singular_packets", result.certificate.valid)
        expected = q @ (signal - estimate) + estimate
        audit.check("singular_packets", np.allclose(result.reconstruction, expected))

    for _ in range(300):
        length = int(rng.integers(10, 200))
        first = rng.random(length) < rng.uniform(0.1, 0.8)
        second = rng.random(length) < rng.uniform(0.1, 0.8)
        a = float(rng.normal())
        b = float(rng.normal())
        p = float(rng.choice([1.0, 2.0, 3.0, 4.0]))
        direct = float(np.linalg.norm(a * first.astype(float) - b * second.astype(float), ord=p))
        formula = scalar_region_lp_error(
            a,
            b,
            float(np.sum(first & second)),
            float(np.sum(first & ~second)),
            float(np.sum(~first & second)),
            p=p,
        )
        audit.check("singular_packets", np.isclose(formula, direct, atol=1e-11, rtol=1e-11))

    for _ in range(220):
        count = 240
        spacing = 1.0 / count
        cells = np.sort(rng.choice(np.arange(20, count - 20), size=3, replace=False))
        if np.min(np.diff(cells)) < 5:
            continue
        locations = (cells + rng.uniform(0.1, 0.9, size=3)) * spacing
        amplitudes = rng.choice([-1.0, 1.0], size=3) * rng.uniform(0.4, 0.9, size=3)
        slope = float(rng.uniform(-0.03, 0.03))
        grid = np.arange(count + 1) * spacing
        signal = slope * grid
        for location, amplitude in zip(locations, amplitudes, strict=True):
            signal = signal + amplitude * (grid >= location)
        result = recover_separated_jumps(
            signal,
            spacing=spacing,
            lipschitz_bound=abs(slope),
            noise_bound=0.0,
            minimum_jump=float(np.min(np.abs(amplitudes))),
            minimum_separation=5 * spacing,
        )
        audit.check("jump_recovery", result.certificate.valid)
        audit.check("jump_recovery", result.cells == tuple(int(np.floor(location / spacing)) for location in locations))
        audit.check("jump_recovery", np.max(np.abs(result.amplitudes - amplitudes)) <= result.amplitude_error_bound + 1e-11)

    # Nonlinear residue products and Smith chart group law.
    for _ in range(360):
        left = rng.normal(size=int(rng.integers(1, 30)))
        right = rng.normal(size=int(rng.integers(1, 30)))
        order = int(rng.integers(1, 9))
        result = packet_product(left, right, order)
        audit.check("nonlinear_signals", result.certificate.valid)
        audit.check("nonlinear_signals", np.allclose(sum(result.value, np.zeros(left.size + right.size - 1)), np.convolve(left, right)))
        audit.check("nonlinear_signals", nonlinear_shift_filter(left).shape == left.shape)

    for _ in range(420):
        impedance = rng.uniform(0.05, 5.0, size=8) + 1j * rng.uniform(-2.0, 2.0, size=8)
        gamma = gamma_from_impedance(impedance)
        audit.check("smith_chart", np.allclose(impedance_from_gamma(gamma), impedance))
        order = int(rng.integers(2, 15))
        first = int(rng.integers(0, order))
        second = int(rng.integers(0, order))
        composed = phase_mobius(phase_mobius(impedance, order, first), order, second)
        target = phase_mobius(impedance, order, first + second)
        audit.check("smith_chart", np.allclose(composed, target, atol=2e-10, rtol=2e-10))

    # Semigroup laws and Duhamel defects.
    for _ in range(220):
        dimension = int(rng.integers(1, 5))
        generator = rng.normal(scale=0.3, size=(dimension, dimension))
        semigroup = MatrixSemigroup(generator)
        first = float(rng.uniform(0.0, 1.0))
        second = float(rng.uniform(0.0, 1.0))
        audit.check("semigroups", semigroup.semigroup_certificate(first, second, tolerance=2e-9).valid)

    for _ in range(160):
        fine_dimension = int(rng.integers(2, 6))
        coarse_dimension = int(rng.integers(1, fine_dimension + 1))
        fine = rng.normal(scale=0.2, size=(fine_dimension, fine_dimension))
        coarse = rng.normal(scale=0.2, size=(coarse_dimension, coarse_dimension))
        coarsen = rng.normal(size=(coarse_dimension, fine_dimension))
        comparison = SemigroupRefinement(coarsen, fine, coarse)
        time = float(rng.uniform(0.0, 1.0))
        result = comparison.duhamel(time, quadrature_order=36)
        audit.check("semigroup_refinement", result.certificate.valid, result.certificate.message)
        audit.check("semigroup_refinement", np.linalg.norm(comparison.direct_defect(time), 2) <= comparison.norm_bound(time) + 1e-10)

    # Exact Markov aggregation and Bayesian identities.
    for _ in range(900):
        block_sizes = tuple(int(rng.integers(1, 4)) for _ in range(int(rng.integers(1, 5))))
        generator, labels = random_lumpable_generator(rng, block_sizes)
        lumping = MarkovLumping(generator, labels)
        audit.check("markov", lumping.certificate().valid)
        coarse = lumping.coarse_generator()
        audit.check("markov", np.allclose(lumping.aggregation @ generator, coarse @ lumping.aggregation, atol=1e-11))
        probability = rng.uniform(size=generator.shape[0])
        probability /= probability.sum()
        time = float(rng.uniform(0.0, 2.0))
        direct = lumping.aggregate(lumping.chain.evolve(probability, time))
        reconstructed = lumping.coarse_chain().evolve(lumping.aggregate(probability), time)
        audit.check("markov", np.allclose(direct, reconstructed, atol=2e-9, rtol=2e-9))

    for _ in range(360):
        size = int(rng.integers(2, 20))
        labels = tuple(int(value) for value in rng.integers(0, max(1, size // 3), size=size))
        partition = BayesianPartition(labels)
        prior = rng.uniform(0.1, 1.0, size=size)
        prior /= prior.sum()
        block_values = rng.uniform(0.1, 3.0, size=partition.block_count)
        likelihood = np.array([block_values[label] for label in partition.labels])
        result = partition.coarse_posterior(prior, likelihood)
        audit.check("bayesian", result.certificate.valid)
        audit.check("bayesian", partition.is_fixed_sufficient([likelihood]))
        posterior = BayesianUpdate(prior, likelihood).posterior
        audit.check("bayesian", np.allclose(partition.aggregate(posterior), result.value))
        second = rng.uniform(0.1, 1.0, size=size)
        second /= second.sum()
        entropy = relative_entropy_decomposition(prior, second, partition.labels)
        audit.check("bayesian", entropy.certificate.valid)
        audit.check("bayesian", np.isclose(entropy.value["total"], relative_entropy(prior, second)))

    # Conditional expectation packets.
    for _ in range(180):
        size = 8
        weights = rng.uniform(0.1, 1.0, size=size)
        tower = ConditionalExpectationTower(weights, [[0] * size, [0, 0, 0, 0, 1, 1, 1, 1], list(range(size))])
        audit.check("martingales", tower.certificate().valid)
        audit.check("martingales", tower.energy(rng.normal(size=size)).certificate.valid)

    # Fractional calculus and Green kernels.
    for _ in range(350):
        dimension = int(rng.integers(1, 8))
        values = rng.uniform(0.2, 8.0, size=dimension)
        alpha = float(rng.uniform(0.05, 1.8))
        matrix = np.diag(values)
        power = fractional_power(matrix, alpha)
        audit.check("fractional", np.allclose(np.diag(power), values**alpha))

    for _ in range(240):
        fine_values = rng.uniform(0.5, 7.0, size=5)
        coarse_values = fine_values[:3] + rng.normal(scale=0.05, size=3)
        coarse_values = np.maximum(coarse_values, 0.2)
        coarsen = np.c_[np.eye(3), np.zeros((3, 2))]
        alpha = float(rng.uniform(0.05, 0.95))
        result = fractional_defect(coarsen, np.diag(fine_values), np.diag(coarse_values), alpha)
        audit.check("fractional", result.certificate.valid)
        audit.check("fractional", np.linalg.norm(result.defect, 2) <= result.bound + result.certificate.tolerance)
        exact_coarse = np.diag(fine_values[:3])
        audit.check("fractional", fractional_naturality_certificate(coarsen, np.diag(fine_values), exact_coarse, alpha).valid)

    for points in range(1, 61):
        green = DirichletGreen1D(points)
        audit.check("green", green.certificate(tolerance=2e-9).valid)
        for mode in range(1, min(points, 8) + 1):
            error = (mode * np.pi) ** 2 - green.eigenvalues()[mode - 1]
            audit.check("green", -1e-10 <= error <= green.eigenvalue_error_bound(mode) + 1e-10)

    # Feynman Kac, Feshbach, memory, and phase type survival.
    for _ in range(160):
        generator, labels = random_lumpable_generator(rng, (2, 2))
        potential_values = rng.uniform(0.0, 1.0, size=2)
        potential = np.array([potential_values[label] for label in labels])
        model = FiniteFeynmanKac(generator, potential)
        first = float(rng.uniform(0.0, 1.0))
        second = float(rng.uniform(0.0, 1.0))
        audit.check("feynman_kac", np.allclose(model.operator(first) @ model.operator(second), model.operator(first + second), atol=2e-9))
        audit.check("feynman_kac", model.coarse_model(labels).certificate.valid)

    for _ in range(180):
        retained = 2
        hidden = 2
        a_pp = rng.normal(scale=0.2, size=(retained, retained))
        a_qq = -np.diag(rng.uniform(0.6, 2.0, size=hidden)) + rng.normal(scale=0.03, size=(hidden, hidden))
        a_pq = rng.normal(scale=0.2, size=(retained, hidden))
        a_qp = rng.normal(scale=0.2, size=(hidden, retained))
        matrix = np.block([[a_pp, a_pq], [a_qp, a_qq]])
        reduction = FeshbachReduction(matrix, [0, 1])
        z = complex(3.0, rng.uniform(0.1, 1.0))
        audit.check("feshbach", reduction.resolvent_certificate(z, tolerance=2e-9).valid)
        kernel = reduction.memory_kernel()
        audit.check("feshbach", np.allclose(kernel.at(0.0), a_pq @ a_qp))
        audit.check("feshbach", kernel.certificate(z, horizon=8.0, points=100).valid)

    for _ in range(24):
        first = MemoryKernel(rng.normal(size=(2, 2)), -np.diag(rng.uniform(0.5, 2.0, size=2)), rng.normal(size=(2, 2)))
        second = MemoryKernel(rng.normal(size=(2, 2)), -np.diag(rng.uniform(0.5, 2.0, size=2)), rng.normal(size=(2, 2)))
        bound = memory_kernel_error_bound(first, second, horizon=2.0, samples=81)
        high_resolution = memory_kernel_error_bound(first, second, horizon=2.0, samples=321)
        audit.check("memory", bound.certificate.valid)
        audit.check("memory", bound.value + 1e-10 >= high_resolution.certificate.residuals["quadrature_estimate"])

    for _ in range(220):
        dimension = int(rng.integers(1, 6))
        off = rng.uniform(0.0, 0.25, size=(dimension, dimension))
        np.fill_diagonal(off, 0.0)
        absorption = rng.uniform(0.1, 0.8, size=dimension)
        subgenerator = off.copy()
        for column in range(dimension):
            subgenerator[column, column] = -float(np.sum(off[:, column]) + absorption[column])
        initial = rng.uniform(size=dimension)
        model = PhaseTypeSurvival(subgenerator, initial)
        audit.check("survival", model.certificate().valid)
        audit.check("survival", np.isclose(model.survival(0.0), 1.0))
        audit.check("survival", np.isclose(model.laplace_transform(0.0), 1.0, atol=2e-10))
        time = float(rng.uniform(0.01, 3.0))
        audit.check("survival", model.survival(time) <= 1.0 + 1e-10)
        audit.check("survival", model.density(time) >= -1e-10)


    # Hardening checks for exact boundaries and optional dependency fallbacks.
    for _ in range(240):
        scalar = float(rng.uniform(-0.8, 0.8))
        off_diagonal = float(rng.uniform(-1.0, 1.0))
        matrix = np.array([[scalar, off_diagonal], [0.0, scalar]])
        expected = np.exp(scalar) * np.array([[1.0, off_diagonal], [0.0, 1.0]])
        audit.check(
            "hardening",
            np.allclose(matrix_exponential(matrix, method="taylor"), expected, atol=2e-13, rtol=2e-13),
            "NumPy only matrix exponential",
        )

    for _ in range(320):
        length = int(rng.integers(2, 35))
        kernel_length = int(rng.integers(1, 16))
        signal = rng.normal(size=length)
        kernel = rng.normal(size=kernel_length)
        full = np.convolve(signal, kernel, mode="full")
        start = (kernel_length - 1) // 2
        audit.check(
            "hardening",
            np.allclose(convolve_signal(signal, kernel, mode="same"), full[start : start + length]),
            "signal first same convention",
        )
        valid_length = max(length - kernel_length + 1, 0)
        expected_valid = full[kernel_length - 1 : kernel_length - 1 + valid_length]
        audit.check(
            "hardening",
            np.allclose(convolve_signal(signal, kernel, mode="valid"), expected_valid),
            "signal first valid convention",
        )
        audit.check(
            "hardening",
            convolution_matrix(kernel, length, mode="circular").shape == (length, length),
            "circular shape",
        )

    def reconstruct_compact(channels: tuple[np.ndarray, ...], order: int) -> np.ndarray:
        length = max(
            (order * (channel.size - 1) + residue + 1 for residue, channel in enumerate(channels)),
            default=0,
        )
        value = np.zeros(length, dtype=np.result_type(*(channel.dtype for channel in channels), float))
        for residue, channel in enumerate(channels):
            value[residue : residue + order * channel.size : order] = channel
        return value

    for _ in range(520):
        order = int(rng.integers(2, 8))
        signal = rng.normal(size=int(rng.integers(1, 30)))
        kernel = rng.normal(size=int(rng.integers(1, 18)))
        source = polyphase_analyze(signal, order).channels
        kernels = polyphase_filter_kernels(kernel, order)
        target_channels: list[np.ndarray] = []
        for target in range(order):
            terms = [
                np.convolve(source[source_index], kernels[target][source_index].coefficients)
                if source[source_index].size
                else np.zeros(1)
                for source_index in range(order)
            ]
            size = max(term.size for term in terms)
            combined = np.zeros(size)
            for term in terms:
                combined[: term.size] += term
            target_channels.append(combined)
        direct = np.convolve(signal, kernel)
        reconstructed = reconstruct_compact(tuple(target_channels), order)
        audit.check(
            "hardening",
            np.allclose(reconstructed[: direct.size], direct, atol=2e-11, rtol=2e-11),
            "infinite polyphase kernel law",
        )

    for _ in range(320):
        order = int(rng.integers(2, 8))
        left = tuple(rng.normal(size=int(rng.integers(1, 8))) for _ in range(order))
        right = tuple(rng.normal(size=int(rng.integers(1, 8))) for _ in range(order))
        result = packet_product(left, right, order)
        direct = np.convolve(reconstruct_compact(left, order), reconstruct_compact(right, order))
        audit.check("hardening", result.certificate.valid, "compact packet product certificate")
        audit.check(
            "hardening",
            np.allclose(reconstruct_compact(result.value, order)[: direct.size], direct, atol=2e-11, rtol=2e-11),
            "compact packet carry",
        )

    for _ in range(120):
        dimension = int(rng.integers(2, 7))
        samples = np.zeros((30, dimension))
        amplitude = rng.normal(size=dimension)
        amplitude = amplitude / np.linalg.norm(amplitude) * float(rng.uniform(3.0, 6.0))
        cell = int(rng.integers(5, 24))
        samples[cell + 1 :] += amplitude
        result = recover_vector_jumps_subgaussian(
            samples,
            spacing=1.0,
            lipschitz_bound=0.0,
            sigma=0.0,
            minimum_jump=float(np.linalg.norm(amplitude) * 0.8),
            failure_probability=0.05,
        )
        audit.check("hardening", result.certificate.valid and result.cells == (cell,), "vector jump recovery")

    exact_profiles = [np.array([1.0, 1.0, 2.0, 2.0]), np.array([3.0, 3.0, 4.0, 4.0])]
    exact_partition = coarsest_sufficient_partition(exact_profiles)
    audit.check("hardening", exact_partition.labels == (0, 0, 1, 1), "exact sufficient partition")
    approximate_partition = approximate_likelihood_partition([np.array([0.0, 0.09, 0.18])], tolerance=0.1)
    audit.check("hardening", approximate_partition.labels == (0, 0, 0), "approximate profile clustering")
    audit.check(
        "hardening",
        not approximate_partition.is_fixed_sufficient([np.array([0.0, 0.09, 0.18])]),
        "approximate clustering is not promoted to sufficiency",
    )

    coarse_transition = np.array([[0.8, 0.3], [0.2, 0.7]])
    transition = np.kron(coarse_transition, np.eye(2))
    filter_model = HiddenMarkovFilter(transition)
    for _ in range(160):
        prior = rng.uniform(0.1, 2.0, size=4)
        likelihood = np.array([rng.uniform(0.2, 2.0)] * 2 + [rng.uniform(0.2, 2.0)] * 2)
        result = filter_model.coarse_step(prior, likelihood, [0, 0, 1, 1])
        audit.check("hardening", result.certificate.valid, "transition based hidden Markov lumping")

    term = TruncatedPowerTerm(-1.0, 2.0, 0)
    audit.check("hardening", term.location_bound(-0.5, interval=(0.0, 1.0)) == 0.0, "finite interval step boundary")
    unstable = PhaseTypeSurvival(np.zeros((1, 1)), [1.0], validate=False)
    audit.check("hardening", not unstable.certificate().valid, "strict survival transience")

    # Source and public API integrity.
    python_files = tuple(sorted(SRC.rglob("*.py")))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            audit.check("source_integrity", True)
        except SyntaxError as exc:
            audit.check("source_integrity", False, f"{path}: {exc}")
    for package_name in ("signals", "dynamics"):
        module = __import__(f"komorebi_math.{package_name}", fromlist=["*"])
        for name in module.__all__:
            audit.check("source_integrity", hasattr(module, name), f"missing {package_name}.{name}")

    return audit.close()


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)
