"""Twelve independently checked examples from the new machinery manual.

From a source checkout with the scientific extra installed, run:
    PYTHONPATH=src python examples/release_manual_machinery.py

The assertions check these finite examples, not universal mathematical claims.
Each function's body is reproduced verbatim in docs/NEW_MACHINERY_GUIDE.md.
"""

from __future__ import annotations


def example_positive_tail() -> None:
    from decimal import Decimal
    from fractions import Fraction
    import mpmath as mp
    from komorebi_math.quadratic_forms import certified_positive_logsf

    tail = certified_positive_logsf(
        "30", ("0.25" for _ in range(32)), relative_accuracy="1e-9",
        initial_K=64, max_K=512, initial_L="10", max_L=14,
        initial_interval_dps=40, max_interval_dps=80, working_dps=120,
    )
    assert tail.certified, tail.failure_reason
    assert Decimal(tail.achieved_relative_ratio_minus_one) <= Decimal("1e-9")
    lower, upper = map(Decimal, tail.log10_interval[1:-1].split(","))

    # Q = 0.25 * chi-square(32): its survival is exp(-60)*sum(60**k/k!).
    term = total = Fraction(1)
    for k in range(1, 16):
        term *= Fraction(60, k)
        total += term
    ctx = mp.mp.clone()
    ctx.dps = 110
    reference = (-60 + ctx.log(ctx.mpf(total.numerator) / total.denominator))
    reference = Decimal(ctx.nstr(reference / ctx.log(10), 100))
    assert lower <= reference <= upper


def example_stored_factor() -> None:
    from decimal import Decimal
    import mpmath as mp
    import numpy as np
    from komorebi_math.quadratic_forms import certified_float64_factor_logsf

    result = certified_float64_factor_logsf(
        "20", np.eye(8), relative_accuracy="1e-3",
        initial_candidate_digits=25, max_candidate_digits=25, interval_dps=40,
        tail_accuracy_fraction="0.5",
        tail_kwargs=dict(initial_K=128, max_K=512, initial_L="10", max_L=12,
                         initial_interval_dps=40, max_interval_dps=60,
                         working_dps=100),
    )
    assert result.certified, result.failure_reason
    spectrum = result.spectral_enclosure
    assert spectrum.rank == 8
    for lower, upper in zip(spectrum.lower, spectrum.upper, strict=True):
        assert Decimal(lower) <= 1 <= Decimal(upper)
    # A = I gives chi-square(8): exp(-10)*(1 + 10 + 50 + 1000/6).
    ctx = mp.mp.clone()
    ctx.dps = 100
    reference = (-10 + ctx.log(ctx.mpf(683) / 3)) / ctx.log(10)
    lower, upper = map(Decimal, result.log10_interval[1:-1].split(","))
    assert lower <= Decimal(ctx.nstr(reference, 90)) <= upper
    assert Decimal(result.achieved_relative_ratio_minus_one) <= Decimal("1e-3")


def example_exact_families() -> None:
    from komorebi_math.sequences import (
        derangement_sequence, fubini_sequence, kbonacci_closed_form,
        kbonacci_sequence, reciprocal_coefficients,
        universal_denominator_coefficients,
    )

    # These are compressed secant-number coefficients, not Taylor coefficients.
    assert reciprocal_coefficients(2, 6) == (1, 1, 5, 61, 1385, 50521)
    assert universal_denominator_coefficients(2, 6) == (1, 0, -1, 0, 1, 0, -1)
    assert reciprocal_coefficients(3, 4, power=0) == (1, 0, 0, 0)
    assert fubini_sequence().first(5) == (1, 1, 3, 13, 75)
    assert derangement_sequence().first(6) == (1, 0, 1, 2, 9, 44)

    # Check the finite binomial formula against direct recurrence execution.
    recurrence = kbonacci_sequence(3).first(12)
    assert recurrence[:7] == (0, 0, 1, 1, 2, 4, 7)
    assert tuple(kbonacci_closed_form(3, n) for n in range(12)) == recurrence


def example_local_oeis() -> None:
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from komorebi_math.sequences import compare_oeis_prefix, load_local_oeis

    # A user-supplied local snapshot; this example makes no network request.
    payload = {
        "sequences": [{"id": "A000045", "terms": [1, 1, 2, 3, 5, 8],
                       "offset": 1, "name": "Fibonacci numbers"}],
        "provenance": {"purpose": "manual finite-window example"},
    }
    with TemporaryDirectory() as temporary:
        location = Path(temporary) / "snapshot.json"
        location.write_text(json.dumps(payload), encoding="utf-8")
        corpus = load_local_oeis(location, format="json")
    reference = corpus.records[0]
    matched = compare_oeis_prefix((2, 3, 5), reference,
                                  query_offset=3, minimum_terms=3)
    assert matched.passed and matched.index_alignment_verified
    exhausted = compare_oeis_prefix((5, 8, 13), reference,
                                    query_offset=5, minimum_terms=2)
    assert exhausted.agrees_on_overlap and not exhausted.passed
    assert exhausted.status == "reference-exhausted-prefix"
    assert corpus.provenance["purpose"] == "manual finite-window example"


def example_exact_special_functions() -> None:
    from fractions import Fraction as F
    from komorebi_math.special_functions import (
        bernoulli_packet_integer, q_racah_lattice, q_racah_scalar,
    )

    a = F(3, 7)
    packet_sum = sum(bernoulli_packet_integer(4, a, 3, r) for r in range(3))
    # Independent polynomial control: B_4(x) = x^4 - 2*x^3 + x^2 - 1/30.
    assert packet_sum == a**4 - 2 * a**3 + a**2 - F(1, 30)
    assert bernoulli_packet_integer(1, 0, 1, 0) == F(-1, 2)

    # A terminating q-Racah table can have repeated sites and no orthogonal rows.
    q, alpha, beta, gamma, delta = F(1, 2), F(1, 2), F(1, 2), F(4), F(1)
    assert gamma * q == q**-1
    assert tuple(q_racah_lattice(y, gamma, delta, q) for y in range(2)) == (3, 3)
    table = tuple(tuple(q_racah_scalar(n, y, alpha, beta, gamma, delta, q)
                        for y in range(2)) for n in range(2))
    assert table == ((1, 1), (1, 1))


def example_gamma_shapes() -> None:
    import mpmath as mp
    from komorebi_math.special_functions import (
        beta_binomial_refinement, gamma_sech_ambiguity,
        log_gamma_fourier, meixner_density,
    )

    ctx = mp.mp.clone()
    ctx.dps = 45
    shape, omega = ctx.mpf("1.75"), ctx.mpf("0.6")
    # Direct Fourier integral after u = exp(y), independent of the Gamma formula.
    def integrand(u):
        return ctx.exp((shape * ctx.log(u) - u) / 2) * u**(-ctx.j * omega) / u
    integral = ctx.quad(integrand, [0, ctx.mpf(".25"), 1, 4, ctx.inf])
    integral /= ctx.sqrt(2 * ctx.pi * ctx.gamma(shape))
    assert abs(log_gamma_fourier(shape, omega, dps=45) - integral) < ctx.mpf("1e-32")
    assert abs(gamma_sech_ambiguity(1, 1, 0, 0, dps=45) - 1) < ctx.mpf("1e-40")
    density = meixner_density(1, omega, dps=45)
    assert abs(density - 1 / ctx.cosh(ctx.pi * omega)) < ctx.mpf("1e-40")
    amplitudes = beta_binomial_refinement(4, "1.25", "2.75", dps=45)
    assert abs(sum(value**2 for value in amplitudes) - 1) < ctx.mpf("1e-40")


def example_fourier_jumps() -> None:
    import numpy as np
    from komorebi_math.signals.fourier_recovery import recover_fourier_jumps

    period = 3.0
    locations = np.array([0.2, 1.1, 2.4])
    amplitudes = np.array([2.0, -0.75, -1.25])
    # Normalized coefficients use 2*pi*i*n for every period.
    measurements = {
        n: np.sum(amplitudes * np.exp(-2j * np.pi * n * locations / period))
           / (2j * np.pi * n)
        for n in range(4, 14)
    }
    result = recover_fourier_jumps(measurements, background_degree=3,
                                   maximum_jumps=5, known_order=3, period=period)
    result.require_recovery()
    assert result.resolution_status == "conditional"
    assert result.order == 3
    assert np.allclose(result.packet.locations, locations, rtol=0, atol=1e-10)
    assert np.allclose(result.packet.amplitudes, amplitudes, rtol=0, atol=1e-10)
    assert np.allclose(result.predicted_moments, result.measured_moments,
                       rtol=0, atol=1e-10)


def example_no_jumps() -> None:
    import numpy as np
    from komorebi_math.signals.fourier_recovery import recover_fourier_jumps

    result = recover_fourier_jumps({n: 0j for n in range(1, 5)},
                                   background_degree=0, maximum_jumps=2, known_order=0)
    result.require_recovery()
    assert result.order == 0
    assert np.array_equal(result.packet.evaluate([0, 0.5, 3]), np.zeros(3))
    assert result.packet.fourier(0) == 0


def example_generated_congruence() -> None:
    from komorebi_math.core.semilattice import FiniteJoinSemilattice

    # Bit masks label the four subsets of a two-element set; union is bitwise OR.
    source = FiniteJoinSemilattice(
        tuple(tuple(left | right for right in range(4)) for left in range(4))
    )
    result = source.generated_congruence(((0, 1),))
    result.certificate().require()
    assert result.classes == ((0, 1), (2, 3))
    assert result.projection == (0, 0, 1, 1)
    assert result.quotient.join_table == ((0, 1), (1, 1))
    for left in range(4):
        for right in range(4):
            assert result.projection[left | right] == result.quotient.join(
                result.projection[left], result.projection[right])


def example_supplied_quotient() -> None:
    from komorebi_math.core.semilattice import FiniteJoinSemilattice

    # The join of a finite chain is maximum; a convex partition gives a quotient.
    chain = FiniteJoinSemilattice(
        tuple(tuple(max(left, right) for right in range(4)) for left in range(4))
    )
    result = chain.quotient((10, 10, -7, -7))
    result.certificate().require()
    assert result.partition == (0, 0, 2, 2)
    assert result.projection == (0, 0, 1, 1)
    assert result.classes == ((0, 1), (2, 3))
    assert result.quotient.join_table == ((0, 1), (1, 1))
    assert result.derivations is None  # This is not a leastness witness.


def example_dirac_evolution() -> None:
    import numpy as np
    from komorebi_math.quantum.dirac import dirac_propagator, dirac_symbol

    momentum, mass, time = [0.3, -0.8, 1.2], -0.4, 0.7
    hamiltonian = dirac_symbol(momentum, mass=mass)
    evolution = dirac_propagator(momentum, time=time, mass=mass)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    reference = (eigenvectors * np.exp(-1j * time * eigenvalues)) @ eigenvectors.conj().T
    assert np.allclose(evolution, reference, rtol=0, atol=2e-14)
    assert np.allclose(evolution.conj().T @ evolution, np.eye(4), rtol=0, atol=2e-14)
    energy_squared = mass**2 + sum(value**2 for value in momentum)
    assert np.allclose(hamiltonian @ hamiltonian, energy_squared * np.eye(4),
                       rtol=0, atol=2e-14)


def example_dirac_conventions() -> None:
    import numpy as np
    from komorebi_math.quantum.dirac import (
        central_mode_momentum, dirac_matrices, dirac_propagator,
    )

    matrices = dirac_matrices()
    for i, left in enumerate(matrices):
        assert not left.flags.writeable
        for j, right in enumerate(matrices):
            expected = 2 * np.eye(4) if i == j else np.zeros((4, 4))
            assert np.array_equal(left @ right + right @ left, expected)
    assert np.array_equal(central_mode_momentum([1, 0, -2]), 2 * np.pi * np.array([1, 0, -2]))
    central = central_mode_momentum([1, 0, 0], mesh=1 / 12)
    assert np.allclose(central, [12 * np.sin(np.pi / 6), 0, 0], rtol=0, atol=1e-14)
    assert np.array_equal(dirac_propagator([0, 0, 0], time=1e100, mass=0), np.eye(4))


EXAMPLES = (
    example_positive_tail,
    example_stored_factor,
    example_exact_families,
    example_local_oeis,
    example_exact_special_functions,
    example_gamma_shapes,
    example_fourier_jumps,
    example_no_jumps,
    example_generated_congruence,
    example_supplied_quotient,
    example_dirac_evolution,
    example_dirac_conventions,
)


def main() -> None:
    if not __debug__:
        raise RuntimeError("Run without -O: these examples need their assertions enabled.")
    for example in EXAMPLES:
        example()
        print(f"PASS {example.__name__}")
    print(f"{len(EXAMPLES)} worked cases passed")


if __name__ == "__main__":
    main()
